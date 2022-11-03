import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from dataloader import read_csv
from dataloader import create_k_fold_validation
from kaggle_submission import create_submission

from preprocessing import DataPreprocessor

from pprint import pprint, pformat
from tabulate import tabulate

# pd.set_option('display.max_columns', None)

trainX, trainY = read_csv('data/train.csv', ylabel='price')
testX, _ = read_csv('data/test.csv')

auxSubzone, _ = read_csv('data/auxiliary-data/sg-subzones.csv')

auxInfraDict = {}
Infralist = ['sg-commerical-centres', 'sg-mrt-stations', 'sg-primary-schools', 'sg-secondary-schools', 'sg-shopping-malls']
for ele in Infralist:
    auxInfra, _ = read_csv('data/auxiliary-data/' + ele + '.csv')
    auxInfraDict[ele] = auxInfra

data_preprocessor = DataPreprocessor(auxSubzone, auxInfraDict)
trainX, trainY = data_preprocessor.fit_transform(trainX, trainY, drop_property_details=True, use_min_max_scaling=True)
testX = data_preprocessor.transform(testX)
col_names = list(trainX.columns)

assert not trainY.isnull().values.any() # Just a check to make sure all labels are available

# Convert data to float and normalize
trainX, trainY, testX = trainX.astype(float).to_numpy(), trainY.astype(float).to_numpy(), testX.astype(float).to_numpy()

# Random shuffling
trainX, trainY = shuffle(trainX, trainY, random_state=0)

gridsearch = True
if gridsearch:
    parameters = {
        'n_estimators': [20, 50, 100, 200, 500],
        'max_depth': [5, 10, 15, 20, 50],
        'min_samples_split': [2, 10, 20, 50, 100],
    }
else:
    parameters = {}

# regressor = DummyRegressor(strategy='mean')
regressor = DecisionTreeRegressor(random_state=0) # The best tuned model
# regressor = Ridge(random_state=0, alpha=5.0, solver='lsqr', tol=0.01)
# regressor = MLPRegressor(random_state=0, learning_rate_init=0.1, max_iter=500

parameters = {}

def rmse(Y_true, Y_pred):
    rmse = mean_squared_error(Y_true, Y_pred, squared=False)
    return rmse

k_fold = 10
gridsearch_regressor = GridSearchCV(TransformedTargetRegressor(RandomForestRegressor(random_state=0, n_jobs=-1), transformer=MinMaxScaler()), param_grid=parameters, scoring=make_scorer(rmse, greater_is_better=False), n_jobs=-1, cv=k_fold, return_train_score=True, verbose=2)
gridsearch_regressor.fit(trainX, trainY) 

def print_results(gridsearch_regressor):
    best_estimator_index = gridsearch_regressor.best_index_
    grid_search_results = gridsearch_regressor.cv_results_
    k_fold = gridsearch_regressor.cv

    train_scores = []
    cross_validation_scores = []
    for i in range(k_fold):
        train_split_score_key = f'split{i}_train_score'
        train_scores.append(abs(grid_search_results[train_split_score_key][best_estimator_index]))
        test_split_score_key = f'split{i}_test_score'
        cross_validation_scores.append(abs(grid_search_results[test_split_score_key][best_estimator_index]))

    rmse_table_rows = [
        ['Minimum', np.min(train_scores), np.min(cross_validation_scores)],
        ['Mean', np.mean(train_scores), np.mean(cross_validation_scores)],
        ['Median', np.median(train_scores), np.median(cross_validation_scores)],
        ['Maximum', np.max(train_scores), np.max(cross_validation_scores)]
    ]

    rmse_table = tabulate(rmse_table_rows, headers=['Metric', 'Train', 'Validation'], tablefmt='github', floatfmt='.4f')
    base_regressor = gridsearch_regressor.best_estimator_.regressor_
    feature_importance = dict(sorted(zip(col_names, base_regressor.feature_importances_)), key=lambda k: k[1], reverse=True)

    print('\n'.join([
        'Best hyperparameters:',
        f'{pformat(gridsearch_regressor.best_params_)}',
        '',
        'RMSE score metrics:',
        f'{rmse_table}',
        '',
        'Feature importance:',
        f'{pformat(feature_importance)}'
    ]))

print_results(gridsearch_regressor)

best_regressor = gridsearch_regressor.best_estimator_
testY = best_regressor.predict(testX)
# testY = gs_regressor.predict(testX)
create_submission(testY, 'baseline-submission.csv')
