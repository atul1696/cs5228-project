import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from dataloader import read_csv
from dataloader import create_k_fold_validation
from preprocessing import preprocess_data_for_classification
from kaggle_submission import create_submission

# pd.set_option('display.max_columns', None)

trainX, trainY = read_csv('data/train.csv', ylabel='price')
testX, _ = read_csv('data/test.csv')

auxSubzone, _ = read_csv('data/auxiliary-data/sg-subzones.csv')

auxInfraDict = {}
Infralist = ['sg-commerical-centres', 'sg-mrt-stations', 'sg-primary-schools', 'sg-secondary-schools', 'sg-shopping-malls']
for ele in Infralist:
    auxInfra, _ = read_csv('data/auxiliary-data/' + ele + '.csv')
    auxInfraDict[ele] = auxInfra

trainX, trainY, testX = preprocess_data_for_classification(trainX, trainY, testX, auxSubzone=auxSubzone, auxInfraDict=auxInfraDict)
col_names = list(trainX.columns)

assert not trainY.isnull().values.any() # Just a check to make sure all labels are available

# Convert data to float and normalize
trainX, trainY, testX = trainX.astype(float).to_numpy(), trainY.astype(float).to_numpy(), testX.astype(float).to_numpy()

scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
scalerX.fit(trainX)
trainX, testX = scalerX.transform(trainX), scalerX.transform(testX)

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

# regressor = DummyRegressor(strategy="mean")
regressor = DecisionTreeRegressor(random_state=0) # The best tuned model
# regressor = Ridge(random_state=0, alpha=5.0, solver='lsqr', tol=0.01)
# regressor = MLPRegressor(random_state=0, learning_rate_init=0.1, max_iter=500

parameters = {}
def rmse(Y_true, Y_pred):
    rmse = mean_squared_error(Y_true, Y_pred, squared=False)
    print("RMSE: ", rmse)
    return rmse
gridsearch_regressor = GridSearchCV(TransformedTargetRegressor(RandomForestRegressor(random_state=0, n_jobs=-1), transformer=MinMaxScaler()), param_grid=parameters, scoring=make_scorer(rmse, greater_is_better=False), n_jobs=-1, cv=10, verbose=3)

results = gridsearch_regressor.fit(trainX, trainY)

from pprint import pprint
pprint(gridsearch_regressor.cv_results_)
pprint(gridsearch_regressor.best_params_)

regressor = gridsearch_regressor.best_estimator_.regressor_

feature_importance = sorted(dict(zip(col_names, regressor.feature_importances_)).items(), key=lambda k: k[1], reverse=True)
print(feature_importance)

regressor.fit(trainX, trainY)
testY = regressor.predict(testX)
# testY = gs_regressor.predict(testX)
create_submission(testY, 'baseline-submission.csv')
