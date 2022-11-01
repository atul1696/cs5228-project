import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
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

trainX, trainY, testX = preprocess_data_for_classification(trainX, trainY, testX, auxSubzone=auxSubzone)
col_names = list(trainX.columns)

assert not trainY.isnull().values.any() # Just a check to make sure all labels are available

# Convert data to float and normalize
trainX, trainY, testX = trainX.astype(float).to_numpy(), trainY.astype(float).to_numpy(), testX.astype(float).to_numpy()

scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
scalerX.fit(trainX)
trainX, testX = scalerX.transform(trainX), scalerX.transform(testX)
scalerY.fit(trainY.reshape(-1, 1))
trainY = scalerY.transform(trainY.reshape(-1, 1)).reshape(-1)

# Random shuffling
trainX, trainY = shuffle(trainX, trainY, random_state=0)

gridsearch = False
if not gridsearch:
    regressor = DecisionTreeRegressor(random_state=0)
    kfold_iterator = create_k_fold_validation(trainX, trainY, k=10)
    rmse_arr = []
    rmse_train_arr = []
    for k, (X, Y, Xval, Yval) in enumerate(tqdm(kfold_iterator)):
        regressor.fit(X, Y)

        Ypred = regressor.predict(Xval)
        Ypred, Yval = scalerY.inverse_transform(Ypred.reshape(-1, 1)).reshape(-1), scalerY.inverse_transform(Yval.reshape(-1, 1)).reshape(-1)
        rmse = mean_squared_error(Yval, Ypred, squared=False)
        rmse_arr.append(rmse)

        Ypred_train = regressor.predict(X)
        Ypred_train, Y = scalerY.inverse_transform(Ypred_train.reshape(-1, 1)).reshape(-1), scalerY.inverse_transform(Y.reshape(-1, 1)).reshape(-1)
        rmse_train = mean_squared_error(Y, Ypred_train, squared=False)
        rmse_train_arr.append(rmse_train)

    print("Mean K-Fold Validation Error : ", np.mean(rmse_arr))
    print("Median K-Fold Validation Error : ", np.median(rmse_arr))
    print("Maximum K-Fold Validation Error : ", np.max(rmse_arr))
    print("Mean K-Fold Train Error : ", np.mean(rmse_train_arr))
    print("Median K-Fold Train Error : ", np.median(rmse_train_arr))
    print("Maximum K-Fold Train Error : ", np.max(rmse_train_arr))

else:
    # regressor = DummyRegressor(strategy="mean")
    regressor = DecisionTreeRegressor(random_state=0) # The best tuned model
    # regressor = Ridge(random_state=0, alpha=5.0, solver='lsqr', tol=0.01)
    # regressor = MLPRegressor(random_state=0, learning_rate_init=0.1, max_iter=500
    parameters = {
        'n_estimators': [20, 50, 100, 200, 500],
        'max_depth': [5, 10, 15, 20, 50],
        'min_samples_split': [2, 10, 20, 50, 100],
    }
    def rmse(Y_true, Y_pred):
        rmse = mean_squared_error(Y_true, Y_pred, squared=False)
        print("RMSE: ", rmse)
        return rmse
    regressor = GridSearchCV(RandomForestRegressor(random_state=0, n_jobs=-1), param_grid=parameters, scoring=make_scorer(rmse, greater_is_better=False), n_jobs=-1, cv=10, verbose=3)

    results = regressor.fit(trainX, trainY)

    from pprint import pprint
    pprint(regressor.cv_results_)
    pprint(regressor.best_params_)

    feature_importance = sorted(dict(zip(col_names, regressor.feature_importances_)).items(), key=lambda k: k[1], reverse=True)

regressor.fit(trainX, trainY)
testY = regressor.predict(testX)
# testY = gs_regressor.predict(testX)
testY = scalerY.inverse_transform(testY.reshape(-1, 1)).reshape(-1)

create_submission(testY, 'baseline-submission.csv')
