import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from dataloader import read_csv
from dataloader import create_k_fold_validation
from preprocessing import preprocess_data_for_classification
from kaggle_submission import create_submission

# pd.set_option('display.max_columns', None)

trainX, trainY = read_csv('data/train.csv', ylabel='price')
testX, _ = read_csv('data/test.csv')

trainX, trainY, testX = preprocess_data_for_classification(trainX, trainY, testX)
col_names = list(trainX.columns)

print(col_names)

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
if gridsearch:
    # estimator = DecisionTreeRegressor(random_state=0)
    # parameters = {"criterion" : ["squared_error", "friedman_mse", "poisson"],
    #               "max_depth" : [None, 1, 5, 10, 20],
    #               "min_samples_split" : [2, 5, 10],
    #               "min_samples_leaf" : [1, 2, 5, 10]}
    estimator = Ridge(random_state=0)
    parameters = {"alpha" : [0.1, 0.5, 1., 2., 5.],
                  "tol" : [1e-2, 1e-3, 1e-4],
                  "solver" : ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
    regressor = GridSearchCV(estimator, parameters, scoring='neg_root_mean_squared_error', verbose=4)
    regressor.fit(trainX, trainY)
    print("Best Hypereparameter Setting")
    print(regressor.best_params_)
    exit()
else:
    # regressor = DummyRegressor(strategy="mean")
    regressor = DecisionTreeRegressor(random_state=0) # The best tuned model
    # regressor = Ridge(random_state=0, alpha=5.0, solver='lsqr', tol=0.01)
    # regressor = MLPRegressor(random_state=0, learning_rate_init=0.1, max_iter=500)


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

regressor.fit(trainX, trainY)
testY = regressor.predict(testX)
# testY = gs_regressor.predict(testX)
testY = scalerY.inverse_transform(testY.reshape(-1, 1)).reshape(-1)

create_submission(testY, 'baseline-submission.csv')
