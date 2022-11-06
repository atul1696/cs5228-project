import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from src.utils import read_csv, create_submission
from src.utils import GridSearchRegressor
from src.data_preprocessor import DataPreprocessor
from src.gridsearch_config import gridsearch_config


################################ Load Dataset ###############################
trainX, trainY = read_csv('data/train.csv', ylabel='price')
testX, _ = read_csv('data/test.csv')

auxSubzone, _ = read_csv('data/auxiliary-data/sg-subzones.csv')

auxInfraDict = {}
Infralist = ['sg-commerical-centres', 'sg-mrt-stations', 'sg-primary-schools', 'sg-secondary-schools', 'sg-shopping-malls']
for ele in Infralist:
    auxInfra, _ = read_csv('data/auxiliary-data/' + ele + '.csv')
    auxInfraDict[ele] = auxInfra


###################### Data Cleaning and Preprocessing ######################
data_preprocessor = DataPreprocessor(auxSubzone, auxInfraDict)
trainX, trainY = data_preprocessor.fit_transform_for_regression(trainX, trainY)
testX = data_preprocessor.transform_for_regression(testX)
col_names = list(trainX.columns)


######################### Prepare Data for Regressor ########################
trainX, trainY, testX = trainX.astype(float).to_numpy(), trainY.astype(float).to_numpy(), testX.astype(float).to_numpy()
trainX, trainY = shuffle(trainX, trainY, random_state=0)


#################### Define Regressor and Training Setup ####################
regressor_name = 'random_forest'
gridsearch = False

if gridsearch:
    parameters = gridsearch_config[regressor_name]['parameters']
    regressor = gridsearch_config[regressor_name]['regressor']
else:
    parameters = {}
    regressor = gridsearch_config[regressor_name]['best_param_regressor']

k_fold = 10
gridsearch_regressor = GridSearchRegressor(regressor_name, regressor, parameters, k_fold)

#################### Train Model and Submit Predictions ####################
best_regressor = gridsearch_regressor.fit(trainX, trainY, col_names, print_results=True)

testY = best_regressor.predict(testX)
create_submission(testY, 'submission.csv')
