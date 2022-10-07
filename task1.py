import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from dataloader import read_csv
from dataloader import remove_columns, convert_to_categorical, convert_to_continuous, convert_to_lowercase
from dataloader import extract_unit_types, fill_lat_lng_knn, replace_corrupted_lat_lng
from dataloader import create_k_fold_validation

from submission import create_submission

trainX, trainY = read_csv('data/train.csv', ylabel='price')
testX, _ = read_csv('data/test.csv')

# Convert all strings to lowercase for easy processing later
trainX, testX = convert_to_lowercase(trainX), convert_to_lowercase(testX)

labels_to_remove = ['listing_id', 'title', 'property_details_url', 'elevation', 'floor_level']
# TODO Suggestion : Can we extract some information from the title?
# TODO Suggestion : Can we mine some information from the url? Probably not, since the URLs lead to 404 error in most cases
# elevation is simply 0 for all entries
# floor_level is NaN for more than 80% of the data
# TODO Suggestion : Maybe we can still use the floor_level information of the rest 20% examples
trainX, testX = remove_columns(trainX, col_labels=labels_to_remove), remove_columns(testX, col_labels=labels_to_remove)

# Remove corrupted lat lng Values
trainX, testX = replace_corrupted_lat_lng(trainX), replace_corrupted_lat_lng(testX)

labels_to_category = ['address', 'property_name', 'property_type', 'tenure', 'furnishing', 'subzone', 'planning_area']
# TODO : property_type also contains information about types of available units, which needs to separately extracted
trainX, category_to_int_dict = convert_to_categorical(trainX, col_labels=labels_to_category)
testX, _ = convert_to_categorical(testX, col_labels=labels_to_category, category_to_int_dict=category_to_int_dict)

# These categories do not contain any NaN values : address, property_name, property_type, furnishing, lat, lng, size_sqft
# Handling NaN values : tenure - Let the NaNs be their own category

# Handling NaN values : subzone and planning_area - Use k-nearest neighbors and lat/lng values to find their subzone and planning_area
nan_index = category_to_int_dict['subzone'][np.nan]
trainX, knngraph_subzone = fill_lat_lng_knn(trainX, 'subzone', nan_index)
testX, _ = fill_lat_lng_knn(testX, 'subzone', nan_index, knngraph=knngraph_subzone)

nan_index = category_to_int_dict['planning_area'][np.nan]
trainX, knngraph_planning_area = fill_lat_lng_knn(trainX, 'planning_area', nan_index)
testX, _ = fill_lat_lng_knn(testX, 'planning_area', nan_index, knngraph=knngraph_planning_area)

# Handling NaN values : built_year - Just provide them with the average value
# Handling NaN values : num_beds - Just provide them with the average value
# Handling NaN values : num_baths - Just provide them with the average value
# Handling NaN values : total_num_units - Just provide them with the average value

# labels_to_continuous = ['built_year', 'num_beds', 'num_baths', 'size_sqft', 'total_num_units', 'lat', 'lng'] # Not required right now

# Handling available_unit_types
trainX, testX = extract_unit_types(trainX), extract_unit_types(testX)

## TODO : Temporary handling of missing entries and NaNs!! Needs to be revisited
trainX = trainX.fillna(trainX.mean())
testX = testX.fillna(trainX.mean())

assert not trainY.isnull().values.any() # Just a check to make sure all labels are available

trainX, trainY, testX = trainX.astype(float).to_numpy(), trainY.astype(float).to_numpy(), testX.astype(float).to_numpy()

scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
scalerX.fit(trainX)
trainX, testX = scalerX.transform(trainX), scalerX.transform(testX)
scalerY.fit(trainY.reshape(-1, 1))
trainY = scalerY.transform(trainY.reshape(-1, 1)).reshape(-1)

trainX, trainY = shuffle(trainX, trainY, random_state=0)

# Attempt to Add GridSearchCV. Incomplete
# regressor = DecisionTreeRegressor(random_state=0)
# parameters = {"criterion" : ["squared_error", "friedman_mse", "poisson"],
#               "max_depth" : [1, 5, 10, 20],
#               "min_samples_split" : [2, 5, 10],
#               "min_samples_leaf" : [1, 2, 5, 10]}
# gs_regressor = GridSearchCV(regressor, parameters, scoring='neg_root_mean_squared_error', verbose=1)
# gs_regressor.fit(trainX, trainY)
#
# print("Best Parameters")
# print(gs_regressor.best_params_)
# {'criterion': 'poisson', 'max_depth': 1, 'min_samples_leaf': 1, 'min_samples_split': 2}

regressor = DecisionTreeRegressor(random_state=0)
# regressor = DecisionTreeRegressor(random_state=0, criterion='poisson', max_depth=1)
# regressor = LinearRegression()

kfold_iterator = create_k_fold_validation(trainX, trainY, k=10)
rmse_arr = []
for (X, Y, Xval, Yval) in tqdm(kfold_iterator):
    regressor.fit(X, Y)
    Ypred = regressor.predict(Xval)
    rmse = mean_squared_error(Yval, Ypred, squared=False)
    rmse_arr.append(rmse)

print("Mean K-Fold Validation Error : ", np.mean(rmse_arr))
print("Median K-Fold Validation Error : ", np.median(rmse_arr))
print("Maximum K-Fold Validation Error : ", np.max(rmse_arr))

regressor.fit(trainX, trainY)
testY = regressor.predict(testX)
# testY = gs_regressor.predict(testX)
testY = scalerY.inverse_transform(testY.reshape(-1, 1)).reshape(-1)

create_submission(testY, 'baseline-submission.csv')
