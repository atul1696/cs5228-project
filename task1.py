import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

from dataloader import read_csv
from dataloader import remove_columns, convert_to_categorical, convert_to_continuous
from dataloader import create_k_fold_validation

from submission import create_submission

trainX, trainY = read_csv('data/train.csv', ylabel='price')
testX, _ = read_csv('data/test.csv')

labels_to_remove = ['listing_id', 'title', 'property_details_url', 'elevation', 'available_unit_types']
labels_to_category = ['address', 'property_name', 'property_type', 'tenure', 'floor_level', 'furnishing', 'subzone', 'planning_area']
# labels_to_continuous = ['built_year', 'num_beds', 'num_baths', 'size_sqft', 'total_num_units', 'lat', 'lng'] # Not required right now

# listing_id, title and property_details_url are unique for every entry
# elevation is simply 0 for all entries
# available_unit_types removed for now as it will require complicated processing
trainX = remove_columns(trainX, col_labels=labels_to_remove)
testX = remove_columns(testX, col_labels=labels_to_remove)

## Currently NaN values also become their own category. Need to handle that later
trainX, category_to_int_dict = convert_to_categorical(trainX, col_labels=labels_to_category)
testX, _ = convert_to_categorical(testX, col_labels=labels_to_category, category_to_int_dict=category_to_int_dict)

## TODO : Temporary handling of missing entries and NaNs!! Needs to be revisited
trainX = trainX.fillna(trainX.mean())
testX = testX.fillna(trainX.mean())

assert not trainY.isnull().values.any() # Just a check to make sure all labels are available

trainX, trainY, testX = trainX.astype(float).to_numpy(), trainY.astype(float).to_numpy(), testX.astype(float).to_numpy()
trainX, trainY = shuffle(trainX, trainY, random_state=0)

kfold_iterator = create_k_fold_validation(trainX, trainY)

rmse_arr = []
for (X, Y, Xval, Yval) in kfold_iterator:
    regressor = DecisionTreeRegressor(random_state=0)
    regressor = regressor.fit(X, Y)
    Ypred = regressor.predict(Xval)
    rmse = mean_squared_error(Yval, Ypred, squared=False)
    rmse_arr.append(rmse)

print("Mean K-Fold Validation Error : ", np.mean(rmse_arr))
print("Median K-Fold Validation Error : ", np.median(rmse_arr))
print("Maximum K-Fold Validation Error : ", np.max(rmse_arr))

regressor = DecisionTreeRegressor(random_state=0)
regressor = regressor.fit(trainX, trainY)
testY = regressor.predict(testX)

create_submission(testY, 'baseline-submission.csv')
