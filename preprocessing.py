import numpy as np
import pandas as pd

from dataloader import remove_columns, convert_to_categorical, convert_to_continuous, convert_to_lowercase, convert_to_onehot
from dataloader import extract_unit_types, fill_lat_lng_knn, replace_corrupted_lat_lng

def preprocess_data_for_classification(trainX, trainY, testX):

    # Convert all strings to lowercase for easy processing later
    trainX, testX = convert_to_lowercase(trainX), convert_to_lowercase(testX)

    # There are 10 values for the furnishing feature set as 'na' in the training set. Changing them to 'unspecified'.
    trainX['furnishing'] = trainX['furnishing'].replace('na', 'unspecified')
    testX['furnishing'] = testX['furnishing'].replace('na', 'unspecified')

    labels_to_remove = ['listing_id', 'title', 'property_details_url', 'elevation', 'floor_level', 'address', 'property_name']
    # TODO Suggestion : Can we extract some information from the title?
    # TODO Suggestion : Can we mine some information from the url? Probably not, since the URLs lead to 404 error in most cases
    # elevation is simply 0 for all entries
    # floor_level is NaN for more than 80% of the data
    # TODO Suggestion : Maybe we can still use the floor_level information of the rest 20% examples
    # address and property_name are also dropped for now
    trainX, testX = remove_columns(trainX, col_labels=labels_to_remove), remove_columns(testX, col_labels=labels_to_remove)

    # Remove corrupted lat lng Values
    trainX, testX = replace_corrupted_lat_lng(trainX), replace_corrupted_lat_lng(testX)

    labels_to_category = ['property_type', 'tenure', 'furnishing', 'subzone', 'planning_area']
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

    # Done after filling subzone values
    labels_to_onehot = labels_to_category
    trainX = convert_to_onehot(trainX, col_labels=labels_to_onehot, category_to_int_dict=category_to_int_dict)
    testX = convert_to_onehot(testX, col_labels=labels_to_onehot, category_to_int_dict=category_to_int_dict)

    # Handling NaN values : built_year - Just provide them with the average value
    # Handling NaN values : num_beds - Just provide them with the average value
    # Handling NaN values : num_baths - Just provide them with the average value
    # Handling NaN values : total_num_units - Just provide them with the average value

    # Handling available_unit_types
    trainX, testX = extract_unit_types(trainX), extract_unit_types(testX)

    ## TODO : Temporary handling of missing entries and NaNs!! Needs to be revisited
    trainX = trainX.fillna(trainX.mean())
    testX = testX.fillna(trainX.mean())

    # pd.set_option('display.max_columns', None)
    # print(trainX.head())

    print("Training Data Shape : ", np.shape(trainX))

    return trainX, trainY, testX
