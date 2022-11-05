import numpy as np
import math
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.utils import read_csv, reverse_dict

from src.preprocessor_utils import remove_columns, convert_to_lowercase
from src.preprocessor_utils import convert_to_categorical, convert_to_onehot, get_target_encoding_dict, apply_target_encoding
from src.preprocessor_utils import extract_floor_level, extract_tenure, apply_lat_lng_knn, replace_corrupted_lat_lng, get_lat_lng_knn
from src.preprocessor_utils import append_auxiliary_data_infra, append_auxiliary_data_subzone

from sklearn.impute import KNNImputer


class DataTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, labels_to_ordinal_encode, ordinal_encoding_dict, auxiliary_infrastructure_dict):
        self.labels_to_encode = labels_to_ordinal_encode
        self.ordinal_encoding_dict = ordinal_encoding_dict
        self.auxiliary_infrastructure_dict = auxiliary_infrastructure_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = remove_columns(X, col_labels=[
            'listing_id', 'title', 'property_details_url', 'elevation', 'available_unit_types'])
        X = convert_to_lowercase(X)
        X['furnishing'] = X['furnishing'].replace('na', 'unspecified')
        X['property_type'] = X['property_type'].replace(
            r'hdb.*', 'hdb', regex=True)
        X = extract_floor_level(X)
        X = extract_tenure(X)
        X = replace_corrupted_lat_lng(X)
        for key, value in self.auxiliary_infrastructure_dict.items():
            X = append_auxiliary_data_infra(X, key, value)
        X, _ = convert_to_categorical(
            X, self.labels_to_encode, self.ordinal_encoding_dict)
        return X

class DataInverseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, ordinal_encoded_labels, one_hot_encoded_labels, ordinal_encoding_dict):
        self.ordinal_encoded_labels = ordinal_encoded_labels
        self.ordinal_encoding_dict = ordinal_encoding_dict
        self.one_hot_encoded_labels = one_hot_encoded_labels

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in self.one_hot_encoded_labels:
            one_hot_cols = list(i for i in X.columns if col in i)
            X[col] = X[one_hot_cols].idxmax(1).apply(lambda k: k.split("_")[-1])
            X = remove_columns(X, col_labels=one_hot_cols)

        for col in self.ordinal_encoded_labels:
            if col in X.columns and col not in self.one_hot_encoded_labels:
                X[col] = X[col].map(reverse_dict(self.ordinal_encoding_dict[col]))
        return X

class DataCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        data = pd.concat([X, y], axis=1)
        duplicate_index_list = data[data.duplicated()].index
        X.drop(duplicate_index_list, inplace=True)
        y.drop(duplicate_index_list, inplace=True)
        return self

    def transform(self, X, y=None):
        return X


class LatLngImputer(BaseEstimator, TransformerMixin):

    def __init__(self, ordinal_encoding_dict, feature, **kwargs):
        self.ordinal_encoding_dict = ordinal_encoding_dict
        self.feature = feature
        self.auxiliary_subzone = None
        if 'auxiliary_subzone' in kwargs:
            self.auxiliary_subzone = kwargs['auxiliary_subzone']

    def fit(self, X, y=None):
        self.feature_nan_index = self.ordinal_encoding_dict[self.feature][np.nan]
        self.knngraph_feature = get_lat_lng_knn(
            X, self.feature, self.feature_nan_index)
        return self

    def transform(self, X, y=None):
        X = apply_lat_lng_knn(
            X, self.feature, self.feature_nan_index, self.knngraph_feature)
        if self.feature == 'subzone' and self.auxiliary_subzone is not None:
            X = append_auxiliary_data_subzone(X, reverse_dict(
                self.ordinal_encoding_dict['subzone']), self.auxiliary_subzone)
        return X


class DataImputer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_property_details, knn_neighbors=7):
        self.knn_imputer = KNNImputer(
            n_neighbors=knn_neighbors, weights='distance')
        self.drop_property_details = drop_property_details

    def fit(self, X, y=None):
        self.knn_imputer.fit(X)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(self.knn_imputer.transform(X), columns=X.columns)
        labels_to_round_off = ['built_year', 'num_beds',
                               'num_baths', 'tenure_duration', 'total_num_units']
        for col in labels_to_round_off:
            X[col] = X[col].apply(np.round)

        cols = ['tenure']
        if self.drop_property_details:
            cols.extend(['address', 'property_name'])

        X = remove_columns(X, cols)

        return X


class DataOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, labels_to_onehot, ordinal_encoding_dict):
        self.ordinal_encoding_dict = ordinal_encoding_dict
        self.labels_to_onehot = labels_to_onehot

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = convert_to_onehot(X, self.labels_to_onehot,
                              self.ordinal_encoding_dict)
        return X


class DataTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, labels_to_target_encode, target_encoding_dict):
        self.labels_to_target_encode = labels_to_target_encode
        self.category_to_target_dict = target_encoding_dict

    def fit(self, X, y=None):
        self.category_to_target_dict.update(get_target_encoding_dict(
            X, y, col_labels=self.labels_to_target_encode))
        return self

    def transform(self, X, y=None):
        X = apply_target_encoding(
            X, self.labels_to_target_encode, self.category_to_target_dict)
        return X


class NanHandler(BaseEstimator, TransformerMixin):

    def __init__(self, col_names):
        self.col_names = col_names

    def fit(self, X, y=None):
        self.mean_values = X.mean()
        self.col_names.extend(X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.mean_values)

class FeatureGenerator(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def calculate_remaining_tenure(self, row):
        if not (isinstance(row['tenure_duration'], float) and math.isnan(row['tenure_duration'])):
            if row['built_year'] <= 2022:
                return (row['tenure_duration'] - (2022 - row['built_year']))

        return row['tenure_duration']

    def fit_transform(self, X, y=None):
        X['tenure_left'] = X.apply(self.calculate_remaining_tenure, axis=1)
        X['price_per_sqft'] = y / X['size_sqft']
        return X


class DataPreprocessor:

    def __init__(self, auxiliary_subzone=None, auxiliary_infrastructure_dict={}):
        self.auxiliary_subzone = auxiliary_subzone
        self.auxiliary_infrastructure_dict = auxiliary_infrastructure_dict

    def drop_outliers(self, X, y):
        # Check outliers.ipynb for more details
        index_list_to_remove = [14218, 15027, 5976, 4347, 16264, 2701, 18446, 663, 4287, 15637, 9750, 13461, 19587]
        index_list_to_remove.extend(X.index[~(
            (X['size_sqft'] > 300) & (X['size_sqft'] <= 30000))].tolist())
        index_list_to_remove.extend(
            y.index[~((y > 0) & (y <= 2 * 10 ** 8))].tolist())

        X = X.drop(index=index_list_to_remove)
        y = y.drop(index=index_list_to_remove)

        return X, y

    def fit_transform(self, X, y, drop_property_details=False, use_min_max_scaling=False):
        X, y = self.drop_outliers(X, y)
        self.ordinal_encoded_labels = ['property_type', 'subzone', 'planning_area',
                                       'tenure', 'furnishing', 'floor_level', 'address', 'property_name']
        self.ordinal_encoding_dict = {}
        self.one_hot_encoded_labels = ['furnishing', 'floor_level']
        self.target_encoded_labels = ['property_type', 'subzone', 'planning_area']
        self.target_encoding_dict = {}

        self.col_names = []

        pipeline_steps = [
            DataTransformer(self.ordinal_encoded_labels,
                            self.ordinal_encoding_dict, self.auxiliary_infrastructure_dict),
            DataCleaner(),
            LatLngImputer(self.ordinal_encoding_dict, 'subzone',
                          auxiliary_subzone=self.auxiliary_subzone),
            LatLngImputer(self.ordinal_encoding_dict, 'planning_area'),
            DataOneHotEncoder(self.one_hot_encoded_labels, self.ordinal_encoding_dict),
            DataImputer(drop_property_details),
            DataTargetEncoder(self.target_encoded_labels, self.target_encoding_dict),
            NanHandler(self.col_names),
        ]

        if use_min_max_scaling:
            pipeline_steps.append(MinMaxScaler())

        self.preprocessing_pipeline = make_pipeline(*pipeline_steps)

        X = pd.DataFrame(self.preprocessing_pipeline.fit_transform(
            X, y), columns=self.col_names)
        y = y.reset_index(drop=True)

        return X, y

    def transform(self, X):
        X = pd.DataFrame(self.preprocessing_pipeline.transform(
            X), columns=self.col_names)
        return X

    def fit_transform_for_recommendations(self, X, y, drop_property_details=False):
        X, y = self.drop_outliers(X, y)
        self.ordinal_encoded_labels = ['subzone', 'planning_area']
        self.ordinal_encoding_dict = {}
        self.one_hot_encoded_labels = []

        pipeline_steps = [
            DataTransformer(self.ordinal_encoded_labels,
                            self.ordinal_encoding_dict, self.auxiliary_infrastructure_dict),
            DataCleaner(),
            LatLngImputer(self.ordinal_encoding_dict, 'subzone',
                          auxiliary_subzone=self.auxiliary_subzone),
            LatLngImputer(self.ordinal_encoding_dict, 'planning_area'),
            DataInverseTransformer(self.ordinal_encoded_labels, self.one_hot_encoded_labels,
                                  self.ordinal_encoding_dict),
            FeatureGenerator()
        ]

        self.preprocessing_pipeline = make_pipeline(*pipeline_steps)

        X = self.preprocessing_pipeline.fit_transform(X, y).reset_index(drop=True)
        y = y.reset_index(drop=True)

        return X, y


if __name__ == '__main__':
    trainX, trainY = read_csv('data/train.csv', ylabel='price')
    col_names = list(trainX.columns)
    auxSubzone, _ = read_csv('data/auxiliary-data/sg-subzones.csv')

    auxInfraDict = {}
    Infralist = ['sg-commerical-centres', 'sg-mrt-stations', 'sg-primary-schools', 'sg-secondary-schools', 'sg-shopping-malls']
    for ele in Infralist:
        auxInfra, _ = read_csv('data/auxiliary-data/' + ele + '.csv')
    auxInfraDict[ele] = auxInfra
    dt_prep = DataPreprocessor(auxSubzone, auxInfraDict)
    trainX, trainY = dt_prep.fit_transform(trainX, trainY, drop_property_details=False)
    print(trainX.isna().sum())
    print(trainX.shape)
    print(trainY.isna().sum())
    print(trainY.shape)
