import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from dataloader import append_auxiliary_data_infra, append_auxiliary_data_subzone, read_csv, remove_columns, convert_to_categorical, convert_to_lowercase, convert_to_onehot, get_ordinal_encoding_dict, reverse_dict
from dataloader import extract_floor_level, extract_tenure, apply_lat_lng_knn, replace_corrupted_lat_lng, get_lat_lng_knn, apply_ordinal_encoding, get_target_encoding_dict, apply_target_encoding

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
    def __init__(self, knn_neighbors=7):
        self.knn_imputer = KNNImputer(
            n_neighbors=knn_neighbors, weights='distance')

    def fit(self, X, y=None):
        self.knn_imputer.fit(X)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(self.knn_imputer.transform(X), columns=X.columns)
        labels_to_round_off = ['built_year', 'num_beds',
                               'num_baths', 'tenure_duration', 'total_num_units']
        for col in labels_to_round_off:
            X[col] = X[col].apply(np.round)

        X = remove_columns(X, ['address', 'property_name', 'tenure'])

        return X


class DataOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, ordinal_encoding_dict):
        self.ordinal_encoding_dict = ordinal_encoding_dict
        self.labels_to_onehot = ['furnishing', 'floor_level']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = convert_to_onehot(X, self.labels_to_onehot,
                              self.ordinal_encoding_dict)
        return X


class DataTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.labels_to_target_encode = [
            'property_type', 'subzone', 'planning_area']

    def fit(self, X, y=None):
        self.category_to_target_dict = get_target_encoding_dict(
            X, y, col_labels=self.labels_to_target_encode)
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


class DataPreprocessor:

    def __init__(self, auxiliary_subzone=None, auxiliary_infrastructure_dict={}):
        self.auxiliary_subzone = auxiliary_subzone
        self.auxiliary_infrastructure_dict = auxiliary_infrastructure_dict

    # Check outliers.ipynb for more details
    def drop_outliers(self, X, y):
        index_list_to_remove = []
        index_list_to_remove.extend(
            [14218, 15027, 5976, 4347, 16264, 2701, 18446, 663, 4287, 15637, 9750, 13461, 19587])
        index_list_to_remove.extend(X.index[~(
            (X['size_sqft'] > 300) & (X['size_sqft'] <= 30000))].tolist())
        index_list_to_remove.extend(
            y.index[~((y > 0) & (y <= 2 * 10 ** 8))].tolist())

        X = X.drop(index=index_list_to_remove)
        y = y.drop(index=index_list_to_remove)

        return X, y

    def fit_transform(self, X, y):
        X, y = self.drop_outliers(X, y)
        self.ordinal_encoded_labels = ['property_type', 'subzone', 'planning_area',
                                       'tenure', 'furnishing', 'floor_level', 'address', 'property_name']
        self.ordinal_encoding_dict = {}
        self.col_names = []

        self.preprocessing_pipeline = make_pipeline(
            DataTransformer(self.ordinal_encoded_labels,
                            self.ordinal_encoding_dict, self.auxiliary_infrastructure_dict),
            LatLngImputer(self.ordinal_encoding_dict, 'subzone',
                          auxiliary_subzone=self.auxiliary_subzone),
            LatLngImputer(self.ordinal_encoding_dict, 'planning_area'),
            DataOneHotEncoder(self.ordinal_encoding_dict),
            DataImputer(),
            DataTargetEncoder(),
            NanHandler(self.col_names),
            # MinMaxScaler()
        )

        X = pd.DataFrame(self.preprocessing_pipeline.fit_transform(
            X, y), columns=self.col_names)
        return X, y

    def transform(self, X):
        X = pd.DataFrame(self.preprocessing_pipeline.transform(
            X), columns=self.col_names)
        return X


if __name__ == '__main__':
    trainX, trainY = read_csv('data/train.csv', ylabel='price')
    testX, _ = read_csv('data/test.csv')
    col_names = list(trainX.columns)
    dt_prep = DataPreprocessor()
    trainX = dt_prep.fit_transform(trainX, trainY)
    testX = dt_prep.transform(testX)
    print(trainX.head())
    print(testX.head())
