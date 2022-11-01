import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from dataloader import read_csv, remove_columns, convert_to_categorical, convert_to_lowercase, convert_to_onehot, get_ordinal_encoding_dict
from dataloader import extract_floor_level, extract_tenure, apply_lat_lng_knn, replace_corrupted_lat_lng, get_lat_lng_knn, apply_ordinal_encoding, get_target_encoding_dict, apply_target_encoding

from sklearn.impute import KNNImputer


class DataTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, labels_to_ordinal_encode, ordinal_encoding_dict):
        self.labels_to_encode = labels_to_ordinal_encode
        self.ordinal_encoding_dict = ordinal_encoding_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = remove_columns(X, col_labels=[
            'listing_id', 'title', 'property_details_url', 'elevation', 'available_unit_types'])
        X = convert_to_lowercase(X)
        X['furnishing'] = X['furnishing'].replace('na', 'unspecified')
        X['property_type'] = X['property_type'].replace(r'hdb.*', 'hdb', regex=True)
        X = extract_floor_level(X)
        X = extract_tenure(X)
        X = replace_corrupted_lat_lng(X)
        X, _ = convert_to_categorical(X, self.labels_to_encode, self.ordinal_encoding_dict)
        return X


class LatLngImputer(BaseEstimator, TransformerMixin):

    def __init__(self, ordinal_encoding_dict, feature):
        self.ordinal_encoding_dict = ordinal_encoding_dict
        self.feature = feature

    def fit(self, X, y=None):
        self.feature_nan_index = self.ordinal_encoding_dict[self.feature][np.nan]
        self.knngraph_feature = get_lat_lng_knn(X, self.feature, self.feature_nan_index)
        return self

    def transform(self, X, y=None):
        X = apply_lat_lng_knn(
            X, self.feature, self.feature_nan_index, self.knngraph_feature)
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
                               'num_baths', 'tenure', 'total_num_units']
        for col in labels_to_round_off:
            X[col] = X[col].apply(np.ceil)

        X = remove_columns(X, ['address', 'property_name'])

        return X


class DataEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, ordinal_encoding_dict, col_names=[]):
        self.ordinal_encoding_dict = ordinal_encoding_dict
        self.labels_to_onehot = ['furnishing', 'floor_level']
        self.labels_to_target_encode = ['subzone', 'planning_area']
        self.col_names = col_names

    def fit(self, X, y=None):
        self.category_to_target_dict = get_target_encoding_dict(
            X, y, col_labels=self.labels_to_target_encode)

        groups = X.groupby(['property_type', 'num_beds', 'num_baths', 'subzone']).indices
        groups.update(X.groupby(['property_type', 'subzone']).indices)
        groups.update(X.groupby(['property_type']).indices)
        self.property_type_encoding_dict = {
            group: np.mean(np.take(y, index_list)) for group, index_list in groups.items()
        }
        return self

    def property_type_map(self, row):
        key1 = (row['property_type'], row['num_beds'],
                row['num_baths'], row['subzone'])
        key2 = (row['property_type'], row['subzone'])
        if key1 in self.property_type_encoding_dict:
            return self.property_type_encoding_dict[key1]
        elif key2 in self.property_type_encoding_dict:
            return self.property_type_encoding_dict[key2]
        else:
            return self.property_type_encoding_dict[row['property_type']]

    def transform(self, X, y=None):
        X = convert_to_onehot(X, self.labels_to_onehot,
                              self.ordinal_encoding_dict)
        X['property_type'] = X.apply(
            lambda r: self.property_type_map(r), axis=1)
        X = apply_target_encoding(
            X, self.labels_to_target_encode, self.category_to_target_dict)
        self.col_names.update(set(X.columns))
        return X


class NanHandler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.mean_values = X.mean()
        return self

    def transform(self, X, y=None):
        return X.fillna(self.mean_values)


class DataPreprocessor:

    def drop_outliers(self, X, y):
        index_list_to_remove = []
        index_list_to_remove.extend(X.index[~(
            (X['size_sqft'] > 300))].tolist())
        index_list_to_remove.extend(
            y.index[~((y > 0) & (y < 2 * 10 ** 8))].tolist())
        index_list_to_remove.extend([14218, 15027, 4347, 663, 19587, 13461])

        X.drop(index=index_list_to_remove, inplace=True)
        y.drop(index=index_list_to_remove, inplace=True)

    def fit_transform(self, X, y):
        self.drop_outliers(X, y)
        self.ordinal_encoded_labels = ['property_type', 'subzone', 'planning_area',
                                       'tenure', 'furnishing', 'floor_level', 'address', 'property_name']
        self.ordinal_encoding_dict = {}
        self.col_names = set()

        self.preprocessing_pipeline = make_pipeline(
            DataTransformer(self.ordinal_encoded_labels, self.ordinal_encoding_dict), 
            LatLngImputer(self.ordinal_encoding_dict, 'subzone'), 
            LatLngImputer(self.ordinal_encoding_dict, 'planning_area'), 
            DataImputer(), 
            DataEncoder(self.ordinal_encoding_dict, self.col_names), 
            NanHandler(), 
            MinMaxScaler()
        )

        X = pd.DataFrame(self.preprocessing_pipeline.fit_transform(X, y), columns=sorted(self.col_names))
        return X

    def transform(self, X):
        X = pd.DataFrame(self.preprocessing_pipeline.transform(X), columns=sorted(self.col_names))
        return X


if __name__ == "__main__":
    trainX, trainY = read_csv('data/train.csv', ylabel='price')
    testX, _ = read_csv('data/test.csv')
    col_names = list(trainX.columns)
    dt_prep = DataPreprocessor()
    trainX = dt_prep.fit_transform(trainX, trainY)
    testX = dt_prep.transform(testX)
    print(trainX.head())
    print(testX.head())
