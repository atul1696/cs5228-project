import numpy as np
import pandas as pd
import math

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

def read_csv(filename, ylabel=None):
    df = pd.read_csv(filename)

    if ylabel is not None:
        dfY = df[ylabel]
        dfX = df.drop([ylabel], axis=1)
        return dfX, dfY
    return df, None

def remove_columns(df, col_labels=[]):
    df = df.drop(col_labels, axis=1)
    return df

def convert_to_categorical(df, col_labels=[], category_to_int_dict={}):
    for col in col_labels:
        categories = df[col].unique()
        if col not in category_to_int_dict:
            category_to_int_dict[col] = {name: n for n, name in enumerate(categories)}

        category_to_int = category_to_int_dict[col]
        df[col] = df[col].map(category_to_int)

    return df, category_to_int_dict

def use_target_encoding(df, target, col_labels=[], category_to_int_dict={}):
    for col in col_labels:
        if col not in category_to_int_dict:
            category_to_int_dict[col] = {name: np.mean(np.take(target, indices)) for name, indices in df.groupby(col).indices.items()}

        category_to_int = category_to_int_dict[col]
        df[col] = df[col].map(category_to_int, na_action='ignore')

    return df, category_to_int_dict

def convert_to_onehot(df, col_labels=[], category_to_int_dict={}):
    for col in col_labels:
        if col not in category_to_int_dict:
            raise "Category labels expected here. Should have been filled during categorical conversion"

        for ele in category_to_int_dict[col]:
            # if isinstance(ele, float) and math.isnan(ele):
            #     continue
            column_name = col + '_' + str(ele)
            df[column_name] = (df[col] == category_to_int_dict[col][ele])

        df = df.drop(col,axis = 1)

    return df

def convert_to_continuous(df, col_labels=[], max_min_dict={}):
    for col in col_labels:
        df = df.fillna(df.mean()) # Temporary handling of NaN values

        if col not in max_min_dict:
            max, min = df[col].max(), df[col].min()
            max_min_dict[col] = (max, min)

        max, min = max_min_dict[col]
        df[col] = (df[col] - min)/(max - min)

    return df, max_min_dict

def convert_to_lowercase(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.lower()

    return df

def extract_unit_types(df):

    def extract_units_arr(inp_str):
        if isinstance(inp_str, float) and math.isnan(inp_str):
            return []

        raw_unit_types_arr = inp_str.split(',')
        unit_types_arr = []
        for ele in raw_unit_types_arr:
            if ele=='studio':
                unit_types_arr.append(ele)
                continue

            ele = ele.replace("br", "")
            unit_types_arr.append(int(ele))

        return unit_types_arr

    df['available_unit_types'] = df['available_unit_types'].apply(lambda x: extract_units_arr(x))

    def check_membership(inp, list):
        if inp in list:
            return True
        else:
            return False

    for i in range(1, 11):
        column_name = str(i) + ' br'
        df[column_name] = df['available_unit_types'].apply(lambda x: check_membership(i, x))
    df['studio'] = df['available_unit_types'].apply(lambda x: check_membership('studio', x))

    df = df.drop(['available_unit_types'], axis=1)
    return df

def extract_floor_level(df):

    def remove_additional_info(inp_str):
        if isinstance(inp_str, float) and math.isnan(inp_str):
            return inp_str

        return inp_str.split(" ")[0]

    df['floor_level'] = df['floor_level'].apply(lambda x: remove_additional_info(x))

    return df

def extract_tenure(df):

    def remove_additional_info(inp_str):
        if isinstance(inp_str, float) and math.isnan(inp_str):
            return inp_str

        if "freehold" in inp_str:
            return -1
        else:
            return inp_str.split("-")[0]

    df['tenure_duration'] = df['tenure'].apply(lambda x: remove_additional_info(x))
    df['is_freehold'] = (df['tenure'].str.contains("freehold")) & (~df['tenure'].isna())

    return df

def replace_corrupted_lat_lng(df):
    corrupted_lat_lng_dict = {(14.4848138, 121.0232316) : (1.316519, 103.857510),
                              (38.9427759, -77.06536425) : (1.312767, 103.886961),
                              (69.4867678, 20.1844341) : (1.314094, 103.806833)}

    def replace_corrupted_values(x, dict):
        if x in dict:
            return dict[x]
        return x

    corrupted_lat_dict = {k[0] : v[0] for k, v in corrupted_lat_lng_dict.items()}
    corrupted_lng_dict = {k[1] : v[1] for k, v in corrupted_lat_lng_dict.items()}
    df['lat'] = df['lat'].apply(lambda x: replace_corrupted_values(x, corrupted_lat_dict))
    df['lng'] = df['lng'].apply(lambda x: replace_corrupted_values(x, corrupted_lng_dict))

    return df

def fill_lat_lng_knn(df, col_label, nan_index, knngraph=None):

    lat = df['lat'].astype(float).to_numpy()
    lng = df['lng'].astype(float).to_numpy()
    target = df[col_label].astype(float).to_numpy()

    if knngraph is None:
        lat_cleaned = lat[target!=nan_index]
        lng_cleaned = lng[target!=nan_index]
        target_cleaned = target[target!=nan_index]

        knngraph = KNeighborsClassifier(n_neighbors=5)
        knngraph.fit(np.stack([lat_cleaned, lng_cleaned], axis=-1), target_cleaned)

    lat_nan = lat[target==nan_index]
    lng_nan = lng[target==nan_index]

    pred_target = knngraph.predict(np.stack([lat_nan, lng_nan], axis=-1))

    target[target==nan_index] = pred_target
    df[col_label] = target

    return df, knngraph

def append_auxiliary_data_subzone(df, int_to_category_dict, aux):
    def subzone_map(inp, col_str):
        if isinstance(inp, float) and math.isnan(inp):
            return inp
        row_index = aux.index[aux['name'] == int_to_category_dict[inp]].tolist()
        return aux[col_str].iloc[row_index[0]]

    df['subzone_area_size'] = df['subzone'].apply(lambda x: subzone_map(x, 'area_size'))

    return df

def append_auxiliary_data_infra(df, col_name, aux):
    lat = aux['lat'].astype(float).to_numpy()
    lng = aux['lng'].astype(float).to_numpy()
    knngraph = NearestNeighbors(n_neighbors=1)
    knngraph.fit(np.stack([lat, lng], axis=-1), np.ones(lat.shape))

    lat_df = df['lat'].astype(float).to_numpy()
    lng_df = df['lng'].astype(float).to_numpy()

    distances, indices = knngraph.kneighbors(np.stack([lat_df, lng_df], axis=-1))
    df['nearest_' + col_name] = distances

    return df

def create_k_fold_validation(X, Y, k=10):
    chunk_size = len(X)//k
    for ite in range(k):
        trainX = np.concatenate((X[:ite*chunk_size], X[(ite+1)*chunk_size:]), axis=0)
        trainY = np.concatenate((Y[:ite*chunk_size], Y[(ite+1)*chunk_size:]), axis=0)
        valX = X[ite*chunk_size:(ite+1)*chunk_size]
        valY = Y[ite*chunk_size:(ite+1)*chunk_size]

        yield trainX, trainY, valX, valY
