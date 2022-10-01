import numpy as np
import pandas as pd
import math

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

def convert_to_continuous(df, col_labels=[], max_min_dict={}):
    for col in col_labels:
        df = df.fillna(df.mean()) # Temporary handling of NaN values

        if col not in max_min_dict:
            max, min = df[col].max(), df[col].min()
            max_min_dict[col] = (max, min)

        max, min = max_min_dict[col]
        df[col] = (df[col] - min)/(max - min)

    return df, max_min_dict

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

def create_k_fold_validation(X, Y, k=10):
    chunk_size = len(X)//k
    for ite in range(k):
        trainX = np.concatenate((X[:ite*chunk_size], X[(ite+1)*chunk_size:]), axis=0)
        trainY = np.concatenate((Y[:ite*chunk_size], Y[(ite+1)*chunk_size:]), axis=0)
        valX = X[ite*chunk_size:(ite+1)*chunk_size]
        valY = Y[ite*chunk_size:(ite+1)*chunk_size]

        yield trainX, trainY, valX, valY
