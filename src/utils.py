import csv
import numpy as np
import pandas as pd

from pprint import pprint, pformat
from tabulate import tabulate

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

def read_csv(filename, ylabel=None):
    df = pd.read_csv(filename)

    if ylabel is not None:
        dfY = df[ylabel]
        dfX = df.drop([ylabel], axis=1)
        return dfX, dfY
    return df, None

def create_submission(testY, filename):
    f = open(filename, 'w')
    writer = csv.writer(f)
    writer.writerow(['Id', 'Predicted'])

    for i, ele in enumerate(testY):
        writer.writerow([i, ele])

    f.close()
    return

def reverse_dict(inp_dict):
    return {v:k for k, v in inp_dict.items()}

def rmse(Y_true, Y_pred):
    rmse = mean_squared_error(Y_true, Y_pred, squared=False)
    return rmse


class GridSearchRegressor():
    def __init__(self, regressor, parameters={}, k_fold=10):
        self.k_fold = k_fold
        transformed_regressor = TransformedTargetRegressor(regressor, transformer=MinMaxScaler())
        scoring_fn = make_scorer(rmse, greater_is_better=False)
        self.gridsearch_regressor = GridSearchCV(transformed_regressor, param_grid=parameters, scoring=scoring_fn,
                                            n_jobs=-1, cv=k_fold, return_train_score=True, verbose=2)

    def fit(self, trainX, trainY, col_names=[], print_results=False):
        self.gridsearch_regressor.fit(trainX, trainY)
        if print_results:
            self.print_results(col_names)

        return self.gridsearch_regressor.best_estimator_

    def print_results(self, col_names):
        best_estimator_index = self.gridsearch_regressor.best_index_
        grid_search_results = self.gridsearch_regressor.cv_results_

        train_scores = []
        cross_validation_scores = []
        for i in range(self.k_fold):
            train_split_score_key = f'split{i}_train_score'
            train_scores.append(abs(grid_search_results[train_split_score_key][best_estimator_index]))
            test_split_score_key = f'split{i}_test_score'
            cross_validation_scores.append(abs(grid_search_results[test_split_score_key][best_estimator_index]))

        rmse_table_rows = [
            ['Minimum', np.min(train_scores), np.min(cross_validation_scores)],
            ['Mean', np.mean(train_scores), np.mean(cross_validation_scores)],
            ['Median', np.median(train_scores), np.median(cross_validation_scores)],
            ['Maximum', np.max(train_scores), np.max(cross_validation_scores)]
        ]

        rmse_table = tabulate(rmse_table_rows, headers=['Metric', 'Train', 'Validation'], tablefmt='github', floatfmt='.4f')

        base_regressor = self.gridsearch_regressor.best_estimator_.regressor_
        feature_importance = dict(sorted(zip(col_names, base_regressor.feature_importances_)), key=lambda k: k[1], reverse=True)

        print('\n'.join([
            'Best hyperparameters:',
            f'{pformat(self.gridsearch_regressor.best_params_)}',
            '',
            'RMSE score metrics:',
            f'{rmse_table}',
            '',
            'Feature importance:',
            f'{pformat(feature_importance)}'
        ]))
