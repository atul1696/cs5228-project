from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor

gridsearch_config = {}

########################## Random Forest Regressor ##########################
regressor_name = 'random_forest'
parameters = {
    'regressor__n_estimators': [20, 50, 100, 200, 500],
    'regressor__max_depth': [5, 10, 15, 20, 50],
    'regressor__min_samples_split': [2, 10, 20, 50, 100]
}
regressor = RandomForestRegressor(random_state=0, n_jobs=-1)
# Save the best regressor parameters manually once we are done with the grid search, so we don't need to repeat it for other experiments.
# Currently just a copied default regressor
best_param_regressor = RandomForestRegressor(random_state=0, n_jobs=-1, max_depth=20, min_samples_split=2, n_estimators=200)

gridsearch_config[regressor_name] = {'parameters': parameters, 'regressor': regressor, 'best_param_regressor': best_param_regressor}
#############################################################################

########################## Decision Tree Regressor ##########################
regressor_name = 'decision_tree'
parameters = {
    'regressor__max_depth': [None, 5, 10, 15, 20, 50, 100, 200],
    'regressor__min_samples_split': [2, 10, 20, 50, 100],
    'regressor__max_features': [None, 5, 10],
    'regressor__min_samples_leaf': [1, 3, 5, 10],
    'regressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
}
regressor = DecisionTreeRegressor(random_state=0)
best_param_regressor = DecisionTreeRegressor(random_state=0, max_depth=20, min_samples_split=2, max_features=10, min_samples_leaf=3, criterion='poisson')

gridsearch_config[regressor_name] = {'parameters': parameters, 'regressor': regressor, 'best_param_regressor': best_param_regressor}
#############################################################################

########################## AdaBoost Regressor ##########################
regressor_name = 'ada_boost'
parameters = {
    'regressor__n_estimators': [100, 200, 500, 1000],
    'regressor__learning_rate': [0.001, 0.01, 0.1, 0.5, 1],
    'regressor__loss': ['linear', 'square', 'exponential']
}
regressor = AdaBoostRegressor(base_estimator=gridsearch_config['decision_tree']['best_param_regressor'], random_state=0)
best_param_regressor = AdaBoostRegressor(base_estimator=gridsearch_config['decision_tree']['best_param_regressor'], random_state=0)

gridsearch_config[regressor_name] = {'parameters': parameters, 'regressor': regressor, 'best_param_regressor': best_param_regressor}
#############################################################################

############################# Rigde Regressor ###############################
regressor_name = 'ridge'
parameters = {
    'regressor__alpha': [0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0, 100.0],
    'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
}
regressor = Ridge(random_state=0)
best_param_regressor = Ridge(random_state=0)

gridsearch_config[regressor_name] = {'parameters': parameters, 'regressor': regressor, 'best_param_regressor': best_param_regressor}
#############################################################################
