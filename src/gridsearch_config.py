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
    'regressor__min_samples_split': [2, 10, 20, 50, 100],
}
regressor = RandomForestRegressor(random_state=0)
# Save the best regressor parameters manually once we are done with the grid search, so we don't need to repeat it for other experiments.
# Currently just a copied default regressor
best_param_regressor = RandomForestRegressor(random_state=0)

gridsearch_config[regressor_name] = {'parameters': parameters, 'regressor': regressor, 'best_param_regressor': best_param_regressor}
#############################################################################

########################## Decision Tree Regressor ##########################
regressor_name = 'decision_tree'
parameters = {
    'regressor__max_depth': [5, 10, 15, 20, 50, 100, 200],
    'regressor__min_samples_split': [2, 10, 20, 50, 100],
    'regressor__max_features': [None, 5, 10],
    'regressor__min_samples_leaf': [1, 3, 5, 10],
}
regressor = DecisionTreeRegressor(random_state=0)
best_param_regressor = DecisionTreeRegressor(random_state=0)

gridsearch_config[regressor_name] = {'parameters': parameters, 'regressor': regressor, 'best_param_regressor': best_param_regressor}
#############################################################################
