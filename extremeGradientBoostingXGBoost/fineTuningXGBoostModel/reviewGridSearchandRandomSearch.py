### REVIEW OF GRID SEARCH AND RANDOM SEARCH ###
-how to find optimal hyperparam vals simulataneously? > leading to lowest loss; when their values interact in non-obvious and non-linear ways
    -GridSearch and RandomSearch
# Grid Search
-exhaustively searching through a collection of given hyperparams
    -once per set of hyperparams
-num of models = num of distinct values per hyper param MULTIPLIED across each hyperparam

-pick FINAL model hyperparam vals that give BEST Cross-Validated Eval METRIC val


# Grid Search Example

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
housing_data = pd.read_csv('ames_housing_trimmed_processed.csv')
X, y = housing_data[housing_data.columns.tolist()[:-1]], housing_data[housing_data.columns.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X, label=y)
gbm_param_grid = {'learning_rate': [0.01, 0.1, 0.5, 0.9], 'n_estimators': [200], 'subsample': [0.3, 0.5, 0.9]}
gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, scoring='neg_mean_squared_error', cv=4, verbose=1)
grid_mse.fit(X, y)
print('Best params found: ', grid_mse.best_params_)
print('Loweest rmse found: ', np.sqrt(np.abs(grid_mse.best_score_)))

Best params found: {'learning_rate': 0.1, 'n_estimators': 200, 'subsample': 0.5}
Lowest rmse found: 28530.1829341


# Random Search
-Create a (possibly infinite) range of hyperparam vals PER hyperparam that we want to search over
    -num of models doesn't grow
-set the num of iterations we would like for the Random Search to continue
    -we decide how many models (iterations) we want to try before stop
-during EACH iteration, RANDOMLY draw a val in the range of specified vals for EACH hyperparam searched over & train/eval a model with those hyperparams

-after reaching MAX num of iterations, SELECT the hyperparam CONFIG with BEST eval score


# Random Search Example

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
housing_data = pd.read_csv('ames_housing_trimmed_processed.csv')
X, y = housing_data[housing_data.columns.tolist()[:-1]], housing_data[housing_data.columsn.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X, label=y)
gbm_param_grid = {'learning_rate': np.arrange(0.05, 1.05, 0.05), 'n_estimators': [200], 'subsample': np.arrange(0.05, 1.05, 0.05)}
gbm = xgb.XGBRegressor()
randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid, n_iter=25, scoring='neg_mean_squared_error', cv=4, verbose=1)
randomized_mse.fit(X, y)
print(f'Best params found: ', randomized_mse.best_params_)
print(f'Lowest rmse found: ', np.sqrt(np.abs(randomized_mse.best_score_)))

Best params found: {'subsample': 0.60000000000000009, 'n_estimators': 200, 'learning_rate': 0.20000000000000001}
Lowest rmse found: 28300.2374291
####################################

# PRACTICE

# Grid Search with XGBoost

Create a parameter grid called gbm_param_grid that contains a list of "colsample_bytree" values (0.3, 0.7), 
a list with a single value for "n_estimators" (50), and a list of 2 "max_depth" (2, 5) values.

Instantiate an XGBRegressor object called gbm.

Create a GridSearchCV object called grid_mse, passing in: 
the parameter grid to param_grid, the XGBRegressor to estimator, "neg_mean_squared_error" to scoring, and 4 to cv. 
Also specify verbose=1 so you can better understand the output.

Fit the GridSearchCV object to X and y.
Print the best parameter values and lowest RMSE, using the .best_params_ and .best_score_ attributes, respectively, of grid_mse.


gbm_param_grid = {'colsample_bytree': [0.3, 0.7], 'n_estimators': [50], 'max_depth': [2, 5]}
gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, scoring='neg_mean_squared_error', cv=4, verbose=1)
grid_mse.fit(X, y)
print(f'Best params found: ', grid_mse.best_params_)
print(f'Lowest rmse found: ', np.sqrt(np.abs(grid_mse.best_score_)))

<script.py> output:
    Fitting 4 folds for each of 4 candidates, totalling 16 fits
    Best parameters found:  {'colsample_bytree': 0.3, 'max_depth': 5, 'n_estimators': 50}
    Lowest RMSE found:  29916.017850830365
###########################################################

# Random Search with XGBoost
Create a parameter grid called gbm_param_grid that contains a list with a single value for 'n_estimators' (25), 
and a list of 'max_depth' values between 2 and 11 for 'max_depth' - use range(2, 12) for this.

Create a RandomizedSearchCV object called randomized_mse, passing in: 
the parameter grid to param_distributions, the XGBRegressor to estimator, "neg_mean_squared_error" to scoring, 5 to n_iter, and 4 to cv. 

Also specify verbose=1 so you can better understand the output.
Fit the RandomizedSearchCV object to X and y.

gbm_param_grid = {'n_estimators': [25], 'max_depth': range(2, 12)}
gbm = xgb.XGBRegressor(n_estimators=10)
randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid, scoring='neg_mean_squared_error', n_iter=5, cv=4, verbose=1)
randomized_mse.fit(X, y)
print(f'Best params found: ', randomized_mse.best_params_)
print(f'Lowest RMSE found: ', np.sqrt(np.abs(randomized_mse.best_score_)))


<script.py> output:
    Fitting 4 folds for each of 5 candidates, totalling 20 fits
    Best parameters found:  {'n_estimators': 25, 'max_depth': 6}
    Lowest RMSE found:  31412.365221128253
####################################################