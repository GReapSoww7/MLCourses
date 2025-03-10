### TUNING AN RF's HYPERPARAMETERS ###


# RF Hyperparams
-CART hyperparams
    -the Ensemble is characterized by:
    -number of estimators
    -choose bootstrap or not and other hyperparams

# TUNING IS EXPENSIVE
    -computationally expensive
    -sometimes leads to little performance improvement
-weigh the impact of tuning on the whole project to see if it is worth pursuing


# Inspect RF Hyperparams
from sklearn.ensemble import RandomForestRegressor
SEED = 1
rf = RandomForestRegressor(random_state=SEED)
print(rf.get_params())

# Optimizing Hyperparams ( GridSearchCV on auto dataset)

from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV

# define grid of hyperparams 'params_rf'
params_rf = {'n_estimators': [300, 400, 500],
             'max_depth': [4, 6, 8],
             'min_samples_leaf': [0.1, 0.2],
             'max_features': ['log2', 'sqrt']
            }

# instantiate 'grid_rf'
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)


# fit 'grid_rf' to Train Set
grid_rf.fit(X_train, y_train)
# extract best hyperparams from 'grid_rf'
best_hyperparams = grid_rf.best_params_
print('Best hyperparameters:\n', best_hyperparams)
Best hyperparameters:
    {'max_depth': 4,
     'max_features': 'log2',
     'min_samples_leaf': 0.1,
     'n_estimators': 400}


# EVAL Best model performance
# extract best model from 'grid_rf'
best_model = grid_rf.best_estimator_

# predict Test Set labels
y_pred = best_model.predict(X_test)
# eval Test Set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

Test set RMSE of rf: 3.89

    -if we had trained an UNTUNED model the RMSE would have been greater

#############################################################

# PRACTICE

# Set Hyperparam Grid of RF
# Define the dictionary 'params_rf'
params_rf = {'n_estimators': [100,350,500], 'max_features': ['log2', 'auto', 'sqrt'], 'min_samples_leaf': [2,10,30]}

# Search for Optimal Forest Tree

from sklearn.model_selection import GridSearchCV
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)


# Eval (find RMSE of best model)

from sklearn.metrics import mean_squared_error as MSE
best_model = grid_rf.best_estimator_
y_pred = best_model.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)
print('Test RMSE of best model: {:.3f}'.format(rmse_test))

Test RMSES of best model: 50.558

#############################################