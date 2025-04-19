### Gradient Boosting Flavors ###


# Variations of Gradient Boosting / Gradient Boosting Algorithms:

-XGBoost
    -optimized or Distributed Computing
    -Parallel training by nature
    -scalable, portable, and accurate

-LightGBM
    -faster training, more efficient
    -lighter in terms of memory
    -optimized for parallel and GPU processing
    -useful for big dataset problems with resource constraints

-Categorical Boosting
    -open-source (Yandex)
    -built-in handling of cat features
    -accurate and robust
    -fast and scalable
    -user-friendly API

############################

# PRACTICE

# Movie Revenue Prediction with CatBoost

Will CatBoost beat AdaBoost? We'll try to use a similar set of parameters to have a fair comparison.

Recall that these are the features we have used so far: 'budget', 'popularity', 'runtime', 'vote_average', and 'vote_count'.

catboost has been imported for you as cb.

OBS: be careful not to use a classifier, or your session might expire!

Build and fit a CatBoostRegressor using 100 estimators, a learning rate of 0.1, and a max depth of 3.
Calculate the predictions for the test set and print the RMSE.


import catboost as cb
reg_cat = cb.CatBoostRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=500)
reg_cat.fit(X_train, y_train)
pred = reg_cat.predict(X_test)
rmse_cat = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE (CatBoost): {:.3f}'.format(rmse_cat))
RMSE (CatBoost): 5.115

-This made improvement compared to the RMSE of the AdaBoost model

###########################

# Boosting Context: LightGBM vs XGBoost

CatBoost is highly recommended when there are categorical features. In this case, all features are numeric, 
therefore one of the other approaches might produce better results.

As we are building regressors, we'll use an additional parameter, objective, which specifies the learning function to be used.
To apply a squared error, we'll set objective to 'reg:squarederror' for XGBoost and 'mean_squared_error' for LightGBM.

In addition, we'll specify the parameter n_jobs for XGBoost to improve its computation runtime.

OBS: be careful not to use classifiers, or your session might expire!

Build an XGBRegressor using the parameters: max_depth = 3, learning_rate = 0.1, n_estimators = 100, and n_jobs=2.
Build an LGBMRegressor using the parameters: max_depth = 3, learning_rate = 0.1, and n_estimators = 100.


reg_xgb = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, n_jobs=2, objective='reg:squarederror', random_state=500)
reg_xgb.fit(X_train, y_train)

reg_lgb = lgb.LGBMRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, objective='mean_squared_error', seed=500)
reg_lgb.fit(X_train, y_train)

pred_xgb = reg_xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
pred_lgb = reg_lgb.predict(X_test)
rmse_lgb = np.sqrt(mean_squared_error(y_test, pred_lgb))

print('Extreme: {:.3f}, Light: {:.3f}'.format(rmse_xgb, rmse_lgb))
Extreme: 5.122, Light: 5.142

-XGB performed better in reducing the RMSE than the LGBM; however LGBM performed FASTER

################################