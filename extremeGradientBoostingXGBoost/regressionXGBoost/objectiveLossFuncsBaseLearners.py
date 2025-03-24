### OBJECTIVE (LOSS) FUNCTIONS AND BASE LEARNERS ###


# Objective Funcs and Why We Use Them
-quantifies how far off PRED is from ACTUAL result for a given data point
-measures the DIFFERENCE between ESTIMATED (pred) and TRUE (target) values for some collection of data
-Goal: find the model that YIELDS the MIN val of LOSS func across ALL data points passed


# Common Loss Funcs and XGBoost
-reg:squarederror - regression
-reg:logistic - binary classification (when we want a category (decision), not probability)
-binary:logistic - when we DO want probability


# Base Learners and Why We Need Them
-each individual model of the Ensemble meta-model (XGBoost) = base learner

-we want base learners that when combined create FINAL PRED that is NON-LINEAR

-each base learner should be good at distinguishing or predicting DIFFERENT parts of dataset

-TWO base learner types:
    -TREE
    -LINEAR


# Trees sklearn API

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

<dataFrame> = pd.read_csv('boston_housing.csv')
X, y = <dataFrame>.iloc[:,:-1],<dataFrame>.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=123)

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10, seed=123)

xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print('RMSE: %f' % (rmse))

RMSE: 129043.2314

# Linear sklearn API

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

<dataFrame> = pd.read_csv('boston_housing.csv')

X, y = <dataFrame>.iloc[:,:-1],<dataFrame>.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

DM_train = xgb.DMatrix(data=X_train,label=y_train)
DM_test = xgb.DMatrix(data=X_test,label=y_test)
params = {'booster':'gblinear', 'objective':'reg:squarederror'}

xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=10)
preds = xg_reg.predict(DM_test)

rmse = np.sqrt(mean_squared_error(y_test,preds))
print(f'RMSE: %f' % (rmse))
RMSE: 124326.24465

#####################################

# PRACTICE

# DT as Base Learners

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(booster='gbtree', objective='reg:squarederror', n_estimators=10, seed=123)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f'RMSE: %f' % (rmse))

<script.py> output:
    RMSE: 28106.463641
#############################

# Linear Base Learners

DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test = xgb.DMatrix(data=X_test, label=y_test)
params = {'booster':'gblinear', 'objective':'reg:squarederror'}
xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=5)
preds = xg_reg.predict(DM_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f'RMSE: %f' % (rmse))

<script.py> output:
    RMSE: 44989.101138
###################################

# Eval Model Quality

-RMSE

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics='rmse', as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-rmse-mean"]).tail(1))
   train-rmse-mean  train-rmse-std  test-rmse-mean  test-rmse-std
0       141767.533         429.451      142980.435       1193.795
1       102832.548         322.472      104891.395       1223.157
2        75872.617         266.474       79478.939       1601.345
3        57245.652         273.624       62411.921       2220.150
4        44401.299         316.424       51348.280       2963.378
4    51348.28
Name: test-rmse-mean, dtype: float64

-MAE

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics='mae', as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-mae-mean"]).tail(1))

   train-mae-mean  train-mae-std  test-mae-mean  test-mae-std
0      127343.480        668.307     127633.999      2404.006
1       89770.056        456.964      90122.501      2107.910
2       63580.789        263.405      64278.559      1887.568
3       45633.157        151.884      46819.169      1459.818
4       33587.090         86.998      35670.647      1140.607
4    35670.647
Name: test-mae-mean, dtype: float64
###########################################################################