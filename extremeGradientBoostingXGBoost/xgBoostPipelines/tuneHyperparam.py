### TUNING XGBOOST HYPERPARAMS IN A PIPELINE ###


# Tuning

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

names = ['crime', 'zone', 'industry', 'charles', 'no', 'rooms', 'age', 'distance', 'radial', 'tax', 'pupil', 'aam', 'lower', 'med_price']
data = pd.read_csv('boston_housing.csv', names=names)
X, y = data.iloc[:,:-1], data.iloc[:, -1]
xgb_pipeline = Pipeline[('st_scaler', StandardScaler()), ('xgb_model', xgb.XGBRegressor())]
gbm_param_grid = {
    'xgb_model__subsample': np.arange(0.05, 1, 0.05),
    'xgb_model__max_depth': np.arange(3, 20, 1),
    'xgb_model__colsamples_bytree': np.arange(0.1, 1.05, 0.05)
}
randomized_neg_mse = RandomizedSearchCV(estimator=xgb_pipeline, param_distributions=gbm_param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=4)

randomized_neg_mse.fit(X, y)

print(f'Best rmse: ', np.sqrt(np.abs(randomized_neg_mse.best_score_)))
Best rmse: 3.9966784203040677

print(f'Best model: ', randomized_neg_mse.best_estimator_)
Best model: <model info>

#############

# PRACTICE

# Bringing it all together
Alright, it's time to bring together everything you've learned so far! 
In this final exercise of the course, you will combine your work from the previous exercises into one end-to-end XGBoost pipeline 
to really cement your understanding of preprocessing and pipelines in XGBoost.

Your work from the previous 3 exercises, where you preprocessed the data and set up your pipeline, has been pre-loaded. 
Your job is to perform a randomized search and identify the best hyperparameters.


Set up the parameter grid to tune 'clf__learning_rate' (from 0.05 to 1 in increments of 0.05), 
'clf__max_depth' (from 3 to 10 in increments of 1), and 'clf__n_estimators' (from 50 to 200 in increments of 50).

Using your pipeline as the estimator, perform 2-fold RandomizedSearchCV with an n_iter of 2. 
Use "roc_auc" as the metric, and set verbose to 1 so the output is more detailed. Store the result in randomized_roc_auc.

Fit randomized_roc_auc to X and y.
Compute the best score and best estimator of randomized_roc_auc.

# create param grid
gbm_param_grid = {
    'clf__learning_rate': np.arange(0.05, 1, 0.05),
    'clf__max_depth': np.arange(3, 10, 1),
    'clf__n_estimators': np.arange(50, 200, 50)
}

# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(estimator=xgb_pipeline, param_distributions=gbm_param_grid, n_iter=2, scoring='roc_auc', cv=2, verbose=1)
# Fit the estimator
randomized_roc_auc.fit(X, y)

# print metrics
print(f'Best auc: ', np.mean(randomized_roc_auc.best_score_))
print(f'Best model: ', randomized_roc_auc.best_estimator_)

<script.py> output:
    Fitting 2 folds for each of 2 candidates, totalling 4 fits
    Best auc:  0.9965333333333333
    Best model:  Pipeline(steps=[('featureunion',
                     FeatureUnion(transformer_list=[('num_mapper',
                                                     DataFrameMapper(df_out=True,
                                                                     features=[(['age'],
                                                                                SimpleImputer(strategy='median')),
                                                                               (['bp'],
                                                                                SimpleImputer(strategy='median')),
                                                                               (['sg'],
                                                                                SimpleImputer(strategy='median')),
                                                                               (['al'],
                                                                                SimpleImputer(strategy='median')),
                                                                               (['su'],
                                                                                SimpleImputer(strategy='median')),
                                                                               (['bgr'],
                                                                                SimpleImputer(s...
                                   gamma=0, gpu_id=-1, grow_policy='depthwise',
                                   importance_type=None, interaction_constraints='',
                                   learning_rate=0.9500000000000001, max_bin=256,
                                   max_cat_to_onehot=4, max_delta_step=0,
                                   max_depth=4, max_leaves=0, min_child_weight=1,
                                   missing=nan, monotone_constraints='()',
                                   n_estimators=100, n_jobs=0, num_parallel_tree=1,
                                   predictor='auto', random_state=0, reg_alpha=0,
                                   reg_lambda=1, ...))])

###################################