### WHAT IS BOOSTING ###


# Overview
    -not a specific ML algorithm
    -Concept that can be applied to a SET of ML Models
        -Boosting is:
            -an Ensemble Meta-Algo used to convert many WEAK learners into a STRONG learner

# Weak Learners and Strong Learners

-Weak Learner: ML algo that is slightly better than CHANCE
    -e.g. Decision Tree with predictions slightly above 50%

-Boosting converts a COLLECTION of weak learners into Strong Learners
-Strong Learner:
    -ANY algorithm that can be TUNED to achieve good performance for some SL problems

-boosting is accomplished:
    -by ITERATIVELY learning a set of weak models on SUBSETS of the data
    -weighing EACH weak pred according to EACH weak learner's performance
    -COMBINE the WEIGHTED preds to obtain a SINGLE weighted pred
        -which is much better than the INDIVIDUAL preds themselves


# Model eval through CrossValidation

-CV: robust methods for estimating the performance of ML model on unseen data
    -generates many non-overlapping train/test splits on train data
    -reports the AVERAGE test set performance across ALL data SPLITS

# CV in XGBoost Example

import xgboost as xgb
import pandas as pd
churn_data = pd.read_csv('classification_data.csv')

# convert dataset into optimized data struct
churn_dmatrix = xgb.DMatrix(data=churn_data.iloc[:,:-1], label=churn_data.month_5_still_here)

# create param dict for passing into CV; because CV method does not know what kind of XGBoos Model we are using so it needs key:value pair dict
params = {"objective":"binary:logistic","max_depth":"4"} # function we would like to use; max depth every tree can grow to


cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=4, num_boost_round=10, metrics="error", as_pandas=True)
print("Accuracy: %f" %((1-cv_results["test-error-mean"]).iloc[-1]))

Accuracy: 0.88315

#####################################

# PRACTICE

# Measuring Accuracy

# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:1], churn_data.iloc[:,-1]

# create the DMatrix from X and y: churn_DMatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# create the param dict: params
params = {'objective':'reg:logistic', 'max_depth':3}

# perform CV: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3, num_boost_round=5, metrics='error', as_pandas=True, seed=123)
print(cv_results)

# print accuracy
print(((1-cv_results['test-error-mean']).iloc[-1]))

<script.py> output:
       train-error-mean  train-error-std  test-error-mean  test-error-std
    0             0.282            0.002            0.284       1.932e-03
    1             0.270            0.002            0.272       1.932e-03
    2             0.256            0.003            0.258       3.963e-03
    3             0.251            0.002            0.254       3.827e-03
    4             0.247            0.002            0.249       9.344e-04
    0.751480015401492
##############################################

# Measuring AUC

# perform CV: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3, num_boost_round=5, metrics='auc', as_pandas=True, seed=123)

print(cv_results)
print((cv_results['test-auc-mean']).iloc[-1])

<script.py> output:
       train-auc-mean  train-auc-std  test-auc-mean  test-auc-std
    0           0.769      1.544e-03          0.768         0.003
    1           0.791      6.758e-03          0.789         0.007
    2           0.816      3.900e-03          0.814         0.006
    3           0.823      2.018e-03          0.822         0.004
    4           0.828      7.694e-04          0.826         0.002
    0.8261911413597645
#################################################