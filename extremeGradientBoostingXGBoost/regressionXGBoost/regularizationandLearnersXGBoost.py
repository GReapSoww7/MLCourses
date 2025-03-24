### REGULARIZATION AND BASE LEARNERS IN XGBOOST ###


-Regularization is a control on model complexity
    -aspect of Loss Functions that accounts for complexity
        -wants models that are accurate and simple

-Params:
    -gamma - min loss reduction allowed for a split to occur
    -alpha - L1 regularization on leaf weights , larger values mean more regularization (strong sparsity constraints)
    -lambda - L2 regularization on leaf weights (smoother penalty) > smooth decrease


# L1 Regularization in XGBoost (tuning)

import xgboost as xgb
import pandas as pd
boston_data = pd.read_csv("boston_data.csv")
X, y = boston_data.iloc[:,:-1], boston_data.iloc[:,-1]
boston_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective":"reg:squarederror", "max_depth":4}
l1_params = [1, 10, 100]
rmses_l1 = []
for reg in l1_params:
    params["alpha"] = reg
    cv_results = xgb.cv(dtrain=boston_dmatrix, params=params, nfold=4, num_boost_round=10, metrix='rmse', as_pandas=True, seed=123)
    rmses_l1.append(cv_results['test-rmse-mean'].tail(1).values[0])
print(f'Best rmse as function of l1:')
print(pd.DataFrame(list(zip(l1_params, rmses_l1)), columns=['l1', 'rmse']))


Best rmse as a function of l1:
    l1      rmse
0   1   69572.517742
1   10  73721.967141
2   100 82312.312413

# Base Learners in XGBoost
-Linear Base Learner:
    -SUM of linear terms
    -Boosted model is weighted SUM of linear models (linear itself)
        -rarely used

-Tree Base Learner:
    -Decision Trees
    -Boosted model is weighted SUM of decision trees (nonlinear)
        -almost exclusive to XGBoost


# Creating DataFrames from Multiple Equal-length Lists

pd.DataFrame(list(zip(list1, list2)), columns=['list1', 'list2'])

-zip creates a generator of PARALLEL values (out of multiple EQUAL length lists):
    -zip([1,2,3],['a','b','c']) = [1, 'a'], [2, 'b'], [3, 'c']
    -GENERATORS need to be completely instantiated BEFORE used in DF objects

-list() INSTANTIATES the FULL generator and passing that into the DF CONVERTS the WHOLE expression

######################################

# PRACTICE

# Using Regularization in XGBoost

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

reg_params = [1, 10, 100]

# Create the initial parameter dictionary for varying l2 strength: params
params = {"objective":"reg:squarederror","max_depth":3}

# Create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []

# Iterate over reg_params
for reg in reg_params:

    # Update l2 strength
    params["lambda"] = reg
    
    # Pass this updated param dictionary into cv
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)
    
    # Append best rmse (final round) to rmses_l2
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

# Look at best rmse per l2 param
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))


<script.py> output:
    Best rmse as a function of l2:
        l2       rmse
    0    1  52275.357
    1   10  57746.064
    2  100  76624.628
###########################

# Visualizing Individual XGBoost Trees

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:squarederror", "max_depth":2}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

# Plot the first tree
xgb.plot_tree(xg_reg, num_trees=0)
plt.show()

# Plot the fifth tree
xgb.plot_tree(xg_reg, num_trees=4)
plt.show()

# Plot the last tree sideways
xgb.plot_tree(xg_reg, num_trees=9, rankdir='LR')
plt.show()
##########################################

# Visualizing Feature Importances: What Features are Most Important in my Dataset

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {'objective':'reg:squarederror', 'max_depth':4}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()
################################################