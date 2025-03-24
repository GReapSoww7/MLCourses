### OVERVIEW OF XGBOOST'S HYPERPARAMS ###


# Tunable Params in XGBoost

-Common Tree Tunable Params
    -learning rate: learning rate/eta
        -affects how quickly the model fits the residual error > using additional base learners
        -a LOW learning rate reqs MORE boosting rounds to achieve the same REDUCTION in Residual Error as a HIGH Learning Rate model

-gamma: MIN loss reduction to create new tree split
-lambda: L2 reg on leaf weights
-alpha: L1 reg on leaf weights
    -all have an effect on how strong regularization is
-max_depth: max depth per tree (how deep tree can grow during boosting rounds)
-subsample: % samples used per tree
    -used in any given boosting round
        -LOW = fraction of training data per round will be low = UNDERFITTING
        -HIGH = OVERFITTING
-colsample_bytree: % (fraction) of features used per tree in any round (value = between 0 and 1)
    -LARGE value = almost ALL features can be used
        -may OVERFIT
    -SMALL value = fraction of features used is small
        -providing additional regularization

# Linear Tunable Params
-lambda: L2 reg on weights
-alpha: L1 reg on weights
-lambda_bias: L2 reg term on BIAS

-we can tune the num of ESTIMATORS used for BOTH Base Model types
    -num_boost_round > num of trees we build (base learners) is TUNABLE

############################################

# PRACTICE

# Tuning ETA

It's time to practice tuning other XGBoost hyperparameters in earnest and observing their effect on model performance!
You'll begin by tuning the "eta", also known as the learning rate.

The learning rate in XGBoost is a parameter that can range between 0 and 1, with higher values of "eta" penalizing feature weights more strongly, causing much stronger regularization.


Create a list called eta_vals to store the following "eta" values: 0.001, 0.01, and 0.1.
Iterate over your eta_vals list using a for loop.
In each iteration of the for loop, set the "eta" key of params to be equal to curr_val. Then, perform 3-fold cross-validation with early stopping (5 rounds), 10 boosting rounds, a metric of "rmse", and a seed of 123. Ensure the output is a DataFrame.
Append the final round RMSE to the best_rmse list.


# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree (boosting round)
params = {"objective":"reg:squarederror", "max_depth":3}

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]
best_rmse = []

# Systematically vary the eta 
for curr_val in eta_vals:

    params["eta"] = curr_val
    
    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, metrics='rmse', num_boost_round=10, nfold=3, early_stopping_rounds=5, as_pandas=True, seed=123)
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta","best_rmse"]))

<script.py> output:
         eta   best_rmse
    0  0.001  195736.403
    1  0.010  179932.184
    2  0.100   79759.412

#########################################

# Tuning max_depth

Create a list called max_depths to store the following "max_depth" values: 2, 5, 10, and 20.
Iterate over your max_depths list using a for loop.
Systematically vary "max_depth" in each iteration of the for loop and perform 2-fold cross-validation with early stopping (5 rounds), 10 boosting rounds, 
a metric of "rmse", and a seed of 123. 
Ensure the output is a DataFrame.


# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Create the parameter dictionary
params = {'objective':'reg:squarederror', 'max_depth':20}

# Create list of max_depth values
max_depths = [2, 5, 10, 20]
best_rmse = []

# Systematically vary the max_depth
for curr_val in max_depths:

    params["max_depth"] = curr_val
    
    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, metrics='rmse', num_boost_round=10, nfold=2, early_stopping_rounds=5, as_pandas=True, seed=123)
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(max_depths, best_rmse)),columns=["max_depth","best_rmse"]))


<script.py> output:
       max_depth  best_rmse
    0          2  37957.469
    1          5  35596.600
    2         10  36065.547
    3         20  36739.576
############################################

# Tuning colsample_bytree

Now, it's time to tune "colsample_bytree".
You've already seen this if you've ever worked with scikit-learn's RandomForestClassifier or RandomForestRegressor, where it just was called max_features.
In both xgboost and sklearn, this parameter (although named differently) simply specifies the fraction of features to choose from at every split in a given tree. 
In xgboost, colsample_bytree must be specified as a float between 0 and 1.

Create a list called colsample_bytree_vals to store the values 0.1, 0.5, 0.8, and 1.
Systematically vary "colsample_bytree" and perform cross-validation, exactly as you did with max_depth and eta previously.

housing_dmatrix = xgb.DMatrix(data=X, label=y)
params = {'objective':'reg:squarederror', 'max_depth':3}
colsample_bytree_vals = [0.1, 0.5, 0.8, 1]
best_rmse = []

for curr_val in colsample_bytree_vals:
     
     params['colsample_bytree'] = curr_val

     cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, metrics='rmse', num_boost_round=10, nfold=2, early_stopping_rounds=5, as_pandas=True, seed=123)

     best_rmse.append(cv_results['test-rmse-mean'].tail().values[-1])

print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)), columns=['colsample_bytree', 'best_rmse']))


<script.py> output:
       colsample_bytree  best_rmse
    0               0.1  50033.735
    1               0.5  35656.186
    2               0.8  36399.002
    3               1.0  35836.044

# NOTE: There are several other individual parameters that you can tune, such as "subsample", which dictates the fraction of the training data that is used during any given boosting round.
##############################################