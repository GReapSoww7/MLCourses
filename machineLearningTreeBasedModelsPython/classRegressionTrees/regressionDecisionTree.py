### DECISION-TREE FOR REGRESSION


# Auto-mpg Dataset
    -6 features (car characteristics)

-Predict mpg consumption of car based on these features
    -analysis is restricted to one feature (displacement)

# Regression-Tree with scikit-learn

# import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
# import train_test_split
from sklearn.model_selection import train_test_split
# import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
# split data into 80% train, 20% test
X_train, y_train, X_test, y_test= train_test_split(X, y, test_size=0.2, random_state=3)

# Instantiate a DecisionTreeRegressor 'dt'
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.1, random_state=3)

# fit 'dt' to the training-set
dt.fit(X_train, y_train)
# predict test-set labels
y_pred = dt.predict(X_test)
# compute test-set MSE
mse_dt = MSE(y_test, y_pred)
# compute test-set RMSE
rmse_dt = mse_dt**(1/2)
# print rmse_dt
print(rmse_dt)
5.1023068889

# Information Criterion for Regression-Tree
-when a RegressionTree is trained on a dataset
    -IMPURITY of a NODE is measured using the mean_squared_error of the TARGETS in that NODE
        -meaning the RegressionTree tries to find SPLITS that PRODUCES the LEAFS where in each LEAF the TARGET_VALUE are on AVG the CLOSEST possible to the LABELS in that particular leaf


I(node) = MSE(node) = 1/Nnode SIGMA [iEnode] (y^(i) - ynode)^2

ynode (mean-target-value) = 1/Nnode SIGMA [iEnode] y^(i)

# Prediction
-as a NEW instance traverses the tree and reaches a certain leaf
    -its target variable is computed as the avg of the target vars contained in the leaf

ypred(leaf) = 1/Nleaf SIGMA y^(i)


# Linear-Regression vs Regression-Tree
-Regression-Tree can capture non-linear trends exhibited by data
    -more flexible than Linear-Regression


####################################################################

# PRACTICE

# Train First Regression-Tree

In this exercise, you'll train a regression tree to predict the mpg (miles per gallon) consumption of cars in the auto-mpg dataset using all the six available features.

The dataset is processed for you and is split to 80% train and 20% test. The features matrix X_train and the array y_train are available in your workspace.

Import DecisionTreeRegressor from sklearn.tree.
Instantiate a DecisionTreeRegressor dt with maximum depth 8 and min_samples_leaf set to 0.13.
Fit dt to the training set.

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)
dt.fit(X_train, y_train)

###################################################################

# Evaluate the Regression Tree

In this exercise, you will evaluate the test set performance of dt using the Root Mean Squared Error (RMSE) metric. 
The RMSE of a model measures, on average, how much the model's predictions differ from the actual labels. 
The RMSE of a model can be obtained by computing the square root of the model's Mean Squared Error (MSE).

The features matrix X_test, the array y_test, as well as the decision tree regressor dt that you trained in the previous exercise are available in your workspace.

Import the function mean_squared_error as MSE from sklearn.metrics.
Predict the test set labels and assign the output to y_pred.
Compute the test set MSE by calling MSE and assign the result to mse_dt.
Compute the test set RMSE and assign it to rmse_dt.


from sklearn.metrics import mean_squared_error as MSE

y_pred = dt.predict(X_test)
mse_dt = MSE(y_test, y_pred)
rmse_dt = mse_dt**(1/2)
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))
####################################################

# Linear Regression vs Regression Tree

In this exercise, you'll compare the test set RMSE of dt to that achieved by a linear regression model. 
We have already instantiated a linear regression model lr and trained it on the same dataset as dt.

The features matrix X_test, the array of labels y_test, the trained linear regression model lr, mean_squared_error function which was imported under the alias MSE 
and rmse_dt from the previous exercise are available in your workspace.

Predict test set labels using the linear regression model (lr) and assign the result to y_pred_lr.

Compute the test set MSE and assign the result to mse_lr.

Compute the test set RMSE and assign the result to rmse_lr.

y_pred_lr = lr.predict(X_test)
mse_lr = MSE(y_test, y_pred_lr)
rmse_lr = mse_lr**(1/2)
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))

###################################################################################