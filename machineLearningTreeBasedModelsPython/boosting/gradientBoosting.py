### GRADIENT BOOSTING ###


# Gradient Boosted Trees
-Sequential correction of predecessor's errors
-does NOT tweak the WEIGHTS of training instances
-Fit = EACH predictor is Trained using its predecessor's RESIDUAL errors as LABELS
-Gradient Boosted Trees: a CART is used as a BASE LEARNER

# Gradient Boosted Trees for REGRESSION: Training

Tree1 > ... > TreeN
    -here Tree1 is trained using (X_train, y_train)
        -from the prediction (y1 - y1 (hat)) are used to determine r1 (training set RESIDUAL ERRORS)

-then Tree2 > ... > TreeN is trained using (X_train, r1)
    -r1 > ... > rN-1


# SHRINKAGE PARAM

-refers to the fact that the prediction of each Tree in ensemble is shrunk AFTER it is multiplied by a LEARNING RATE 'etah' (a number between 0 and 1)

-Decreasing the LEARNING RATE 'etah' needs to be compensated by INCREASING the number of ESTIMATORS
    -for the ENSEMBLE to reach a certain performance


# PREDICTION
-Regression:
y_pred = y1 + 'etah'*r1 + ... + 'etah'*rN
    -in sklearn: GradientBoostingRegressor

-Classification:
    -in sklearn: GradientBoostingClassifier

# Gradient Boosting in sklearn (auto dataset)

# import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# set SEED
SEED = 1
# split dataset 70% train 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# instantiate a GradientBoostinRegressor 'gbt'
gbt = GradientBoostingRegressor(n_estimators=300, max_depth=1, random_state=SEED)

# Fit 'gbt' to Train Set
gbt.fit(X_train, y_train)

# Predict Test set labels
y_pred = gbt.predict(X_test)

# Eval Test Set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)

# print
print('Test set RMSE: {:.2f}'.format(rmse_test))
Test set RMSE: 4.01

#######################################

# PRACTICE

# Define the GB Regressor
You'll now revisit the Bike Sharing Demand dataset that was introduced in the previous chapter. 
Recall that your task is to predict the bike rental demand using historical weather data from the Capital Bikeshare program in Washington, D.C.. 
For this purpose, you'll be using a gradient boosting regressor.

As a first step, you'll start by instantiating a gradient boosting regressor which you will train in the next exercise.


Import GradientBoostingRegressor from sklearn.ensemble.
Instantiate a gradient boosting regressor by setting the parameters:
max_depth to 4
n_estimators to 200


from sklearn.ensemble import GradientBoostingRegressor
gbt = GradientBoostinRegressor(n_estimators=200, max_depth=4, random_state=2)


# Train GB Regressor
You'll now train the gradient boosting regressor gb that you instantiated in the previous exercise and predict test set labels.
The dataset is split into 80% train and 20% test. 
Feature matrices X_train and X_test, as well as the arrays y_train and y_test are available in your workspace. 
In addition, we have also loaded the model instance gb that you defined in the previous exercise.


Fit gb to the training set.
Predict the test set labels and assign the result to y_pred.

gbt.fit(X_train, y_train)
y_pred = gbt.predict(X_test)


# Eval GB Regressor
Now that the test set predictions are available, you can use them to evaluate the test set Root Mean Squared Error (RMSE) of gb.
y_test and predictions y_pred are available in your workspace.

Import mean_squared_error from sklearn.metrics as MSE.
Compute the test set MSE and assign it to mse_test.
Compute the test set RMSE and assign it to rmse_test.

from sklearn.metrics import mean_squared_error as MSE

mse_test = MSE(y_test, y_pred)
rmse_test = MSE(y_test, y_pred)**(1/2)
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))
Test set RMSE of gb: 52.071
#################################################################