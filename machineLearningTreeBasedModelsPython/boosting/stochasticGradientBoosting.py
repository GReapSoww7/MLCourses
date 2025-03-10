### STOCHASTIC GRADIENT BOOSTING (SGB) ###


# Gradient Boosting: Cons
-GB involves exhaustive SEARCH PROCEDURE
    -each CART is Trained to FIND the BEST split points and features
        -which may lead CARTs to use the SAME split points and maybe same features


# to Mitigate with SGB

-an algorithm where:
    -Each CART (tree) is Trained on a RANDOM Subset of Rows of the Train Set

-the sampled instances (40%-80% of the Train Set) are sampled WITHOUT replacement

-Features are sampled (without replacement) when choosing Split Points
    -RESULT: further Ensemble DIVERSITY
    -EFFECT: ADDING more Variance to the Ensemble of Trees


-the Residual Errors
    r1 = y1 - y(hat)1 take the place of y_train for the next tree to learn from its predecessor

(X_train, 'etah'*r1) > fed to next Tree in Ensemble


# SGB in sklearn (auto dataset)

# import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_square_error as MSE
# set SEED
SEED = 1
# split dataset 70% train 30% test
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)


# instantiate a Stochastic GradientBoostingRegressor 'sgbt'
sgbt = GradientBoostingRegressor(max_depth=1
                                 subsample = 0.8 # uses 80% of the Train Set data
                                 max_features = 0.2 # 20% of features to be used in each tree to find split points
                                 n_estimators=300,
                                 random_state=SEED)
# fit 'sgbt' to train set
sgbt.fit(X_train, y_train)
# predict Test Set labels
y_pred = sgbt.predict(X_test)
# eval Test Set RMSE 'rmse_test'
rmse_test = MSE(y_test, y_pred)**(1/2)
# print
print('Test set RMSE: {:.2f}'.format(rmse_test))
#############################################################

# PRACTICE

# Regression with SGB

As in the exercises from the previous lesson, you'll be working with the Bike Sharing Demand dataset. 
In the following set of exercises, you'll solve this bike count regression problem using stochastic gradient boosting.


Instantiate a Stochastic Gradient Boosting Regressor (SGBR) and set:
max_depth to 4 and n_estimators to 200,
subsample to 0.9, and
max_features to 0.75.


from sklearn.ensemble import GradientBoostingRegressor
sgbr = GradientBoostingRegressor(max_depth=4, subsample=0.9, max_features=0.75, n_estimators=200, random_state=2)


# train

sgbr.fit(X_train, y_train)
y_pred = sgbr.predict(X_test)


# Eval

from sklearn.metrics import mean_squared_error as MSE
mse_test = MSE(y_test, y_pred)
rmse_test = MSE(y_test, y_pred)**(1/2)
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))
Test set RMSE of sgbr: 49.621

-the RMSE of sgbr is lower than of the GradientBoostingRegressor (52.071)
    -this means the Root Mean Squared Error is less resulting in a higher prediction success
#########################################################################