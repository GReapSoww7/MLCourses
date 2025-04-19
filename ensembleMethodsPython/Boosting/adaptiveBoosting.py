### ADAPTIVE BOOSTING ###


# About AdaBoost
-proposed by Yoav Freund and Robert Schapire (1997)
-Winner of the Godel Prize in (2003)
-The FIRST practical Boosting Algo
-Highly used and well known Ensemble Method

# AdaBoost Properties
1. Instances are drawn using a sample distribution
    -difficult instances have HIGHER Weights
    -initialized to be UNIFORM

2. Estimators are COMBINED with a WEIGHTED Majority Voting
    -Good Estimators are given HIGHER Weights

3. Guaranteed to improve
4. Classification and Regression


# AdaBoost Classifier with sklearn

from sklearn.ensemble import AdaBoostClassifier
clf_ada = AdaBoostClassifier(
    base_estimator,
    n_estimators,
    learning_rate
)

-Params:
    -base_estimator:
        -Default: Decision Tree (max_depth=1)
    -n_estimators:
        -Default: 50
    -learning_rate:
        -Default: 1.0
        -Trade-off between n_estimators and learning_rate

# AdaBoost Regressor

from sklearn.ensemble import AdaBoostRegressor
reg_ada = AdaBoostRegressor(
    base_estimator,
    n_estimators,
    learning_rate,
    loss
)

-Params:
    -base_estimator:
        -Default: Decision Tree (max_depth=3)
    -loss:
        -linear (default)
        -square
        -exponential

###########################################

# PRACTICE

# AdaBoost Model

In this exercise, you'll build your first AdaBoost model - an AdaBoostRegressor - in an attempt to improve performance even further.

The movies dataset has been loaded and split into train and test sets.
Here you'll be using the 'budget' and 'popularity' features,
which were already standardized for you using StandardScaler() from sklearn.preprocessing module.


Instantiate the default linear regression model.
Build and fit an AdaBoostRegressor, using the linear regression as the base model and 12 estimators.
Calculate the predictions on the test set.


reg_lm = LinearRegression()
reg_ada = AdaBoostRegressor(reg_lm, n_estimators=12, random_state=500)
reg_ada.fit(X_train, y_train)

pred = reg_ada.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))

RMSE: 7.179
-this is better than the custom model built in gradualLearning section
###########################

# Tree-based AdaBoost Regression

AdaBoost models are usually built with decision trees as the base estimators.
Let's give this a try now and see if model performance improves even further.

We'll use twelve estimators as before to have a fair comparison.
There's no need to instantiate the decision tree as it is the base estimator by default.

Build and fit an AdaBoostRegressor using 12 estimators. You do not have to specify a base estimator.
Calculate the predictions on the test set.

reg_ada = AdaBoostRegressor(n_estimators=12, random_state=500)
reg_ada.fit(X_train, y_train)
pred = reg_ada.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))

RMSE: 5.443
-Using a decision tree instead of linear regression as the base_estimator reduced the RMSE even more

#################################

# Making the Most of AdaBoost

In this exercise, you'll specify some parameters to extract even more performance.
In particular, you'll use a lower learning rate to have a smoother update of the hyperparameters.
Therefore, the number of estimators should increase.
Additionally, the following features have been added to the data: 'runtime', 'vote_average', and 'vote_count'.

Build an AdaBoostRegressor using 100 estimators and a learning rate of 0.01.
Fit reg_ada to the training set and calculate the predictions on the test set.


reg_ada = AdaBoostRegressor(n_estimators=100, learning_rate=0.01, random_state=500)
reg_ada.fit(X_train, y_train)
pred = reg_ada.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))

RMSE: 5.150
-introducing a lower learning rate and a larger number of estimators reduced the RMSE even more

###############################