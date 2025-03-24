### INTRO TO ENSEMBLE METHODS (Supervised Learning) ###


# Choosing the Best Model
-choosing based on a single metric can hinder ability to select best model
-combining models and their selected metric will provide a meta model which is better than a single model by itself

# MLxtend
-we will be introduced into this


# sklearn MetaEstimator

from sklearn.ensemble import MetaEstimator
# base estimators
est1 = Model1()
est2 = Model2()
estN = ModelN()

# Meta Estimator
est_combined = MetaEstimator(estimators=[est1, est2, ..., estN], <additionalParams>)
# Train and Test
est_combined.fit(X_train, y_train)
pred = est_combined.predict(X_test)

########################

# PRACTICE

# Predicting the Rating of an App

Having explored the Google apps dataset in the previous exercise, 
let's now build a model that predicts the rating of an app given a subset of its features.

To do this, you'll use scikit-learn's DecisionTreeRegressor. 
As decision trees are the building blocks of many ensemble models, refreshing your memory of how they work will serve you well throughout this course.

We'll use the MAE (mean absolute error) as the evaluation metric.
This metric is highly interpretable, as it represents the average absolute difference between actual and predicted ratings.

All required modules have been pre-imported for you. 
The features and target are available in the variables X and y, respectively.


Use train_test_split() to split X and y into train and test sets. Use 20%, or 0.2, as the test size.
Instantiate a DecisionTreeRegressor(), reg_dt, with the following hyperparameters: min_samples_leaf = 3 and min_samples_split = 9.
Fit the regressor to the training set using .fit().
Predict the labels of the test set using .predict().

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg_dt = DecisionTreeRegressor(min_samples_leaf=3, min_samples_split=9, random_state=500)
reg_dt.fit(X_train, y_train)
y_pred = reg_dt.predict(X_test)
print(f'MAE: {:.3f}'.format(mean_absolute_error(y_test, y_pred)))

<script.py> output:
    MAE: 0.609
#####################