### RANDOM FORESTS ###


# Bagging
-Base Estimator can be ANY model
    -Decision Tree, Logistic Regression, Neural Net, etc
-EACH estimator is TRAINED on a DISTINCT bootstrap sample of the training set
    -Estimators use ALL features for Train and Prediction


# Random Forest Estimator
-the Base Estimator = DecisionTreeClassifier
    -EACH estimator is trained on a DIFFERENT bootstrap sample, HAVING the SAME size as the Train Set

-RF introduces further randomization (compared to Bagging) in the Training of individual trees

- 'd' (features) are sampled at EACH node WITHOUT replacement
    ('d' < total number of Features)

    -the NODE is then split using the SAMPLED Feature that MAXIMIZES information gain (IG)

-in sklearn 'd' default to the square root of the number of features
    -ex. if there is 100 features; 'd' = 10 (only 10 features are sampled at EACH node)

-ONCE TRAINED
    -predictions can be made on NEW instances
        -when new instances are FED to the DIFFERENT base_estimators > each outputs a PREDICTION > predictions are collected be the RF meta-model > final prediction made depending on the nature of the problem

# Classification & Regression

-Classification: final prediction is made by MAJORITY Voting
    from sklearn.ensemble import RandomForestClassifier

-Regression: aggregates predictions through AVERAGING
    from sklearn.ensemble import RandomForestRegressor

-IN GENERAL RANDOM FOREST achieves a LOWER Variance than INDIVIDUAL TREES


# Random Forests Regressor in sklearn (auto dataset)

# basic imports

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# set SEED
SEED = 1
# split dataset 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)


# instantiate a random forests regressor 'rf' 400 estimators
rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=SEED)
# fit 'rf' to Train Set
rf.fit(X_train, y_train)
# predict the Test Set labels 'y_pred'
y_pred = rf.predict(X_test)
# Evaluate the Test Set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# print the test set RMSE
print('Test Set RMSE of rf: {:.2f}'.format(rmse_test))

Test Set RMSE of rf: 3.98
    -this Error is smaller than that achieved of a single Regression Tree (which is 4.243)
#######################################


# Feature Importance in Tree-based Method Training

-Tree-based Methods: ENABLE measuring the importance of EACH Feature in Prediction

-in sklearn:
    -how MUCH the Tree Nodes use a particular Feature (weighted average) to REDUCE impurity
        -expressed as a pct indicating the weight of the feature in training and prediction
    -accessed using/extracting the Attribute feature_importances_


# Feature Importance in sklearn

import pandas as pd
import matplotlib.pyplot as plt

# create a pd.Series of features importances
importances_rf = pd.Series(rf.feature_importances_, index = X.columns)

# sort importances_rf
sorted_importances_rf = importances_rf.sort_values()

# make a horizontal bar plot (to visualize the feature importance results)
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()

###############################################################

# PRACTICE

# Train an RF Regressor

In the following exercises you'll predict bike rental demand in the Capital Bikeshare program in Washington, D.C 
using historical weather data from the Bike Sharing Demand dataset available through Kaggle. 
For this purpose, you will be using the random forests algorithm. 
As a first step, you'll define a random forests regressor and fit it to the training set.

The dataset is processed for you and split into 80% train and 20% test. The features matrix X_train and the array y_train are available in your workspace.


Import RandomForestRegressor from sklearn.ensemble.
Instantiate a RandomForestRegressor called rf consisting of 25 trees.
Fit rf to the training set.

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=25, random_state=2)
rf.fit(X_train, y_train)


# Evaluate the RF Regressor

You'll now evaluate the test set RMSE of the random forests regressor rf that you trained in the previous exercise.

The dataset is processed for you and split into 80% train and 20% test. 
The features matrix X_test, as well as the array y_test are available in your workspace. 
In addition, we have also loaded the model rf that you trained in the previous exercise.


Import mean_squared_error from sklearn.metrics as MSE.
Predict the test set labels and assign the result to y_pred.
Compute the test set RMSE and assign it to rmse_test.


from sklearn.metrics import mean_squared_error as MSE

y_pred = rf.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# Visualizing Feature Importances

In this exercise, you'll determine which features were the most predictive according to the random forests regressor rf that you trained in a previous exercise.
For this purpose, you'll draw a horizontal barplot of the feature importance as assessed by rf. Fortunately, this can be done easily thanks to plotting capabilities of pandas.
We have created a pandas.Series object called importances containing the feature names as index and their importances as values. 
In addition, matplotlib.pyplot is available as plt and pandas as pd.


Call the .sort_values() method on importances and assign the result to importances_sorted.
Call the .plot() method on importances_sorted and set the arguments:
kind to 'barh'
color to 'lightgreen'


import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(data=rf.feature_importances_, index=X_train.columns)
importances_sorted = importances.sort_values()
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()
###########################################################################