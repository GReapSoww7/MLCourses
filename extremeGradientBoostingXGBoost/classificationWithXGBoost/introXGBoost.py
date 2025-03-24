### XGBoost ###


# Supervised Learning
-relies on labeled data
-have understanding on PAST behavior


-Example:
    -Face Recognition
        -training data: vectors of pixel values
        -Labels: 1 or 0

    -Classification
        -predicting binary or multiclass outcomes
        -Binary:
            -AUC (area under the Receiver Operating Characteristic curve)
                -common/versatile eval metric to judge classification model quality
                -the probability that a randomly chosen POS data point will have a HIGHER rank than a random NEG data point
                    -higher AUC = more sensitive/better performing model
        -Multi-Class:
            -Accuracy Score (higher is better)
            -look at the overall Confusion Matrix to eval model quality

        -common Algorithms:
            -LogisticRegression and DecisionTrees

-ALL Supervised Learning Models REQUIRE:
    -the data to be structured as a table of Feature Vectors
        -Features (attributes/predictors) must be either numeric or categorical
            -numeric features are SCALED to aid feature interpretation or ensure proper model training (Z-scored)
                -e.g. numeric feature scaling is essential to properly train Support Vector Machine models
            -categorical features are ENCODED before apply SL algorithms (e.g. one-hot encoding)


-Ranking Problems:
    -predicting and ordering on a set of choices
        -e.g. google search suggestions

-Recommendation:
    -recommending an item or set of items based on data collection
        -e.g. Netflix, Amazon, etc

######################################################################

### XGBoost ###

-Optimized Gradient-Boosted ML Lib
    -originally as C++ application
        -bindings (functions) leverage core C++ code [APIs]:
            -Python, R, Scala, Julia, Java

-speed and performance
-core algorithm is parallizable across networks, multi-core CPUs and GPUs
    -makes large dataset training possible
-outperforms single-algorithm models
-state of the art performance on ML benchmark dataset tasks


import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class_data = pd.read_csv('classification_data.csv')

X, y = class_data.iloc[:,:-1], class_data.iloc[:,-1] 
    # splits dataset into a matrix of Samples by FEATURES (X by convention) and a Vector of Target Values (called y by convention)

# train_test_split prevents overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# instantiate with params
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)
xg_cl.fit(X_train, y_train)

preds = xg_cl.predict(X_test)
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]

print('accuracy: %f' % (accuracy))

accuracy: 0.78333

######################################

# PRACTICE

# XGBoost: Fit/Predict

Your goal is to use the first month's worth of data to predict whether the app's users will remain users of the service at the 5 month mark. 
This is a typical setup for a churn prediction problem. 
To do this, you'll split the data into training and test sets, fit a small xgboost model on the training set,
and evaluate its performance on the test set by computing its accuracy.

pandas and numpy have been imported as pd and np, and train_test_split has been imported from sklearn.model_selection. 
Additionally, the arrays for the features and the target have been created as X and y.


Import xgboost as xgb.
Create training and test sets such that 20% of the data is used for testing. Use a random_state of 123.
Instantiate an XGBoostClassifier as xg_cl using xgb.XGBClassifier(). 
Specify n_estimators to be 10 estimators and an objective of 'binary:logistic'. 
Do not worry about what this means just yet, you will learn about these parameters later in this course.
Fit xg_cl to the training set (X_train, y_train) using the .fit() method.
Predict the labels of the test set (X_test) using the .predict() method and hit 'Submit Answer' to print the accuracy.

# Import xgboost
import xgboost as xgb

# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

accuracy: 0.758200
############################