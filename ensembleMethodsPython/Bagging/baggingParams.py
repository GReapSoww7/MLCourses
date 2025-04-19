### BAGGING PARAMETERS: TIPS AND TRICKS ###


# Basic Params for Bagging
-Basic Params:
    -base_estimator
    -n_estimators
    -oob_score
        -est_bag.oob_score_


# Additional Params
-Additional Params:
    -max_samples: the num of Samples to DRAW for EACH Estimator
    -max_features: the num of Features to DRAW for EACH Estimator
        -Classification ~ sqrt(num_of_features)
        -Regression ~ num_of_features / 3
    -bootstrap: whether Samples are drawn WITH Replacement
        -True --> max_samples = 1.0
        -False --> max_samples < 1.0

# Random Forest
-Classification:
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(
    # params
)

-Regression
from sklearn.ensemble import RandomForestRegressor
clf_rf = RandomForestRegressor(
    # params
)

-Bagging Params:
    -n_estimators
    -max_features
    -oob_score

-Tree-specific Params:
    -max_depth
    -min_samples_split
    -min_samples_leaf
    -class_weight ('balanced')

# Bias-Variance Tradeoff
-Simple model has LOW Variance but HIGH BIAS
-more complex model may REDUCE the BIAS but INCREASE VARIANCE
-we MUST OPTIMIZE the Params of the Ensemble Models to MINIMIZE the TOTAL ERROR and find BALANCE between BIAS and VARIANCE


################################

# PRACTICE

# A More Complex Bagging Model

As the target has a high class imbalance, use a "balanced" logistic regression as the base estimator here.

We will also reduce the computation time for LogisticRegression with the parameter solver='liblinear', which is a faster optimizer than the default.

Instantiate a logistic regression to use as the base classifier with the parameters: class_weight='balanced', solver='liblinear', and random_state=42.
Build a bagging classifier using the logistic regression as the base estimator, 
specifying the maximum number of features as 10, and including the out-of-bag score.
Print the out-of-bag score to compare to the accuracy.


clf_lr = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)

clf_bag = BaggingClassifier(clf_lr, max_features=10, oob_score=True, random_state=500)
clf_bag.fit(X_train, y_train)
pred = clf_bag.predict(X_test)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, pred)))
print('OOB-Score: {:.2f}'.format(clf_bag.oob_score_))

# Print the Confusion Matrix
print(confusion_matrix(y_test, pred))

<script.py> output:
    Accuracy:  0.71
    OOB-Score: 0.60
    [[423 162]
     [ 20  22]]
###############################

# Tuning Bagging HyperParams

Build a bagging classifier using the default parameters, it is highly recommended that you tune these in order to achieve optimal performance. 
Ideally, these should be optimized using K-fold cross-validation.

In this exercise, let's see if we can improve model performance by modifying the parameters of the bagging classifier.

Here we are also passing the parameter solver='liblinear' to LogisticRegression to reduce the computation time.

Build a bagging classifier using as base the logistic regression, with 20 base estimators, 10 maximum features, 0.65 (65%) maximum samples (max_samples), 
and sample without replacement.
Use clf_bag to predict the labels of the test set, X_test.

clf_base = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
clf_bag = BaggingClassifier(clf_lr, n_estimators=20, max_features=10, max_samples=0.65, bootstrap=False, random_state=500)
clf_bag.fit(X_train, y_train)
y_pred = clf_bag.predict(X_test)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, pred)))

print(classification_report(y_test, pred))


<script.py> output:
    Accuracy:  0.72
                  precision    recall  f1-score   support
    
              -1       0.95      0.74      0.83       585
               1       0.11      0.43      0.17        42
    
        accuracy                           0.72       627
       macro avg       0.53      0.59      0.50       627
    weighted avg       0.89      0.72      0.79       627
################################################