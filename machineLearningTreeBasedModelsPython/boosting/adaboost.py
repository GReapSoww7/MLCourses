### ADABOOST ###


-an Ensemble method where many predictors are trained
    -each predictor learns from the ERRORS of its predecessor


# Boosting
-Boosting: Ensemble Method COMBINING several WEAK learners to FORM a STRONG learner
    -Weak Learner: Model doing SLIGHTLY better than RANDOM guessing
        -ex. Decision Stump (CART whose max_depth=1)

-Trains an ensemble of predictors sequentially
-each predictor tries to CORRECT its predecessor

-Most Popular Boosting Methods:
    -AdaBoost (Adaptive Boosting)
    -Gradient Boosting

# AdaBoost
-EACH predictor pays MORE attention to the Instances WRONGLY predicted by its predecessor
    -achieved by changing the weight of training instances
-each predictor is assigned a Coeff 'alpha'
    -the Coeff depends on the predictor's Training Error

# Training

Train Set > Train > Predictor1 > Predict > Error used to find Coeff 'alpha' > Coeff is used to find Weights W(2) > Predictor2 Train Set

Predictor1 > ... > PredictorN
Weight2 > ... > WeightN

    -Predictors pay additional attention to the incorrectly predicted instances


# Learning Rate

0 < 'etah' <= 1
-used to shrinking 'alpha' of a trained predictor
    -smaller value 'etah' = greater number of Estimators



# Prediction
-once ALL Predictors are trained
    -label of new instance can be predicted

-Classification:
    -weighted majority voting
    -sklearn: AdaBoostClassifier

-Regression:
    -weighted average
    -sklearn: AdaBoostRegressor

-individual predictors do NOT have to be CARTs (but CARTs are often used because of their High Variance)

############################################################################

# AdaBoost Classification in sklearn (Breast Cancer dataset)

# import models and utility functions
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# set SEED
SEED = 1

# split data 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

# instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)
# instantiate an AdaBoost classifier 'adb_clf'
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
# fit 'adab_clf' to the Train Set
adb_clf.fit(X_train, y_train)
# predict Test Set PROBABILITIES of POSITIVE class
y_pred_proba = adb_clf.pred_proba(X_test)[:,1]
# Eval Test Set roc_auc_score
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)
# Print adb_clf_roc_auc_score
print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))

ROC AUC score: 0.99


#######################################################

# PRACTICE

# Define the AdaBoost Classifier

In the following exercises you'll revisit the Indian Liver Patient dataset which was introduced in a previous chapter. 
Your task is to predict whether a patient suffers from a liver disease using 10 features including Albumin, age and gender. 
However, this time, you'll be training an AdaBoost ensemble to perform the classification task. 
In addition, given that this dataset is imbalanced, you'll be using the ROC AUC score as a metric instead of accuracy.

As a first step, you'll start by instantiating an AdaBoost classifier.


Import AdaBoostClassifier from sklearn.ensemble.
Instantiate a DecisionTreeClassifier with max_depth set to 2.
Instantiate an AdaBoostClassifier consisting of 180 trees and setting the base_estimator to dt.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

dt = DecisionTreeClassifier(max_depth=2, random_state=1)
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)

##################

# Train the AdaBoost Classifier

Now that you've instantiated the AdaBoost classifier ada, it's time train it. 
You will also predict the probabilities of obtaining the positive class in the test set. This can be done as follows:

Once the classifier ada is trained, call the .predict_proba() method by passing X_test as a parameter 
and extract these probabilities by slicing all the values in the second column as follows:

ada.predict_proba(X_test)[:,1]
The Indian Liver dataset is processed for you and split into 80% train and 20% test. 
Feature matrices X_train and X_test, as well as the arrays of labels y_train and y_test are available in your workspace. 
In addition, we have also loaded the instantiated model ada from the previous exercise.


Fit ada to the training set.
Evaluate the probabilities of obtaining the positive class in the test set.


ada.fit(X_train, y_train)
y_pred_proba = ada.predict_proba(X_test)[:,1]


# Eval the AdaBoost Classifier
Now that you're done training ada and predicting the probabilities of obtaining the positive class in the test set, it's time to evaluate ada's ROC AUC score. 
Recall that the ROC AUC score of a binary classifier can be determined using the roc_auc_score() function from sklearn.metrics.

The arrays y_test and y_pred_proba that you computed in the previous exercise are available in your workspace.


Import roc_auc_score from sklearn.metrics.
Compute ada's test set ROC AUC score, assign it to ada_roc_auc, and print it out.

from sklearn.metrics import roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))

###################################################