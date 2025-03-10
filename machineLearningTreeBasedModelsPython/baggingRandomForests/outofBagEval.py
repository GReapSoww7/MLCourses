### OUT OF BAG EVALUATION


# Bagging
-some instances may be sampled several times for ONE model
-OTHER instances may NOT be sampled at ALL

# Out of Bag (OOB) Instances
-on AVG, for each model, 63% of the training instances are sampled.
    -the remaining 37% constitute the OOB instances (unseen; used to estimate ensemble performance WITHOUT cross-validation)

OOB score = OOB1 + ... + OOBn / N (average of the OOB evaluations)


# OOB Eval in sklearn (Breast Cancer Dataset)


# import models and split utility function
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set SEED
SEED = 1
# split data into 70% train and 30% test

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

# instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)

# instantiate a BaggingClassifier 'bc'; set oob_score=True
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, oob_score=True, n_jobs=-1)
    # oob_score CORRESPONDS to the accuracy_score for Classifiers and the r^2 score for Regressors

# fit 'bc' to the train set
bc.fit(X_train, y_train)

# predict the test set labels
y_pred = bc.predict(X_test)


# eval test set accuracy
test_accuracy = accuracy_score(y_test, y_pred)

# extract OOB accuracy from 'bc' and assign
oob_accuracy = bc.oob_score_

# print test accuracy
print('Test set accuracy: {:.3f}'.format(test_accuracy))

Test set accuracy: 0.936

# print OOB accuracy
print('OOB accuracy: {:.3f}'.format(oob_accuracy))

OOB accuracy: 0.925

-OOB can be an efficient performance estimate of a Bagged Ensemble on UNSEEN data WITHOUT performing Cross-Validation

############################################

# PRACTICE

# Prepare the Ground

In the following exercises, you'll compare the OOB accuracy to the test set accuracy of a bagging classifier trained on the Indian Liver Patient dataset.
In sklearn, you can evaluate the OOB accuracy of an ensemble classifier by setting the parameter oob_score to True during instantiation. 
After training the classifier, the OOB accuracy can be obtained by accessing the .oob_score_ attribute from the corresponding instance.
In your environment, we have made available the class DecisionTreeClassifier from sklearn.tree.

Import BaggingClassifier from sklearn.ensemble.
Instantiate a DecisionTreeClassifier with min_samples_leaf set to 8.
Instantiate a BaggingClassifier consisting of 50 trees and set oob_score to True.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

dt = DecisionTreeClassifier(min_samples_leaf=8, random_state=1)
bc = BaggingClassifier(base_estimator=dt, n_estimator=50, oob_score=True, random_state=1)


#####################################################

# OOB Score vs Test Set Score

Now that you instantiated bc, you will fit it to the training set and evaluate its test set and OOB accuracies.
The dataset is processed for you and split into 80% train and 20% test. 
The feature matrices X_train and X_test, as well as the arrays of labels y_train and y_test are available in your workspace. 
In addition, we have also loaded the classifier bc instantiated in the previous exercise and the function accuracy_score() from sklearn.metrics.

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Fit bc to the training set and predict the test set labels and assign the results to y_pred.
Evaluate the test set accuracy acc_test by calling accuracy_score.
Evaluate bc's OOB accuracy acc_oob by extracting the attribute oob_score_ from bc.

bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
acc_oob = bc.oob_score_
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))

#########################################################################