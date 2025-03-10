#### BOOT STRAP AGGREGATION (BAGGING)


# Ensemble Methods
-Voting Classifier
    -same training set
    -NOT = to algorithms (different algos)

-Bagging
    -ONE algorithm (uses same algorithm)
    -NOT = to subsets of the train set (each model is trained on a different subset)


# Bootstrap Aggregation

-uses bootstrap technique
    -reduces Variance of individual models in the ensemble


-Original Set
A B C 

    -a sample drawn from above with a replacement (any bowl can be drawn anytime)

-Bootstrap Samples

B B B
A B A

A C C

# Training

-drawing N different bootstrap samples from the Train Set
    -then samples are used to train N models which use the SAME algorithm

-The Meta-model then collects the models' predictions and produces a final prediction


# Classification & Regression

-Classification:
    -Aggregates predictions by MAJORITY Voting

-BaggingClassifier in sklearn


-Regression:
    -Aggregates predictions through AVERAGING

-BaggingRegressor in sklearn


# Bagging Classifier in sklearn (Breast-Cancer dataset)

# import models and utlity functions
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# set SEED
SEED = 1

# split data into 70% train and 30% test
X_train, y_train, X_test, y_test= train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)


# instantiate a classification tree 'dt'
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
# instantiate a BaggingClassifier 'bc'
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)
# fit 'bc' to the training set
bc.fit(X_train, y_train)
# predict Test set labels
y_pred = bc.predict(X_test)

# eval and print Test Set accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))

Accuracy of Bagging Classifier: 0.936

-training dt to the SAME train set = 88.9%
    -BAGGING outperforms this

##################################################

# PRACTICE

# Define the Bagging Classifier
In the following exercises you'll work with the Indian Liver Patient dataset from the UCI machine learning repository. 
Your task is to predict whether a patient suffers from a liver disease using 10 features including Albumin, age and gender. 
You'll do so using a Bagging Classifier.


Import DecisionTreeClassifier from sklearn.tree and BaggingClassifier from sklearn.ensemble.
Instantiate a DecisionTreeClassifier called dt.
Instantiate a BaggingClassifier called bc consisting of 50 trees.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier(random_state=1)
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)
####################################################

# Evaluate Bagging Performance
Now that you instantiated the bagging classifier, it's time to train it and evaluate its test set accuracy.
The Indian Liver Patient dataset is processed for you and split into 80% train and 20% test. 
The feature matrices X_train and X_test, as well as the arrays of labels y_train and y_test are available in your workspace.
In addition, we have also loaded the bagging classifier bc that you instantiated in the previous exercise and the function accuracy_score() from sklearn.metrics.

Fit bc to the training set.
Predict the test set labels and assign the result to y_pred.
Determine bc's test set accuracy.

bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
print(f'Test set accuracy of bc: {:.2f}'.format(acc_test))

<script.py> output:
    Test set accuracy of bc: 0.67
    
###############################