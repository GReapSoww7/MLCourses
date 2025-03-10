### ENSEMBLE LEARNING


# Advantages of CARTs
-simple to understand/interpret
    -easy to use
-flexibility: describe non-linear dependencies
-preprocessing: NO need to Standardize or Normalize FEATURES

# Limitations
-Classification: can ONLY produce orthogonal decision boundaries
-sensitive to SMALL variations in the train set
-High variance: unconstrained CARTs may OVERFIT the train set
-Solution: Ensemble learning

# Ensemble Learning
-train different modles on SAME dataset
-let each model make its own Predictions
-Meta-model: aggregates predictions of individual models
-Final prediction: more robutst and less prone to errors
-Best Results: models are skillful in DIFFERENT ways


# Feed Training Set to DIFFERENT Classifiers (P1, P2, P3, P4, ...) > Meta-model > Final Ensemble Prediction
    -Decision Tree, Logistic Regression, KNN, etc


# Ensemble Technique: Voting Classifier

-Binary Classification Task
-N classifiers make predictions (P1, P2, P3, P4, ..., Pn) with Pi = 0 or 1
-Meta-model Prediction: HARD VOTING

# Hard Voting
-consider a Voting Classifier that consists of N classifiers
    -Predictions: 1, 0, 1 > prediction equals 1

# Voting Classifier in sklearn (Breast-Cancer dataset)

# import functions to compute accuracy and split data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# import models, including VotingClassifier meta-model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier

# Set SEED
SEED = 1


# split data into 70% train and 30% test
X_train, y_train, X_test, y_test= train_test_split(X, y, test_size=0.3, random_state=SEED)

# instantiate individual classifiers
lr = LogisticRegression(random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)

# define a list called classifiers that contains the tuples (classifier_name, classifier)
classifiers = [('LogisticRegression', lr),
               ('K Nearest Neighbors', knn),
               ('Classification Tree', dt)]


# iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
    # fit clf to the train set
    clf.fit(X_train, y_train)

    # predict the labels of the Test Set
    y_pred = clf.predict(X_test)

    # eval the accuracy of clf on the Test Set
    print(f'{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))


output:

Logistic Regression: 0.947
K Nearest Neighbors: 0.930
Classification Tree: 0.930



# instantiate the VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers)

# fit 'vc' to the Train Set and Predict Test Set labels
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)

# eval the Test Set accuracy of 'vc'
print(f'Voting Classifier: {.3f}'.format(accuracy_score(y_test, y_pred)))

Voting Classifier: 0.953
    -results in a higher accuracy than the individual Classifiers ALONE
####################################################################################

# PRACTICE

# Define the Ensemble

In the following set of exercises, you'll work with the Indian Liver Patient Dataset from the UCI Machine learning repository.
In this exercise, you'll instantiate three classifiers to predict whether a patient suffers from a liver disease using all the features present in the dataset.
The classes LogisticRegression, DecisionTreeClassifier, and KNeighborsClassifier under the alias KNN are available in your workspace.

Instantiate a Logistic Regression classifier and assign it to lr.
Instantiate a KNN classifier that considers 27 nearest neighbors and assign it to knn.
Instantiate a Decision Tree Classifier with the parameter min_samples_leaf set to 0.13 and

from sklearn.neighbors import KNeighborsClassifier as KNN
lr = LogisticRegression(random_state=SEED)
knn = KNN(n_neighbors=27)
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)
# define the list classifiers
classifiers = [('Logistic Regression', lr),
               ('K Nearest Neighbors', knn),
               ('Classification Tree', dt)]
########################################################################################

# Eval Individual Classifiers

In this exercise you'll evaluate the performance of the models in the list classifiers that we defined in the previous exercise. 
You'll do so by fitting each classifier on the training set and evaluating its test set accuracy.

The dataset is already loaded and preprocessed for you (numerical features are standardized) and it is split into 70% train and 30% test. 
The features matrices X_train and X_test, as well as the arrays of labels y_train and y_test are available in your workspace. 
In addition, we have loaded the list classifiers from the previous exercise, as well as the function accuracy_score() from sklearn.metrics.

Iterate over the tuples in classifiers. Use clf_name and clf as the for loop variables:
Fit clf to the training set.
Predict clf's test set labels and assign the results to y_pred.
Evaluate the test set accuracy of clf and print the result.

for clf_name, clf in classifiers:
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f'{:s} : {:.3f}'.format(clf_name, accuracy))

<script.py> output:
    Logistic Regression : 0.741
    K Nearest Neighbours : 0.701
    Classification Tree : 0.707

####################################################################################

# Better Performance with a Voting Classifier

Finally, you'll evaluate the performance of a voting classifier that takes the outputs of the models defined in the list classifiers and assigns labels by majority voting.
X_train, X_test,y_train, y_test, the list classifiers defined in a previous exercise, as well as the function accuracy_score from sklearn.metrics are available in your workspace.

Import VotingClassifier from sklearn.ensemble.
Instantiate a VotingClassifier by setting the parameter estimators to classifiers and assign it to vc.
Fit vc to the training set.
Evaluate vc's test set accuracy using the test set predictions y_pred.

from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Voting Classifier {:.3f}'.format(accuracy))

<script.py> output:
    Voting Classifier: 0.764
###################################################################################################