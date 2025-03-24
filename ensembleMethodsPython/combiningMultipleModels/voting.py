### VOTING ###


# Majority Voting
-Properties:
    -Classification problems
    -Majority Voting: Mode
    -Odd Num of Classifiers

-Wise Crowd Characteristics:
    -Diverse: different algorithms or datasets
    -Independent and Uncorrelated
    -Use Individual Knowledge
    -Aggregate Individual Predictions

# Voting Ensemble sklearn

from sklearn.ensemble import VotingClassifier
clf_voting = VotingClassifier(estimators=[('label1', clf_1), ('label2', clf_2), ('labelN', clf_N)])
# create individual models
clf_knn = KNeighborsClassifier(5)
clf_dt = DecisionTreeClassifier()
clf_lr = LogisticRegression()
# create voting classifier
clf_voting = VotingClassifier(estimators=[('knn', clf_knn), ('dt', clf_dt), ('lr', clf_lr)])
# Fit
clf_voting.fit(X_train, y_train)
y_pred = clf_voting.predict(X_test)
# Eval
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {:0.3f}'.format(acc))
##########

# PRACTICE

# Choosing the best model
In this exercise, you'll compare different classifiers and choose the one that performs the best.

The dataset here - already loaded and split into train and test sets - consists of Pokémon - 
their stats, types, and whether or not they're legendary. The objective of our classifiers is to predict this 'Legendary' variable.

Three individual classifiers have been fitted to the training set:

clf_lr is a logistic regression.
clf_dt is a decision tree.
clf_knn is a 5-nearest neighbors classifier.
As the classes here are imbalanced - only 65 of the 800 Pokémon in the dataset are legendary - 
we'll use F1-Score to evaluate the performance.
Scikit-learn's f1_score() has been imported for you.

1.
Predict the labels of X_test using each of the classifiers - clf_lr, clf_dt, and clf_knn.

# predict the labels of the test set
pred_lr = clf_lr.predict(X_test)
pred_dt = clf_dt.predict(X_test)
pred_knn = clf_knn.predict(X_test)


# Eval
score_lr = f1_score(y_test, pred_lr)
score_dt = f1_score(y_test, pred_dt)
score_knn = f1_score(y_test, pred_knn)

# print scores
print(score_lr)
print(score_dt)
print(score_knn)

<script.py> output:
    0.5882352941176471
    0.5833333333333334
    0.47619047619047616

-LogisticRegression model scored the best


#################

# Assembling your first ensemble
It's time to build your first ensemble model! The Pokémon dataset from the previous exercise has been loaded and split into train and test sets.

Your job is to leverage the voting ensemble technique using the sklearn API. 
It's up to you to instantiate the individual models and pass them as parameters to build your first voting classifier.


Instantiate a KNeighborsClassifier called clf_knn with 5 neighbors (specified using n_neighbors).
Instantiate a "balanced" LogisticRegression called clf_lr (specified using class_weight).
Instantiate a DecisionTreeClassifier called clf_dt with min_samples_leaf = 3 and min_samples_split = 9.
Build a VotingClassifier using the parameter estimators to specify the following list of (str, estimator) tuples: 
'knn', clf_knn, 'lr', clf_lr, and 'dt', clf_dt.


clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_lr = LogisticRegression(class_weight='balanced')
clf_dt = DecisionTreeClassifier(min_samples_leaf=3, min_samples_split=9, random_state=500)

clf_vote = VotingClassifier(estimators=[('knn', clf_knn), ('lr', clf_lr), ('dt', clf_dt)])
clf_vote.fit(X_train, y_train)


# Evaluating your ensemble
In the previous exercise, you built your first voting classifier.
Let's now evaluate it and compare it to that of the individual models.

The individual models (clf_knn, clf_dt, and clf_lr) and the voting classifier (clf_vote) have already been loaded and trained.

Remember to use f1_score() to evaluate the performance.
In addition, you'll create a classification report on the test set (X_test, y_test) using the classification_report() function.

Will your voting classifier beat the 58% F1-score of the decision tree?


Use the voting classifier, clf_vote, to predict the labels of the test set, X_test.
Calculate the F1-Score of the voting classifier.
Calculate the classification report of the voting classifier by passing in y_test and pred_vote to classification_report().

pred_vote = clf_vote.predict(X_test)
score_vote = f1_score(y_test, pred_vote)
print(f'F1-Score: {:.3f}'.format(score_vote))
report = classification_report(y_test, pred_vote)
print(report)

<script.py> output:
    F1-Score: 0.583
                  precision    recall  f1-score   support
    
           False       0.98      0.95      0.97       150
            True       0.50      0.70      0.58        10
    
        accuracy                           0.94       160
       macro avg       0.74      0.83      0.77       160
    weighted avg       0.95      0.94      0.94       160
#####################