### AVERAGING ###


# Counting Jelly Beans

-Guessing (Random Num)
-Volume Approximation
-Actual Val ~ mean(estimates)

# Averaging (Soft Voting)
-Properties:
    -Classification & Regression problems
    -Soft Voting: Mean
        -Regression: Mean of Pred Vals
        -Classification: Mean of Pred Probabilities
-NEED at LEAST 2 Estimators

# Averaging Ensemble with sklearn
-Averaging Classifier

from sklearn.ensemble import VotingClassifier
clf_voting = VotingClassifier(
    estimators=[('label1', clf_1), ('label2', clf_2), ..., ('labelN', clf_N)], 
    voting='soft', 
    weights=[w_1, w_2, ..., w_N]
)

-Averaging Regressor

from sklearn.ensemble import VotingRegressor
reg_voting = VotingRegressor(
    estimators=[('label1', reg_1), ('label2', reg_2), ..., ('labelN', reg_N)],
    weights=[w_1, w_2, ..., w_N]
)

# Example

# Instantiate the Individual Models
clf_knn = KNeighborsClassifier(5)
clf_dt = DecisionTreeClassifier()
clf_lr = LogisticRegression()

# Create an Averaging Classifier
clf_voting = VotingClassifier(
    estimators=[('knn', clf_knn), ('dt', clf_dt), ('lr', clf_lr)],
    voting='soft',
    weights=[1, 2, 1]
)

#####################

# PRACTICE

# Predicting GoT Deaths

While the target variable does not have any missing values, other features do. As the focus of the course is not on data cleaning and preprocessing, we have already done the following preprocessing for you:

Replaced NA values with 0.
Replace negative values of age with 0.
Replace NA values of age with the mean.
Let's now build an ensemble model using the averaging technique.
The following individual models have been built:

Logistic Regression (clf_lr).
Decision Tree (clf_dt).
Support Vector Machine (clf_svm).
As the target is binary, all these models might have good individual performance. 
Your objective is to combine them using averaging.
Recall from the video that this is the same as a soft voting approach, so you should still use the VotingClassifier().


Set up the list of (string, estimator) tuples. Use 'lr' for clf_lr, 'dt' for clf_dt, and 'svm' for clf_svm.
Build an averaging classifier called clf_avg. Be sure to specify an argument for the voting parameter.


# Build the individual models
clf_lr = LogisticRegression(class_weight='balanced')
clf_dt = DecisionTreeClassifier(min_samples_leaf=3, min_samples_split=9, random_state=500)
clf_svm = SVC(probability=True, class_weight='balanced', random_state=500)

# List of (string, estimator) tuples
estimators = [('lr', clf_lr), ('dt', clf_dt), ('svm', clf_svm)]

# Build and fit an averaging classifier
clf_avg = VotingClassifier(
    estimators, voting='soft'
)
clf_avg.fit(X_train, y_train)

# Eval model performance
acc_avg = accuracy_score(y_test, clf_avg.predict(X_test))
print('Accuracy: {:.2f}'.format(acc_avg))

Accuracy: 0.82

########################

# Soft vs. hard voting
You've now practiced building two types of ensemble methods: Voting and Averaging (soft voting).
Which one is better? It's best to try both of them and then compare their performance.
Let's try this now using the Game of Thrones dataset.

Three individual classifiers have been instantiated for you:

A DecisionTreeClassifier (clf_dt).
A LogisticRegression (clf_lr).
A KNeighborsClassifier (clf_knn).
Your task is to try both voting and averaging to determine which is better.


Prepare the list of (string, estimator) tuples. Use 'dt' as the label for clf_dt, 'lr' for clf_lr, and 'knn' for clf_knn.
Build a voting classifier called clf_vote.
Build an averaging classifier called clf_avg.


# List of estimators
estimators = [('dt', clf_dt), ('lr', clf_lr), ('knn', clf_knn)]

# Build and fit an averaging classifier
clf_vote = VotingClassifier(
    estimators
)
clf_vote.fit(X_train, y_train)


# Build and fit an averaging classifier
clf_avg = VotingClassifier(
    estimators, voting='soft'
)
clf_avg.fit(X_train, y_train)

# Eval performance of both models
acc_vote = accuracy_score(y_test, clf_vote.predict(X_test))
acc_avg = accuracy_score(y_test, clf_avg.predict(X_test))
print('Voting: {:.2f}, Averaging: {:.2f}'.format(acc_vote, acc_avg))


Voting: 0.80, Averaging: 0.81

#######################################