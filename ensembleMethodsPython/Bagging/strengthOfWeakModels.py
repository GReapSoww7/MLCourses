### THE STRENGTH OF "WEAK" MODELS ###


# "Weak" Model
-Voting and Averaging:
    -small number of estimators
    -Fine-tuned estimators
    -individually trained

# Properties of "weak" models
-Weak Estimator
    -performance BETTER than Random Guessing
    -Light Model
    -Low Training and Eval time
-Ex. Decision Tree


# Examples of "weak" models
-Some "weak" models:
    -Decision tree: SMALL DEPTH
    -Logistic Regression
    -Linear Regression
    -Other restricted models


# Sample Code
model = DecisionTreeClassifier(
    max_depth=3
)
model = LogisticRegression(
    max_iter=50, C=100.0
)
model = LinearRegression()
##################################

# PRACTICE

# Restricted and Unrestricted Decision Trees

Here, you will build two separate decision tree classifiers.
In the first, you will specify the parameters min_samples_leaf and min_samples_split,
but not a maximum depth, so that the tree can fully develop without any restrictions.

In the second, you will specify some constraints by limiting the depth of the decision tree.
By then comparing the two models, you'll better understand the notion of a "weak" learner.


# Build an unrestricted decision tree using the parameters min_samples_leaf=3, min_samples_split=9, and random_state=500.

# Build unrestricted decision tree
clf = DecisionTreeClassifier(min_samples_leaf=3, min_samples_split=9, random_state=500)
clf.fit(X_train, y_train)

# Predict the labels
pred = clf.predict(X_test)

# Print the confusion matrix
cm = confusion_matrix(y_test, pred)
print('Confusion matrix:\n', cm)

# Print the F1 score
pred = clf.predict(X_test)

# Print the confusion matrix
score = f1_score(y_test, pred)
print('F1-Score: {:.3f}'.format(score))

# Print the F1 score
score = f1_score(y_test, pred)
print('F1-Score: {:.3f}'.format(score))

<script.py> output:
    Confusion matrix:
     [[143   7]
     [  3   7]]
    F1-Score: 0.583

# Build a restricted tree by replacing min_samples_leaf and min_samples_split with max_depth=4 and max_features=2.

clf = DecisionTreeClassifier(max_depth=4, max_features=2, random_state=500)

<script.py> output:
    Confusion matrix:
     [[146   4]
     [  5   5]]
    F1-Score: 0.526

Restricted Decision Tree performs worse than Unrestricted

###########################