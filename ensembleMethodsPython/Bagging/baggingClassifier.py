### BAGGINGCLASSIFIER: NUTS AND BOLTS ###


# Heterogeneous vs Homogeneous Functions

-Heterogeneous Ensemble Function
het_est = HeterogeneousEnsemble(
    estimators=[('est1', est1), ('est2', est2), ..., ('estN', estN)],
    # additional parameters
)

-Homogeneous Ensemble Function
hom_est = HomoegeneousEnsemble(
    est_base,
    n_estimators=chosen_number,
    # additional parameters
)


# BaggingClassifier
-Bagging Classifier example:

# Instantiate the base estimator ('weak' model)
clf_dt = DecisionTreeClassifier(max_depth=3)

# Build the Bagging Classifier with 5 estimators
clf_bag = BaggingClassifier(
    clf_dt,
    n_estimators=5
)

# Fit the Bagging model to the training set
clf_bag.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf_bag.predict(X_test)



# BaggingRegressor

-BaggingRegressor example:

# Instantiate the base estimator ('weak' model)
reg_lr = LinearRegression()

# Build the Bagging Regressor with 10 estimators
reg_bag = BaggingRegressor(
    reg_lr
)

# Fit the Bagging model to the training set
reg_bag.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg_bag.predict(X_test)

# Out-of-Bag Score

-Calculate the individual Predictions using ALL Estimators for which an Instance was OUT of the Sample
-COMBINE the individual Predictions
-Eval the Metric on those Predictions:
    -Classification: accuracy
    -Regression: R^2

clf_bag = BaggingClassifier(
    clf_dt,
    oob_score=True
)
clf_bag.fit(X_train, y_train)
print(clf_bag.oob_score_)
0.9328125

pred = clf_bag.predict(X_test)
print(accuracy_score(y_test, pred))
0.9625

######################

# PRACTICE

# Bagging: the sklearn way

You obtained an F1 score of around 0.63 with your custom bagging ensemble.

Will BaggingClassifier() beat it? Time to find out!

Instantiate the base model, clf_dt: a "restricted" decision tree with a max depth of 4.
Build a bagging classifier with the decision tree as base estimator, using 21 estimators.
Predict the labels of the test set.

# Instantiate the base model
clf_dt: DecisionTreeClassifier(max_depth=4)

# Build the BaggingClassifier
clf_bag = BaggingClassifier(clf_dt, n_estimators=21, random_state=500)

# Fit the Bagging model to the train set
clf_bag.fit(X_train, y_test)

# Predict the labels of the test set
pred = clf_bag.predict(X_test)

# Show the F1-Score
print(f'F1-Score: {:.3f}'.format(f1_score(y_test, pred)))
F1-Score: 0.667

-this BaggingClassifier performance better than the Custom Bagging Ensemble

##################################

# Checking the out-of-bag score

Let's now check the out-of-bag score for the model from the previous exercise.

So far you've used the F1 score to measure performance.
However, in this exercise you should use the accuracy score so that you can easily compare it to the out-of-bag score.

The decision tree classifier from the previous exercise, clf_dt, is available in your workspace.

The pokemon dataset is already loaded for you and split into train and test sets.
In addition, the decision tree classifier was fit and is available for you as clf_dt to use it as base estimator.


Build the bagging classifier using the decision tree as base estimator and 21 estimators.
This time, use the out-of-bag score by specifying an argument for the oob_score parameter.
Print the classifier's out-of-bag score.

# Build and Train the BaggingClassifier
clf_bag = BaggingClassifier(clf_dt, n_estimators=21, oob_score=True, random_state=500)
clf_bag.fit(X_train, y_train)

# Print the oob_score
print(f'OOB-Score: {:.3f}'.format(clf_bag.oob_score_))

# Eval the performance on the test set to compare
pred = clf_bag.predict(X_test)
print(f'Accuracy: {:.3f}'.format(accuracy_score(y_test, pred)))

###############################################