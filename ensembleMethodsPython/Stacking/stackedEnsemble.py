### STACKING ENSEMBLE ###


# Stack Models with sklearn

1. sklearn provides stacking Estimators
2. compatible with other sklearn Estimators
3. FINAL Estimator is trained through Cross-Validation

-General Steps:
    1. prep dataset
    2. build first-layer Estimators
    3. APPEND the Preds to the dataset
    4. build the second-layer Meta Estimator
    5. use the Stacked Ensemble for Preds


# Stacking Classifier
from sklearn.ensemble import StackingClassifier

# Instantiate the first-layer classifiers
classifiers = [
    ('clf1', Classifier1(params1)),
    ('clf2', Classifier2(params2)),
    ...,
    ('clfN', ClassifierN(paramsN))
]

# Instantiate the 2nd-layer classifier
clf_meta = ClassifierMeta(paramsMeta)

# Build the Stacking Classifier
clf_stack = StackingClassifier(
    estimators=classifiers,
    fina_estimator=clf_meta,
    cv=5,
    stack_method='predict_proba',
    passthrough=False
)

# use the fit and predict methods
clf_stack.fit(X_train, y_train)
pred = clf_stack.predict(X_test)


# Stacking Regressor
from sklearn.ensemble import StackingRegressor
regressors = [
    ('reg1', Regressor1(params1)),
    ('reg2', Regressor2(params2)),
    ...,
    ('regN', RegressorN(paramsN))
]

reg_meta = RegressorMeta(paramsMeta)

reg_stack = StackingRegressor(
    estimators=regressors,
    final_estimator=reg_meta,
    cv=5,
    passthrough=False
)

reg_stack.fit(X_train, y_train)
pred = reg_stack.predict(X_test)


##########################

# PRACTICE

# Applying Stacking to Predict App Ratings

Build and fit a decision tree classifier with: min_samples_leaf: 3 and min_samples_split: 9.
Build and fit a 5-nearest neighbors classifier using: algorithm: 'ball_tree' (to expedite the processing).
Evaluate the performance of each estimator using the accuracy score on the test set.

clf_dt = DecisionTreeClassifier(min_samples_leaf=3, min_samples_split=9, random_state=500)
clf_dt.fit(X_train, y_train)

clf_knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
clf_knn.fit(X_train, y_train)

pred_dt = clf_dt.predict(X_test)
pred_knn = clf_knn.predict(X_test)

print('Decision Tree: {:0.4f}'.format(accuracy_score(y_test, pred_dt)))
print('5-Nearest Neighbors: {:0.4f}'.format(accuracy_score(y_test, pred_knn)))

Decision Tree: 0.5930
5-Nearest Neighbors: 0.5684
-Decision Tree performed better than KNN

#########################

# Building the Stacking Classifier

Prepare the list of tuples with the first-layer classifiers: clf_dt and clf_knn (specifying the appropriate labels and order).
Instantiate the second-layer meta estimator: a LogisticRegression.

Build the stacking classifier passing: the list of tuples, the meta classifier, 
with stack_method='predict_proba' (to use class probabilities), and passthrough = False (to only use predictions as features).

classifiers = [
    ('dt', clf_dt),
    ('knn', clf_knn)
]

clf_meta = LogisticRegression()

clf_stack = StackingClassifier(
    estimators=classifiers,
    final_estimator=clf_meta,
    stack_method='predict_proba',
    passthrough=False
)

# Stack Preds for App Ratings

Fit the stacking classifier on the training set.
Calculate the final predictions from the stacking estimator on the test set.
Evaluate the performance on the test set using the accuracy score.

clf_stack.fit(X_train, y_train)
pred_stack = clf_stack.predict(X_test)
print('Accuracy: {:0.4f}'.format(accuracy_score(y_test, pred_stack)))

Accuracy: 0.6424

-Stacking improved the accuracy from 0.5930 to 0.6424

######################################