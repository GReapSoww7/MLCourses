### MLXTEND ###

# MLxtend
    -Machine Learning Extensions
    -Utilities and Tools for Data Science Tasks:
        -Feature Selection
        -Ensemble Methods
        -Visualization
        -Model Evaluation
    -Friendly API
    -Compatible with sklearn Estimators


# MLxtend Stacking Implementation
-Characteristics:
    -Individual Estimators are trained on COMPLETE Features
    -the Meta-Estimator is trained using the Preds as the ONLY Meta-Features
    -the Meta-Estimator can be trained with Labels or Probabilities as Target

# MLxtend StackingClassifier
from mlxtend.classifier import StackingClassifier

# Instantiate the 1st-layer estimators
clf1 = Classifier1(params1)
clf2 = Classifier2(params2)
...
clfN = ClassifierN(paramsN)

# Instantiate the 2nd-layer classifier
clf_meta = ClassifierMeta(paramsMeta)

# Build the StackingClassifier
clf_stack = StackingClassifier(
    classifiers=[clf1, clf2, ..., clfN],
    meta_classifier=clf_meta,
    use_probas=False,
    use_features_in_secondary=False
)

# Use the Fit and Predict Methods
clf_stack.fit(X_train, y_train)
pred = clf_stack.predict(X_test)

# MLxtend StackingRegressor
from mlxtend.regressor import StackingRegressor
reg1 = Regressor1(params1)
reg2 = Regressor2(params2)
...
regN = RegressorN(paramsN)

reg_meta = RegressorMeta(paramsMeta)

reg_stack = StackingRegressor(
    regressors=[reg1, reg2, ..., regN],
    meta_regressor=reg_meta,
    use_features_in_secondary=False
)
reg_stack.fit(X_train, y_train)
pred = reg_stack.predict(X_test)


######################################

# PRACTICE

# MLxtend

Instantiate a decision tree classifier with min_samples_leaf = 3 and min_samples_split = 9.
Instantiate a 5-nearest neighbors classifier using the 'ball_tree' algorithm.
Build a StackingClassifier passing: the list of classifiers, the meta classifier, use_probas=True (to use probabilities), 
and use_features_in_secondary = False (to only use the individual predictions).
Evaluate the performance by computing the accuracy score.

clf_dt = DecisionTreeClassifier(min_samples_leaf=3, min_samples_split=9, random_state=500)
clf_knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')

clf_meta = LogisticRegression()

clf_stack = StackingClassifier(
    classifiers=[clf_dt, clf_knn],
    meta_classifier=clf_meta,
    use_probas=True,
    use_features_in_seconday=False
)
clf_stack.fit(X_train, y_train)
pred_stack = clf_stack.predict(X_train)
print('Accuracy: {:0.4f}'.format(accuracy_score(y_test, pred_stack)))
Accuracy: 0.6050

##################

# Stacking Regressor

To practice using the StackingRegressor, we'll go back to the regression approach.
As usual, the input features have been standardized for you with a StandardScaler().

The MAE (mean absolute error) is the evaluation metric. In Chapter 1, the MAE was around 0.61.
Let's see if the stacking ensemble method can reduce that error.

Instantiate a decision tree regressor with: min_samples_leaf = 11 and min_samples_split = 33.
Instantiate the default linear regression.
Instantiate a Ridge regression model with random_state = 500.
Build and fit a StackingRegressor, passing the regressors and the meta_regressor.


reg_dt = DecisionTreeRegressor(min_samples_leaf=11, min_samples_split=33, random_state=500)
reg_lr = LinearRegression()
reg_ridge = Ridge(random_state=500)

reg_meta = LinearRegression()

clf_stack = StackingRegressor(
    regressors=[reg_dt, reg_lr, reg_ridge],
    meta_regressor=reg_meta,
)
clf_stack.fit(X_train, y_train)

pred = clf_stack.predict(X_test)
print('MAE: {:.3f}'.format(mean_absolute_error(y_test, pred)))
MAE: 0.587
-this stacking could reduce the error, which is an improvement


#############

# Mushrooms

Instantiate the first-layer estimators: a 5-nearest neighbors using the ball tree algorithm, 
a decision tree classifier with parameters min_samples_leaf = 5 and min_samples_split = 15, and a Gaussian Naive Bayes classifier.

Build and fit a stacking classifier, using the parameters classifiers - a list containing the first-layer classifiers - 
and meta_classifier - the default logistic regression.

clf_knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
clf_dt = DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=15, random_state=500)
clf_nb = GaussianNB()

clf_lr = LogisticRegression()

clf_stack = StackingClassifier(
    classifiers=[clf_knn, clf_dt, clf_nb],
    meta_classifier=clf_lr
)
clf_stack.fit(X_train, y_train)
pred_stack = clf_stack.predict(X_test)
print('Accuracy: {:0.4f}'.format(accuracy_score(y_test, pred_stack)))
Accuracy: 1.0000
-This is a quality Ensemble; we have increased the accuracy completely

#####################################################