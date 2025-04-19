### INTUITION BEHIND STACKING ###


# Stacking
-analogous to a relay race
-individual Models pass Preds together with the Input Features to next Model

Dataset > model1, model2, ..., modelN (first-layer estimators) > pred1, pred2, ..., predN > Combiner Model (second-layer estimators) > Final Preds

###################

# PRACTICE

# Predicting Mushroom Edibility

As both the features and the target are categorical, these have been transformed into "dummy" binary variables for you.

Let's begin with Naive Bayes (using scikit-learn's GaussianNB) and see how this algorithm performs on this problem.

Instantiate a GaussianNB classifier called clf_nb.
Fit clf_nb to the training data X_train and y_train.
Calculate the predictions on the test set. These predictions will be used to evaluate the performance using the accuracy score.

clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)
pred = clf_nb.predict(X_test)
print('Accuracy: {:0.4f}'.format(accuracy_score(y_test, pred)))
Accuracy: 0.9671

#########################

# K-nearest Neighbors for Mushrooms

In this case, the algorithm to use is a 5-nearest neighbors classifier.
As the dummy features create a high-dimensional dataset, use the Ball Tree algorithm to make the model faster.
Let's see how this model performs!

Build a KNeighborsClassifier with 5 neighbors and algorithm = 'ball_tree' (to expedite the processing).
Fit the model to the training data.
Evaluate the performance on the test set using the accuracy score.

clf_knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
clf_knn.fit(X_train, y_train)
pred = clf_knn.predict(X_test)
print('Accuracy: {:0.4f}'.format(accuracy_score(y_test, pred)))
Accuracy: 1.0000

##########################################