### SCIKIT-LEARN REFRESH

# Fitting and Predicting

import sklearn.datasets

newsgroups = sklearn.datasets.fetch_20newsgroups_vectorized()

X, y = newsgroups.data, newsgroups.target

X.shape
    -about 11,000 training examples

    -about 130,000 features
        -derived from the words appearing in news articles

y.shape
    -article topics which we are trying to predict

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
    -instantiate and store in knn
    -this is where we specify model HYPERPARAMS (ex. number of neighbors for knn)

knn.fit(X,y)
    -standard syntax for fitting a model


y_pred = knn.predict(X)
    -make predictions on any dataset (including original training dataset 'X')
    -contains one entry per row of X with the prediction from the trained classifier

# Model Eval

-Compute the score (Pre-split)

knn.score(X,y)
    -number is not very meaningful
    -we want to know how the model GENERALIZES to unseen data
        -ability to generalize is measured with a VALIDATION SET

0.99991


-Splitting Data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_test/y_test = VALIDATION SET


knn.fit(X_train, y_train)
    -before computing the test set; our model is training on the training set only


-Compute test score
knn.score(X_test, y_test)

0.66242

-This is much lower than before (testing accuracy)
    -This measures the model's ability to classify new data


# PRACTICE

### KNN CLASSIFICATION

Create a KNN model with default hyperparameters.
Fit the model.
Print out the prediction for the test example 0.
    


from sklearn.neighbors import KNeighborsClassifier

# Create and fit the model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

# Predict on the test features, print the results
pred = knn.predict(X_test)[0]
print("Prediction for test example 0:", pred)

NOTE: OVERFITTING is where the Training accuracy is much greater than the Testing accuracy

######################################################################################################

# APPLYING LOGISTIC REGRESSION AND SVM

-Using LogisticRegression
    -A LINEAR CLASSIFIER THAT IS LEARNED WITH THE LOGISTIC LOSS FUNCTION

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.predict(X_test)
lr.score(X_test, y_test)


# EXAMPLE

import sklearn.datasets

wine = sklearn.datasets.load_wine()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(wine.data, wine.target)
lr.score(wine.data, wine.target)

0.966

lr.predict_proba(wine.data[:1]) # Confidence scores rather than hard or definite predictions

array([[9.966e-01, 2.74e-03, 6.787e-04]])
    -reporting over 99% confidence for the first class


# Using LinearSVC (Basic SVM Classifier)
    -Linear Support Vector Classifier
    -works same as LogisticRegression

import sklearn.datasets
wine = sklearn.datasets.load_wine()

svm = LinearSVC()

svm.fit(wine.data, wine.target)
svm.score(wine.data, wine.target)

0.955

-Can repeat for the SVC class [fits a NONLINEAR SVM by default]
    -using default HyperParams

import sklearn.datasets
wine = sklearn.datasets.load_wine()
from sklearn.svm import SVC
svm = SVC()
svm.fit(wine.data, wine.target);
svm.score(wine.data, wine.target)

0.708
-such a classifier could be overfitting (risk when using more complex models like NONLINEAR SVM)

# NOTES:
        -a choice about the model we make before fitting to the data; often controls model complexity
            -UNDERFITTING:
                model is too simple = low training accuracy (unable to capture data patterns)
            -OVERFITTING:
                model is too complex = low TEST accuracy (may learn peculiarities of our training set)

# PRACTICE

# Running LogisticRegression and SVC

Apply logistic regression and SVM (using SVC()) to the handwritten digits data set using the provided train/validation split.
For each classifier, print out the training and validation accuracy.

from sklearn import datasets
digits = datasets.load_digits()
X_train, X_test, y_train, y_test, = train_test_split(digits.data, digits.target)

# Apply logistic regression and print scores
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression
lr.fit(digits.data, digits.target)
print(lr.score(digits.data, digits.target))
print(lr.score(digits.data, digits.target))

# Apply SVM and print scores

from sklearn.svm import SVC
svm = SVC()
svm.fit(digits.data, digits.target)
print(svm.score(digits.data, digits.target))
print(svm.score(digits.data, digits.target))

###########################

# Sentiment Analysis for Movie Reviews

In this exercise you'll explore the probabilities outputted by logistic regression on a subset of the Large Movie Review Dataset.

The variables X and y are already loaded into the environment. X contains features based on the number of times words appear in the movie reviews, and y contains labels for whether the review sentiment is positive (+1) or negative (-1).

Train a logistic regression model on the movie review data.
Predict the probabilities of negative vs. positive for the two given reviews.
Feel free to write your own reviews and get probabilities for those too!

# Instantiate logistic regression and train
lr = LogisticRegression()
lr.fit(X, y)

# Predict Sentiment for a positive review
review1 = "LOVED IT ! This movie was amazing. Top 10 this year."
review1_features = get_features(review1)
print("Review:", review1)
print("Probability of positive review:", lr.predict_proba(review1_features)[0,1])

# Predict Sentiment for a negative review
review2 = "Total junk! I'll never watch a film by that director again, no matter how good the reviews."
review2_features = get_features(review2)
print("Review:", review2)
print("Probability of positive review:", lr.predict_proba(review2_features)[0,1])

#####################################

# LINEAR CLASSIFIERS

# Linear Decision Boundaries
-Decision Boundaries tell us what class our classifier will predict for any value of X

-This definition extends to more than 2 features
    -with 5 features the space of possible X values is 5 dimensional (the boundary will be higher dimensional hyperplane)

-A Nonlinear boundary
    -sometimes leads to non-contiguous regions of a certain prediction


# LogisticRegression and SVM are Linear Classifiers (in their basic forms)

# Vocab

-Classification: supervised learning when y values are categories (to predict categories)
    -this is in contrast with Regression

-Decision Boundary: The surface separating different predicted classes

-Linear Classifier: a Classifier that learns linear decision boundaries
    -e.g. Logistic Regression, linear SVM

-Linearly Separable: a data set can be perfectly explained by a linear classifier

##################################

# Visualizing Decision Boundaries

Create the following classifier objects with default hyperparameters: LogisticRegression, LinearSVC, SVC, KNeighborsClassifier.
Fit each of the classifiers on the provided data using a for loop.
Call the plot_4_classifers() function (similar to the code here), passing in X, y, and a list containing the four classifiers.


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Define the classifiers
classifiers = [LogisticRegression(),
               SVC(),
               LinearSVC(),
               KNeighborsClassifier()
]

# Fit the classifiers
for c in classifiers:
    c.fit(X,y)

# Plot the classifiers
plot_4_classifiers(X, y, classifiers)
plt.show()