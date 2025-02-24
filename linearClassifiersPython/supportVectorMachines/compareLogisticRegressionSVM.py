### COMPARING LOGISTIC REGRESSION AND SVM


# Logistic Regression:
    -Linear Classifier
    -can use kernels, very slow
    -outputs meaningful probabilities
    -can be extended to multi-class
    -all data points affect the model FITTING
    -L2 or L1 regularization and Logistic Loss Function

# Support Vector Machine:
    -Linear Classifier
    -can use with kernels, and fast
    -does NOT naturally output probabilities
    -can be extended to multi-class
    -only "support vectors" affect the FIT
    -conventionally just L2 regularization and Hinge Loss Function



# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

-Hyperparams
    -C (inverse regularization strength)
    -penalty (type of regularization)
    -multi_class (type of multi_class)

# SVM
from sklearn.svm import LinearSVC, SVC

-Hyperparams
    -C (inverse regularization strength)
    -kernel (type of kernel)
    -gamma (inverse RBF smoothness)

# SGDClassifier (Stochastic Gradient Descent)
-scales well to large datasets better than SVM and LogisticRegression
    -a Linear Classifier
    -can switch between LogisticRegression and SVM by setting the loss hyperparam

from sklearn.linear_model import SGDClassifier

logreg = SGDCLassifier(loss='log_loss')

linsvm = SGDClassifier(loss='hinge')

-Hyperparams
    -alpha is the inverse of C hyperparam

###########################################

# PRACTICE

# Using SGDClassifier

In this final coding exercise, you'll do a hyperparameter search over the regularization strength and the loss (logistic regression vs. linear SVM) using SGDClassifier().

Instantiate an SGDClassifier instance with random_state=0.
Search over the regularization strength and the hinge vs. log_loss losses.

linear_classifier = SGDClassifier(random_state=0)

parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
              'loss':['log_loss', 'hinge']}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(X_train, y_train)

# Report the best params and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
#########################################################################################

# CONCLUSION

-Data Science
    -the process of answering questions and making decision based on data
        -the process combines data collection, data prep, database design, visualization, communication, software engineering, MACHINE LEARNING, etc.

-Machine Learning
    -several branches:
        -Supervised Learning
        -Unsupervised Learning
        -Reinforcement Learning

-Supervised Learning
    -trying to predict a Target value from features when given a labeled dataset
    -focused on Classification
        -prediction is categorical rather than continuous in nature

-Linear Classifiers (LogisticRegression and SVMs)
#####################################################