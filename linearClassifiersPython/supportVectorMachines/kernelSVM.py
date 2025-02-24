### KERNEL SVMs


# Transforming your Features

transformed feature = (original feature)^2

-fit a Linear SVM into these transformed features

-then what does the linear boundary look like in the original feature set space (unsquare)
    -produces an ellipse

-fitting a linear model in a transformed space corresponds to fitting a Non-linear model in the original space

-in general the transformation will NOT always be squaring and the boundary will NOT always be an ellipse


-a NEW Space often has a different number of Dimensions than the original space

# Kernel SVMs
    -implement feature transformations in a computationally efficient way

from sklearn.svm import SVC

svm = SVC(gamma=1) # default is kernel='rbf' = radial basis function kernel
    -not computed this way > think of this as an extremely complicated transformation of features
        -followed by fitting a linear boundary in the new space


-we can control the shape of the boundary using HYPERPARAMETERS
    -C value (regularization)
    -gamma (smoothness of the boundary)
        -decrease == smoother
        -increase == more complex

-higher gamma can cause overfitting

##############################################

# PRACTICE

# GridSearchCV warm-up

In the video we saw that increasing the RBF kernel hyperparameter gamma increases training accuracy. 
In this exercise we'll search for the gamma that maximizes cross-validation accuracy using scikit-learn's GridSearchCV. 
A binary version of the handwritten digits dataset, in which you're just trying to predict whether or not an image is a "2", is already loaded into the variables X and y.

Create a GridSearchCV object.
Call the fit() method to select the best value of gamma based on cross-validation accuracy.


# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X,y)

# Report the best parameters
print("Best CV params", searcher.best_params_)
####################################################

# Jointly Tuning Gamma and C with GridSearchCV

In the previous exercise the best value of gamma was 0.001 using the default value of C, which is 1. 
In this exercise you'll search for the best combination of C and gamma using GridSearchCV.

As in the previous exercise, the 2-vs-not-2 digits dataset is already loaded, but this time it's split into the variables X_train, y_train, X_test, and y_test. 
Even though cross-validation already splits the training set into parts, it's often a good idea to hold out a separate test set to make sure the cross-validation results are sensible.


Run GridSearchCV to find the best hyperparameters using the training set.
Print the best values of the parameters.
Print out the accuracy on the test set, which was not used during the cross-validation procedure.


# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X_train,y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test,y_test))
###################################################################