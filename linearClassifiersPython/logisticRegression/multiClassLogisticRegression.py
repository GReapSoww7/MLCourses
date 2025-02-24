### MULTI-CLASS LOGISTIC REGRESSION


# Combining Binary Classifiers with One-vs-Rest

-3 logistic regression classifiers

lr0.fit(X, y==0)
    -returns an array the same size as y that's TRUE when y=0 and FALSE if y!=0
        -the Classifier learns to predict these True/False values
        -a Binary Classifier learning to discriminate between Class [i] or not Class [i]

# Get Raw Model Output
lr0.decision_function(X)[0]
6.124

lr1.fit(X, y==1)
lr1.decision_function(X)[0]
-5.429

lr2.fit(X, y==2)
lr2.decision_function(X)[0]
-7.532

-in order to make predictions
    -we take the Class whose Classifier provides the LARGEST raw model output (or DECISION FUNCTION)


-Classifier lr0 provides the largest DECISION FUNCTION VALUE
    -meaning it is more confident that the Class is == 0 than any other classifiers

# Fitting LogisticRegression model onto the original multi-class dataset

lr = LogisticRegression(multi_class='ovr')
lr.fit(X, y)
lr.predict(X)[0]

0

# Another Way to Achieve Multi-Class Classification with LogisticRegression
    -modify the Loss Function so it directly tries to optimize ACCURACY on the multi-class PROBLEM


-One-vs-Rest
    -fit a binary classifier for EACH class
    -predict with ALL, take LARGEST output
    -pro: simple, modular
    -con: NOT directly optimizing accuracy
    -common for SVMs as well
    -can produce probabilities


Contrasted against the following:

-Multinomial LogisticRegression
-Softmax
-Cross-Entropy Loss

    -fit a SINGLE classifier for ALL classes
    -prediction directly outputs BEST class
    -con: more complicated, new code
    -pro: tackle the problem directly
    -possible for SVMs; but less common
    -can produce probabilities

#######

# Model Coefficients for Mult-Class

# ONE-vs-REST
lr_ovr = LogisticRegression(multi_class='ovr')
lr_ovr.fit(X,y)

lr_ovr.coef_.shape
(3,13) # 3 by 13 array
    -3 entire binary classifiers (1 coefficient per feature per class)

lr_ovr.intercept_.shape
(3,)
    -one intercept per class

# Multinomial

lr_mn = LogisticRegression(multi_class="multinomial") # default for non-binary classification
lr_mn.fit(X,y)

lr_mn.coef_.shape
(3,13)
lr_mn.intercept_.shape
(3,)

#################################

# PRACTICE

# Fitting Multi-Class Logistic Regression

In this exercise, you'll fit the two types of multi-class logistic regression, one-vs-rest and softmax/multinomial, on the handwritten digits data set and compare the results. 
The handwritten digits dataset is already loaded and split into X_train, y_train, X_test, and y_test.

Fit a one-vs-rest logistic regression classifier by setting the multi_class parameter and report the results.
Fit a multinomial logistic regression classifier by setting the multi_class parameter and report the results.

# Fit one-vs-rest logistic regression classifier
lr_ovr = LogisticRegression(multi_class='ovr')
lr_ovr.fit(X_train, y_train)

print("OVR training accuracy:", lr_ovr.score(X_train, y_train))
print("OVR test accuracy    :", lr_ovr.score(X_test, y_test))

# Fit softmax classifier
lr_mn = LogisticRegression(multi_class='multinomial')
lr_mn.fit(X_train, y_train)

print("Softmax training accuracy:", lr_mn.score(X_train, y_train))
print("Softmax test accuracy    :", lr_mn.score(X_test, y_test))

###############################################

# Visualizing Multi-Class Logistic Regression

In this exercise we'll continue with the two types of multi-class logistic regression, but on a toy 2D data set specifically designed to break the one-vs-rest scheme.
The data set is loaded into X_train and y_train. The two logistic regression objects,lr_mn and lr_ovr, are already instantiated (with C=100), fit, and plotted.
Notice that lr_ovr never predicts the dark blue class… yikes! Let's explore why this happens by plotting one of the binary classifiers that it's using behind the scenes.

Create a new logistic regression object (also with C=100) to be used for binary classification.
Visualize this binary classifier with plot_classifier… does it look reasonable?


# Print training accuracies
print("Softmax     training accuracy:", lr_mn.score(X_train, y_train))
print("One-vs-rest training accuracy:", lr_ovr.score(X_train, y_train))

# Create the binary classifier (class 1 vs. rest)
lr_class_1 = LogisticRegression(C=100)
lr_class_1.fit(X_train, y_train==1)

# Plot the binary classifier (class 1 vs. rest)
plot_classifier(X_train, y_train==1, lr_class_1)
###################################################################

# One-vs-Rest SVM

As motivation for the next and final chapter on support vector machines, we'll repeat the previous exercise with a non-linear SVM. 
Once again, the data is loaded into X_train, y_train, X_test, and y_test .

Instead of using LinearSVC, we'll now use scikit-learn's SVC object, which is a non-linear "kernel" SVM (much more on what this means in Chapter 4!). 
Again, your task is to create a plot of the binary classifier for class 1 vs. rest.

Fit an SVC called svm_class_1 to predict class 1 vs. other classes.
Plot this classifier.


# We'll use SVC instead of LinearSVC from now on
from sklearn.svm import SVC

# Create/plot the binary classifier (class 1 vs. rest)
svm_class_1 = SVC()
svm_class_1.fit(X_train, y_train==1)
plot_classifier(X_train, y_train==1, svm_class_1)

########################################################