### SUPPORT VECTORS


# What is an SVM

-Linear Classifiers that are learned with the Hinge Loss Function
    -trained using the Hinge Loss and L2 Regularization


-Support Vector: a TRAIN example NOT in the FLAT part of the LOSS diagram
    -examples that is INCORRECTLY classified or  CORRECTLY classified examples that are CLOSE to the boundary
    -if an example is NOT a support vector, removing it has no effect on the model

-having a SMALL number of support vectors makes KERNEL SVMs REALLY FAST
    -run time scales based on number of SVMs rather than the training data

-the NON Support Vectors are interesting when compared with Logistic Regression = all data points matter to the FIT


##############

# Max-margin Viewpoint

-the SVM maximizes the 'margin' for LINEARLY separable datasets
    
-Margin = distance from the boundary to the closest points

#######################################################

# PRACTICE

# Effect of Removing Examples (non-support vectors)

Support vectors are defined as training examples that influence the decision boundary. 
In this exercise, you'll observe this behavior by removing non support vectors from the training set.
The wine quality dataset is already loaded into X and y (first two features only). 
(Note: we specify lims in plot_classifier() so that the two plots are forced to use the same axis limits and can be compared directly.)

Train a linear SVM on the whole data set.
Create a new data set containing only the support vectors.
Train a new linear SVM on the smaller data set.


# Train a linear SVM
svm = SVC(kernel="linear")
svm.fit(X, y)
plot_classifier(X, y, svm, lims=(11,15,0,6))

# Make a new data set keeping only the support vectors
print("Number of original examples", len(X))
print("Number of support vectors", len(svm.support_))
X_small = X[svm.support_]
y_small = y[svm.support_]

# Train a new SVM using only the support vectors
svm_small = SVC(kernel="linear")
svm_small.fit(X_small, y_small)
plot_classifier(X_small, y_small, svm_small, lims=(11,15,0,6))

##################################################################