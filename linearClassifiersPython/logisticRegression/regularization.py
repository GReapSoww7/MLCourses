### LOGISTIC REGRESSION AND REGULARIZATION


# Regularized Logistic Regression

-the hyperparam 'c/C' is the inverse of the regularization strength

c = more regularization
C = less regularization

# How does regularization affect training accuracy?
lr_weak_reg = LogisticRegression(C=100)
lr_strong_reg = LogisticRegression(c=0.01)

lr_weak_reg.fit(X_train, y_train)
lr_strong_reg.fit(X_train, y_train)

lr_weak_reg.score(X_train, y_train)
lr_strong_reg.score(X_train, y_train)

1.0
0.92

-model with the weaker regularization has the higher training accuracy
    -regularization is an extra term that we add to the loss function > which penalizes large values of coefficients

regularized loss = original loss + large coefficient penalty
    -distracts from the goal of optimizing accuracy
-more regularization: lower training accuracy
    -the more we deviate from goal of maximizing training accuracy

# Test Accuracy

lr_weak_reg.score(X_test, y_test)
0.86
lr_strong_reg.score(X_test, y_test)
0.88


regularized loss = original loss + large coefficient penalty

-more regularization = lower TRAIN accuracy
-more regularization = (almost always) higher TEST accuracy
    -regularization causes you to overfit LESS

# L1 vs L2 Regularization

-Lasso = linear regression with L1 regularization
-Ridge = linear regression with L2 regularization
-for other models like Logistic Regression we just say L1, L2, etc

lr_L1 = LogisticRegression(solver='liblinear', pnealty='l1')
lr_L2 = LogisticRegression() # penalty='l2' by default

lr_L1.fit(X_train, y_train)
lr_L2.fit(X_train, y_train)

plt.plot(lr_L1.coef_.flatten())
plt.plot(lr_L2.coef_.flatten())

-solver arg = controls the optimization method used to find the coefficients
    -the default solver is NOT compatible with L1 Regularization

##############################################################

# PRACTICE

# Regularized Logistic Regression

The handwritten digits dataset is already loaded, split.
Stored in the variables X_train, y_train, X_valid, and y_valid. The variables train_errs and valid_errs are already initialized as empty lists.


Loop over the different values of C_value, creating and fitting a LogisticRegression model each time.
Save the error on the training set and the validation set for each model.
Create a plot of the training and testing error as a function of the regularization parameter, C.
Looking at the plot, what's the best value of C?


# Train and validaton errors initialized as empty list
train_errs = list()
valid_errs = list()

# Loop over values of C_value
for C_value in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    # Create LogisticRegression object and fit
    lr = LogisticRegression(C=C_value, solver='liblinear')
    lr.fit(X_train, y_train)
    
    # Evaluate error rates and append to lists
    train_errs.append( 1.0 - lr.score(X_train, y_train) )
    valid_errs.append( 1.0 - lr.score(X_valid, y_valid) )
    
# Plot results
plt.semilogx(C_values, train_errs, C_values, valid_errs)
plt.legend(("train", "validation"))
plt.show()
################################################################

# Logistic Regression and Feature Selection

In this exercise we'll perform feature selection on the movie review sentiment data set using L1 regularization. 
The features and targets are already loaded for you in X_train and y_train.

We'll search for the best value of C using scikit-learn's GridSearchCV(), which was covered in the prerequisite course.


Instantiate a logistic regression object that uses L1 regularization.
Find the value of C that minimizes cross-validation error.
Print out the number of selected features for this value of C.

# Specify L1 regularization
lr = LogisticRegression(solver='liblinear', penalty='l1')

# Instantiate the GridSearchCV object and run the search
searcher = GridSearchCV(lr, {'C':[0.001, 0.01, 0.1, 1, 10]})
searcher.fit(X_train, y_train)

# Report the best parameters
print("Best CV params", searcher.best_params_)

# Find the number of nonzero coefficients (selected features)
best_lr = searcher.best_estimator_
coefs = best_lr.coef_
print("Total number of features:", coefs.size)
print("Number of selected features:", np.count_nonzero(coefs))

############################################################################

# Identifying the MOST Positive and Negative Words

In this exercise we'll try to interpret the coefficients of a logistic regression fit on the movie review sentiment dataset. 
The model object is already instantiated and fit for you in the variable lr.
In addition, the words corresponding to the different features are loaded into the variable vocab. 
For example, since vocab[100] is "think", that means feature 100 corresponds to the number of times the word "think" appeared in that movie review.

Find the words corresponding to the 5 largest coefficients.
Find the words corresponding to the 5 smallest coefficients.

# Get the indices of the sorted cofficients
inds_ascending = np.argsort(lr.coef_.flatten()) 
inds_descending = inds_ascending[::-1]

# Print the most positive words
print("Most positive words: ", end="")
for i in range(5):
    print(vocab[inds_descending[i]], end=", ")
print("\n")

# Print most negative words
print("Most negative words: ", end="")
for i in range(5):
    print(vocab[inds_ascending[i]], end=", ")
print("\n")

##################################################