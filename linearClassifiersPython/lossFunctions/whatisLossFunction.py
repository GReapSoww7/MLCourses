### WHAT IS A LOSS FUNCTION


# Least Squares: the Squared Loss (LinearRegression)
    -minimizes a loss:
    -Minimization is with RESPECT to COEFFICIENTS or Params of the model

-minimizes the SUM of Squares of the ERRORS made on our training set

-Error = difference between TRUE ith target value and the predicted ith target value
    - i=1

-minimization accomplished with coefficients (or param) adjustments of the model until the error is as small as possible
    -ex. the FIT FUNCTION minimizes the loss

-the LOSS function is used to fit the model on the data
-the SCORE function is used to see how well we are doing


# Classification errors: the 0-1 Loss

-a NATURAL loss for class problems is the number of errors we have made.

-It's 0 for correct prediction or 1 if prediction is incorrect
    -by summing this function over all the training examples we get the number of errors we made on the training set


-more conceptual but difficult to minimize loss in practicality

#########################################

# Min a Loss

from scipy.optimize import minimize

minimize(np.square, 0).x # second arg is our INITIAL GUESS
array([0.])


minimize(np.square, 2).x
array([-1.88846401e-08])


-we don't expect EXACTLY the right answer but something VERY close

-inputs for exercises will be the MODEL'S coefficients
    -what values for the coeffs will make my squared error as SMALL as possible?


############################################################

# PRACTICE

# Minimizing a Loss Function

Fill in the loss function for least squares linear regression.
Print out the coefficients from fitting sklearn's LinearRegression.


# the squared error, summed over trianing examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        # Get the true and predicted target values for example 'i'
        y_i_true = y[i]
        y_i_pred = w@X[i]
        s = s + (y_i_true - y_i_pred)**2
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LinearRegression coefficients
lr = LinearRegression(fit_intercept=False).fit(X,y)
print(lr.coef_)

###########################

# Loss Function Diagrams

-Raw Model Output

-Logistic Loss Diagram (LogisticRegression) and Hinge Loss Diagram (SVM)

###########################

# PRACTICE

# Comparing the Logistic and Hinge Losses

Evaluate the log_loss() and hinge_loss() functions at the grid points so that they are plotted.

# Mathematical functions for logistic and hinge losses
def log_loss(raw_model_output):
    return np.log(1+np.exp
(-raw_model_output))
def hinge_loss(raw_model_output):
    return np.maximum(0,1-raw_model_output)

# Create a grid of values and plot
grid = np.linspace(-2,2,1000)
plt.plot(grid, log_loss(grid), label='logistic')
plt.plot(grid, hinge_loss(grid), label='hinge')
plt.legend()
plt.show()

###################################

# Implementing Logistic Regression

The log_loss() function from the previous exercise is already defined in your environment.
The sklearn breast cancer prediction dataset (first 10 features, standardized) is loaded into the variables X and y.

Input the number of training examples into range().
Fill in the loss function for logistic regression.
Compare the coefficients to sklearn's LogisticRegression.

# The logistic loss, summed over trianing examples
def my_loss(w):
    s = 0
    for i in range(len(X)):
        raw_model_output = w@X[i]
        s = s + log_loss(raw_model_output * y[i])
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LogisticRegression
lr = LogisticRegression(fit_intercept=False, C=1000000).fit(X,y)
print(lr.coef_)
###################################################################################