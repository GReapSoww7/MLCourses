### LINEAR CLASSIFIERS: THE COEFFICIENTS


# Prediction Equations

-Dot Products using NumPy arrays

x = np.arange(3)
x
array([0,1,2])

y = np.arange(3,6)
y
array([3,4,5])

x*y
array([0,4,10])

np.sum(x*y)
14

x@y
14

-x@y is called the DOT PRODUCT of x and y


# Linear Classifer Prediction
    -using dot products we can express how they make predictions

-raw model output = coefficients@features + intercept

-Linear classifier prediction: compute raw model output, check the SIGN
    -if pos, predict one class
    -if neg, predict the other class

-This is the SAME for Logistic Regression and Linear SVM
        -FIT is different but PREDICT is the SAME



# How LogisticRegression makes predictions

lr = LogisticRegression()

lr.fit(X,y)

lr.predict(X)[10]
0
lr.predict(X)[20]
1


lr.coef_ @ X[10] + lr.intercept_ # raw model output
array([-33.78572166]) # negative sign = predict other class '0'

lr.coef_ @ X[20] + lr.intercept_ # raw model output
array([0.08050621]) # positive sign = predict class '1'

-In general the PREDICT FUNC computes the raw model output > checks sign > returns the result based on the NAMES OF CLASSES IN THE dataset



# PRACTICE

# Changing the Model Coefficients

Set the two coefficients and the intercept to various values and observe the resulting decision boundaries.
Try to build up a sense of how the coefficients relate to the decision boundary.
Set the coefficients and intercept such that the model makes no errors on the given training data.

# Set the coefficients
model.coef_ = np.array([[0,1]]) # change this np.array([[x,y]]) where x is between -6->-18 and y is between 5->21
model.intercept_ = np.array([0]) # change to between -20->-41
            # THE ABOVE VARIES BY THE COMBINATION OF WHAT YOU PLACE AS THE COEFFICIENTS AND THE INTERCEPT
# Plot the data and decision boundary
plot_classifier(X,y,model)

# Print the number of errors
num_err = np.sum(y != model.predict(X))
print("Number of errors:", num_err)
###############################################