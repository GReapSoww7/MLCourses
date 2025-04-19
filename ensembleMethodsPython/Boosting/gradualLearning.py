### GRADUAL LEARNING ###


# Collective vs Gradual Learning

-Collective Learning:
    -Principle: wisdom of the crowd
    -Independent Estimators
    -Learning the SAME task for the SAME goal
    -Parallel Building

-Gradual Learning:
    -Principle: iterative learning
    -Dependent Estimators
    -Learning DIFFERENT tasks for the SAME goal
    -Sequential Building

# Gradual Learning
-Possible Steps:
    -First Attempt (INITIAL model)
    -Feedback (Model Eval)
    -Correct Errors (Subsequent Model)

# Fitting to Noise
-White Noise:
    -Uncorrelated Errors
    -Unbiased Errors and with Constant Variance

-Improvement Tolerance:
    -if Performance Difference < Improvement Threshold:
        -STOP Training


##################

# PRACTICE

# Predicting Move Revenue

Estimate the log-revenue of movies based on the 'budget' feature.
The metric you will use here is the RMSE (root mean squared error).
To calculate this using scikit-learn, you can use the mean_squared_error() function from the sklearn.metrics module 
and then take its square root using numpy.

The movies dataset has been loaded for you and split into train and test sets.
Additionally, the missing values have been replaced with zeros. We also standardized the input feature by using StandardScaler().
Check out DataCamp's courses on cleaning data and feature engineering if you want to learn more about preprocessing for machine learning.

Instantiate the default LinearRegression model.
Calculate the predictions on the test set.
Calculate the RMSE. The mean_squared_error() function requires two arguments: y_test, followed by the predictions.


reg_lm = LinearRegression()
reg_lm.fit(X_train, y_train)
pred = reg_lm.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))

RMSE: 7.335

####################

# Boosting for Predicted Revenue

The initial model got an RMSE of around 7.34. Let's see if we can improve this using an iteration of boosting.

You'll build another linear regression, but this time the target values are the errors from the base model, calculated as follows:

y_train_error = pred_train - y_train
y_test_error = pred_test - y_test
For this model you'll use 'popularity' feature instead, hoping that it can provide more informative patterns than with the 'budget' feature alone.
This is available to you as X_train_pop and X_test_pop. As in the previous exercise, the input features have been standardized for you.


Fit a linear regression model to the previous errors using X_train_pop and y_train_error.
Calculate the predicted errors on the test set, X_test_pop.
Calculate the RMSE, like in the previous exercise, using y_test_error and pred_error.

reg_error = LinearRegression()
reg_error.fit(X_train_pop, y_train_error)
pred_error = reg_error.predict(X_test_pop)
rmse_error = np.sqrt(mean_squared_error(y_test_error, pred_error))
print('RMSE: {:.3f}'.format(rmse_error))

RMSE: 7.277
-We LOWERED the RMSE by boosting using the errors of the previous model
##########################################