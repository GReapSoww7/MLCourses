### REGRESSION REVIEW ###


# Regression Basics
-outcome of Regression is a REAL value


-Common Regression Metrics:
    -Root mean squared error (RMSE)
        -take the difference between the ACTUAL and PREDICTED values (error) > squaring the differences (error^2) > compute the mean (MSE) > taking the value square root
        -allows us to treat negative and positive differences equally
            -punishes LARGER differences between predicted and actual vals much more than smaller differences
    
    -Mean Absolute Error (MAE)
        -SUMS the ABSOLUTE differences between predict and actual vals across ALL of the Samples we build our model on
            -lacks math properties = less used as an eval metric

# Common Regression Algorithms
-Linear
-Decision Trees
    -DT used for both Regression and Classification
#############################################