### DIAGNOSING BIAS AND VARIANCE PROBLEMS


# Estimating the Generalization Error
-how do we estimate the generalization error of the model?
    -cannot be directly done:
        -f [model] is unknown
        -usually we only have ONE dataset
        -NOISE is unpredictable


# Solution

-split the data into train and test sets
    -model f can be fit to the train set
    -eval the error of [model] f on the UNSEEN Test set
    -generalization error of [model] f approximately = Test set ERROR of [model] f


# Better Model Eval with Cross-Validation
-test set should NOT be touched until we are confident about [model] f performance

-evaluating [model] f on Train set: Biased estimate, [model] f has already SEEN ALL Training Points

-Solution -> Cross-Validation (CV)
    -K-Fold CV
        CV error = E1 + ... + E10/10 (training set folds)

    -Hold-Out CV


# Diagnose Variance Problems
-if [model] f suffers from HIGH Variance: CV error of [model] f > training set error of [model] f
-[model] f is said to OVERFIT the Train Set. To remedy:
    -DECREASE model complexity
        -for ex: DECREASE max_depth, INCREASE min_samples_leaf
    -gather MORE data


# Diagnose Bias Problems
-if [model] f suffers from HIGH Bias: CV error of [model] approximately = training set error of [model] f >> desired error
-[model] f is said to UNDERFIT Train Set. To remedy:
    -INCREASE complexity
        -for ex: INCREASE max_depth, DECREASE min_samples_leaf
    -gather more relevant FEATURES


# K-Fold CV in sklearn on the Auto Dataset

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

# set SEED for reproducibility
SEED = 123
# split data into 70% Train and 30% Test
X_train, y_train, X_test, y_test= train_test_split(X,y,test_size=0.3,random_state=SEED)

# instantiate decision tree regressor and assign it to 'dt'
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.14, random_state=SEED)

# eval the list of MSE obtained by 10-fold CV
# set n_jobs to -1 in order to explot ALL CPU cores in computation
MSE_CV = cross_val_score(dt, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

    -Cross_val_score does not allow computing the MSE directly

-result is a NumPy array of the 10 negative_mean_squared_errors achieved on the Folds
    -multiply the result by -1 to obtain an array of the CV MSE

# fit dt to Train Set
dt.fit(X_train, y_train)
# predict the labels of Train Set
y_predict_train = dt.predict(X_train)
# predict the labels of the Test Set
y_predict_test = dt.predict(X_test)


# CV MSE
print('CV MSE: {:.2f}'.format(MSE_CV.mean()))

CV MSE: 20.51

# Train Set MSE
print('Train MSE: {:.2f}'.format(MSE(y_train, y_predict_train)))

TRAIN MSE: 15.30
    -we may deduce that the model OVERFITS the Train Set and suffers from HIGH Variance

# Test Set MSE
print('Test MSE: {:,.2f}'.format(MSE(y_test, y_predict_test)))

TEST MSE: 20.92

####################################################################

# PRACTICE

# Instantiate the Model

In the following set of exercises, you'll diagnose the bias and variance problems of a regression tree. 
The regression tree you'll define in this exercise will be used to predict the mpg consumption of cars from the auto dataset using all available features.

We have already processed the data and loaded the features matrix X and the array y in your workspace. 
In addition, the DecisionTreeRegressor class was imported from sklearn.tree.

Import train_test_split from sklearn.model_selection.
Split the data into 70% train and 30% test.
Instantiate a DecisionTreeRegressor with max depth 4 and min_samples_leaf set to 0.26.

from sklearn.model_selection import train_test_split
SEED = 1
X_train, y_train, X_test, y_text= train_test_split(X, y, test_size=0.3, random_state=SEED)
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED)
#####################################################################

# Evaluate the 10-fold CV Error

In this exercise, you'll evaluate the 10-fold CV Root Mean Squared Error (RMSE) achieved by the regression tree dt that you instantiated in the previous exercise.

In addition to dt, the training data including X_train and y_train are available in your workspace. 
We also imported cross_val_score from sklearn.model_selection.

Note that since cross_val_score has only the option of evaluating the negative MSEs, its output should be multiplied by negative one to obtain the MSEs. 
The CV RMSE can then be obtained by computing the square root of the average MSE.

Compute dt's 10-fold cross-validated MSE by setting the scoring argument to 'neg_mean_squared_error'.

Compute RMSE from the obtained MSE scores.

MSE_CV_scores = cross_val_score(dt, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
RMSE_CV = (MSE_CV_scores.mean())**(1/2)

print('CV RMSE: {:,2f}'.format(RMSE_CV))
############################################################

# Eval the Training Error

You'll now evaluate the training set RMSE achieved by the regression tree dt that you instantiated in a previous exercise.
In addition to dt, X_train and y_train are available in your workspace.
Note that in scikit-learn, the MSE of a model can be computed as follows:

MSE_model = mean_squared_error(y_true, y_predicted)
where we use the function mean_squared_error from the metrics module and pass it the true labels y_true as a first argument, 
and the predicted labels from the model y_predicted as a second argument.

Import mean_squared_error as MSE from sklearn.metrics.
Fit dt to the training set.
Predict dt's training set labels and assign the result to y_pred_train.
Evaluate dt's training set RMSE and assign it to RMSE_train.


from sklearn.metrics import mean_squared_error
dt.fit(X_train, y_train)
y_pred_train = dt.predict(X_train)
RMSE_train = (mean_squared_error(y_train, y_pred_train))**(1/2)
print('Train RMSE: {:.2f}'.format(RMSE_train))

<script.py> output:
    Train RMSE: 5.15
##################################################################