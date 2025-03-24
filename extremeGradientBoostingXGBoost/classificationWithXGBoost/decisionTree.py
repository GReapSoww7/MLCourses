### WHAT IS A DECISION TREE ###


# Decision Trees as Base Learners
-Base Learner = individual learning algorithm in an ensemble algorithm (XGBoost is an Ensemble Learning Method)
    -composed of a series of binary questions
    -predictions happen at leaf nodes of the tree


# Decision Trees and CART
-constructed iteratively (one decision at a time)
    -until a stopping criterion is met (split point)
-each leaf node will have a single category in the majority or exclusively of one category

-individual trees are Low Bias High Variance Learning Models
    -good at learning relationships with variety of data used to trained
        -tend to Overfit and generalize to NEW data poorly


# XGBoost uses CART (Classification and Regression Trees)
    -each leaf node ALWAYS contain a real-valued score (rather than a decision value)
        -scores can then be converted into categories later on

##############################

# PRACTICE

# Decision Trees

Your task in this exercise is to make a simple decision tree using scikit-learn's DecisionTreeClassifier on the breast cancer dataset that comes pre-loaded with scikit-learn.

This dataset contains numeric measurements of various dimensions of individual tumors (such as perimeter and texture) 
from breast biopsies and a single outcome value (the tumor is either malignant, or benign).

We've preloaded the dataset of samples (measurements) into X and the target values per tumor into y.
Now, you have to split the complete dataset into training and testing sets, and then train a DecisionTreeClassifier.
You'll specify a parameter called max_depth. Many other parameters can be modified within this model, and you can check all of them out here.
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the classifier: dt_clf_4
dt_clf_4 = DecisionTreeClassifier(max_depth=4, random_state=123)

# Fit the classifier to the training set
dt_clf_4.fit(X_train, y_train)

# Predict the labels of the test set: y_pred_4
y_pred_4 = dt_clf_4.predict(X_test)

# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred_4==y_test))/y_test.shape[0]
print("accuracy:", accuracy)

<script.py> output:
    accuracy: 0.9649122807017544
##########################################