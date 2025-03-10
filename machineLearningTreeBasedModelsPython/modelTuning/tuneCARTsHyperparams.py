### TUNING a CART's HYPERPARAMETERS ###


# to Obtain Better Performance

# Hyperparameters
-Machine Learning Model:
    -parameters: LEARNED FROM DATA through training
        -CART ex: Split-Point of a Node, Split-Feature of a Node, ...

    -hyperparamters: NOT learned from data, SET prior to training
        -ex: max_depth, min_samples_leaf, splitting criterion, etc

# What is Hyperparam Tuning?
-Problem: search for a set of OPTIMAL hyperparams for a learning algo
-Solution: find a set of OPTIMAL hyperparams that RESULTS in an OPTIMAL model
-Optimal Model: YIELDS an optimal SCORE
-Score: in sklearn DEFAULTS to accuracy (classification) and R^2 (regression)
    -measure the agreement between the TRUE labels and the model's prediction
-Cross Validation is USED to ESTIMATE the Generalization Performance

# Why TUNE?
-in sklearn, a model's default hyperparams are not optimal for ALL problems
-tuning to achieve BEST model performance

# Approaches to Hyperparam Tuning
-Grid Search
-Random Search
-Bayesian Optimization
-Genetic Algorithms, and more ...

# Grid Search Cross Validation
-MANUALLY set a grid of discrete hyperparam values
-set a metric for scoring model performance
-search exhaustively through the grid
-for each set of hyperparams, eval each model's CV score
-the optimal hyperparams are those of best CV score model

    -grid search suffers from Recursive Dimensionality:
        -the larger the grid > the longer it takes to find solution

# Example of Grid Search Cross Validation of a CART
-hyperparams grids:
    -max_depth = {2,3,4}
    -min_samples_leaf = {0.05, 0.1}

-hyperparam space = { (2,0.05), (2,0.01), (3,0.05), and so on ...}

-CV Scores = { score(2,0.05), ...}

-optimal hyperparams = SET of hyperparams corresponding to BEST CV score


# Inspecting hyperparams

from sklearn.tree import DecisionTreeClassifier
SEED = 1
dt = DecisionTreeClassifier(random_state=SEED)

print(dt.get_params())
    -prints the dict where keys are the hyperparameters names


# Example with Breast Cancer Dataset

# import GridSearchCV
from sklearn.model_selection import GridSearchCV
# define the grid of hyperparams 'params_dt'
params_dt = {'max_depth': [<val>, <val>, <val>],
             'min_samples_leaf': [<val>, <val>, <val>],
             'max_features': [0.2, 0.4, 0.6, 0.8]
             }
# instantiate a 10-fold CV grid search object 'grid_dt'
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='accuracy', cv=10, n_jobs=-1)

# fit grid_dt to train set
grid_dt.fit(X_train, y_train)

# extract BEST hyperparams from 'grid_dt'
best_hyperparams = grid_dt.best_params_
print('Best hyperparameters:\n', best_hyperparams)
Best hyperparameters:
    {'max_depth': 3, 'max_features': 0.4, 'min_samples_leaf': 0.06}

# extract BEST CV score from 'grid_dt'
best_CV_score = grid_dt.best_score_
print('Best CV accuracy'.format(best_CV_score))
Best CV accuracy: 0.938

# extract best model from 'grid_dt'
best_model = grid_dt.best_estimator_

# eval test set accuracy
test_acc = best_model.score(X_test, y_test)

print('Test set accuracy of best model: {:.3f}'.format(test_acc))
Test set accuracy of best model: 0.947
##################################################################

# PRACTICE

# Set the Tree's Hyperparam Grid
params_dt = {'max_depth': [2,3,4], 'min_samples_leaf': [0.12,0.14,0.16,0.18]}

# search for the optimal tree (model)

from sklearn.model_selection import GridSearchCV
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='roc_auc', cv=5, n_jobs=-1)

# eval the optimal tree

from sklearn.metrics import roc_auc_score

# extract best estimator from grid_dt assign to best_model
best_model = grid_dt.best_estimator_
# predict the test set probabilities of obtaining the positive class y_pred_proba
y_pred_proba = best_model.predict_proba(X_test)[:,1]
# compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))

Test set ROC AUC score: 0.610

########################################################