### BOOTSTRAP AGGREGATING ###


# Hetergeneous vs Homogeneous Ensembles

-Heterogeneous:
    -Different Algorithms (fine-tuned)
    -Small amount of estimators
    -Voting, Averaging, Stacking

-Homogeneous:
    -the SAME algorithm ('weak' model)
    -Large amount of estimators
    -Bagging and Boosting

# Condorcet's Jury Theorem
-Requirements:
    -Models are Independent
    -Each model performs better than random guessing
    -All individual models have similar performance

-Conclusion:
Adding MORE models IMPROVES the PERFORMANCE of the ENSEMBLE (Voting or Averaging), and this approaches 1 (100%)


# Bootstrapping
-Bootstrapping requires:
    -Random subsamples
    -Using replacement

-Bootstrapping guarantees:
    -Diverse crowd: different datasets
    -Independent: separately sampled


# Pros and Cons of Bagging
-Pros:
    -Bagging usually reduces variance
    -Overfitting can be avoided by the ensemble itself
    -more stability and robustness

-Cons:
    -it is computationally expensive
########################

# PRACTICE

# Training with Bootstrapping

Let's now build a "weak" decision tree classifier and train it on a sample of the training set drawn with replacement.
This will help you understand what happens on every iteration of a bagging ensemble.

To take a sample, you'll use pandas' .sample() method, which has a replace parameter.
For example, the following line of code samples with replacement from the whole DataFrame df:

df.sample(frac=1.0, replace=True, random_state=42)

Take a sample drawn with replacement (replace=True) from the whole (frac=1.0) training set, X_train.
Build a decision tree classifier using the parameter max_depth = 4.
Fit the model to the sampled training data.

# take a sample with replacement
X_train_samples = X_train.sample(frac=1.0, replace=True, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

# build a 'weak' Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=4, random_state=500)

# Fit the model to the training sample
clf.fit(X_train_sample, y_train_sample)

###############################

# A First Attempt at Bagging
You've seen what happens in a single iteration of a bagging ensemble. Now let's build a custom bagging model!

Two functions have been prepared for you:

def build_decision_tree(X_train, y_train, random_state=None):
    # Takes a sample with replacement,
    # builds a "weak" decision tree,
    # and fits it to the train set

def predict_voting(classifiers, X_test):
    # Makes the individual predictions 
    # and then combines them using "Voting"
Technically, the build_decision_tree() function is what you did in the previous exercise.
Here, you will build multiple such trees and then combine them. Let's see if this ensemble of "weak" models improves performance!


Build the individual models by calling build_decision_tree(), passing the training set and the index i as the random state.
Predict the labels of the test set using predict_voting(), with the list of classifiers clf_list and the input test features.

# Build the list of individual models
clf_list = []
for i in range(21):
    weak_dt = build_decision_tree(X_train, y_train, random_state=i)
    clf_list.append(weak_dt)

# Predict on the test set
pred = predict_voting(clf_list, X_test)

# Print the F1 score
print('F1 score: {:.3f}'.format(f1_score(y_test, pred)))

F1 score: 0.632

-Built a CUSTOM Bagging Ensemble; better performance than a SINGLE 'weak' model (only using 21 models)

###########################################