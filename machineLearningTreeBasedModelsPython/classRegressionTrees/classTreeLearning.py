### CLASSIFICATION TREE LEARNING


# Building Blocks of a Decision Tree
-Decision Tree: data structure consisting of a hierarchy of nodes
    -Node: question or prediction

-3 kinds of nodes:
    -Root: NO parent node, question giving rise to 2 children nodes
    -Internal: ONE parent node, question giving rise to 2 children nodes
    -Leaf: ONE parent node, NO children nodes --> Prediction

-NODES are GROWN RECURSIVELY
    -tree asks the question f(eature) < sp(lit point) at each node; f < sp
        -it knows what feature and split point to pick by maximizing Information Gain
            -obtained after each SPLIT

# Information Gain (IG)

IG( f, sp ) = I(parent) - (Nleft/N I(left) + Nright/N I(right))

-Criteria to measure the IMPURITY of a node I(node):
    -gini index
    -entropy

# Classification-Tree Learning (unconstrained trees)
-when an unconstrained Tree is TRAINED:
    -Nodes are grown Recursively

-at each (non-leaf) Node, split the data based on:
    -feature f and split-point sp to maximize IG(node)

-if IG(node) = 0, the Node is declared a LEAF (no child nodes)


# Revisit the Breast Cancer 2D dataset

# import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# import train_test_split
from sklearn.model_selection import train_test_split
# import accuracy_score
from sklearn.metrics import accuracy_score
# Split the dataset into 80% train and 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.2, stratify=y, random_state=1)

# Instantiate dt, set 'criterion' to 'gini'
dt = DecisionTreeClassifier(criterion='gini', random_state=1)

# Information Criterion in scikit-learn

# fit dt to the training set
dt.fit(X_train, y_train)
# predict test-set labels
y_pred = dt.predict(X_test)
# eval test-set accuracy
accuracy_score(y_test, y_pred)
0.92105263157894735

#########################################

# PRACTICE

# Using Entropy as a Criterion
In this exercise, you'll train a classification tree on the Wisconsin Breast Cancer dataset using entropy as an information criterion. 
You'll do so using all the 30 features in the dataset, which is split into 80% train and 20% test.

X_train as well as the array of labels y_train are available in your workspace.

Import DecisionTreeClassifier from sklearn.tree.

Instantiate a DecisionTreeClassifier dt_entropy with a maximum depth of 8.

Set the information criterion to 'entropy'.

Fit dt_entropy on the training set.



from sklearn.tree import DecisionTreeClassifier
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)
dt_entropy.fit(X_train, y_train)

#####################################################################

# Entropy vs Gini Index
In this exercise you'll compare the test set accuracy of dt_entropy to the accuracy of another tree named dt_gini. 
The tree dt_gini was trained on the same dataset using the same parameters except for the information criterion which was set to the gini index using the keyword 'gini'.

X_test, y_test, dt_entropy, as well as accuracy_gini which corresponds to the test set accuracy achieved by dt_gini are available in your workspace.


Import accuracy_score from sklearn.metrics.
Predict the test set labels of dt_entropy and assign the result to y_pred.
Evaluate the test set accuracy of dt_entropy and assign the result to accuracy_entropy.
Review accuracy_entropy and accuracy_gini.



from sklearn.metrics import accuracy_score
y_pred = dt_entropy.predict(X_test)
accuracy_entropy = accuracy_score(y_test, y_pred)
accuracy_gini = accuracy_score(y_test, y_pred)
print(f'Accuracy achieved using entropy: {accuracy_entropy:.3f}')
print(f'Accuracy achieved using the gini index: {accuracy_gini:.3f}')
######################################################################