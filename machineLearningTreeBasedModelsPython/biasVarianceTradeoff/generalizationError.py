### GENERALIZATION ERROR


# Supervised Learning - Under the Hood
-Supervised Learning: y = f(x), f is UNKNOWN function that we want to determine
    -data generation is ALWAYS a companion with RANDOMNESS (noise)


# Goals of Supervised Learning
-find a MODEL that BEST approximates f: f (model) approximately = f (function)
    -f (model) can be Logistic Regression, Decision Tree, Neural Network ...
-Discard NOISE as much as possible
-END GOAL: f (model) should achieve a LOW predictive error on UNSEEN datasets


# Difficulties in Approximating f (model)
-Overfitting = [model] f(x) fits the training set NOISE
    -its predictive power on UNSEEN datasets is LOW
        -it memorizes too much = LOW Training Set error and HIGH Test Set error
-Underfitting = [model] f(x) is NOT flexible enough to approximate f (function)
    -the Training Set error is roughly equal to Test Set error
        -but both are relatively HIGH
            -now the Trained model is not flexible enough to capture the complex feature/label dependency
                -like teaching a 3 year old calculus

# Generalization Error
-Generalization Error of f [model]: does f [model] generalize well on UNSEEN data?

-decomposed as follows:
f [model] = bias^2 + variance + irreducible error

-irreducible error = error contribution of NOISE

-BIAS: error term that tells us, on AVG, how much f [model] is NOT = f [function]
    -the differences between them
        -HIGH Bias models = Underfitting

-VARIANCE: how much f [model] is inconsistent over DIFFERENT Train Sets
    -HIGH Variance models = Overfitting


# Model Complexity

-sets the model flexibility to approximate the true function f
    -ex. Max tree depth, minimum samples per leaf, etc

# Bias-Variance Tradeoff
-Best Complexity corresponds to the LOWEST Generalization Error
    -as Complexity increases = Variance increases; bias decreases
    -decrease = bias increase; variance decrease

########################################################