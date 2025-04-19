### GRADIENT BOOSTING ###


# Intro Gradient Boosting Machine
-suppose we want to ESTIMATE an Objective Func
y = f(X)
-Iterations:
1. Initial Model (weak estimator): y ~ f1(X)
    -this is fit to the dataset

2. New model fits to residuals: y - f1(X) ~ f2(X) (the calculation of error)

3. New ADDITIVE model: y ~ f1(X) + f2(X)
    -improvement on the previous estimate

4. Repeat n times or until error is small enough
    -such that the difference in performance is neglibile

5. Final ADDITIVE model:
y ~ f1(X) + f2(X) + ... + fn(X) = n over Sigma; i=1 under Sigma * fi(X)
    -the INDIVIDUAL Estimators are NOT Combined through Voting or Averaging but through ADDITION
        -this is because ONLY the FIRST model is FITTED to the Target Var
        -the remaining models are Estimates of the RESIDUAL ERRORS

# Equivalence to Gradient Descent
-Gradient Boosting Method is equivalent to applying Gradient Descent as the Optimization Algorithm

# Gradient Descent
-Residuals: y - Fi(X)
    -represents the ERROR that the Model has at iteration 'i'

-it is an Iterative Optimization Algo to MINIMIZE the LOSS of an Estimator

-Loss: (Fi(X) - y)^2 / 2
    -the square loss = square of the Residuals divided by 2
    -every Iteration steps in the direction of the Neg Gradient (points towards the MIN)

-Gradient: derivativeLoss / derivativeFi(X) = Fi(X) - y
    -the Derivative LOSS with respect to the Approximate Func
-this expression is the OPPOSITE of the Residuals
    -we can notice the equivalence which is:
-Residuals = Negative Gradient

-IMPROVING the Model using Gradient Descent on EACH ITERATION

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
clf_gbm = GradientBoostinClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split,
    min_samples_leaf,
    max_features
)

-PARAMS:
    -n_estimators
        -Default: 100
    -learning_rate
        -Default: 0.1
    -max_depth
        -Default: 3
    -min_samples_split
    -min_samples_leaf
    -max_features

-the GBClassifier we DO NOT specify base_estimator
    -as GB is implemented and optimized with Regression Trees as the individual estimators
    -in Classification the Trees are FITTED to the Class Probabilities
-in Gradient Boosting it is recommended to use ALL FEATURES

# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
reg_gbm = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split,
    min_samples_leaf,
    max_features
)

############################

# PRACTICE

# Sentiment Analysis with GBM

predict the sentiment of a review given its text.

We will not pass the raw text as input for the model. The following pre-processing has been done for you:

Remove reviews with missing values.
Select data from the top 5 apps.
Select a random subsample of 500 reviews.
Remove "stop words" from the reviews.
Transform the reviews into a matrix, in which each feature represents the frequency of a word in a review.


Build a GradientBoostingClassifier with 100 estimators and a learning rate of 0.1.
Calculate the predictions on the test set.
Compute the accuracy to evaluate the model.
Calculate and print the confusion matrix.

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

reviews = pd.read_csv('reviews.csv')
reviews_clean = df.dropna()

top_5_apps = reviews_clean['app_name'].value_counts().head(5).index
reviews_top_5 = reviews_clean[reviews_clean['app_name'].isin(top_5_apps)]

reviews_rand_sample = reviews_top_5.sample(n=500, random_state=42)

stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

reviews_rand_sample['cleaned_review'] = reviews_rand_sample['review_text'].apply(remove_stop_words)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(reviews_rand_sample['cleaned_review'])

# Transform with CountVectorizer
reviews_vectorized = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
# Transform with TfidfVectorizer
reviews_vectorizer = TfidfVectorizer()
X_tfidf = reviews_vectorizer.fit_transform(reviews_rand_sample['cleaned_review'])
reviews_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=reviews_vectorizer.get_feature_names_out())

clf_gbm = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=500
)
clf_gbm.fit(X_train, y_train)

pred = clf_gbm.predict(X_test)

acc = accuracy_score(y_test, pred)
print('Accuracy: {:.3f}'.format(acc))

# Get and show the Confusion Matrix
cm = confusion_matrix(y_test, pred)
print(cm)

<script.py> output:
    Accuracy: 0.920
    [[29  0  5]
     [ 0  2  1]
     [ 2  0 61]]

##########################################