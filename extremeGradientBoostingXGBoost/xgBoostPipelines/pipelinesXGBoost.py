### INCORPORATING XGBOOST INTO PIPELINES ###


# sklearn Example with XGBoost

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

names = ['crime', 'zone', 'industry', 'charles', 'no', 'rooms', 'age', 'distance', 'radial', 'tax', 'pupil', 'aam', 'lower', 'med_price']

data = pd.read_csv('boston_housing.csv', names=names)

X, y = data.iloc[:,:-1], data.iloc[:,-1]
xgb_pipeline = Pipeline[('st_scaler', StandardScaler()), ('xgb_model', xgb.XGBRegressor())]

scores = cross_val_score(xgb_pipeline, X, y, scoring='neg_mean_squared_error', cv=10)
final_avg_rmse = np.mean(np.sqrt(np.abs(scores)))
print(f'Final XGB RMSE:', final_avg_rmse)

Final RMSE: 4.02719593323

# Additional Components Introduced for Pipelines

-sklearn_pandas: bridging the gaps between the libraries
    -DataFrameMapper - Interoperability between pandas and sklearn

-sklearn.impute:
    -SimpleImputer - Native imputation of numerical and categorical COLUMNS in sklearn

-sklearn.pipeline:
    -FeatureUnion - combine multiple pipelines of features into a SINGLE pipeline of features
######################

# PRACTICE

# Cross-Validating Your XGBoost Model

Create a pipeline called xgb_pipeline using steps.
Perform 10-fold cross-validation using cross_val_score().
You'll have to pass in the pipeline, X (as a dictionary, using .to_dict("records")), y, the number of folds you want to use,
and scoring ("neg_mean_squared_error").
Print the 10-fold RMSE.

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

X.LotFrontage = X['LotFrontage'].fillna(0)
# pipeline steps: steps
steps = [('ohe_onestep', DictVectorizer(sparse=False)), ('xgb_model', xgb.XGBRegressor(max_depth=2, objective='reg:squarederror'))]
xgb_pipeline = Pipeline(steps)
cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict('records'), y, scoring='neg_mean_squared_error', cv=10)
print(f'Final XGB RMSE: ', np.mean(np.sqrt(np.abs(cross_val_scores))))

<script.py> output:
    10-fold RMSE:  27683.04157118635

###########################

# Kidney Disease Case Study I: Categorical Imputer

You'll now continue your exploration of using pipelines with a dataset that requires significantly more wrangling.
The chronic kidney disease dataset contains both categorical and numeric features, but contains lots of missing values. 
The goal here is to predict who has chronic kidney disease given various blood indicators as features.

As Sergey mentioned in the video, you'll be introduced to a new library, sklearn_pandas,
that allows you to chain many more processing steps inside of a pipeline than are currently supported in scikit-learn.

Specifically, you'll be able to use the DataFrameMapper() class to apply any arbitrary sklearn-compatible transformer on DataFrame columns,
where the resulting output can be either a NumPy array or DataFrame.

We've also created a transformer called a Dictifier that encapsulates converting a DataFrame using .to_dict("records")
without you having to do it explicitly (and so that it works in a pipeline). 

Finally, we've also provided the list of feature names in kidney_feature_names, the target name in kidney_target_name,
the features in X, and the target in y.

In this exercise, your task is to apply sklearn's SimpleImputer to impute all of the categorical columns in the dataset.
You can refer to how the numeric imputation mapper was created as a template. Notice the keyword arguments input_df=True and df_out=True? 
This is so that you can work with DataFrames instead of arrays. 
By default, the transformers are passed a numpy array of the selected columns as input, and as a result, 
the output of the DataFrame mapper is also an array. Scikit-learn transformers have historically been designed to work with numpy arrays, 
not pandas DataFrames, even though their basic indexing interfaces are similar.


Apply the categorical imputer using DataFrameMapper() and SimpleImputer(). SimpleImputer() does not need any arguments to be passed in. 
The columns are contained in categorical_columns. Be sure to specify input_df=True and df_out=True, 
and use category_feature as your iterator variable in the list comprehension.

# Import necessary modules
from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer

# Check number of nulls in each feature column
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()


# Apply Numeric Imputer
numeric_imputation_mapper = DataFrameMapper(
    [([numeric_feature], SimpleImputer(strategy='median')) for numeric_feature in non_categorical_columns], input_df=True, df_out=True
)
# Apply Categorical Imputer
categorical_imputation_mapper = DataFrameMapper(
    [([category_feature], SimpleImputer(strategy='median')) for category_feature in categorical_columns], input_df=True, df_out=True
)
<output>
age        9
bp        12
sg        47
al        46
su        49
bgr       44
bu        19
sc        17
sod       87
pot       88
hemo      52
pcv       71
wc       106
rc       131
rbc      152
pc        65
pcc        4
ba         4
htn        2
dm         2
cad        2
appet      1
pe         1
ane        1
dtype: int64

#########################

# Kidney Disease Case Study II: Feature Union

Having separately imputed numeric as well as categorical columns, your task is now to use scikit-learn's
FeatureUnion to concatenate their results, which are contained in two separate transformer objects - numeric_imputation_mapper, 
and categorical_imputation_mapper, respectively.

You may have already encountered FeatureUnion in Machine Learning with the Experts: School Budgets. 
Just like with pipelines, you have to pass it a list of (string, transformer) tuples, 
where the first half of each tuple is the name of the transformer.


Import FeatureUnion from sklearn.pipeline.
Combine the results of numeric_imputation_mapper and categorical_imputation_mapper using FeatureUnion(), 
with the names "num_mapper" and "cat_mapper" respectively.


from sklearn.pipeline import FeatureUnion

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
    ('num_mapper', numeric_imputation_mapper),
    ('cat_mapper', categorical_imputation_mapper)
])
####################

# Kidney disease case study III: Full pipeline
It's time to piece together all of the transforms along with an XGBClassifier to build the full pipeline!

Besides the numeric_categorical_union that you created in the previous exercise, there are two other transforms needed: 
the Dictifier() transform which we created for you, and the DictVectorizer().

After creating the pipeline, your task is to cross-validate it to see how well it performs.


Create the pipeline using the numeric_categorical_union, Dictifier(), 
and DictVectorizer(sort=False) transforms, and xgb.XGBClassifier() estimator with max_depth=3. 

Name the transforms "featureunion", "dictifier" "vectorizer", and the estimator "clf".

Perform 3-fold cross-validation on the pipeline using cross_val_score().
Pass it the pipeline, pipeline, the features, kidney_data, the outcomes, y. Also set scoring to "roc_auc" and cv to 3.


# Create Full Pipeline
pipeline = Pipeline([
    ('featureunion', numeric_categorical_union),
    ('dictifier', Dictifier()),
    ('vectorizer', DictVectorizer(sort=False)),
    ('clf', xgb.XGBClassifier(max_depth=3))
])

# Perform Cross-Validation
cross_val_scores = cross_val_score(pipeline, kidney_data, y, scoring='roc_auc', cv=3)
# Print AVG auc
print(f'3-fold AUC: ', np.mean(cross_val_scores))
<script.py> output:
    3-fold AUC:  0.998237712755785
##############################



# SUMMARY

-Using XGBoost for CLASSIFICATION TASKS
-Using XGBoost for REGRESSION TASKS
-Tuning XGBoost hyperparams
-incorporate XGBoost into sklearn pipelines


# What We have NOT Covered
-how to use XGBoost for Ranking/Recommendation problems
    -done by modifying the LOSS function
-more sophisticated hyperparam tuning strategies
    -Bayesian Optimization
-using XGBoost as part of an Ensemble of OTHER models for Regression/Classification
###############