### REVIEW OF PIPELINES USING SKLEARN ###


# Pipelines
-are sklearn OBJECTS that takes a LIST of named 2-TUPLES (name, pipeline_step) as INPUT
-TUPLES can contain ANY arbitrary sklearn compatible ESTIMATOR or TRANSFORMER OBJECT
-pipeline implements FIT/PREDICT methods
-can be used as INPUT Estimator into GRID/RANDOMIZED Search and cross_val_score methods

# Example

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

names = ['crime', 'zone', 'industry', 'charles', 'no', 'rooms', 'age', 'distance', 'radial', 'tax', 'pupil', 'aam', 'lower', 'med_price']
data = pd.read_csv('boston_housing.csv', names=names)
X, y = data.iloc[:,:-1], data.iloc[:,-1] # feature matrix and target vector
rf_pipeline = Pipeline[('st_scaler', StandardScaler()), ('rf_model', RandomForestRegressor())]
scores = cross_val_score(rf_pipeline, X, y, scoring='neg_mean_squared_error', cv=10)
    # 'neg_mean_squared_error' is sklearn's method to calc MSE in an API compatible way

final_avg_rmse = np.mean(np.sqrt(np.abs(scores)))
print('Final RMSE: ', final_avg_rmse)
Final RMSE: 4.54530686529

# Preprocessing 1: LabelEncoder and OneHotEncoder Classes

-LabelEncoder: converts a categorical COLUMN of strings into int
-OneHotEncoder: takes the COLUMN of ints (that are treated as categorical vals) and encodes them as DUMMY vars
-this 2 step method cannot be done with a pipeline
    
# Preprocessing 2: DictVectorizer Class
-can accomplish both steps above in ONE line of code
-used in text processing pipelines
-CONVERTS a list of Feature mappings into VECTORS
-need to convert DF into a list of dict entries

#####################################################

# PRACTICE

# Encoding Categorical Columns 1: LabelEncoder

Import LabelEncoder from sklearn.preprocessing.
Fill in missing values in the LotFrontage column with 0 using .fillna().

Create a boolean mask for categorical columns. You can do this by checking for whether df.dtypes equals object.
Create a LabelEncoder object. You can do this in the same way you instantiate any scikit-learn estimator.

Encode all of the categorical columns into integers using LabelEncoder().
To do this, use the .fit_transform() method of le in the provided lambda function.


from sklearn.preprocessing import LabelEncoder
# fill missing vals
df.LotFrontage = df['LotFrontage'].fillna(0)
# create BOOL mask for cat columns
categorical_mask = (df.dtypes == object)
categorical_columns = df.columns[categorical_mask].tolist() # list of cat column names (casting DataFrame to list)

# print the head of the cat columns
print(df[categorical_columns].head())

# instantiate LabelEncoder as le
le = LabelEncoder()

# apply le to the cat columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform((x)))

# print the head of the LabelEncoded cat columns
print(df[categorical_columns].head())


<script.py> output:
      MSZoning PavedDrive Neighborhood BldgType HouseStyle
    0       RL          Y      CollgCr     1Fam     2Story
    1       RL          Y      Veenker     1Fam     1Story
    2       RL          Y      CollgCr     1Fam     2Story
    3       RL          Y      Crawfor     1Fam     2Story
    4       RL          Y      NoRidge     1Fam     2Story
       MSZoning  PavedDrive  Neighborhood  BldgType  HouseStyle
    0         3           2             5         0           5
    1         3           2            24         0           2
    2         3           2             5         0           5
    3         3           2             6         0           5
    4         3           2            15         0           5

# NOTE: A BldgTpe of 1Fam is encoded as 0, while a HouseStyle of 2Story is encoded as 5.

################################################

# Encoding Cat Columns 2: OneHotEncoder

Import OneHotEncoder from sklearn.preprocessing.
Instantiate a OneHotEncoder object called ohe. 
Specify the keyword argument sparse=False.

Using its .fit_transform() method, apply the OneHotEncoder to df and save the result as df_encoded. 
The output will be a NumPy array.
Print the first 5 rows of df_encoded, and then the shape of df as well as df_encoded to compare the difference.



from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
df_encoded = ohe.fit_transform(df)

print(df_encoded[:5, :])
print(df.shape)
print(df_encoded.shape)

<script.py> output:
    [[0. 0. 0. ... 0. 0. 0.]
     [1. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]
    (1460, 21)
    (1460, 3369)
####################################

# Encoding Cat Columns 3: DictVectorizer

Import DictVectorizer from sklearn.feature_extraction.
Convert df into a dictionary called df_dict using its .to_dict() method with "records" as the argument.

Instantiate a DictVectorizer object called dv with the keyword argument sparse=False.
Apply the DictVectorizer on df_dict by using its .fit_transform() method.

Hit 'Submit Answer' to print the resulting first five rows and the vocabulary.

from sklearn.feature_extraction import DictVectorizer
df_dict = df.to_dict('records')
dv = DictVectorizer(sparse=False)
df_encoded = dv.fit_transform(df_dict)
print(df_encoded[:5, :])
print(dv.vocabulary_)
# NOTE: Besides simplifying the process into one step, DictVectorizer has useful attributes 
# such as vocabulary_ which maps the names of the features to their indices. 
# With the data preprocessed, it's time to move onto pipelines!

######################################

# Preprocessing within a Pipeline
Import DictVectorizer from sklearn.feature_extraction and Pipeline from sklearn.pipeline.
Fill in any missing values in the LotFrontage column of X with 0.

Complete the steps of the pipeline with DictVectorizer(sparse=False) for "ohe_onestep" and xgb.XGBRegressor() for "xgb_model".

Create the pipeline using Pipeline() and steps.
Fit the Pipeline. 

Don't forget to convert X into a format that DictVectorizer understands by calling the to_dict("records") method on X


from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

X.LotFrontage = X['LotFrontage'].fillna(0)

# setup the pipeline steps: steps
steps = [('ohe_onestep', DictVectorizer(sparse=False)), ('xgb_model', xgb.XGBRegressor())]

# Create pipeline
xgb_pipeline = Pipeline(steps)

xgb_pipeline.fit(X.to_dict('records'), y)

Pipeline(steps=[('ohe_onestep', DictVectorizer(sparse=False)),
                ('xgb_model',
                 XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
                              colsample_bylevel=1, colsample_bynode=1,
                              colsample_bytree=1, early_stopping_rounds=None,
                              enable_categorical=False, eval_metric=None,
                              gamma=0, gpu_id=-1, grow_policy='depthwise',
                              importance_type=None, interaction_constraints='',
                              learning_rate=0.300000012, max_bin=256,
                              max_cat_to_onehot=4, max_delta_step=0,
                              max_depth=6, max_leaves=0, min_child_weight=1,
                              missing=nan, monotone_constraints='()',
                              n_estimators=100, n_jobs=0, num_parallel_tree=1,
                              predictor='auto', random_state=0, reg_alpha=0,
                              reg_lambda=1, ...))])
#########################################################