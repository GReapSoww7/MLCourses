### TRANSFORMING FEATURES FOR BETTER CLUSTERINGS


# Piedmont Wines Dataset
-178 samples from 3 distinct varieties of red wine: Barolo, Grignolino, and Barbera
-FEATURES measure chem composition e.g. alcohol content
-VISUAL properties like "color intensity"


# Clustering the wines

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
labels = model.fit_predict(samples)

# Clusters vs varieties

df = pd.DataFrame({'labels': labels, 'varieties': varieties})

ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)

-the KMeans clusters don't correspond well with the wine varieties


# Feature Variances
    -the wine features have very different variances
-Variance of a FEATURE measures spread of its values

# StandardScaler
-in KMeans: feature variance = feature influence
    -we need features to have EQUAL variance

-StandardScaler TRANSFORMS each feature to have mean 0 and variance 1
-Features are said to be "standardized"


# sklearn StandardScaler

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaled = scaler.transform(samples)

# Similar Methods

-StandardScaler and KMeans have similar methods
    -StandardScaler TRANSFORMS data
        -uses .fit() and .transform()
    -KMeans ASSIGNS cluster labels to data
        -uses .fit() and .predict()

# StandardScaler, THEN KMeans

-two steps: StandardScaler, then use KMeans

-use a sklearn PIPELINE to COMBINE multiple steps
-data flows from one step into the next


# Pipelines Combine Multiple Steps

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)

Pipeline(steps=...)

labels = pipeline.predict(samples)


# Feature Standardization Improves Clustering

-With Feature Standardization = GOOD

-Without = BAD

# sklearn Preprocessing Steps

-StandardScaler is a "preprocessing" step
-MaxAbsScaler and Normalizer are OTHER EXAMPLES
#############################################################

# PRACTICE

# Scaling Fish Data for Clustering

You are given an array 'samples' giving measurements of fish. Each row represents an individual fish. 
The measurements, such as weight in grams, length in centimeters, and the percentage ratio of height to length, have very different scales. 
In order to cluster this data effectively, you'll need to standardize these features first. In this exercise, you'll build a pipeline to standardize and cluster the data.

These fish measurement data were sourced from the Journal of Statistics Education.

Import:
make_pipeline from sklearn.pipeline.
StandardScaler from sklearn.preprocessing.
KMeans from sklearn.cluster.
Create an instance of StandardScaler called scaler.
Create an instance of KMeans with 4 clusters called kmeans.
Create a pipeline called pipeline that chains scaler and kmeans. To do this, you just need to pass them in as arguments to make_pipeline().

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
kmeans = KMeans(n_clusters=4)
pipeline = make_pipeline(scaler, kmeans)

################################################

# Clustering the Fish Data

You'll now use your standardization and clustering pipeline from the previous exercise to cluster the fish by their measurements, and then create a cross-tabulation to compare the cluster labels with the fish species.

As before, samples is the 2D array of fish measurements. Your pipeline is available as pipeline, and the species of every fish sample is given by the list species.

Import pandas as pd.
Fit the pipeline to the fish measurements samples.
Obtain the cluster labels for samples by using the .predict() method of pipeline.
Using pd.DataFrame(), create a DataFrame df with two columns named 'labels' and 'species', using labels and species, respectively, for the column values.
Using pd.crosstab(), create a cross-tabulation ct of df['labels'] and df['species'].

import pandas as pd
pipeline.fit(samples)
labels = pipeline.predict(samples)
df = pd.DataFrame({'labels': labels, 'species': species})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)
##########################################################

# Clustering Stocks using KMeans

In this exercise, you'll cluster companies using their daily stock price movements (i.e. the dollar difference between the closing and opening prices for each trading day). 
You are given a NumPy array movements of daily price movements from 2010 to 2015 (obtained from Yahoo! Finance), where each row corresponds to a company, and each column corresponds to a trading day.

Some stocks are more expensive than others. To account for this, include a Normalizer at the beginning of your pipeline. 
The Normalizer will separately transform each company's stock price to a relative scale before the clustering begins.

Note that Normalizer() is different to StandardScaler(), which you used in the previous exercise. 
While StandardScaler() standardizes features (such as the features of the fish data from the previous exercise) by removing the mean and scaling to unit variance, Normalizer() rescales each sample - here, each company's stock price - independently of the other.

KMeans and make_pipeline have already been imported for you.


Import Normalizer from sklearn.preprocessing.
Create an instance of Normalizer called normalizer.
Create an instance of KMeans called kmeans with 10 clusters.
Using make_pipeline(), create a pipeline called pipeline that chains normalizer and kmeans.
Fit the pipeline to the movements array.

from sklearn.preprocessing import Normalizer

normalizer = Normalizer()
kmeans = KMeans(n_clusters=10)
pipeline = make_pipeline(normalizer, kmeans)
pipeline.fit(movements)
##################################################

# Which Stocks Move Together?

In the previous exercise, you clustered companies by their daily stock price movements. 
So which company have stock prices that tend to change in the same way? 
You'll now inspect the cluster labels from your clustering to find out.

Your solution to the previous exercise has already been run. 
Recall that you constructed a Pipeline pipeline containing a KMeans model and fit it to the NumPy array movements of daily stock movements. 
In addition, a list companies of the company names is available.


Import pandas as pd.
Use the .predict() method of the pipeline to predict the labels for movements.
Align the cluster labels with the list of company names companies by creating a DataFrame df with labels and companies as columns. This has been done for you.
Use the .sort_values() method of df to sort the DataFrame by the 'labels' column, and print the result.
Hit submit and take a moment to see which companies are together in each cluster!


import pandas as pd
labels = pipeline.predict(movements)
df = pd.DataFrame({'labels': labels, 'companies': companies})
print(df.sort_values('labels'))
#######################################################################