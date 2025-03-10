### VISUALIZING THE PCA TRANSFORMATION


# Dimension Reduction
-finds patterns in data > uses patterns to re-express it in a COMPRESSED form
    -more efficient storage/computation

-removes less-informative 'noise' FEATURES
    -'noise' creates problems for prediction tasks (classification, regression)


# Principal Component Analysis
    -PCA
    -fundamental dimension reduction technique
        -first step 'decorrelation' > second step reduces dimension


# PCA Aligns Data with Axes
-rotates data samples to be aligned with axes
-shifts data samples so they have mean 0
-no info is lost


# PCA follows the fit/transform pattern

-PCA is a scikit-learn component
-fit() learns the transformation from given data
-transform() APPLIES the learned transformation
    -can also be APPLIED to NEW (unseen) data


# Using scikit-learn PCA

-'samples' = array of two Features (2D)

from sklearn.decomposition import PCA
model = PCA()
model.fit(samples)

transformed = model.transform(samples)
print(transformed) # PCA FEATURES

-Rows of transformed correspond to 'samples'
-Columns of transformed are the PCA Features
-Row gives PCA Feature VALUES of corresponding sample from 'samples'


# PCA Features are NOT Correlated
-Features of a dataset are often correlated, e.g. <feature1> and <feature2>

-PCA aligns the data with axes
-Resulting PCA features are NOT linearly correlated ("decorrelation")

# Pearson Correlation
-measures linear correlation of Features
    -value between -1 and 1
    -value of 0 means no linear correlation


# Principal Components
-PCA learns the principal components of the data > aligns them with the Axes (Features)

-"Principal Components" = directions of variance

-after a model has been fit > Principal Components are made available as the 'components_' attribute of a PCA object
    -each Row defines DISPLACEMENT from MEAN

#########################################################

# Correlated Data in Nature

You are given an array grains giving the width and length of samples of grain. 
You suspect that width and length will be correlated. 
To confirm this, make a scatter plot of width vs length and measure their Pearson correlation.


Import:
matplotlib.pyplot as plt.
pearsonr from scipy.stats.
Assign column 0 of grains to width and column 1 of grains to length.
Make a scatter plot with width on the x-axis and length on the y-axis.
Use the pearsonr() function to calculate the Pearson correlation of width and length.

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

width = grains[:,0]
length = grains[:,1]

plt.scatter(width, length)
plt.axis('equal')
plt.show()

correlation, pvalue = pearsonr(width, length)
print(correlation)
##########################################################

# Decorrelating the Grain Measurements with PCA

You observed in the previous exercise that the width and length measurements of the grain are correlated. 
Now, you'll use PCA to decorrelate these measurements, then plot the decorrelated points and measure their Pearson correlation.

Import PCA from sklearn.decomposition.
Create an instance of PCA called model.
Use the .fit_transform() method of model to apply the PCA transformation to grains. Assign the result to pca_features.
The subsequent code to extract, plot, and compute the Pearson correlation of the first two columns pca_features has been written for you, so hit submit to see the result!

from sklearn.decomposition import PCA
model = PCA()
pca_features = model.fit_transform(grains)
xs = pca_features[:,0]
ys = pca_features[:,1]
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()
correlation, pvalue = pearsonr(xs, ys)
print(correlation)











