### INTRINSIC DIMENSION


-number of Features needed to approximate the dataset
-essential idea behind dimension reduction
    -what is the most compact representation of the SAMPLES

-Can be detected with PCA

# PCA identifies intrinsic dimension

-scatter plots work ONLY if samples have 2 or 3 Features
-PCA IDs intrinsic dimension when samples have ANY number of Features
-Intrinsic Dimension = number of PCA Features with significant Variance
    
-PCA rotates and shifts the samples to line up with the coordinate axes
    -Features are ordered by Variance DESCENDING


# Variance and Intrinsic Dimension
-intrinsic dimension is number of PCA Features with significant variance


# plotting the variance of PCA Features

-'samples' = array of samples

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(samples)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()

# Intrinsic Dimension can be Ambiguous
-it is an IDEALIZATION
    -there is NOT always ONE correct answer

###########################################################

# PRACTICE

# The First Principal Component

The first principal component of the data is the direction in which the data varies the most. 
In this exercise, your job is to use PCA to find the first principal component of the length and width measurements of the grain samples, 
and represent it as an arrow on the scatter plot.

The array grains gives the length and width of the grain samples. PyPlot (plt) and PCA have already been imported for you.


Make a scatter plot of the grain measurements. This has been done for you.
Create a PCA instance called model.
Fit the model to the grains data.
Extract the coordinates of the mean of the data using the .mean_ attribute of model.
Get the first principal component of model using the .components_[0,:] attribute.
Plot the first principal component as an arrow on the scatter plot, using the plt.arrow() function. You have to specify the first two arguments - mean[0] and mean[1].

plt.scatter(grains[:,0], grains[:,1])
model = PCA()
model.fit(grains)
mean = model.mean_
first_pc = model.components_[0,:]
plt.arrow(mean[0], mean[1], first_pc[0], color='red', width=0.01) # creates an arrow in the scatter plot showing the direction in which the data varies the MOST
plt.axis('equal')
plt.show()
#########################################

# Variance of the PCA Features

The fish dataset is 6-dimensional. But what is its intrinsic dimension? 
Make a plot of the variances of the PCA features to find out. As before, samples is a 2D array, where each row represents a fish. 
You'll need to standardize the features first.


Create an instance of StandardScaler called scaler.
Create a PCA instance called pca.
Use the make_pipeline() function to create a pipeline chaining scaler and pca.
Use the .fit() method of pipeline to fit it to the fish samples samples.
Extract the number of components used using the .n_components_ attribute of pca. Place this inside a range() function and store the result as features.
Use the plt.bar() function to plot the explained variances, with features on the x-axis and pca.explained_variance_ on the y-axis.

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(samples)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

#######################################################