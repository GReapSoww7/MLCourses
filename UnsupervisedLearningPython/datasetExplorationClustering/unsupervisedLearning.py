### UNSUPERVISED LEARNING


-Unsupervised learning finds patterns in data
    -e.g. clustering customers by their purchases
    -compressing the data using purchase patterns (dimension reduction)

# Supervised vs Unsupervised Learning

-Supervised learning finds patterns for a prediction task
    -e.g. classify tumors as benign or cancerous (labels)

-Unsupervised learning finds patterns in data
    -but without a specific prediction task in mind (no labels)

# Iris Dataset

-measurements of many iris plants

-3 species of iris:
    -setosa
    -versicolor
    -virginica

-Petal length, petal width, sepal length, sepal width (the FEATURES of the dataset)


# Arrays, features & samples

-datasets like this will be written as 2D NumPy arrays
    -columns are measurements (the FEATURES)
    -rows represent iris plants (the SAMPLES)

# Iris data is 4D

-the samples of the iris dataset have 4 measurements
    -iris samples are POINTS in 4 dimensional space
-Dimension = number of FEATURES
    -dimension is TOO HIGH to visualize
-but unsupervised learning gives insights to the data

# K-means clustering
-finds clusters of SAMPLES
-number of Clusters must be specified
-implemented in sklearn


print(samples)
[array]

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples)

KMeans(n_clusters=3)

labels = model.predict(samples)
print(labels)
[0 0 1 1 0 1 2 1 0 1 ...]

# Cluster Labels for New Samples
-new samples can be assigned to existing clusters
-k-means remembers the MEAN of each cluster (THE CENTROID)
-finds the NEAREST Centroid to EACH NEW sample


print(new_samples)
[array]

new_labels = model.predict(new_samples)
print(new_labels)
[0 2 1]

# Scatter Plots

-scatter plot of sepal length vs petal length
    -each point represents an iris sample
    -color points by cluster labels
    -PyPlot (matplotlib.pyplot)


import matplotlib.pyplot as plt

xs = samples[:,0]
ys = samples[:,2]
plt.scatter(xs, ys, c=labels)
plt.show()
#####################################

# PRACTICE

# How Many Clusters?
You are given an array points of size 300x2, where each row gives the (x, y) co-ordinates of a point on a map. 
Make a scatter plot of these points, and use the scatter plot to guess how many clusters there are.

matplotlib.pyplot has already been imported as plt.

Create an array called xs that contains the values of points[:,0] - that is, column 0 of points.
Create an array called ys that contains the values of points[:,1] - that is, column 1 of points.
Make a scatter plot by passing xs and ys to the plt.scatter() function.
Call the plt.show() function to show your plot.
How many clusters do you see?



import matplotlib.pyplot as plt

xs = points[:,0]
ys = points[:,1]
plt.scatter(xs, ys)
plt.show()

3 CLUSTERS
######################################

# Clustering 2D Points

From the scatter plot of the previous exercise, you saw that the points seem to separate into 3 clusters. 
You'll now create a KMeans model to find 3 clusters, and fit it to the data points from the previous exercise. 
After the model has been fit, you'll obtain the cluster labels for some new points using the .predict() method.

You are given the array points from the previous exercise, and also an array new_points.


Import KMeans from sklearn.cluster.
Using KMeans(), create a KMeans instance called model to find 3 clusters. To specify the number of clusters, use the n_clusters keyword argument.
Use the .fit() method of model to fit the model to the array of points points.
Use the .predict() method of model to predict the cluster labels of new_points, assigning the result to labels.
Hit submit to see the cluster labels of new_points.


from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(points)
labels = model.predict(new_points)
print(labels)
#######################################

# Inspect your Clustering

Let's now inspect the clustering you performed in the previous exercise!

A solution to the previous exercise has already run, so new_points is an array of points and labels is the array of their cluster labels.


Import matplotlib.pyplot as plt.
Assign column 0 of new_points to xs, and column 1 of new_points to ys.
Make a scatter plot of xs and ys, specifying the c=labels keyword arguments to color the points by their cluster label. Also specify alpha=0.5.
Compute the coordinates of the centroids using the .cluster_centers_ attribute of model.
Assign column 0 of centroids to centroids_x, and column 1 of centroids to centroids_y.
Make a scatter plot of centroids_x and centroids_y, using 'D' (a diamond) as a marker by specifying the marker parameter. Set the size of the markers to be 50 using s=50.

import matplotlib.pyplot as plt

xs = new_points[:,0]
ys = new_points[:,1]

plt.scatter(xs, ys, c=labels, alpha=0.5)

centroids = model.cluster_centers_

centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()
#########################################################