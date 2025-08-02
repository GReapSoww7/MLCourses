### UNSUPERVISED LEARNING: BASICS ###


# Labeled and Unlabeled Data

No Label:
    -Point 1: (1,2)
    -Point 2: (2,2)
    -Point 3: (3,1)

Label:
    -Point 1: (1,2), Label: Danger Zone
    -Point 2: (2,2), Label: Normal Zone
    -Point 3: (3,1), Label: Normal Zone

# What is Unsup Learning?
-a group of ML algos that FIND patterns in data
-data for algos has NOT been labeled, classified or characterized
-the objective of the algo is to INTERPRET ANY struct in the data
-common Unsup Learning Algos: Clutering, Neural Networks, Anomaly Detection

# What is Clustering?
-the process of grouping items with SIMILAR characteristics
-items in groups similar to each other than in OTHER groups
-ex: distance between points on a 2D plane


# Ex: Plotting Data for Clustering - Sightings
from matplotlib import pyplot as plt
x_coordinates = [80, 93, 86, 98, 86, 9, 15, 3, 10, 20, 44, 56, 49, 62, 44]
y_coordinates = [87, 96, 95, 92, 92, 57, 49, 47, 59, 55, 25, 2, 10, 24, 10]

plt.scatter(x_coordinates, y_coordinates)
plt.show()

# PRACTICE

# Sightings

from matplotlib import pyplot as plt
plt.scatter(x, y)
plt.show()

######################

# Basics of Cluster Analysis

# What is a Cluster?

-group of items with similar chars
-ex. Google News: articles where similar words and word associations appear together
-Customer Segments

# Clustering Algos
-Hierarchical clustering
-K Means Clustering
-DBSCAN, Gaussian Methods


# Clusters (2D plane)

-first ALL points are considered INDIVIDUAL CLUSTERS
    -Cluster Center = MEAN of attributes of ALL data points in a cluster
        -2 attributes (the MEAN of x and y)

-next step
    -distances between ALL pairs of cluster centers are COMPUTED
        -the 2 CLOSEST clusters are MERGED

-each step of clustering -> the number of clusters reduces by ONE

# Hierarchical Clustering

from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib import pyplot as plt
import seaborn as sns, pandas as pd

x_coordinates = [80.1, 93.1, 86.6, 98.5, 86.4, 9.5, 15.2, 3.4, 10.4, 20.3, 44.2, 56.8, 49.2, 62.5, 44.0]
y_coordinates = [87.2, 96.1, 95.6, 92.4, 92.4, 57.7, 49.4, 47.3, 59.1, 55.5, 25.6, 2.1, 10.9, 24.1, 10.3]

df = pd.DataFrame({'x_coordinate': x_coordinates, 'y_coordinate': y_coordinates})

Z = linkage(df, 'ward')
df['cluster_labels'] = fcluster(Z, 3, criterion='maxclust')

sns.scatterplot(x='x_coordinate', y='y_coordinate', hue='cluster_labels', data=df)
plt.show()


# K Means
-random point is generated for EACH cluster
    -distance to cluster center from these points to assign the point to the closest cluster
        -cluster centers are recomputed
        -Iterations for assignment are predefined
    
from scipy.cluster.vq import kmeans, vq
from matplotlib import pyplot as plt
import seaborn as sns, pandas as pd

import random
random.seed((1000, 2000))

x_coordinates = [80.1, 93.1, 86.6, 98.5, 86.4, 9.5, 15.2, 3.4, 10.4, 20.3, 44.2, 56.8, 49.2, 62.5, 44.0]
y_coordinates = [87.2, 96.1, 95.6, 92.4, 92.4, 57.7, 49.4, 47.3, 59.1, 55.5, 25.6, 2.1, 10.9, 24.1, 10.3]

df = pd.DataFrame({'x_coordinate': x_coordinates, 'y_coordinate': y_coordinates})

centroids,_ = kmeans(df, 3)
df['cluster_labels'], _ = vq(df, centroids)

sns.scatterplot(x='x_coordinate', y='y_coordinate', hue='cluster_labels', data=df)
plt.show()

# PRACTICE

# Sighting: Hierarchical Clustering

from scipy.cluster.hierarchy import linkage, fcluster

Z = linkage(df, 'ward')
df['cluster_labels'] = fcluster(Z, 3, criterion='maxclust')
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)
plt.show()


###########################

# Sightings: K Means Clustering

from scipy.cluster.vq import kmeans, vq

centroids,_ = kmeans(df, 2)
df['cluter_labels'], _ = vq(df, centroids)
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)
plt.show()


########################################################