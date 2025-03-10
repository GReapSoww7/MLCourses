### VISUALIZING HIERARCHIES


# Visualizations Communicate Insight

-2 Unsupervised Learning Techniques

-'t-SNE': Creates a 2D map of a dataset (later)

-"Hierarchical Clustering" (now)

# a Hierarchy of Groups
-groups of living things can form a hierarchy
    -clusters are contained in one another

# Eurovision scoring dataset
-countires gave scores to songs performed at the Eurovision 2016
-2D array of scores
-Rows are countries, columns are songs


-creates a tree-like diagram called a dendrogram

# Hiearchical Clustering proceeds in Steps

-Every country begins in a SEPARATE cluster
-at each step, the two closest clusters are MERGED
-Continue until all countries are in a single cluster
-this is "agglomerative" hierarchical clustering


# Dendrogram
    -read from bottom up
    -vertical lines represent clusters

# Hierarchical Clustering with SciPy

-given 'samples' (the array of scores), and 'country_names'

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples, method='complete')
dendrogram(mergings,
            labels=country_names,
            leaf_rotation=90,
            leaf_font_size=6)
plt.show()
##################################

# PRACTICE

# Hierarchical Clustering of the Grain Data

In the video, you learned that the SciPy linkage() function performs hierarchical clustering on an array of samples. 
Use the linkage() function to obtain a hierarchical clustering of the grain samples, and use dendrogram() to visualize the result. 
A sample of the grain measurements is provided in the array samples, while the variety of each grain sample is given by the list varieties.

Import:
linkage and dendrogram from scipy.cluster.hierarchy.
matplotlib.pyplot as plt.
Perform hierarchical clustering on samples using the linkage() function with the method='complete' keyword argument. Assign the result to mergings.
Plot a dendrogram using the dendrogram() function on mergings. Specify the keyword arguments labels=varieties, leaf_rotation=90, and leaf_font_size=6.

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

mergings = linkage(samples, method='complete')
dendrogram(mergings, labels=varieties, leaf_rotation=90, leaf_font_size=6)
plt.show()
#############################################################

# Hierarchies of Stocks

In chapter 1, you used k-means clustering to cluster companies according to their stock price movements. 
Now, you'll perform hierarchical clustering of the companies. 
You are given a NumPy array of price movements movements, where the rows correspond to companies, and a list of the company names companies. 
SciPy hierarchical clustering doesn't fit into a sklearn pipeline, so you'll need to use the normalize() function from sklearn.preprocessing instead of Normalizer.

linkage and dendrogram have already been imported from scipy.cluster.hierarchy, and PyPlot has been imported as plt.

Import normalize from sklearn.preprocessing.
Rescale the price movements for each stock by using the normalize() function on movements.
Apply the linkage() function to normalized_movements, using 'complete' linkage, to calculate the hierarchical clustering. Assign the result to mergings.
Plot a dendrogram of the hierarchical clustering, using the list companies of company names as the labels. In addition, specify the leaf_rotation=90, and leaf_font_size=6 keyword arguments as you did in the previous exercise.

from sklearn.preprocessing import normalize

normalized_movements = normalize(movements)
mergings = linkage(normalized_movements, method='complete')
dendrogram(mergings, labels=companies, leaf_rotation=90, leaf_font_size=6)
plt.show()
###########################################################