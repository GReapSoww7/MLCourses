### LIMITATIONS OF HIERARCHICAL CLUSTERING ###


# Measuring Speed in Hierarchical Clustering
-timeit module
-measure the speed of .linkage() method
-use randomly generated points
-run various iterations to extrapolate

# Use of timeit module
from scipy.cluster.hierarchy import linkage
import pandas as pd
import random, timeit
points = 100
df = pd.DataFrame({'x': random.sample(range(0, points), points), 'y': random.sample(range(0, points), points)})
%timeit linkage(df[['x', 'y']], method = 'ward', metric = 'euclidean')

# Comparison of runtime of linkage method
-increasing runtime with data points
-quadratic increase of runtime
-NOT feasible for LARGE datasets


# PRACTICE

# Fit the data into a hierarchical clustering algorithm
distance_matrix = linkage(fifa[['scaled_sliding_tackle', 'scaled_aggression']], 'ward')

# Assign cluster labels to each row of data
fifa['cluster_labels'] = fcluster(distance_matrix, 3, criterion='maxclust')

# Display cluster centers of each cluster
print(fifa[['scaled_sliding_tackle', 'scaled_aggression', 'cluster_labels']].groupby('cluster_labels').mean())

# Create a scatter plot through seaborn
sns.scatterplot(x='scaled_sliding_tackle', y='scaled_aggression', hue='cluster_labels', data=fifa)
plt.show()


<script.py> output:
                    scaled_sliding_tackle  scaled_aggression
    cluster_labels                                          
    1                                2.99               4.35
    2                                0.74               1.94
    3                                1.34               3.62
##############################################