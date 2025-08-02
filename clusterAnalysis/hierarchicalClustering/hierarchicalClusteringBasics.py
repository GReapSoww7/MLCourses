### BASICS OF HIERARCHICAL CLUSTERING ###


# Distance Matrix using Linkage
-method: how to calc the proximity of clusters
-metric: distance metric
-optimal_ordering: order data points


# Methods
'single': based on two CLOSEST objects
'complete': based on two FARTHEST objects
'average': based on the arithmetic MEAN of ALL objects
'centroid': based on the geometric MEAN of ALL objs
'median': based on the MEDIAN of ALL objs
'ward': based on the SUM of SQUARES

# Create Cluster Labels with fcluster
scipy.cluster.hierarchy.fcluster(distance_matrix, num_clusters, criterion)

-distance_matrix: output of linkage() method
-num_clusters: number of clusters
-criterion: HOW to decide thresholds to FORM clusters


# Single Method

# Import the fcluster and linkage functions
from scipy.cluster.hierarchy import linkage, fcluster

# Use the linkage() function
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method = 'single', metric = 'euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', 
                hue='cluster_labels', data = comic_con)
plt.show()

# Complete Method

# Import the fcluster and linkage functions
from scipy.cluster.hierarchy import linkage, fcluster

# Use the linkage() function
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method='complete', metric='euclidean')

# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion='maxclust')

# Plot clusters
sns.scatterplot(x='x_scaled', y='y_scaled', 
                hue='cluster_labels', data = comic_con)
plt.show()

##################################################