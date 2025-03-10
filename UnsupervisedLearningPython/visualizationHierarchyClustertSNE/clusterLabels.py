### CLUSTER LABELS IN HIERARCHICAL CLUSTERING


# Cluster Labels
-not only a visualization tool
-cluster labels at ANY intermediate stage CAN be RECOVERED
    -for use in e.g. CROSS-TABULATIONS


# Intermediate Clusterings & Height on Dendrogram

-what is the meaning of the HEIGHT?
    -Height on Dendrogram = distance BETWEEN merging clusters
        -e.g. Clusters with ONLY <value> and <value2> had distance approx 6
    -this NEW cluster DISTANCE approx 12 from cluster with ONLY <value3>


-Height on Dendrogram specifies MAXIMUM distance BETWEEN merging clusters

-DON'T merge clusters further apart than this (e.g. 15)

# Distance between Clusters
-defined by a "linkage method"

-in "complete" linkage: distance between clusters is MAX distance between their SAMPLES
-specified via METHOD param, e.g. linkage(samples, method='complete')

-Different linkage METHODS, different hierarchical CLUSTERING!

# Extracting Cluster Labels

-user the fcluster() function

-RETURNS a NumPy array of cluster labels

from scipy.cluster.hierarchy import linkage
mergings = linkage(samples, method='complete')
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 15, criterion='distance')
print(labels)

[ 9 8 11 20 2 1 17 14 ...]

# Aligning Cluster Labels with Country Names

-given a list of strings 'country_names':

import pandas as pd
pairs = pd.DataFrame({'labels': labels, 'countries': country_names}) # create a DataFrame from the list of strings and labels
print(pairs.sort_values('labels'))

###############################################

# PRACTICE

In the video, you learned that the linkage method defines how the distance between clusters is measured. 
In complete linkage, the distance between clusters is the distance between the furthest points of the clusters. 
In single linkage, the distance between clusters is the distance between the closest points of the clusters.

Consider the three clusters in the diagram. Which of the following statements are true?

A. In single linkage, Cluster 3 is the closest cluster to Cluster 2.

B. In complete linkage, Cluster 1 is the closest cluster to Cluster 2.

-both are true

########################################

# PRACTICE

# Different Linkage, Different Hierarchical Clustering

You saw a hierarchical clustering of the voting countries at the Eurovision song contest using 'complete' linkage. 
Now, perform a hierarchical clustering of the voting countries with 'single' linkage, and compare the resulting dendrogram with the one in the video. 
Different linkage, different hierarchical clustering!

You are given an array samples. 
Each row corresponds to a voting country, and each column corresponds to a performance that was voted for. 
The list country_names gives the name of each voting country. This dataset was obtained from Eurovision.


Import linkage and dendrogram from scipy.cluster.hierarchy.
Perform hierarchical clustering on samples using the linkage() function with the method='single' keyword argument. Assign the result to mergings.
Plot a dendrogram of the hierarchical clustering, using the list country_names as the labels. 
In addition, specify the leaf_rotation=90, and leaf_font_size=6 keyword arguments as you have done earlier.

from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples, method='single')
dendrogram(mergings, labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show()
##################################

# Intermediate Clusterings

Displayed is the dendrogram for the hierarchical clustering of the grain samples that you computed earlier. 
If the hierarchical clustering were stopped at height 6 on the dendrogram, how many clusters would there be?


3

################################################

# Extracting the Cluster Labels

In the previous exercise, you saw that the intermediate clustering of the grain samples at height 6 has 3 clusters. 
Now, use the fcluster() function to extract the cluster labels for this intermediate clustering, and compare the labels with the grain varieties using a cross-tabulation.

The hierarchical clustering has already been performed and mergings is the result of the linkage() function. 
The list varieties gives the variety of each grain sample.

Import:
pandas as pd.
fcluster from scipy.cluster.hierarchy.
Perform a flat hierarchical clustering by using the fcluster() function on mergings. Specify a maximum height of 6 and the keyword argument criterion='distance'.
Create a DataFrame df with two columns named 'labels' and 'varieties', using labels and varieties, respectively, for the column values. This has been done for you.
Create a cross-tabulation ct between df['labels'] and df['varieties'] to count the number of times each grain variety coincides with each cluster label.


import pandas as pd
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 6, criterion='distance')
df = pd.DataFrame({'labels': labels, 'varieties': varieties})
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)
############################################