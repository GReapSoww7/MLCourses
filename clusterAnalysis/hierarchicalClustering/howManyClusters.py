### HOW MANY CLUSTERS ###


# Intro to Dendrograms
-Dendrograms help in showing progressions as clusters are MERGED
-a branching diagram; showing how each cluster is composed by branching out into its CHILD nodes

# Create a Dendrogram
from scipy.cluster.hierarchy import dendrogram
Z = linkage(df[['x_whiten', 'y_whiten', method='ward', metric='euclidean']])
dn = dendrogram()
plt.show()

#########################