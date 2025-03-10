### EVALUATING A CLUSTERING


# Cross Tabulation with pandas

-clusters vs species is a "cross-tabulation"
-use pandas lib
-given the species of each sample as a list species

print(species)

# Aligning labels and species

import pandas as pd

df = pd.DataFrame({'labels': labels, 'species': species})
print(df)


# Crosstab of labels and species

ct = pd.crosstab(df['labels'], df['species'])
print(ct)

-how to eval clustering quality, if there were NO species info???


# Measuring clustering quality
    -using ONLY samples and their cluster labels
    -a good clustering has tight clusters
    -samples in each cluster bunched together


# Inertia measures clustering quality
-measures how spread out the clusters are (LOWER is BETTER)
-distance from each sample to centroid of its cluster

from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(samples)
print(model.inertia_)

78.940814261

# The number of clusters
-clusterings of the iris dataset with different number of clusters
    -more clusters means lower inertia 
    -what is the best number of clusters?


# How many clusters to choose?

-a good clustering has low inertia
    -but not too many clusters

-choose an 'elbow' in the inertia plot
    -where inertia begins to DECREASE more slowly

-e.g. for iris dataset, 3 is a good choice
    
##################################################

# PRACTICE

# How Many Clusters of Grain?

You are given an array samples containing the measurements (such as area, perimeter, length, and several others) of samples of grain. 
What's a good number of clusters in this case?

KMeans and PyPlot (plt) have already been imported for you.

This dataset was sourced from the UCI Machine Learning Repository.

For each of the given values of k, perform the following steps:
Create a KMeans instance called model with k clusters.
Fit the model to the grain data samples.
Append the value of the inertia_ attribute of model to the list inertias.
The code to plot ks vs inertias has been written for you, so hit submit to see the plot!


ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

########################################

# Eval the Grain clustering

In the previous exercise, you observed from the inertia plot that 3 is a good number of clusters for the grain data. 
In fact, the grain samples come from a mix of 3 different grain varieties: "Kama", "Rosa" and "Canadian". 
In this exercise, cluster the grain samples into three clusters, and compare the clusters to the grain varieties using a cross-tabulation.

You have the array samples of grain samples, and a list varieties giving the grain variety for each sample. Pandas (pd) and KMeans have already been imported for you.

Create a KMeans model called model with 3 clusters.
Use the .fit_predict() method of model to fit it to samples and derive the cluster labels. Using .fit_predict() is the same as using .fit() followed by .predict().
Create a DataFrame df with two columns named 'labels' and 'varieties', using labels and varieties, respectively, for the column values. This has been done for you.
Use the pd.crosstab() function on df['labels'] and df['varieties'] to count the number of times each grain variety coincides with each cluster label. Assign the result to ct.
Hit submit to see the cross-tabulation!

# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)
#############################################################