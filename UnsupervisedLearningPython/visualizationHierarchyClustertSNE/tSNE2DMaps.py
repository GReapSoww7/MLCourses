### t-SNE FOR 2D MAPS

-t-distributed stochastic neighbor embedding

-maps samples to 2D space (or 3D)

-map approx preserves nearness of samples
    -great for inspecting datasets

# t-SNE on the Iris Dataset

-iris dataset has 4 measurements (4 dimensional)
-t-SNE maps samples to 2D space
    -t-SNE did NOT know that there were different species
        -yet it kept the species mostly separate


# Interpreting t-SNE Scatter Plots

-KMeans inertia plot: could argue that for 2 clusters (<value1> and <value2> data clusters are comingling)
    -or it could argue for 3


# t-SNE in sklearn

-2D NumPy array 'samples'

-List 'species' gives species LABELS as number (0, 1, or 2)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(samples)
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
plt.show()


# t-SNE ONLY has fit_transform() method

-simultaneously fits model/transforms data
    -does NOT have separate fit() or transform() methods
-can NOT extend the map to INCLUDE new data samples
    -MUST START OVER EACH TIME


# learning_rate

-choose different learning rates for different datasets
    -wrong choice = points will bunch together
-try values between 50 -> 200

# Different Every Time
-the axes (features) of a t-SNE plot are DIFFERENT EVERY TIME (no interpretable meaning)


#########################################################

# PRACTICE

# t-SNE Visualization of Grain Dataset

You saw t-SNE applied to the iris dataset. 
In this exercise, you'll apply t-SNE to the grain samples data and inspect the resulting t-SNE features using a scatter plot. 
You are given an array samples of grain samples and a list variety_numbers giving the variety number of each grain sample.


Import TSNE from sklearn.manifold.
Create a TSNE instance called model with learning_rate=200.
Apply the .fit_transform() method of model to samples. Assign the result to tsne_features.
Select the column 0 of tsne_features. Assign the result to xs.
Select the column 1 of tsne_features. Assign the result to ys.
Make a scatter plot of the t-SNE features xs and ys. To color the points by the grain variety, specify the additional keyword argument c=variety_numbers.

from skleanr.manifold import TSNE
model = TSNE(learning_rate=200)
tsne_features = model.fit_transform(samples)
xs = tsne_features[:,0]
ys = tsne_features[:,1]
plt.scatter(xs, ys, c=variety_numbers)
plt.show()

###################################################

# A t-SNE Map of the Stock Market

t-SNE provides great visualizations when the individual samples can be labeled. 
In this exercise, you'll apply t-SNE to the company stock price data. 
A scatter plot of the resulting t-SNE features, labeled by the company names, gives you a map of the stock market! 
The stock price movements for each company are available as the array normalized_movements (these have already been normalized for you). 
The list companies gives the name of each company. PyPlot (plt) has been imported for you.

Import TSNE from sklearn.manifold.
Create a TSNE instance called model with learning_rate=50.
Apply the .fit_transform() method of model to normalized_movements. Assign the result to tsne_features.
Select column 0 and column 1 of tsne_features.
Make a scatter plot of the t-SNE features xs and ys. Specify the additional keyword argument alpha=0.5.
Code to label each point with its company name has been written for you using plt.annotate(), so just hit submit to see the visualization!


from sklearn.manifold import TSNE
model = TSNE(learning_rate=50)
tsne_features = model.fit_transform(normalized_movements)
xs = tsne_features[:,0]
ys = tsne_features[:,1]
plt.scatter(xs, ys, alpha=0.5)

# annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()

###############################################################