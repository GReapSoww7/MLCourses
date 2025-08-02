### VISUALIZE CLUSTERS ###


# Why Visualize?
-try to make sense of clusters formed
-an additional step in VALIDATION of clusters
-Spot TRENDS in data


# Use Seaborn
-based on matplotlib
-contains functions that make data visualization tasks easy in the context of data analytics

# Visualize with matplotlib
from matplotlib import pyplot as plt
df = pd.DataFrame({'x': [2,3,5,6,2], 'y': [1,1,5,5,2], 'labels': ['A', 'A', 'B', 'B', 'A']})
colors = {'A':'red', 'B':'blue'}
df.plot.scatter(x='x', y='y', c=df['labels'].apply(lambda x: colors[x]))
plt.show()


# Visualize with Seaborn
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.DataFrame({'x': [2,3,5,6,2], 'y': [1,1,5,5,2], 'labels': ['A', 'A', 'B', 'B', 'A']})
sns.scatterplot(x='x', y='y', hue='labels', data=df)
plt.show()


# Visualize Clusters with matplotlib

# Import the pyplot class
from matplotlib import pyplot as plt

# Define a colors dictionary for clusters
colors = {1:'red', 2:'blue'}

# Plot a scatter plot
comic_con.plot.scatter(x='x_scaled', 
                	   y='y_scaled',
                       c=comic_con['cluster_labels'].apply(lambda x: colors[x]))
plt.show()

# Visualize Clusters with Seaborn

# Import the seaborn module
import seaborn as sns

# Plot a scatter plot using seaborn
sns.scatterplot(x='x_scaled', 
                y='y_scaled', 
                hue='cluster_labels', 
                data = comic_con)
plt.show()

########################################