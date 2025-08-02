### DATA PREPARATION FOR CLUSTER ANALYSIS ###


# Why do we Prep data for clustering?

-Vars have INCOMPARABLE units (product dimensions in cm, price in $)
-Vars with SAME units have VASTLY different scales and variances (expenditures on cereals, travel)
-data in RAW form may lead to BIAS in Clustering
-Clusters may be heavily dependent on ONE var
-SOLUTION: NORMALIZATION of individual vars

# Normalization
Normalization: the process of RESCALING data to a standard deviation (stddev) of 1

x_new = x / std_dev(x)

from scipy.cluster.vq import whiten
data = [5, 1, 3, 3, 2, 3, 3, 8, 1, 2, 2, 3, 5]
scaled_data = whiten(data)
print(scaled_data)
[2.73, 0.55, 1.64, 1.64, 1.09, 1.64, 1.64, 4.36, 0.55, 1.09, 1.09, 1.64, 2.73]


# Illustration: normalization of data

from matplot lib import pyplot as plt

# init original, scaled data
plt.plot(data, label='original')
plt.plot(scaled_data, label='scaled')
# show legend and display
plt.legend()
plt.show()


# Normalize Basic List Data

from scipy.cluster.vq import whiten
goals_for = [4,3,2,3,1,1,2,0,1,4]
scaled_data = whiten(goals_for)
print(scaled_data)

<script.py> output:
    [3.07692308 2.30769231 1.53846154 2.30769231 0.76923077 0.76923077
     1.53846154 0.         0.76923077 3.07692308]


# Visualize Normalized Data
from matplotlib import pyplot as plt

plt.plot(goals_for, label='original')
plt.plot(scaled_data, label='scaled')
plt.legend()
plt.show()

# Normalization of SMALL Numbers

# Prepare data
rate_cuts = [0.0025, 0.001, -0.0005, -0.001, -0.0005, 0.0025, -0.001, -0.0015, -0.001, 0.0005]

# Use the whiten() function to standardize the data
scaled_data = whiten(rate_cuts)

# Plot original data
plt.plot(rate_cuts, label='original')

# Plot scaled data
plt.plot(scaled_data, label='scaled')

plt.legend()
plt.show()


# Normalize Data

# Scale wage and value
fifa['scaled_wage'] = whiten(fifa['eur_wage'])
fifa['scaled_value'] = whiten(fifa['eur_value'])

# Plot the two columns in a scatter plot
fifa.plot(x='scaled_wage', y='scaled_value', kind = 'scatter')
plt.show()

# Check mean and standard deviation of scaled values
print(fifa[['scaled_wage', 'scaled_value']].describe())


<script.py> output:
           scaled_wage  scaled_value
    count      1000.00       1000.00
    mean          1.12          1.31
    std           1.00          1.00
    min           0.00          0.00
    25%           0.47          0.73
    50%           0.85          1.02
    75%           1.41          1.54
    max           9.11          8.98


##############################################