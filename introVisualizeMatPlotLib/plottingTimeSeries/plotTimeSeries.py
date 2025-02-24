### PLOTTING TIME-SERIES DATA


# Time-series data (detect patterns)

-DateTimeIndex

<dataFrame>.index

-Time-Series Data

<dataFrame>['relative_temp'] <dataFrame>['co2']

-Plotting Time-Series Data

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

ax.plot(<dataFrame>.index, <dataFrame>['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()

-Zooming in on a decade

sixties = <dataFrame>["1960-01-01":"1969-12-31"]

fig, ax = plt.subplots()
ax.plot(sixties.index, sixties['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()

-Zooming in on a year

sixty_nine = <dataFrame>["1969-01-01":"1969-12-31"]

fig, ax = plt.subplots()
ax.plot(sixty_nine.index, sixty_nine['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()

# PRACTICE

# Read Data with a Time Index

Import the pandas library as pd.
Read in the data from a CSV file called 'climate_change.csv' using pd.read_csv.
Use the parse_dates key-word argument to parse the "date" column as dates.
Use the index_col key-word argument to set the "date" column as the index.


# Import pandas as pd
import pandas as pd

# Read the data from file using read_csv
climate_change = pd.read_csv('climate_change.csv', parse_dates=["date"], index_col="date")

##############################

# Plot Time-Series Data

Add the data from climate_change to the plot: use the DataFrame index for the x value and the "relative_temp" column for the y values.
Set the x-axis label to 'Time'.
Set the y-axis label to 'Relative temperature (Celsius)'.
Show the figure.


import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# Add the time-series for "relative_temp" to the plot
ax.plot(climate_change.index, climate_change['relative_temp'])

# Set the x-axis label
ax.set_xlabel('Time')

# Set the y-axis label
ax.set_ylabel('Relative temperature (Celsius)')

# Show the figure
plt.show()
####################################

# Using a Time Index to Zoom In

Use plt.subplots to create a Figure with one Axes called fig and ax, respectively.
Create a variable called seventies that includes all the data between "1970-01-01" and "1979-12-31".
Add the data from seventies to the plot: use the DataFrame index for the x value and the "co2" column for the y values.
    
import matplotlib.pyplot as plt

# Use plt.subplots to create fig and ax
fig, ax = plt.subplots()

# Create variable seventies with data from "1970-01-01" to "1979-12-31"
seventies = climate_change["1970-01-01":"1979-12-31"]

# Add the time-series for "co2" data from seventies to the plot
ax.plot(seventies.index, seventies["co2"])

# Show the figure
plt.show()

###########################################

# Plotting Time-Series with Different Variables

-Plotting two time-series together
    -Using Twin Axes

import matplotlib.pyplot as plt
fix, ax = plt.subplots()

ax.plot(climate_change.index, climate_change['co2'], color='b')
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)', color='b')
ax.tick_params('y', colors='b')
ax2 = ax.twinx()
ax2.plot(climate_change.index, climate_change['relative_temp'], color='r')
ax2.set_ylabel('Relative temperature (Celsius)', color='r')
ax2.tick_params('y', colors='r')
plt.show()

- A Function that Plots Time-Series Data

def plot_timeseries(axes, x, y, color, xlabel, ylabel):
    axes.plot(x, y, color=color)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel, color=color)
    axes.tick_params('y', colors=color)

- Using the Function

fig, ax = plt.subplots()
plot_timeseries(ax, climate_change.index, climate_change['co2'], 'blue', 'Time', 'CO2 (ppm)')
ax2 = ax.twinx()
plot_timeseries(ax2, climate_change.index, climate_change['relative_temp'], 'red', 'Time', 'Relative temperature (Celsius)')


##################

# Plotting Two Variables

Use plt.subplots to create a Figure and Axes objects called fig and ax, respectively.
Plot the carbon dioxide variable in blue using the Axes plot method.
Use the Axes twinx method to create a twin Axes that shares the x-axis.
Plot the relative temperature variable in red on the twin Axes using its plot method.


import matplotlib.pyplot as plt

# Initalize a Figure and Axes
fig, ax = plt.subplots()

# Plot the CO2 variable in blue
ax.plot(climate_change.index, climate_change['co2'], color='b')

# Create a twin Axes that shares the x-axis
ax2 = ax.twinx()

# Plot the relative temperature in red
ax2.plot(climate_change.index, climate_change['relative_temp'], color='r')

#####################################

# Defining a Function that Plots Time-Series Data

Define a function called plot_timeseries that takes as input an Axes object (axes), data (x,y), a string with the name of a color and strings for x- and y-axis labels.
Plot y as a function of in the color provided as the input color.
Set the x- and y-axis labels using the provided input xlabel and ylabel, setting the y-axis label color using color.
Set the y-axis tick parameters using the tick_params method of the Axes object, setting the colors key-word to color.


def plot_timeseries(axes, x, y, color, xlabel, ylabel):
    axes.plot(x, y, color=color)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel, color=color)
    axes.tick_params('y', color=color)

#############################################

# Using a Plotting Function

In the provided ax object, use the function plot_timeseries to plot the "co2" column in blue, with the x-axis label "Time (years)" and y-axis label "CO2 levels".
Use the ax.twinx method to add an Axes object to the figure that shares the x-axis with ax.
Use the function plot_timeseries to add the data in the "relative_temp" column in red to the twin Axes object, with the x-axis label "Time (years)" and y-axis label "Relative temperature (Celsius)".

fig, ax = plt.subplots()

plot_timeseries(ax, climate_change.index, climate_change['co2'], 'blue', 'Time (years)', 'CO2 levels')
ax2 = ax.twinx()
plot_timeseries(ax2, climate_change.index, climate_change['relative_temp'], 'red', 'Time (years)', 'Relative temperature (Celsius)')

#########################################

# Annotating Time-Series Data
-annotations = small pieces of text that refer to a part of the visualization > focusing on a feature of the data and explaining it

-Annotation > positioning text > adding arrows to annotation > customize arrow properties

fig, ax = plt.subplots()

plot_timeseries(ax, climate_change.index, climate_change['co2'], 'blue', 'Time', 'CO2 (ppm)')
ax2 = ax.twinx()
plot_timeseries(ax2, climate_change.index, climate_change['relative_temp'], 'red', 'Time', 'Relative temperature (Celsius)')
ax2.annotate('>1 degree', 
             xy=(pd.Timestamp('2015-10-06'), 1),
             xytext=(pd.Timestamp('2008-10-06'), -0.2),
             arrowprops={'arrowstyle':'->', 'color':'gray'}
)
plt.show()
###############################################

# Annotating a Plot of Time-Series Data

Use the ax.plot method to plot the DataFrame index against the relative_temp column.
Use the annotate method to add the text '>1 degree' in the location (pd.Timestamp('2015-10-06'), 1).

fig, ax = plt.subplots()

ax.plot(climate_change.index, climate_change['relative_temp'])
ax.annotate('>1 degree',
            xy=(pd.Timestamp('2015-10-06'), 1)
)
plt.show()
#####################

# Plotting Time-Series: Putting it All Together

Use the plot_timeseries function to plot CO2 levels against time. Set xlabel to "Time (years)" ylabel to "CO2 levels" and color to 'blue'.
Create ax2, as a twin of the first Axes.
In ax2, plot temperature against time, setting the color ylabel to "Relative temp (Celsius)" and color to 'red'.
Annotate the data using the ax2.annotate method. Place the text ">1 degree" in x=pd.Timestamp('2008-10-06'), y=-0.2 pointing with a gray thin arrow to x=pd.Timestamp('2015-10-06'), y = 1.


fig, ax = plt.subplots()

plot_timeseries(ax, climate_change.index, climate_change['co2'], 'blue', 'Time (years)', 'CO2 levels')
ax2 = ax.twinx()
plot_timeseries(ax2, climate_change.index, climate_change['relative_temp'], 'red', 'Time (years)', 'Relative temp (Celsius)')
ax2.annotate('>1 degree',
             xy=(pd.Timestamp('2015-10-06'), 1),
             xytext=(pd.Timestamp('2008-10-06'), -0.2),
             arrowprops={'arrowstyle':'->', 'color':'gray'}
)
################################################################