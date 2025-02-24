### SMALL MULTIPLES


# Adding too much Data
ax.set_xlabel("Time (months)")
ax.set_ylabel("Precipitation (inches)")
ax.plot(<dataFrame1>["<column1>"], <dataFrame1>["MLY-PRCP-NORMAL"],
        color='r')
ax.plot(<dataFrame1>["<column1>"], <dataFrame1>["MLY-PRCP-25PCTL"],
        linestyle='--', color='r')
ax.plot(<dataFrame1>["<column1>"], <dataFrame1>["MLY-PRCP-75PCTL"],
        linestyle='--', color='r')

ax.plot(<dataFrame2>["<column1>"], <dataFrame2>["MLY-PRCP-NORMAL"],
        color='b')
ax.plot(<dataFrame2>["<column1>"], <dataFrame2>["MLY-PRCP-25PCTL"],
        linestyle='--', color='b')
ax.plot(<dataFrame2>["<column1>"], <dataFrame2>["MLY-PRCP-75PCTL"],
        linestyle='--', color='color')
plt.show()

-WITH THIS WE WANT TO USE SMALL MULTIPLES (multiple small plots that shows similar data over different conditions)

# Small multiples with plt.subplots

fig, ax = plt.subplots(<rows>, <columns>)

-give inputs

fig, ax = plt.subplots(3, 2)

-ax is now an array of axes objects with a ax.shape() of 3 X 2

ax[0,0].plot(<dataFrame1>["<column1>"],
             <dataFrame2>["<column2>"],
             color='b')
plt.show()


# Subplots with data

fig, ax = plt.subplots(2, 1, sharey=True)

ax[0].plot(<dataFrame1>["<column1>"], <dataFrame1>["MLY-PRCP-NORMAL"],
        color='r')
ax[0].plot(<dataFrame1>["<column1>"], <dataFrame1>["MLY-PRCP-25PCTL"],
        linestyle='--', color='r')
ax[0].plot(<dataFrame1>["<column1>"], <dataFrame1>["MLY-PRCP-75PCTL"],
        linestyle='--', color='r')

ax[1].plot(<dataFrame2>["<column1>"], <dataFrame2>["MLY-PRCP-NORMAL"],
        color='b')
ax[1].plot(<dataFrame2>["<column1>"], <dataFrame2>["MLY-PRCP-25PCTL"],
        linestyle='--', color='b')
ax[1].plot(<dataFrame2>["<column1>"], <dataFrame2>["MLY-PRCP-75PCTL"],
        linestyle='--', color='color')

ax[0].set_ylabel("Precipitation (inches)")
ax[1].set_ylabel("Precipitation (inches)")
ax[1].set_xlabel("Time (months)")

plt.show()

##############

# Creating Small Multiples with plt.subplots

Create a Figure and an array of subplots with 2 rows and 2 columns.
Addressing the top left Axes as index 0, 0, plot the Seattle precipitation.
In the top right (index 0,1), plot Seattle temperatures.
In the bottom left (1, 0) and bottom right (1, 1) plot Austin precipitations and temperatures.


# Create a Figure and an array of subplots with 2 rows and 2 columns
fig, ax = plt.subplots(2, 2)

# Addressing the top left Axes as index 0, 0, plot month and Seattle precipitation
ax[0, 0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-NORMAL"])

# In the top right (index 0,1), plot month and Seattle temperatures
ax[0, 1].plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])

# In the bottom left (1, 0) plot month and Austin precipitations
ax[1, 0].plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"])

# In the bottom right (1, 1) plot month and Austin temperatures
ax[1, 1].plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
plt.show()

##########################################

# Small Multiples with Shared y axis

Create a Figure with an array of two Axes objects that share their y-axis range.
Plot Seattle's "MLY-PRCP-NORMAL" in a solid blue line in the top Axes.
Add Seattle's "MLY-PRCP-25PCTL" and "MLY-PRCP-75PCTL" in dashed blue lines to the top Axes.
Plot Austin's "MLY-PRCP-NORMAL" in a solid red line in the bottom Axes and the "MLY-PRCP-25PCTL" and "MLY-PRCP-75PCTL" in dashed red lines.


# Create a figure and an array of axes: 2 rows, 1 column with shared y axis
fig, ax = plt.subplots(2, 1, sharey=True)

# Plot Seattle precipitation data in the top axes
ax[0].plot(seattle_weather["MLY-PRCP-NORMAL"], color = 'b')
ax[0].plot(seattle_weather["MLY-PRCP-25PCTL"], color = 'b', linestyle = '--')
ax[0].plot(seattle_weather["MLY-PRCP-75PCTL"], color = 'b', linestyle = '--')

# Plot Austin precipitation data in the bottom axes
ax[1].plot(austin_weather["MLY-PRCP-NORMAL"], color = 'r')
ax[1].plot(austin_weather["MLY-PRCP-25PCTL"], color = 'r', linestyle = '--')
ax[1].plot(austin_weather["MLY-PRCP-75PCTL"], color = 'r', linestyle = '--')

plt.show()
##########################