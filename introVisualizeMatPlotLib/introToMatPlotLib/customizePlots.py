### CUSTOMIZING YOUR PLOTS


# Customize Data Appearance

-adding Markers

-lowercase 'o' = circles
-lowercase 'v' = triangles
#://matplotlib.org/api/markers_api.html

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(<dataFrame>["<column1>"],
        <dataFrame>["<column2>"],
        marker="o")
plt.show()
#####

# Setting the linestyle

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(<dataFrame>["<column1>"],
        <dataFrame>["<column2>"],
        marker="o", linestyle="--")
plt.show()

-or pass linestyle="None"

# Color


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(<dataFrame>["<column1>"],
        <dataFrame>["<column2>"],
        marker="o", linestyle="--", color="r")
plt.show()

-"r" = red


######

# Customizing axis labels and title

ax.set_xlabel("Time (months)")
ax.set_ylabel("Avg temperature (Fahrenheit degrees)")
ax.set_title("Weather in Seattle")
plt.show()


#####

# PRACTICE

# Customizing data appearance

Call ax.plot to plot "MLY-PRCP-NORMAL" against "MONTHS" in both DataFrames.
Pass the color key-word arguments to these commands to set the color of the Seattle data to blue ('b') and the Austin data to red ('r').
Pass the marker key-word arguments to these commands to set the Seattle data to circle markers ('o') and the Austin markers to triangles pointing downwards ('v').
Pass the linestyle key-word argument to use dashed lines for the data from both cities ('--').
    

# Plot Seattle data, setting data appearance
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-NORMAL"], marker="o", linestyle="--", color="b")

# Plot Austin data, setting data appearance
ax.plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"], marker="v", linestyle="--", color="r")

# Call show to display the resulting plot
plt.show()
##############

# Customizing Axis Labels and Adding Titles

Use the set_xlabel method to add the label: "Time (months)".
Use the set_ylabel method to add the label: "Precipitation (inches)".
Use the set_title method to add the title: "Weather patterns in Austin and Seattle".

ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-NORMAL"])
ax.plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"])

# Customize the x-axis label
ax.set_xlabel("Time (months)")

# Customize the y-axis label
ax.set_ylabel("Precipitation (inches)")

# Add the title
ax.set_title("Weather patterns in Austin and Seattle")

# Display the figure
plt.show()
#######################