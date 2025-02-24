### INTRODUCTION TO DATA VISUALIZATION WITH MATPLOTLIB


# Intro to PYPLOT INTERFACE (OOP interface)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
    -creates a figure object (a container to holds data that is displayed)
    -creates an axis object (holds the data [the canvas])

plt.show()


# Add Data to Axes

<dataFrame>["<column>"]

ax.plot(<dataFrame>["<column1>"], <dataFrame>["<column2>"])
plt.show()

# Adding more data to Axes and putting it all together

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(<dataframe1>["<column1>"], <dataFrame1>["<column2>"])
ax.plot(<dataFrame2>["<column1>"], <dataFrame2>["<column2>"])
plt.show()


# Adding Data to an Axes Object

Import the matplotlib.pyplot submodule as plt.
Create a Figure and an Axes object by calling plt.subplots.
Add data from the seattle_weather DataFrame by calling the Axes plot method.
Add data from the austin_weather DataFrame in a similar manner and call plt.show to show the results.


# Import the matplotlib.pyplot submodule and name it plt
import matplotlib.pyplot as plt

# Create a Figure and an Axes with plt.subplots
fig, ax = plt.subplots()

# Plot MLY-PRCP-NORMAL from seattle_weather against the MONTH
ax.plot(seattle_weather["MONTH"],seattle_weather["MLY-PRCP-NORMAL"])

# Plot MLY-PRCP-NORMAL from austin_weather against MONTH
ax.plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"])

# Call the show function
plt.show()
#########################