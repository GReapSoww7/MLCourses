### VISUALIZING YOUR DATA


# Histograms

-using bins arg to adjust

import matplotlib as plt

<dataFrame>["<column>"].hist(bins=<val>)
plt.show()


# Bar Plots
-can reveal relationships between a categorical var and numeric var (ex. breed, and weight)
    -to compute avg weight of each breed (across columns)
        -group by breed > select weight column > and take  the mean

<newDataFrame> = <dataFrame>.groupby("<column1>")["<column2>"].mean()
print(<newDataFrame>)

<newDataFrame>.plot(kind="bar")
plt.show()

or

<newDataFrame>plot(kind="bar",
                   title="Mean <column> by <subject> <column>")
plt.show()

# Line Plots
-visualize changes in numeric vars over time

<dataFrame VALUE>.head()

<dataFrame VAL>.plot(x="<column1>",
                     y="<column2>",
                     kind="line")
plt.show()


# Rotating Axis Labels
-rotate x-axis to make text easier to read

<dataFrame VAL>.plot(x="date", y="weight_kg", kind="line", rot=45)
plt.show()


# Scatter Plots
-visualize relationships between two numeric vars

<dataFrame>.plot(x="<column1>", y="<column2>", kind="scatter")
plt.show()

# Layering Plots
-plots can be layered on top of one another

<dataFrame>[<dataFrame>["<column>"]=="<VAL>"]["<column2>"].hist()
<dataFrame>[<dataFrame>["<column>"]=="<VAL>"]["<column2>"].hist()
plt.show()

# Add a legend
-a Plot-Dot-Legend (passing in a list of labels) > then call plt.show()
    -so we can differentiate color

<dataFrame>[<dataFrame>["<column1>"]=="<VAL1>"]["<column2>"].hist()
<dataFrame>[<dataFrame>["<column1>"]=="<VAL2>"]["<column2>"].hist()

plt.legend(["<Val1>", "<Val2>"])
plt.show()

# Transparency
<dataFrame>[<dataFrame>["<column1>"]=="<VAL1>"]["<column2>"].hist(alpha=0.7)
<dataFrame>[<dataFrame>["<column1>"]=="<VAL2>"]["<column2>"].hist(alpha=0.7)

plt.legend(["<Val1>", "<Val2>"])
plt.show()


###############################



# PRACTICE

Print the head of the avocados dataset. What columns are available?
For each avocado size group, calculate the total number sold, storing as nb_sold_by_size.
Create a bar plot of the number of avocados sold by size.
Show the plot.

# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Look at the first few rows of data
print(avocados.head())

# Get the total number of avocados sold of each size
nb_sold_by_size = avocados.groupby("size")["nb_sold"].sum()

# Create a bar plot of the number of avocados sold by size
nb_sold_by_size.plot(kind="bar")

# Show the plot
plt.show()

###############################

# CHANGES IN SALES OVER TIME

Get the total number of avocados sold on each date. The DataFrame has two rows for each date—one for organic, and one for conventional. Save this as nb_sold_by_date.
Create a line plot of the number of avocados sold.
Show the plot.

# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Get the total number of avocados sold on each date
nb_sold_by_date = avocados.groupby("date")["nb_sold"].sum()

# Create a line plot of the number of avocados sold by date
nb_sold_by_date.plot(kind="line")

# Show the plot
plt.show()

###############################

# AVOCADO SUPPLY AND DEMAND

Create a scatter plot with nb_sold on the x-axis and avg_price on the y-axis. Title it "Number of avocados sold vs. average price".
Show the plot.


# Scatter plot of avg_price vs. nb_sold with title
avocados.plot(
    x="nb_sold", 
    y="avg_price",
    kind="scatter",
    title="Number of avocados sold vs. average price"
)

# Show the plot
plt.show()


###################################

# PRICE OF CONVENTIONAL VS ORGANIC AVOCADOS

Subset avocados for the "conventional" type and create a histogram of the avg_price column.
Create a histogram of avg_price for "organic" type avocados.
Add a legend to your plot, with the names "conventional" and "organic".
Show your plot.


# Modify histogram transparency to 0.5 
avocados[avocados["type"] == "conventional"]["avg_price"].hist()

# Modify histogram transparency to 0.5
avocados[avocados["type"] == "organic"]["avg_price"].hist()

# Add a legend
plt.legend(["conventional", "organic"])

# Show the plot
plt.show()

###################