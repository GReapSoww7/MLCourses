### AGGREGATING DATA FRAMES


# Summary Stats

<dataFrame>.mean()
    -center of where data is

.median(), .mode(), .min(), .max(), .var(), .std(), .sum(), .quantile()

-you can get summary stats for date columns

<dataFrame>["date_of_birth"].min()

 
.agg() method:
    -allows to compute custom summary Stats


def pct30(column):
    return column.quantile(0.3)

<dataFrame>["weight_kg"].agg(pct30)
    -30th percentile of dog's weight

# summaries on multiple columns

<dataFrame>[["weight_kg", "height_cm"]].agg(pct30)


# multiple summ stats simultaneously

def pct40(column):
    return column.quantile(0.4)

<dataFrame>["weight_kg"].agg([pct30, pct40])


# Cumulative sum

<dataFrame>["weight_kg"]

<dataFrame>["weight_kg"].cumsum()


# Cumulative stats

.cummax()
.cummin()
.cumprod()


# PRACTICE

Explore your new DataFrame first by printing the first few rows of the sales DataFrame.
Print information about the columns in sales.
Print the mean of the weekly_sales column.
Print the median of the weekly_sales column.

# Print the head of the sales DataFrame
print(sales.head())

# Print the info about the sales DataFrame
print(sales.info())

# Print the mean of weekly_sales
print(sales["weekly_sales"].mean())

# Print the median of weekly_sales
print(sales["weekly_sales"].median())

Print the maximum of the date column.
Print the minimum of the date column.

# Print the maximum of the date column
print(sales["date"].max())

# Print the minimum of the date column
print(sales["date"].min())


Use the custom iqr function defined for you along with .agg() to print the IQR of the temperature_c column of sales.
Update the column selection to use the custom iqr function with .agg() to print the IQR of temperature_c, fuel_price_usd_per_l, and unemployment, in that order.
Update the aggregation functions called by .agg(): include iqr and np.median in that order.

# Import NumPy and create custom IQR function
import numpy as np
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)

# Update to print IQR and median of temperature_c, fuel_price_usd_per_l, & unemployment
print(sales[["temperature_c", "fuel_price_usd_per_l", "unemployment"]].agg([iqr, np.median]))


Sort the rows of sales_1_1 by the date column in ascending order.
Get the cumulative sum of weekly_sales and add it as a new column of sales_1_1 called cum_weekly_sales.
Get the cumulative maximum of weekly_sales, and add it as a column called cum_max_sales.
Print the date, weekly_sales, cum_weekly_sales, and cum_max_sales columns.


# Sort sales_1_1 by date
sales_1_1 = sales_1_1.sort_values("date", ascending=True)

# Get the cumulative sum of weekly_sales, add as cum_weekly_sales col
sales_1_1["cum_weekly_sales"] = sales["weekly_sales"].cumsum()

# Get the cumulative max of weekly_sales, add as cum_max_sales col
sales_1_1["cum_max_sales"] = sales["weekly_sales"].cummax()

# See the columns you calculated
print(sales_1_1[["date", "weekly_sales", "cum_weekly_sales", "cum_max_sales"]])

#########################

# Counting


# Dropping duplicate names

<dataFrame>.drop_duplicates(subset="<column>")


# Dropping duplicate pairs

<newDataFrame> = <dataFrame>.drop_duplicates(subset=["<column1>", "<column2>"])
print(<variable>)

# Easy as 1, 2, 3

<newDataFrame>["<column1>"].value_counts()

<newDataFrame>["<column1>"].value_counts(sort=True)


# Proportions

<newDataFrame>["<column>"].value_counts(normalize=True)


# PRACTICE

Remove rows of sales with duplicate pairs of store and type and save as store_types and print the head.
Remove rows of sales with duplicate pairs of store and department and save as store_depts and print the head.
Subset the rows that are holiday weeks using the is_holiday column, and drop the duplicate dates, saving as holiday_dates.
Select the date column of holiday_dates, and print.



# Drop duplicate store/type combinations
store_types = sales.drop_duplicates(subset=["store", "type"])
print(store_types.head())

# Drop duplicate store/department combinations
store_depts = sales.drop_duplicates(subset=["store", "department"])
print(store_depts.head())

# Subset the rows where is_holiday is True and drop duplicate dates
holiday_dates = sales[sales["is_holiday"] == True].drop_duplicates(subset="date")

# Print date col of holiday_dates
print(holiday_dates["date"])



Count the number of stores of each store type in store_types.
Count the proportion of stores of each store type in store_types.
Count the number of stores of each department in store_depts, sorting the counts in descending order.
Count the proportion of stores of each department in store_depts, sorting the proportions in descending order.


# Count the number of stores of each type
store_counts = store_types["type"].value_counts()
print(store_counts)

# Get the proportion of stores of each type
store_props = store_types["type"].value_counts(normalize=True)
print(store_props)

# Count the number of stores for each department and sort
dept_counts_sorted = store_depts["department"].value_counts(sort=True)
print(dept_counts_sorted)

# Get the proportion of stores in each department and sort
dept_props_sorted = store_depts["department"].value_counts(sort=True, normalize=True)
print(dept_props_sorted)

#######################################

# Grouped Summary Stats

-useful to compare different groups
    -gain insights from summaries of different groups

-subset DataFrame into groups based on <condition>s

<dataFrame>[<dataFrame["<column>"] == "<condition>"]["<column2>"].mean()
-replicate with different <condition>

TOO MUCH WORK/COPY PASTE BUGS

# Solution for this/GROUPED SUMMARIES
<dataFrame>.groupby("<columnVariable>")["<column>"].mean()

# MULTIPLE GROUPED SUMMARIES
<dataFrame>.groupby("<columnVar>")["<column>"].agg([min, max, sum])

# GROUPING BY MULTIPLE VARS

<dataFrame>.groupby(["<column1>", "<column2>"])["<column3>"].mean()

# MANY GROUPS, MANY SUMMARIES

<dataFrame>.groupby(["<column1>", "<column2>"])[["<column3>", "<column4>"]].mean()


# PRACTICE

# What percent of sales occurred at each store type? (without .groupby())

Calculate the total weekly_sales over the whole dataset.
Subset for type "A" stores, and calculate their total weekly sales.
Do the same for type "B" and type "C" stores.
Combine the A/B/C results into a list, and divide by sales_all to get the proportion of sales

# Calc total weekly sales
sales_all = sales["weekly_sales"].sum()

# Subset for type A stores, calc total weekly sales
sales_A = sales[sales["type"] == "A"]["weekly_sales"].sum()

# Subset for type B stores, calc total weekly sales
sales_B = sales[sales["type"] == "B"]["weekly_sales"].sum()

# Subset for type C stores, calc total weekly sales
sales_C = sales[sales["type"] == "C"]["weekly_sales"].sum()

# Get proportion for each type
sales_propn_by_type = [sales_A, sales_B, sales_C] / sales_all
print(sales_propn_by_type)

#############################################

# Calculations with .groupby()

Group sales by "type", take the sum of "weekly_sales", and store as sales_by_type.
Calculate the proportion of sales at each store type by dividing by the sum of sales_by_type. Assign to sales_propn_by_type.

# Group by type; calc total weekly sales
sales_by_type = sales.groupby("type")["weekly_sales"].sum()

# Get proportion for each type
sales_propn_by_type = sales_by_type / sum(sales_by_type)
print(sales_propn_by_type)

###################################

# Calculations with .groupby()

Group sales by "type" and "is_holiday", take the sum of weekly_sales, and store as sales_by_type_is_holiday.

# From previous step
sales_by_type = sales.groupby("type")["weekly_sales"].sum()

# Group by type and is_holiday; calc total weekly sales
sales_by_type_is_holiday = sales.groupby(["type", "is_holiday"])["weekly_sales"].sum()
print(sales_by_type_is_holiday)

#############################################

# Multiple Grouped Summaries

Import numpy with the alias np.
Get the min, max, mean, and median of weekly_sales for each store type using .groupby() and .agg(). Store this as sales_stats. Make sure to use numpy functions!
Get the min, max, mean, and median of unemployment and fuel_price_usd_per_l for each store type. Store this as unemp_fuel_stats.


# Import numpy with the alias np
import numpy as np

# For each store type, aggregate weekly_sales: get min, max, mean, and median
sales_stats = sales.groupby("type")["weekly_sales"].agg([min, max, np.mean, np.median])

# Print sales_stats
print(sales_stats)

# For each store type, aggregate unemployment and fuel_price_usd_per_l: get min, max, mean, and median
unemp_fuel_stats = sales.groupby("type")[["unemployment", "fuel_price_usd_per_l"]].agg([min, max, np.mean, np.median])

# Print unemp_fuel_stats
print(unemp_fuel_stats)



#################################################

# Pivot Tables

-calculate group summ stats (ex. spreadsheet)

column1 = "color"
column2 = "weight_kg"

# Group by Pivot Table
<dataFrame>.groupby("<column1>")["<column2>"].mean()

<dataFrame>.pivot_table(values="<column2>", index="<column1>")

# Different Stats

<dataFrame>.pivot_table(values="<column2>", index="<column1>", aggfunc=np.median)

# Multiple Stats

<dataFrame>.pivot_table(values="<column2>", index="<column1>", aggfunc=[np.mean, np.median])

# Pivot on Two Vars

<dataFrame>.groupby(["<column1>", "<column3>"])["<column2>"].mean()

<dataFrame>.pivot_table(values="<column2>", index="<column1>", columns="<column3>")

# Filling missing values in pivot tables

<dataFrame>.pivot_table(values="<column2>", index="<column1>", columns="<column3>", fill_value=0)

# Summing with Pivot Tables
<dataFrame>.pivot_table(values="<column2>", index="<column1>", columns="<column3>", fill_value=0, margins=True)
    -last row and last column provide means of values provided
    -margin=True allows us to see a Summ Stat for multiple levels of the dataset

###############################
# PRACTICE

# Pivoting on one Var

Get the mean weekly_sales by type using .pivot_table() and store as mean_sales_by_type.

# Pivot for mean weekly_sales for each store type
mean_sales_by_type = sales.pivot_table(values="weekly_sales", index="type")

# Print mean_sales_by_type
print(mean_sales_by_type)
##########

Get the mean and median (using NumPy functions) of weekly_sales by type using .pivot_table() and store as mean_med_sales_by_type.

# Import NumPy as np
import numpy as np

# Pivot for mean and median weekly_sales for each store type
mean_med_sales_by_type = sales.pivot_table(values="weekly_sales", index="type", aggfunc=[np.mean, np.median])

# Print mean_med_sales_by_type
print(mean_med_sales_by_type)

##############

Get the mean of weekly_sales by type and is_holiday using .pivot_table() and store as mean_sales_by_type_holiday.

# Pivot for mean weekly_sales by store type and holiday 
mean_sales_by_type_holiday = sales.pivot_table(values="weekly_sales", index="type", columns="is_holiday")

# Print mean_sales_by_type_holiday
print(mean_sales_by_type_holiday)



############################

# FILL IN MISSING VALS AND SUM VALS WITH PIVOT TABLES

Print the mean weekly_sales by department and type, filling in any missing values with 0.

# Print mean weekly_sales by department and type; fill missing values with 0
print(sales.pivot_table(values="weekly_sales", index="department", columns="type", fill_value=0))

#####

Print the mean weekly_sales by department and type, filling in any missing values with 0 and summing all rows and columns.

print(sales.pivot_table(values="weekly_sales", index="department", columns="type", fill_value=0, margins=True))


# NOTE:
Note the subtlety in the value of margins here. 

The column 'All' returns an overall mean for each department, not (A+B)/2. 

(A+B)/2 would be a mean of means, rather than an overall mean per department!
####################################