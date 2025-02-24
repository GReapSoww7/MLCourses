### READING AND WRITING CSVs


# Comma Separated Values
    -store tabulated data (DataFrame-like data)
    -each row of data has its own line; and each value is separated by a comma
    -most database and spreadsheet programs use/create them


# CSV to DataFrame

import pandas as pd
<dataFrame> = pd.read_csv("/path/to/<file>.csv")
print(<dataFrame>)

# DataFrame Manipulation

<dataFrame>["<column>"] = <dataFrame>["<column2>"] / (<dataframe>["<column3>"] / 100) ** 2
print(<newDataFrame>)


# Dataframe to CSV

<dataFrame>.to_csv("/path/to/<file>.csv")

####################

# PRACTICE

# CSV to DataFrame

Read the CSV file "airline_bumping.csv" and store it as a DataFrame called airline_bumping
print the first few rows of airline_bumping

For each airline group, select the nb_bumped, and total_passengers columns, and calculate the sum (for both years). Store this as airline_totals.

Create a new column of airline_totals called bumps_per_10k, which is the number of passengers bumped per 10,000 passengers in 2016 and 2017.

Print airline_totals to see the results of your manipulations.


# From previous steps
airline_bumping = pd.read_csv("airline_bumping.csv")
print(airline_bumping.head())
airline_totals = airline_bumping.groupby("airline")[["nb_bumped", "total_passengers"]].sum()
airline_totals["bumps_per_10k"] = airline_totals["nb_bumped"] / airline_totals["total_passengers"] * 10000

# Print airline_totals
print(airline_totals)
##########################

# DataFrame to CSV

Sort airline_totals by the values of bumps_per_10k from highest to lowest, storing as airline_totals_sorted
Print your sorted DataFrame
Save the sorted DataFrame as a CSV called "airline_totals_sorted.csv"

airline_totals_sorted = airline_totals.sort_values("bumps_per_10k", ascending=False)
print(airline_totals_sorted)
airline_totals_sorted.to_csv("airline_totals_sorted.csv")