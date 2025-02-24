### SLICING AND INDEXING DATAFRAMES

-DataFrames are composed of NumPy array for data, and two Indexes to store Row and Column details

# EXPLICIT INDEXES

.columns
    -contains index object of Column Names

.index
    -contains an index object of Row Numbers

<dataFrame>(['name', 'breed', 'color', 'height_cm', 'weight_kg'], dtype='object')

<dataFrame>(start=0, stop=7, step=1)

# Setting a column as the index

<dataFrameVariable> = <dataFrame>.set_index("<column>")
print(<dataFrameVariable>)

# Removing an index

<newDataFrame>.reset_index()

# Dropping an index

<newDataFrame>.reset_index(drop=True)

# Indexes make subsetting simpler

<dataFrame>[<dataFrame>["<column>"].isin(["<val>", "<val2>"])]

<newDataFrame>.loc[["<val>", "<val2>"]]

# Index Vals DON'T need to be unique

<newDataFrame2> = <dataFrame>.set_index("<column>")
print(<newDataFrame2>)

# Subsetting on Duplicated Index Vals

<newDataFrame2>.loc["<rowVALUE>"]

# Multi-level Indexes aka hierarchical indexes

<newDataFrame3> = <dataFrame>.set_index(["<column>", "<column2>"])
print(<newDataFrame3>)


# Subset the outer level with a list
    -to take a subset of rows at the outer level index, we pass a list of index values to loc

<newDataFrame3>.loc[["<rowVal1>", "<rowVal2>"]]

# Subset Inner Levels with a list of TUPLES

<newDataFrame3>.loc[[("<rowVal>", "<rowVal2>"), ("<rowVal3>", "<rowVal4>")]]

# Sorting by Index Vals

<newDataFrame3>.sort_index()
    -DEFAULT: sorts ALL index levels from OUTER to INNER, in ascending order

# Controlling sort_index
    -can control by passing lists to the LEVEL and ASCENDING args

<newDataFrame3>.sort_index(level=["<column>", "<column2>"], ascending=[True, False])


# WE HAVE 2 PROBLEMS

-Index values are JUST DATA
    -storing data in multiple forms makes it harder to think about
-Indexes violate "tidy data" principles

-two syntaxes = more bugs
    -we don't have to use indexes but are useful to know how to read

# PRACTICE

# Setting and Removing Indexes

Look at temperatures.
Set the index of temperatures to "city", assigning to temperatures_ind.
Look at temperatures_ind. How is it different from temperatures?
Reset the index of temperatures_ind, keeping its contents.
Reset the index of temperatures_ind, dropping its contents.

# Look at temperatures
print(temperatures)

# Set the index of temperatures to city
temperatures_ind = temperatures.set_index("city")

# Look at temperatures_ind
print(temperatures_ind)

# Reset the temperatures_ind index, keeping its contents
print(temperatures_ind.reset_index())

# Reset the temperatures_ind index, dropping its contents
print(temperatures_ind.reset_index(drop=True))


# Subsetting with .loc[]

Create a list called cities that contains "Moscow" and "Saint Petersburg".
Use [] subsetting to filter temperatures for rows where the city column takes a value in the cities list.
Use .loc[] subsetting to filter temperatures_ind for rows where the city is in the cities list.

# Make a list of cities to subset on
cities = ["Moscow", "Saint Petersburg"]

# Subset temperatures using square brackets
print(temperatures[temperatures["city"].isin(cities)])

# Subset temperatures_ind using .loc[]
print(temperatures_ind.loc[cities])

######################################
# Setting Multi-Level indexes

Set the index of temperatures to the "country" and "city" columns, and assign this to temperatures_ind.
Specify two country/city pairs to keep: "Brazil"/"Rio De Janeiro" and "Pakistan"/"Lahore", assigning to rows_to_keep.
Print and subset temperatures_ind for rows_to_keep using .loc[].
    
# Index temperatures by country & city
temperatures_ind = temperatures.set_index(["country", "city"])

# List of tuples: Brazil, Rio De Janeiro & Pakistan, Lahore
rows_to_keep = [("Brazil", "Rio De Janeiro"), ("Pakistan", "Lahore")]

# Subset for rows to keep
print(temperatures_ind.loc[rows_to_keep])

###############################
# Sorting by Index Values

Sort temperatures_ind by the index values.
Sort temperatures_ind by the index values at the "city" level.
Sort temperatures_ind by ascending country then descending city.

# Sort temperatures_ind by index values
print(temperatures_ind.sort_index())

# Sort temperatures_ind by index values at the city level
print(temperatures_ind.sort_index(level="city"))

# Sort temperatures_ind by country then descending city
print(temperatures_ind.sort_index(level=["country", "city"], ascending=[True,False]))

###################################

# SLICING AND SUBSETTING WITH .loc and .iloc

# Slicing Lists

<dataFrame>[<index>:<index>]

# Sort Index()
<newDataFrame> = <dataFrame>.set_index(["<column>", "<column2>"].)sort_index()
print(<newDataFrame)

# Slicing the outer index level
<newDataFrame>.loc["<column>":"<column2>"]

# Slicing the inner index levels badly
<newDataFrame>.loc["<column>":"<column2>"] # returns inefficiently

# Slicing the inner index levels correctly
<newDataFrame>.loc[
    ("<column1Val>", "<column2Val>"):("<column1Val>", "<column2Val>")]

# Slicing Columns
<newDataFrame>.loc[:, "<column>":"<column2>"]

# Slice Twice
<newDataFrame>.loc[
    ("<column1Val>", "<column2Val"):("<column1Val>", "<column2val>"), 
    "<column3>":"<column4>"]

# By Days

<dataFrame> = <dataFrame>.set_index("<date_of_birth>").sort_index()
print(<dataFrame>)

# Slicing By Dates
# Get <dataFrame> with <date_of_birth> between 2014-08-25 and 2016-09-16
<dataFrame>.loc["2014-08-25":"2016-09-16"]

# Slicing by Partial Dates
<dataFrame>.loc["2014":"2016"]

# Subsetting by Row/Column Number
print(<dataFrame>.iloc[2:5, 1:4])
    -two args rows and columns


# PRACTICE

# Slicing Index Values

Sort the index of temperatures_ind.
Use slicing with .loc[] to get these subsets:
from Pakistan to Russia.
from Lahore to Moscow. (This will return nonsense.)
from Pakistan, Lahore to Russia, Moscow.


temperatures_srt = temperatures_ind.sort_index()
print(temperatures_srt.loc["Pakistan":"Russia"])
print(temperatures_srt.loc["Lahore":"Moscow"])
print(temperatures_srt.loc[
    ("Pakistan","Lahore"):("Russia", "Moscow")
    ])


#############################

# Slicing in both directions

Use .loc[] slicing to subset rows from India, Hyderabad to Iraq, Baghdad.
Use .loc[] slicing to subset columns from date to avg_temp_c.
Slice in both directions at once from Hyderabad to Baghdad, and date to avg_temp_c.

print(temperatures_srt.loc[
    ("India", "Hyderabad"):("Iraq", "Baghdad")
])
print(temperatures_srt.loc[:, "date":"avg_temp_c"])
print(temperatures_srt.loc[
    ("India", "Hyderabad"):("Iraq", "Baghdad"),
    "date":"avg_temp_c"
])

# Slicing time series

Use Boolean conditions, not .isin() or .loc[], and the full date "yyyy-mm-dd", to subset temperatures for rows where the date column is in 2010 and 2011 and print the results.
Set the index of temperatures to the date column and sort it.
Use .loc[] to subset temperatures_ind for rows in 2010 and 2011.
Use .loc[] to subset temperatures_ind for rows from August 2010 to February 2011.

temperatures_bool = temperatures[(temperatures["date"] >= "2010-01-01") & (temperatures["date"] <= "2011-12-31")]
print(temperatures_bool)

temperatures_ind = temperatures.set_index("date").sort_index()
print(temperatures_ind.loc["2010":"2011"])
print(temperatures_ind.loc["2010-08":"2011:02"])



# Subsetting by row/column number

Use .iloc[] on temperatures to take subsets.

Get the 23rd row, 2nd column (index positions 22 and 1).
Get the first 5 rows (index positions 0 to 5).
Get all rows, columns 3 and 4 (index positions 2 to 4).
Get the first 5 rows, columns 3 and 4.


temperatures.iloc[22, 1]
temperatures.iloc[0:6]
temperatures.iloc[:, 2:4]
temperatures.iloc[0:6, 2:4]

#####################################


# WORKING WITH PIVOT TABLES


# Pivoting the <dataFrame>

<newDataFrame> = <dataFrame>.pivot_table(
    "<column>", index="<column>", columns="<column>"
)
print(<newDataFrame>)
    -first arg is a value to aggregate
    -index arg is list to groupby and display in rows
    -columns arg lists the columns to groupby and display in columns

# .loc[] + slicing is a power combo
-Pivot Tables are just DFs with SORTED INDEXES.
    -ideal for subsetting pivot tables

<pivotDataFrame>.loc["<columnVal>":"<columnVal2>"]

# the axis argument
-methods for calc summ stats on a DF (ex. mean) have an AXIS ARG
    -default val is "index" (meaning "calc the stats ACROSS rows")
        -mean is calc for EACH color ("ACROSS breeds")

<pivotDataFrame>.mean(axis="index")

# Calculating Summ Stats Across Columns
-To calc a Summ Stat for each row (Across the columns) > you set AXIS to "columns"
    -here the MEAN <column2> is calculated for each <column1> (Across the <column3>)
        -for MOST DFs, setting the AXIS ARG doees NOT make sense; PIVOT TABLES are the SPECIAL CASE

<pivotDataFrame>.mean(axis="columns")

#######

# PRACTICE


# Pivot temp by city and year

Add a year column to temperatures, from the year component of the date column.
Make a pivot table of the avg_temp_c column, with country and city as rows, and year as columns. Assign to temp_by_country_city_vs_year, and look at the result.

temperatures["year"] = temperatures["date"].dt.year

temp_by_country_city_vs_year = temperatures.pivot_table("avg_temp_c", index=["country", "city"], columns="year")
print(temp_by_country_city_vs_year)
###############################

# Subsetting pivot tables
Use .loc[] on temp_by_country_city_vs_year to take subsets.

From Egypt to India.
From Egypt, Cairo to India, Delhi.
From Egypt, Cairo to India, Delhi, and 2005 to 2010.

temp_by_country_city_vs_year.loc[("Egypt", "Cairo"):("India", "Delhi"), ("2005"):("2010")]


# Calculating on a pivot table
Calculate the mean temp for each year, assigning to mean_temp_by_year
Filter mean_temp_by_year for the year that had the highest mean temp
Calculate the mean temp for each city (across columns), assigning to mean_temp_by_city
Filter mean_temp_by_city for the city that had the lowest mean temp


# Get the worldwide mean temp by year
mean_temp_by_year = temp_by_country_city_vs_year.mean(axis="index")

# Filter for the year that had the highest mean temp
print(mean_temp_by_year[mean_temp_by_year == mean_temp_by_year.max()])

# Get the mean temp by city
mean_temp_by_city = temp_by_country_city_vs_year.mean(axis="columns")

# Filter for the city that had the lowest mean temp
print(mean_temp_by_city[mean_temp_by_city == mean_temp_by_city.min()])



















