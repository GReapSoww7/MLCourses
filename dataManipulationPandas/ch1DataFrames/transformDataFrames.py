### TRANSFORMING DATAFRAMES

# pandas is build on NumPy and Matplotlib

-NumPy
    -provides multidimensional array objects

-Matplotlib
    -data visualization


-Rectangular Data
    -represented as a DataFrame in pandas


# Exploring a DataFrame:

print(<dataFrame>.head())

print(<dataFrame>.info())

print(<dataFrame>.shape)

print(<dataFrame>.describe())

print(<dataFrame>.values)

print(<dataFrame>.columns)

<dataFrame>.index


###################

# SORTING AND SUBSETTING

-Sorting
<dataFrame>.sort_values("<columnName>")
##########

-descending order
<dataFrame>.sort_values("<columnName>", ascending=False)
###########

-multiple vars
<dataFrame>.sort_values("<columnName>", "<columnName2>")

<dataFrame>.sort_values(["<columnName>", "<columnName2>"], ascending=[True,False])
#############
#############
-Subsetting columns

<dataFrame>["<columnName>"]
##################

-Subsetting multiple columns

<dataFrame>[["<columnName1>", "columnName2>"]]

cols_to_subset = ["<column1>", "<column2>"]
<dataFrame>[cols_to_subset]
###############

-Subsetting rows

<dataFrame>["<column>"] > <val>
    -returns boolean value (True, False)

<dataFrame>[<dataFrame>["<column>"] > 50 ]
##############

-Subsetting by text

<dataFrame>[<dataFrame>["<column>"] == "<column VALUE>"]
##############

-Subsetting by dates

<dataFrame>[<dataFrame>["<column>"] < "<date>"]
###############

-Subsetting based on multiple conditions

is_<condition> = <dataFrame>["<column>"] == "<column VALUE>"
is_<condition> = <dataFrame>["<column>"] == "<column VALUE>"
<dataFrame>[is_<condition1> & is_<condition2>]

-one liner ^^^
<dataFrame>[ (<dataFrame>["<column>"] == "<column VALUE>") & (<dataFrame>["<column2>"] == "<column2 VALUE>") ]

###############
-Subsetting using .isin()

is_<condition1>_or_<condition2> = <dataFrame>["<column1>"].isin(["<column VALUE>", "<column VALUE2>"])
<dataFrame>[is_<condition1>_or_<condition2>]

-ex.
# The Mojave Desert states
canu = ["California", "Arizona", "Nevada", "Utah"]

# Filter for rows in the Mojave Desert states
mojave_homelessness = homelessness[homelessness["state"].isin(canu)]

# See the result
print(mojave_homelessness)

############################


# NEW COLUMNS

-Adding a New Column

<dataFrame>["<newColumn"] = <dataFrame>["<originalColumn>"] / 100
print(<dataFrame>)


# BMI example

<dataFrame>["bmi"] = <dataFrame>["<column1>"] / <dataFrame>["column2"] ** 2
print(<dataFrame>.head())

# Multiple Manipulations

bmi_lt_100 = <dataFrame>[<dataFrame>["bmi"] < 100]
bmi_lt_100_height = bmi_lt_100.sort_values("<column1>", ascending=False)
bmi_lt_100_height[["<column2>", "<column1>", "bmi"]]


# Exercises

Add a new column to homelessness, named total, containing the sum of the individuals and family_members columns.
Add another column to homelessness, named p_homeless, containing the proportion of the total homeless population to the total population in each state state_pop.


# Add total col as sum of individuals and family_members
homelessness["total"] = homelessness["individuals"] + homelessness["family_members"]

# Add p_homeless col as proportion of total homeless population to the state population
homelessness["p_homeless"] = homelessness["total"] / homelessness["state_pop"]

# See the result
print(homelessness)

# Exercise 2

Add a column to homelessness, indiv_per_10k, containing the number of homeless individuals per ten thousand people in each state, using state_pop for state population.
Subset rows where indiv_per_10k is higher than 20, assigning to high_homelessness.
Sort high_homelessness by descending indiv_per_10k, assigning to high_homelessness_srt.
Select only the state and indiv_per_10k columns of high_homelessness_srt and save as result. Look at the result.


# Create indiv_per_10k col as homeless individuals per 10k state pop
homelessness["indiv_per_10k"] = 10000 * homelessness["individuals"] / homelessness["state_pop"] 

# Subset rows for indiv_per_10k greater than 20
high_homelessness = homelessness[homelessness["indiv_per_10k"] > 20]

# Sort high_homelessness by descending indiv_per_10k
high_homelessness_srt = high_homelessness.sort_values("indiv_per_10k", ascending=False)

# From high_homelessness_srt, select the state and indiv_per_10k cols
result = high_homelessness_srt[["state", "indiv_per_10k"]]

# See the result
print(result)

####################

















