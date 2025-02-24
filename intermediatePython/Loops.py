### Loop Data Structures Part 1

# Dictionary (requires a method)

for var in seq :
    expression


world = { "<key>":<value>,
         "<key>":<value>,
         "<key>":<value> }

for k, v in world.items() :
    print(k + " -- " + str(v))


# NumPy Arrays (use a Function)

import numpy as np

np_height = np.array([<val>, <val>, <val>, <val>, <val>])
np_weight = np.array([<val>, <val>, <val>, <val>, <val>])

bmi = np_weight / np_height ** 2

for val in bmi :
    print(val)

# 2D NumPy Array

np_height = np.array([<val>, <val>, <val>, <val>, <val>])
np_weight = np.array([<val>, <val>, <val>, <val>, <val>])

combo = np.array([np_height, np_weight])

for val in combo :
    print(val)


# get every element of an array

np_height = np.array([<val>, <val>, <val>, <val>, <val>])
np_weight = np.array([<val>, <val>, <val>, <val>, <val>])

combo = np.array([np_height, np_weight])
for val in np.nditer(combo) :
    print(val)


### Loop Data Structures Part 2 (PANDAS)


import pandas as pd

<dataFrame> = pd.read_csv("<file>.csv", index_col = 0)

# iterrows
-generates label of row and the data of the row


<dataFrame> = pd.read_csv("/path/to/<file>.csv", index_col = 0)
for x, y in <dataFrame>.iterrows() :
    print(x)
    print(y)

# can use subsetting (selective printing)

print(x + ": " + y["<column>"])



# Add column

<dataFrame> = pd.read_csv("/path/to/<file>.csv", index_col = 0)

for lab, row in <dataFrame>.iterrows() :
    # - Creating Serires on every iteration (INEFFICIENT)
    <dataFrame>.loc[lab, "<column Name>"] = len(row["<column>"])
print(<dataFrame>)


# a BETTER APPROACH
# APPLY

import pandas as pd

<dataFrame> = pd.read_csv("<file>.csv", index_col = 0)
<dataFrame>["<newColumn>"] = <dataFrame>["originalColumn"].apply(len)
print(<dataFrame>)

## Loop over DataFrame(2)

Using the iterators lab and row, adapt the code in the for loop such that the first iteration prints out "US: 809", the second iteration "AUS: 731", and so on.
The output should be in the form "country: cars_per_cap". Make sure to print out this exact string (with the correct spacing).
You can use str() to convert your integer data to a string so that you can print it in conjunction with the country label.


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
print(cars)
print()
print()
# Adapt for loop
for lab, row in cars.iterrows() :
    print(lab + ": " + str(row['cars_per_cap']))
