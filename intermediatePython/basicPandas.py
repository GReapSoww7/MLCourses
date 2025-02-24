### Dict to DataFrame using pandas
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr = [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Import pandas as pd
import pandas as pd

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict = {'country':names, 'drives_right':dr, 'cars_per_capt':cpc}

# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

# Print cars
print(cars)



### CSV to DataFrame

#import pandas as pd
#cars = pd.read_csv('/path/to/cars.csv', index_col = 0)
#print(cars)

### Pandas, Part 2

# Index and Select Data
    # Square brackets
    # Adv methods:
        # loc and iloc

# Column Access[]

# type(<dataFrame>["<key>"])
# pandas.core.series.Series
# 1D labelled array

# Select column but keep data in DF
# double square brackets

# <dataFrame>[["<key>"]]
# check type
# type(<dataFrame>[["<key>"]])
# pandas.core.frame.DataFrame

# can select multiple columns
# <dataFrame>[["<key>", "<key>"]]
    # diff perspective > putting a list with column labels inside another set a square brackets > end up with a SUB DataFrame

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out country column as Pandas Series
print(cars["country"])

# Print out country column as Pandas DataFrame
print(cars[["country"]])

# Print out DataFrame with country and drives_right columns
print(cars[["country", "drives_right"]])


# can select rows as well
    # specify a slice
# <dataFrame>[<row>:<row>]

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out first 3 observations
print(cars[0:3])

# Print out fourth, fifth and sixth observation
print(cars[3:6])



# Square brackets are limited functionality (good for column access)
# slice for ROW access

# we want similar to 2D NumPy arrays
    # my_array[rows, columns]

# Extend tools

# loc (label-based)
# iloc (integer position-based)

# <dataFrame>.loc["<str>"]
    # row as pandas series

# <dataFrame>.loc[["<str>"]]
    # as a DataFrame

# Row & Column loc
# <dataFrame>.loc[["<row>", "<row>", "<row>"], ["<column>", "<column>"]]

# ALL ROWS, specific COLUMNS
# <dataFrame>.loc[:, ["<column>", "<column>"]]

###########

# subset dataframes with index use iloc

# Row Access iloc
# <dataFrame>.iloc[[<index>]]

# <dataFrame>.iloc[[<index>, <index>, <index>]]

# Row & Column
# <dataFrame>.iloc[[<row>, <row>, <row>], [<column>,<column>]]

# ALL ROWS, specific columns
# <dataFrame>.iloc[:, [0,1]]















