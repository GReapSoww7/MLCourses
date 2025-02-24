# NumPy Boolean Ops

logical_and()
logical_or()
logical_not()

ex.

bmi[np.logical_and(bmi > 21, bmi < 22)]


# Filtering pandas DataFrame

-Step 1: Get column
<dataFrame>["area"]


-Step 2: Do comparison on area column
<dataFrame>["<column>"] > <value>

returns True or False


-Step 3: Use result to select countries (Subset DF)

is_huge = <dataFrame>["<column>"] > <value>
-is a pandas Series

<dataFrame>[is_huge]

# one-liner
<dataFrame>[<dataFrame>['<column>']]

-ex.  # Convert code to a one-liner
sel = cars[cars['drives_right']]

-Boolean Ops:
    -pandas is built on NumPy

import numpy as np

np.logical_and(<dataFrame>["<column>"] > <value>, <dataFrame>["<column>"] < 10)




# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Import numpy, you'll need this
import numpy as np

# Create medium: observations with cars_per_cap between 100 and 500
cpc = cars['cars_per_cap']
between = np.logical_and(cpc > 100, cpc < 500)
medium = cars[between]


# Print medium
print(medium)






