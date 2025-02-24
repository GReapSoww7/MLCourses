### CREATING DATAFRAMES


# Dictionaries

<dictionary> = {
    "key1": val1,
    "key2": val2,
    "key3": val3,
}

-access the vals of keys:

<dict>["<key[i]"]

# Create a DF from a list of dictionaries/a dictionary of lists

-List of Dictionaries - by row

<dict> = [
    {"<key1>": "<val1>", "<key2>": "<val2>", "<key3>": "<val3>", "<key4>": "<val4>", "<key5>": "<val5>"}
    {"<key1>": "<val1>", "<key2>": "<val2>", "<key3>": "<val3>", "<key4>": "<val4>", "<key5>": "<val5>"}
]

-convert into a dataFrame

<dataFrame> = pd.DataFrame(<dict>)
print(<dataFrame>)

-Dictionary of lists - by column
    -key = column name
    -val = list of column values

<dict> = {
    "<key1>": ["<val1>", "<val2>"]
    "<key2>": ["<val1>", "<val2>"]
    "<key3>": ["<val1>", "<val2>"]
    "<key4>": ["<val1>", "<val2>"]
} 
<dataFrame> = pd.DataFrame(<dict>)

print(<dataFrame>)

# PRACTICE

# List of Dictionaries

Create a list of dictionaries with the new data called avocados_list
Convert the list into a DataFrame called avocados_2019
print

# Create a list of dictionaries with new data
avocados_list = [
    {"date": "2019-11-03", "small_sold": 10376832, "large_sold": 7835071},
    {"date": "2019-11-10", "small_sold": 10717154, "large_sold": 8561348},
]

# Convert list into DataFrame
avocados_2019 = pd.DataFrame(avocados_list)

# Print the new DataFrame
print(avocados_2019)

#############################

# Dictionary of Lists

Create a dictionary of lists with the new data called avocados_dict
Convert the dictionary to a DataFrame called avocados_2019
Print

# Create a dictionary of lists with new data
avocados_dict = {
  "date": ["2019-11-17", "2019-12-01"],
  "small_sold": [10859987, 9291631],
  "large_sold": [7674135, 6238096]
}

# Convert dictionary into DataFrame
avocados_2019 = pd.DataFrame(avocados_dict)

# Print the new DataFrame
print(avocados_2019)
########################################