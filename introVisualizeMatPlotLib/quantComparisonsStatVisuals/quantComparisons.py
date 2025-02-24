### QUANTITIVE COMPARISONS AND STATISTICAL VISUALIZATIONS
-comparing parts of data

# Bar Chart

-visualize data

medals = pd.read_csv('medals_by_country_2016.csv', index_col=0)
fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.bar(medals.index, medals['Silver'], bottom=medals['Gold'],
       label='Silver')
ax.bar(medals.index, medals['Bronze'],
       bottom=medals['Gold'] + medals['Silver'],
       label='Bronze')
ax.set_xticklabels(medals.index, rotation=90)
ax.set_ylabel('Number of medals')
ax.legend()
plt.show()


# PRACTICE

# Bar Chart

Call the ax.bar method to plot the "Gold" column as a function of the country.
Use the ax.set_xticklabels to set the x-axis tick labels to be the country names.
In the call to ax.set_xticklabels rotate the x-axis tick labels by 90 degrees by using the rotation key-word argument.
Set the y-axis label to "Number of medals".

fig, ax = plt.subplots()

ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation=90)
ax.set_ylabel('Number of medals')
#########################

# Stacked Bar Chart
Call the ax.bar method to add the "Gold" medals. Call it with the label set to "Gold".
Call the ax.bar method to stack "Silver" bars on top of that, using the bottom key-word argument so the bottom of the bars will be on top of the gold medal bars, and label to add the label "Silver".
Use ax.bar to add "Bronze" bars on top of that, using the bottom key-word and label it as "Bronze".


# Add bars for "Gold" with the label "Gold"
ax.bar(medals.index, medals['Gold'], label='Gold')

# Stack bars for "Silver" on top with label "Silver"
ax.bar(medals.index, medals['Silver'], bottom=medals['Gold'], label='Silver')

# Stack bars for "Bronze" on top of that with label "Bronze"
ax.bar(medals.index, medals['Bronze'], bottom=medals['Gold'] + medals['Silver'], label='Bronze')

# Display the legend
ax.legend()

plt.show()

###############################

# Histograms

fig, ax = plt.subplots()
ax.bar('Rowing', mens_rowing['Height'].mean())
ax.bar('Gymnastics', mens_gymnastics['Height'].mean())
ax.set_ylabel('Height (cm)')
plt.show()

-NOW WE ARE GOING TO DO A HISTOGRAM

fig, ax = plt.subplots()
ax.hist(mens_rowing['Height'], label='Rowing', bins=5, histtype='step')
    -or set BIN BOUNDARIES bins=[150, 160, 170, 180, 190, 200, 210]
ax.hist(mens_gymnastics['Height'], label='Gymnastics', bins=5, histtype='step')
ax.set_xlabel('Height (cm)')
ax.set_ylabel('# of observations')
ax.legend()
plt.show()


# PRACTICE

# Creating Histograms

Use the ax.hist method to add a histogram of the "Weight" column from the mens_rowing DataFrame.
Use ax.hist to add a histogram of "Weight" for the mens_gymnastics DataFrame.
Set the x-axis label to "Weight (kg)" and the y-axis label to "# of observations".

fig, ax = plt.subplots()
ax.hist(mens_rowing['Weight'])
ax.hist(mens_gymnastics['Weight'])
ax.set_xlabel('Weight (kg)')
ax.set_ylabel('# of observations')
plt.show()
##########################

# "STEP" histogram

Use the hist method to display a histogram of the "Weight" column from the mens_rowing DataFrame, label this as "Rowing".
Use hist to display a histogram of the "Weight" column from the mens_gymnastics DataFrame, and label this as "Gymnastics".
For both histograms, use the histtype argument to visualize the data using the 'step' type and set the number of bins to use to 5.
Add a legend to the figure, before it is displayed.


fig, ax = plt.subplots()

ax.hist(mens_rowing['Weight'], label='Rowing', bins=5, histtype='step')
ax.hist(mens_gymnastics['Weight'], label='Gymnastics', bins=5, histtype='step')
ax.set_xlabel('Weight (kg)')
ax.set_ylabel('# of observations')
ax.legend()
plt.show()
###########################

# Statistical Plotting

-Adding error bars to bar charts

fig, ax = plt.subplots()

ax.bar('Rowing',
       mens_rowing['Height'].mean(),
       yerr=mens_rowing['Height'].std())

ax.bar('Gymnastics',
       mens_gymnastics['Height'].mean(),
       yerr=mens_gymnastics['Height'].std())

ax.set_ylabel('Height (cm)')
plt.show()

-Adding error bars to plots

fig, ax = plt.subplots()

ax.errorbar(<dataFrame>['MONTH'],
            <dataFrame>['MLY-TAVG-NORMAL'],
            yerr=<dataFrame>['MLY-TAVG-STDDEV'])

ax.errorbar(<dataFrame2>['MONTH'],
            <dataFrame2>['MLY-TAVG-NORMAL'],
            yerr=<dataFrame2>['MLY-TAVG-STDDEV'])

ax.set_ylabel('Temperature (Fahrenheit)')
plt.show()

################################

-Adding boxplots

fig, ax = plt.subplots()

ax.boxplot([mens_rowing['Height'],
            mens_gymnastics['Height']])
ax.set_xticklabels(['Rowing', 'Gymnastics'])
ax.set_ylabel('Height (cm)')
plt.show()

############################

# PRACTICE

# Adding Error-Bars to a Bar Chart

Add a bar with size equal to the mean of the "Height" column in the mens_rowing DataFrame and an error-bar of its standard deviation.
Add another bar for the mean of the "Height" column in mens_gymnastics with an error-bar of its standard deviation.
Add a label to the the y-axis: "Height (cm)".

fig, ax = plt.subplots()
ax.bar('Rowing', mens_rowing['Height'].mean(), yerr=mens_rowing['Height'].std())
ax.bar('Gymnastics', mens_gymnastics['Height'].mean(), yerr=mens_gymnastics['Height'].std())
ax.set_ylabel('Height (cm)')
plt.show()

#################

# Addint error-bars to plot

Use the ax.errorbar method to add the Seattle data: the "MONTH" column as x values, the "MLY-TAVG-NORMAL" as y values and "MLY-TAVG-STDDEV" as yerr values.
Add the Austin data: the "MONTH" column as x values, the "MLY-TAVG-NORMAL" as y values and "MLY-TAVG-STDDEV" as yerr values.
Set the y-axis label as "Temperature (Fahrenheit)".

fig, ax = plot.subplots()

ax.errorbar(seattle_weather['MONTH'], seattle_weather['MLY-TAVG-NORMAL'], yerr=seattle_weather['MLY-TAVG-STDDEV'])
ax.errorbar(austin_weather['MONTH'], austin_weather['MLY-TAVG-NORMAL'], yerr=austin_weather['MLY-TAVG-STDDEV'])
ax.set_ylabel('Temperature (Fahrenheit)')
plt.show()
#########################

# Creating Boxplots

Create a boxplot that contains the "Height" column for mens_rowing on the left and mens_gymnastics on the right.
Add x-axis tick labels: "Rowing" and "Gymnastics".
Add a y-axis label: "Height (cm)".

fig, ax = plt.subplots()

# Add a boxplot for the "Height" column in the DataFrames
ax.boxplot([mens_rowing['Height'], mens_gymnastics['Height']])

# Add x-axis tick labels:
ax.set_xticklabels(['Rowing', 'Gymnastics'])

# Add a y-axis label
ax.set_ylabel('Height (cm)')

plt.show()
############################

# Scatter Plots

-Introducing Scatter Plots

fig, ax = plt.subplots()
ax.scatter(climate_change['co2'], climate_change['relative_temp'])
ax.set_xlabel('CO2 (ppm)')
ax.set_ylabel('Relative temperature (Celsius)')
plt.show()

-customize Scatter Plots

eighties = climate_change['1980-01-01':'1989-12-31']
nineties = climate_change['1990-01-01':'1999-12-31']
fig, ax = plt.subplots()
ax.scatter(eighties['co2'], eighties['relative_temp'],
           color='r', label='eighties')
ax.scatter(nineties['co2'], nineties['relative_temp'],
           color='b', label='nineties')
ax.legend()

ax.set_xlabel('CO2 (ppm)')
ax.set_ylabel('Relative temperature (Celsius)')
plt.show()
########################

-Encoding a 3rd Var by Color

fig, ax = plt.subplots()
ax.scatter(climate_change['co2'], climate_change['relative_temp'],
c=climate_change.index)
ax.set_xlabel('CO2 (ppm)')
ax.set_ylabel('Relative temperature (Celsius)')
plt.show()
########################

# Simple Scatter Plot

Using the ax.scatter method, add the data to the plot: "co2" on the x-axis and "relative_temp" on the y-axis.
Set the x-axis label to "CO2 (ppm)".
Set the y-axis label to "Relative temperature (C)"

fig, ax = plt.subplots()
ax.scatter(climate_change['co2'], climate_change['relative_temp'], c=climate_change.index)
ax.set_xlabel('CO2 (ppm)')
ax.set_ylabel('Relative temperature (C)')
plt.show()

#######################################

# Encoding Time by Color

Using the ax.scatter method add a scatter plot of the "co2" column (x-axis) against the "relative_temp" column.
Use the c key-word argument to pass in the index of the DataFrame as input to color each point according to its date.
Set the x-axis label to "CO2 (ppm)" and the y-axis label to "Relative temperature (C)".

fig, ax = plt.subplots()
ax.scatter(climate_change['co2'], climate_change['relative_temp'], c=climate_change.index)
ax.set_xlabel('CO2 (ppm)')
ax.set_ylabel('Relative temperature (C)')
############################################