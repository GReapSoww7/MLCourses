### SHARING VISUALIZATIONS WITH OTHERS


# PREPARING YOUR FIGURES TO SHARE WITH OTHERS

-Changing Plot Style > Choosing a Style

import matplotlib.pyplot as plt

plt.style.use('ggplot')
or
plt.style.use('default')

fig, ax = plt.subplots()
ax.plot(seattle_weather['MONTH'], seattle_weather['MLY-TAVG-NORMAL'])
ax.plot(austin_weather['MONTH'], austin_weather['MLY-TAVG-NORMAL'])
ax.set_xlabel('Time (months)')
ax.set_ylabel('Average temperature (Fahrenheit degrees)')
plt.show()

-available styles
#://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html

-The 'bmh' Style

plt.style.use('bmh')

fig, ax = plt.subplots()
ax.plot(seattle_weather['MONTH'], seattle_weather['MLY-TAVG-NORMAL'])
ax.plot(austin_weather['MONTH'], austin_weather['MLY-TAVG-NORMAL'])
ax.set_xlabel('Time (months)')
ax.set_ylabel('Average temperature (Fahrenheit degrees)')
plt.show()


-Seaborn Styles

plt.style.use('seaborn-colorblind')

-GUIDELINES FOR CHOOSING PLOTTING STYLES

    -Dark backgrounds are usually less visible
    -If color is important, consider choosing colorblind-friendly options
        -'seaborn-colorblind' or 'tableau-colorblind10'
    -If you think that someone will want to print your figure, use less ink
    -if it will be printed in black-and-white, use the 'grayscale' style

# PRACTICE

# Switching Between Styles

Select the 'ggplot' style, create a new Figure called fig, and a new Axes object called ax with plt.subplots.
Select the 'Solarize_Light2' style, create a new Figure called fig, and a new Axes object called ax with plt.subplots.


plt.style.use('ggplot')

# Use the "Solarize_Light2" style and create new Figure/Axes
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
plt.show()

############################

# Saving Your Visualizations

fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation=90)
ax.set_ylabel('Number of medals')

fig.savefig('gold_medals.png[.jpg],[svg]', quality=50)

-Resolution
    -use the dpi=<val> arg in the method

-Size
fig.set_size_inches([5, 3])
####################################

# Automating Figures from Data

-Why Automate?
    -Ease and speed
    -Flexibility
    -Robustness
    -Reproducibility

-How many Different Kinds of Data?
    -Getting Unique Values of a Column


summer_2016_medals['Sport'].unique()

sports = summer_2016_medals['Sport'].unique()
print(sports)


-Bar-chart of Heights for all Sports

fig, ax = plt.subplots()

for sport in sports:
    sports_df = summer_2016_medals[summer_2016_medals['Sport'] == sport]
    ax.bar(sport, sport_df['Height'].mean(),
           yerr=sport_df['Height'].std())
ax.set_ylabel('Height (cm)')
ax.set_xticklabels(sports, rotation=90)
plt.show()

####################

# MATPLOT LIB GALLERY OF VISUALIZATIONS

#://matplotlib.org/gallery.html

# pandas + Matplotlib = Seaborn

seaborn.relplot(x='horsepower', y='mpg', hue='origin', size='weight', sizes=(40, 400), alpha=0.5, palette='muted', height=6, data=mpg)

# Seaborn Example Gallery
#://seaborn.pydata.org/examples/index.html


