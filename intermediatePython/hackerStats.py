### RANDOM NUMBERS

-Simulate the process


# Rand Gen

import numpy as np

np.random.rand() # Pseudo-random numbers

# seed setting

np.random.seed(123)
np.random.rand()

np.random.rand() # ensures "reproducibility"

##########


# Coin Toss

game.py

import numpy as np
np.random.seed(123)
coin = np.random.randint(0,2) # Randomly gen 1 or 2
print(coin)
if coin == 0:
    print("heads")
else:
    print("tails")

# Determine next move

# NumPy is imported, seed is set
import numpy as np
np.random.seed(123)

# Starting step
step = 50

# Roll the dice
dice = np.random.randint(1,7)

# Finish the control construct
if dice <= 2 :
    step = step - 1
elif dice <= 5  :
    step = step + 1
else :
    step = step + np.random.randint(1,7)

# Print out dice and step
print(dice)
print(step)


# Random Walk
-gradually build a list with a FOR Loop


BASIC HEADS OR TAILS
-10 iterations (set by 'range')
headtails.py

import numpy as np
np.random.seed(123)
outcomes = []
for x in range(10) :
    coin = np.random.randint(0,2)
    if coin == 0 :
        outcomes.append("heads")
    else :
        outcomes.append("tails")

print(outcomes)


# RANDOM WALK HEADS/TAILS

import numpy as np
np.random.seed(123)
tails = [0]
for x in range(10) :
    coin = np.random.randint(0,2)
    tails.append(tails[x] + coin)
print(tails)


# VISUALIZE RANDOM WALK

# NumPy is imported, seed is set

# Initialization
random_walk = [0]

for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)

    if dice <= 2:
        step = max(0, step - 1) # max() takes 2 args, prevents the var from going below x in max(x, y) while iterating
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    random_walk.append(step)

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Plot random_walk
plt.plot(random_walk)

# Show the plot
plt.show()


# DISTRIBUTION of RANDOM_WALKS

-once we know the distribution we can calculate the odds/chances/probability

# 100 RUNS
distribution.py

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)
final_tails = []
for x in range(10000) : # increase the iterations in order to get a larger sample size
    tails = [0]
    for x in range(10) :
        coin = np.random.randint(0,2)
        tails.append(tails[x] + coin)
    final_tails.append(tails[-1])
#print(final_tails)
plt.hist(final_tails, bins = 10)
plt.show()


# VISUALIZE WALKS


# numpy and matplotlib imported, seed set.

# initialize and populate all_walks
all_walks = []
for i in range(5) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    all_walks.append(random_walk)

# Convert all_walks to NumPy array: np_aw
np_aw = np.array(all_walks)

# Plot np_aw and show
plt.plot(np_aw)
plt.show()

# Clear the figure
plt.clf()

# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()

### IMPLEMENT FAILURE/REGRESSION PROBABILITY

Change the range() function so that the simulation is performed 20 times.
Finish the if condition so that step is set to 0 if a random float is less or equal to 0.005. Use np.random.rand().

# numpy and matplotlib imported, seed set

# clear the plot so it doesn't get cluttered if you run this many times
plt.clf()

# Simulate random walk 20 times
all_walks = []
for i in range(20) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)

        # Implement clumsiness
        if np.random.rand() <= 0.005 : # if RANDOM FLOAT is less or equal to 0.005
            step = 0

        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
plt.show()



































