### LIMITS OF GRID SEARCH AND RANDOM SEARCH ###


# Grid Search
-as num of hyperparams grows = more time it takes GridSearch
    -i.e. num of models we must build with EVERY additional NEW hyperparam grows QUICKLY

# Random Search
-we can specify num of iterations
    -time it takes to finish won't explode as we add hyperparams
      -as we add new hyperparams to search over the SPACE to explore becomes massive
        -we hope a Random hyperparam Config is optimal; random jumping through SPACE for best Config takes more time