### WHY TUNE YOUR MODEL ###


# Untuned Model Example

import pandas as pd
import xgboost as xgb
import numpy as np
housing_data = pd.read_csv('ames_housing_trimmed_processed.csv')
X, y = housing_data[housing_data.columns.tolist()[:-1]], housing_data[housing_data.columns.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X, label=y)
untuned_params = {'objective':'reg:squarederror'}
untuned_cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=untuned_params, nfold=4, metrics='rmse', as_pandas=True, seed=123)
print(f'Untuned rmse: %f' %((untuned_cv_results_rmse['test-rmse-mean']).tail(1)))

Untuned rmse: 34624.229980

# Tuned Model Example

import pandas as pd
import xgboost as xgb
import numpy as np
housing_data = pd.read_csv('ames_housing_trimmed_processed.csv')
X, y = housing_data[housing_data.columns.tolist()[:-1]], housing_data[housing_data.columns.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X, label=y)
tuned_params = {'objective':'reg:squarederror', 'colsample_bytree':0.3, 'learning_rate':0.1, 'max_depth':5}
tuned_cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=tuned_params, nfold=4, num_boost_round=200, metrics='rmse', as_pandas=True, seed=123)
print('Tuned rmse: %f' %((tuned_cv_results_rmse['test-rmse-mean']).tail(1)))

Tuned rmse: 29812.683594

#########################################

# PRACTICE

# Tuning the Number of Boosting Rounds

# Create DMatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)
# Create param dictionary
params = {'objective':'reg:squarederror', 'max_depth':3}
# Create list of number of boosting rounds
num_rounds = [5, 10, 15]
# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []
# Iterate over num_rounds and build one mode per num_boost_round param
for curr_num_rounds in num_rounds:
    # perform CV: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics='rmse', as_pandas=True, seed=123)

    # append final round rmse
    final_rmse_per_round.append(cv_results['test-rmse-mean'].tail().values[-1])

# print the resultant DF
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses, columns=['num _boosting_rounds', 'rmse']))

   num_boosting_rounds       rmse
0                    5  50903.300
1                   10  34774.194
2                   15  32895.099
#############################################

# Automated Boosting Round Selection Using early_stopping

Now, instead of attempting to cherry pick the best possible number of boosting rounds, you can very easily have XGBoost automatically select the number of boosting rounds for you within xgb.cv(). This is done using a technique called early stopping.

Early stopping works by testing the XGBoost model after every boosting round against a hold-out dataset and stopping the creation of additional boosting rounds (thereby finishing training of the model early) if the hold-out metric ("rmse" in our case) does not improve for a given number of rounds. Here you will use the early_stopping_rounds parameter in xgb.cv() with a large possible number of boosting rounds (50). Bear in mind that if the holdout metric continuously improves up through when num_boost_rounds is reached, then early stopping does not occur.

Here, the DMatrix and parameter dictionary have been created for you. Your task is to use cross-validation with early stopping. Go for it!


Perform 3-fold cross-validation with early stopping and "rmse" as your metric. Use 10 early stopping rounds and 50 boosting rounds. Specify a seed of 123 and make sure the output is a pandas DataFrame. Remember to specify the other parameters such as dtrain, params, and metrics.
Print cv_results.

cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, metrics='rmse', num_boost_round=50, nfold=3, early_stopping_rounds=10, as_pandas=True, seed=123)

print(cv_results)

<script.py> output:
        train-rmse-mean  train-rmse-std  test-rmse-mean  test-rmse-std
    0        141871.635         403.633      142640.654        705.560
    1        103057.034          73.768      104907.665        111.117
    2         75975.968         253.727       79262.057        563.767
    3         57420.531         521.658       61620.138       1087.693
    4         44552.956         544.170       50437.561       1846.447
    5         35763.949         681.797       43035.660       2034.471
    6         29861.464         769.571       38600.881       2169.797
    7         25994.675         756.521       36071.818       2109.795
    8         23306.836         759.238       34383.186       1934.547
    9         21459.770         745.625       33509.140       1887.375
    10        20148.721         749.612       32916.807       1850.893
    11        19215.383         641.387       32197.833       1734.457
    12        18627.389         716.256       31770.852       1802.154
    13        17960.695         557.043       31482.782       1779.124
    14        17559.737         631.413       31389.990       1892.320
    15        17205.713         590.172       31302.883       1955.166
    16        16876.572         703.632       31234.059       1880.706
    17        16597.662         703.677       31318.348       1828.861
    18        16330.461         607.274       31323.635       1775.910
    19        16005.972         520.471       31204.135       1739.076
    20        15814.301         518.605       31089.864       1756.022
    21        15493.406         505.616       31047.998       1624.673
    22        15270.734         502.019       31056.916       1668.044
    23        15086.382         503.913       31024.984       1548.985
    24        14917.608         486.206       30983.685       1663.131
    25        14709.589         449.668       30989.477       1686.667
    26        14457.286         376.788       30952.114       1613.172
    27        14185.567         383.103       31066.901       1648.535
    28        13934.067         473.466       31095.642       1709.226
    29        13749.645         473.671       31103.887       1778.880
    30        13549.837         454.899       30976.085       1744.515
    31        13413.485         399.603       30938.469       1746.053
    32        13275.916         415.409       30931.000       1772.469
    33        13085.878         493.793       30929.057       1765.541
    34        12947.181         517.790       30890.629       1786.510
    35        12846.027         547.733       30884.493       1769.729
    36        12702.379         505.523       30833.542       1691.002
    37        12532.244         508.298       30856.688       1771.445
    38        12384.055         536.225       30818.017       1782.785
    39        12198.444         545.166       30839.393       1847.327
    40        12054.584         508.842       30776.965       1912.780
    41        11897.037         477.178       30794.703       1919.675
    42        11756.222         502.992       30780.956       1906.820
    43        11618.847         519.837       30783.755       1951.260
    44        11484.080         578.428       30776.731       1953.448
    45        11356.553         565.369       30758.544       1947.455
    46        11193.558         552.299       30729.972       1985.699
    47        11071.316         604.090       30732.663       1966.997
    48        10950.778         574.863       30712.241       1957.751
    49        10824.865         576.666       30720.854       1950.511
##############################################################################