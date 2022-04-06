#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 03:13:18 2022

@author: SawmonAbo
"""

#####################################################################################
#####################################Modules#########################################
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file (e.g. pd.read_csv)
from pandas import read_csv

from itertools import combinations

from numpy.random import seed

import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV  #, GridSearchCV

import xgboost as xgb

from statsmodels.api import add_constant
import os

#####################################################################################
######################## Data/Model Reading and Setup ###############################

os.chdir('/Users/SawmonAbo/Desktop/housePricePred/csv')
dataset = read_csv('/Users/SawmonAbo/Desktop/housePricePred/csv/sales.csv')
dataset = dataset.drop(['pid','tid'],axis=1)

seed(1234)
dataset = dataset.sample(len(dataset))



#####################################################################################
######################### Data Visualization ########################################

#Check whether there is any null values
dataset.info()


dataset.head()
dataset.describe()

# Data Visualization using Seaborn
plt.figure(figsize=(10,6))
sns.plotting_context('notebook',font_scale=1.2)
g = sns.pairplot(dataset[['price', 'home_size', 'parcel_size', 'beds', 'age',\
                          'pool', 'year', 'cbd_dist', 'x_coord', 'y_coord']] \
                 ,hue='beds',height=2)
g.set(xticklabels=[])


# Data Visualization using Heat Map
plt.figure(figsize=(15,10))
columns =['price', 'home_size', 'parcel_size', 'beds', 'age', 'pool', 'year', \
          'cbd_dist', 'x_coord', 'y_coord']
sns.heatmap(dataset[columns].corr(),annot=True)



###################################################################################
######################### Feature Engineering #####################################

os.chdir('/Users/SawmonAbo/Desktop/housePricePred/DataFrame')

# Split training data into features (x) and labels (Y)
x = pd.read_pickle('features_wZipcode.pkl') 
Y = x[['price']]
x = x.drop(['price'],axis=1)



# New variable creations/interactions:
x['land_to_building'] = x['parcel_size'] / x['home_size']
x['cbdDist_to_landBuilding'] = x['cbd_dist'] / x['land_to_building']

# Why does this increase from 1.933082e+00 -> 9.923470e+00 
x['homesize_to_beds'] = x['home_size'] / x['beds']

# Variable Transformations:
# Ex]
# dataset['cbd_dist_sq'] = np.power(dataset['cbd_dist'], 2)

# Dropping repeated variables for multicollinearaity
# Ex]
# dataset = dataset.drop(['home_size'],axis=1)


# Reformat and Assign new variables to dataset.
# x = x[['home_size', 'parcel_size', 'beds', 'age', 'pool', 'year', \
#              'cbd_dist', 'E_stateplane', 'N_stateplane', 'lattitude', 'longitutde',\
#                  'zipcode', 'land_to_building', 'cbdDist_to_landBuilding', 'homesize_to_beds']]


x = x[['home_size', 'parcel_size', 'beds', 'age', 'pool', 'year', \
             'cbd_dist', 'E_stateplane', 'N_stateplane', 'lattitude', 'longitutde',\
                 'zipcode', 'land_to_building', 'cbdDist_to_landBuilding']]


# Variance inflation factor
def calc_vif(X):

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = x.columns
    vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    return(vif)

calc_vif(x)



# Data Visualization using Seaborn for new dataset features
plt.figure(figsize=(10,6))
sns.plotting_context('notebook',font_scale=1.2)
g = sns.pairplot(x[['home_size', 'parcel_size', 'beds', 'age', 'pool', 'year', \
             'cbd_dist', 'E_stateplane', 'N_stateplane', 'lattitude', \
                 'longitutde', 'zipcode', 'land_to_building', \
                     'cbdDist_to_landBuilding']],hue='beds',height=2)


# Data Visualization using Heat Map for new dataset features
plt.figure(figsize=(15,10))
columns =  ['home_size', 'parcel_size', 'beds', 'age', 'pool', 'year', \
             'cbd_dist', 'E_stateplane', 'N_stateplane', 'lattitude', \
                 'longitutde', 'zipcode', 'land_to_building', 'cbdDist_to_landBuilding']

sns.heatmap(dataset[columns].corr(),annot=True)




#####################################################################################
####################### Liner Regression - OLS/Lasso/Ridge ##########################

# Reformat and Assign new variables to dataset.

x_scaled = preprocessing.scale(x)
x_scaled = pd.DataFrame(x_scaled, columns=('home_size', 'parcel_size', 'beds',\
            'age', 'pool', 'year', 'cbd_dist', 'E_stateplane', 'N_stateplane',\
                'lattitude', 'longitutde', 'zipcode', 'land_to_building',\
                    'cbdDist_to_landBuilding'))

# scaled_Dataset = np.array([x_scaled.home_size, x_scaled.parcel_size, x_scaled.beds, x_scaled.pool, \
#             x_scaled.year, x_scaled.cbd_dist, x_scaled.E_stateplane, x_scaled.N_stateplane, x_scaled.lattitude, \
#                 x_scaled.longitutde, x_scaled.zipcode, x_scaled.land_to_building, \
#                     x_scaled.cbdDist_to_landBuilding])

scaled_Dataset = np.transpose(x_scaled)
scaled_Dataset = add_constant(x_scaled)


x_combos = []
for n in range(1,7):
    combos = combinations(['home_size', 'parcel_size', 'beds', 'age', 'pool', 'year', \
                 'cbd_dist', 'E_stateplane', 'N_stateplane', 'lattitude', 'longitutde', \
                     'zipcode', 'land_to_building', 'cbdDist_to_landBuilding'], n)
    x_combos.extend(combos)

# Create dictionary to store the variable combinations and associated average
# (over folds) test mse values
ols_mse = {}
lasso_mse = {}
ridge_mse = {}

# Initalize the k-fold cross validation.
k = KFold(n_splits = 10, shuffle=True, random_state=10)

# Runtime is around 30mins
for n in range(0, len(x_combos)):
    combo_list = list(x_combos[n])
    x2 = scaled_Dataset[combo_list]

    poly = PolynomialFeatures(3)
    poly_x = poly.fit_transform(x2)
    
        
    ols_cv_scores = cross_validate(LinearRegression(), poly_x, Y, cv=k, scoring=('neg_mean_squared_error'))
    # lasso_cv_scores = cross_validate(Lasso(alpha=5.00), poly_x, Y, cv=k, scoring=('neg_mean_squared_error'))
    ridge_cv_scores = cross_validate(Ridge(alpha=5.75), poly_x, Y, cv=k, scoring=('neg_mean_squared_error'))

    ols_mse[str(combo_list)] = np.mean(ols_cv_scores['test_score'])
    # lasso_mse[str(combo_list)] = np.mean(lasso_cv_scores['test_score'])
    ridge_mse[str(combo_list)] = np.mean(ridge_cv_scores['test_score'])
    
    
        

print("Outcomes from the Best OLS model:")
ols_min_mse = abs(max(ols_mse.values()))
print("Minimum Average OLS Test Mse:", ols_min_mse.round(3))
for possibles, r in ols_mse.items():
    if r == -ols_min_mse:
        print("The OLS combination of variables: ", possibles)

# Poly(3)
# Outcomes from the Best OLS model:
# Minimum Average OLS Test Mse: 7312.977
# The OLS combination of variables:  ['home_size', 'year', 'cbd_dist', 'E_stateplane', 'longitutde', 'cbdDist_to_landBuilding']

print("Outcomes from the Best Lasso model:")
lasso_min_mse = abs(max(lasso_mse.values()))
print("Minimum Average Lasso Test Mse:", lasso_min_mse.round(3))
for possibles, r in lasso_mse.items():
    if r == -lasso_min_mse:
        print("The Lasso combination of variables: ", possibles)


# Outcomes from the Best Lasso model:
# Minimum Average Lasso Test Mse: 9191.795
# The Lasso combination of variables:  ['home_size', 'year', 'cbd_dist', 'E_stateplane', 'lattitude', 'cbdDist_to_landBuilding']



print("Outcomes from the Best Ridge model:")
ridge_min_mse = abs(max(ridge_mse.values()))
print("Minimum Average Ridge Test Mse:", ridge_min_mse.round(3))
for possibles, r in ridge_mse.items():
    if r == -ridge_min_mse:
        print("The Ridge combination of variables: ", possibles)

# Poly(3)
# Outcomes from the Best Ridge model:
# Minimum Average Ridge Test Mse: 7647.505
# The Ridge combination of variables:  ['home_size', 'year', 'cbd_dist', 'E_stateplane', 'N_stateplane', 'cbdDist_to_landBuilding']





#####################################################################################
############################## Gradient Boosting XR #################################

parameters = \
{
"max_depth": [4, 5, 6],
"min_child_weight": [1],
"learning_rate": [.04, .075, .1],
"n_estimators": [1000],
"booster": ["gbtree"],
"gamma": [0, 5, 15, 25, 50],
"subsample": [0.3, 0.6, 0.8],
"colsample_bytree": [0.5, 0.7, 0.8],
"colsample_bylevel": [0.5, 0.7,],
"reg_alpha": [1, 10, 33],
"reg_lambda": [1, 3, 10],
}


# data_dmatrix = xgb.DMatrix(data=x,label=Y)
# dmat_train = xgb.DMatrix(X_train, y_train, feature_names=data_dmatrix.feature_names)
# dmat_test = xgb.DMatrix(X_test, y_test, feature_names=data_dmatrix.feature_names)


# GridSearch Approach
# XGB = xgb.XGBRegressor(random_state=10, max_features="sqrt")
# regr = GridSearchCV(estimator=XGB, param_grid=parameters, scoring='neg_mean_squared_error', cv=10)




# RandomizedSearch Approach (Faster runtime and better predictions made)
k = KFold(n_splits = 10, shuffle=True, random_state=10)
regr = RandomizedSearchCV(xgb.XGBRegressor(random_state=10), parameters, cv=k, \
                          n_jobs=4, scoring="neg_mean_squared_error", random_state=10, n_iter=10)

    


# Fit trained data on best model/parameters from the RandomizedSearchCV
regr.fit(x, Y)
xgb_model = xgb.XGBRegressor(**regr.best_params_)



X_train, X_test, y_train, y_test = train_test_split(x, Y, train_size=0.9, random_state=1234)

xgb_model.fit(X_train, y_train)
y_pred = pd.DataFrame(xgb_model.predict(X_test))
xgb_mse = mean_squared_error(y_test, y_pred)
score = xgb_model.score(X_test, y_test)


print("XGBoost Score=", score)
print("XGBoost Train/Test MSE =", xgb_mse)
print("Coefficient of determination R^2 (Train/Test)", r2_score(y_test, y_pred))



# XGBoost Score= 0.8637660849359128
# XGBoost Train/Test MSE = 2279.020234815399
# Coefficient of determination R^2 (Train/Test) 0.8637660849359128

################################################################################
############################ Validation ########################################

# Re-estimate model on whole dataset.
xgb_model.fit(x, Y)

# validation_set = read_csv('/Users/SawmonAbo/Desktop/UCF/eco_4443/midterm/validation_set-6.csv')
os.chdir('/Users/SawmonAbo/Desktop/housePricePred/DataFrame')

validation_set = pd.read_pickle('features_validation.pkl') 

seed(1234)
validation_set = validation_set.sample(len(validation_set))



val_Y = validation_set[['price']]
val_x = validation_set.drop(['price'],axis=1)



# New variable creations/interactions:
val_x['land_to_building'] = val_x['parcel_size'] / val_x['home_size']
val_x['cbdDist_to_landBuilding'] = val_x['cbd_dist'] / val_x['land_to_building']
# val_x['homesize_to_beds'] = val_x['home_size'] / val_x['beds']


# Reformat and Assign new variables to dataset.
val_x = val_x[['home_size', 'parcel_size', 'beds', 'age', 'pool', 'year', \
             'cbd_dist', 'E_stateplane', 'N_stateplane', 'lattitude', 'longitutde',\
                 'zipcode', 'land_to_building', 'cbdDist_to_landBuilding']]


# Obtaining prediction results on validation set.
valY_pred = pd.DataFrame(xgb_model.predict(val_x))

# Printing result measures:
final_mse = sum((val_Y.price - valY_pred)**2) / len(val_Y)
val_score = xgb_model.score(val_x, val_Y)





print("XGBoost Validation MSE =", final_mse)
print("Coefficient of determination R^2 (Validation)", r2_score(val_Y, valY_pred))
print("XGBoost Score=", val_score)





# XGBoost Validation MSE Self-Calculated = 2499.506514331529
# Coefficient of determination R^2 (Validation) 0.8401775751366231
# XGBoost Score= 0.8401775751366231

#####################################################################################
######################## Validation Visualizations  ############################


# Feature Importance Bar Greaph
xgb.plot_importance(xgb_model)
pyplot.show()


# Numeric Chart
pd.DataFrame({ "Actuals":val_Y[:10].values.ravel(), "Prediction":xgb_model.predict(val_x)[:10]})


# Y.values.ravel()
# .values will give the values in a numpy array (shape: (n,1))
# .ravel will convert that array shape to (n, ) (i.e. flatten it)


# Line Graph
x_Y = range(len(val_Y))
x_ax = range(len(valY_pred))
plt.plot(x_Y , val_Y.values.ravel(), label="original", color='red', marker='o', markersize=5, linewidth=2)
plt.plot(x_ax, valY_pred.values.ravel(), label="predicted", marker='+', linewidth=1)
plt.title("House Price Original and Predicted Data")
# plt.rcParams['figure.figsize'] = [100, 100]
plt.legend()
plt.show()


# Scatter Plot
# plt.figure(figsize=(5,5))
plt.scatter(val_Y.values.ravel(), valY_pred.values.ravel(), c='crimson')
p1 = range(len(val_Y))
p2 = range(len(valY_pred))
plt.plot([p1, p2], [p1, p2], 'g-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

# XGBoost Decision Tree
plt.figure(dpi=1800)
clf = xgb_model.fit(val_x, val_Y)
xgb.plot_tree(clf, filled=True)
plt.title("Decision Tree On All Features", fontsize=50)
plt.rcParams['figure.figsize'] = [100, 100]
plt.show()

