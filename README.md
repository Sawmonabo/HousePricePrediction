# HousePricePrediction

[housePredictionsReport.pdf](https://github.com/Sawmonabo/HousePricePrediction/files/8433441/housePredictionsReport.pdf)


**HOUSE PRICE PREDICTION USING MACHINE LEARNING TECHNIQUES**

Prepared by: Sawmon Abossedgh



## Contents

- [Introduction](#introduction)
- [Data Preparation and Description Using SQLite](#data-preparation-and-description-using-sqlite)
- [Reading and Visualizing the Dataset on Python](#reading-and-visualizing-the-dataset-on-python)
- [Feature Engineering](#feature-engineering)
- [Train and Test Split with Cross Validation](#train-and-test-split-with-cross-validation)
- [Regression Model Implementations](#regression-model-implementations)
  - [Liner Regression with OLS](#a-ordinary-least-squares)
  - [Lasso](#b-lasso)
  - [Ridge](#c-ridge)
  - [Gradient Boosting XR](#d-gradient-boosting-xr)
- [Validation](#validation-of-best-model)
- [Conclusion](#conclusion)
- [Citations](#citations)



## Introduction

<p>The purpose of this project is to research and apply various machine learning regression algorithms to undertake different models with training and testing process to identify a hedonic property price model that can be used to predict residential property prices in Orlando as a function of the physical and locational attributes of the properties. The data is first preprocessed and is then analyzed to summarize the main characteristics of the variables such as their correlation or any observable patterns. These variables are then fed into the different machine learning regression algorithms with the data being split into training and testing sets to estimate a model to predict house prices accurately. The best model will be identified by the measure of MSE (mean squared error) and its accuracy to the validation set. 


## Data Preparation and Description Using SQLite

<p>I was given files with the names prices.csv, characteristics.csv, and locations.csv. The files contain information on 10,000 residential property sales in Orange County, Florida. The first file contains a property identifier (pid), the most recent sales price of the property measured in thousands of dollars (price), and the year that the sale occurred (year). The second file contains a property identifier (pid), property-type code (tid), and numerous property characteristics. The property characteristics include the square feet of heated area (home\_size), the square feet of the parcel of land (parcel\_size), the number of bedrooms (beds), the age of the home in years (age), and an indicator variable (pool) that takes a value of 1 if the property has a pool and a value of 0 if it does not. The third file contains a property-type code (tid), a variable indicating the distance in meters to the central business district in downtown Orlando (cbd\_dist), and coordinates identifying the geographic location of each property (x\_coord and y\_coord). 

<p>In SQLite, I created a database called Sales.db. Next, I made three tables within the database called Prices, Characteristics, and Locations and populated the tables with the data contained in the prices.csv, characteristics.csv, and locations.csv files. Once all data is uploaded into SQLite, I then joined the tables and exported a single file called sales.csv that contains the sale prices and dates from the Prices table, the property characteristics from the Characteristics table, and the location variables from the Locations table. 


## Reading and Visualizing the Dataset on Python

<p>Once I contained a single csv file with the data I required, I read the data into Python as a dataframe. To ensure replicability of the results and to be able to compare the results later, I set the sample seed at 1234. Then I familiarized myself starting by generating summary statistics for our dataset using its function attributes:

- dataset.info() - visualized our datatypes from each of the variables from our dataset. Containing 9999 entries, 10 columns, and data types of integers and floats.
- dataset.describe() – gave us descriptive statistics summary on each variable from our dataset containing statistical measures of the count, mean, std, min, max, and percentiles of 25-50-75%.

<p>With the visual packages python provides I can use histograms, a correlation matrix, and scatter plots to see correlation between variables. More specifically I wanted to see how our feature/predictor variables are correlated with our target variable, price. For instance, in Figure 1 below, I can see how all the predictor variables are shaped with respect to price. In Figure 2 below, I can see the correlation represented as a value between (+/-) 1, where positive one shows the highest positive correlation, negative one shows the highest negative correlation, and zero representing no correlation.



| ![Original Variable Scatter Plots](https://user-images.githubusercontent.com/77422313/162085570-20bb0e01-cfc7-4256-91f8-d5a25c26c2bf.png) | 
|:--:| 
| *Figure 1 - Scatter Plot* |

| ![Original Variable Correlation Heat Map](https://user-images.githubusercontent.com/77422313/162085583-d7b8c6dc-03b3-4d15-903b-92c30b7564be.png) |
|:--:| 
| *Figure 2 - Correlation Heat Map* |


<p>From our heat map in Figure 2, I can see there are many correlated variables. I notice that variables ‘home\_size’ to ‘beds’ have the highest positive correlation, and ‘age’ to ‘cbd\_dist’ have the highest negative correlation which from a practical standpoint makes sense. Most importantly, I need to see the correlation between the predictor variables and our target, price. Looking at the first column of our heat map in Figure 2, I see ‘home\_size’ being the highest positive correlation.

## Feature Engineering


<p>Before I started making new variables I reverse engineered the variables ‘(x/y)\_coord’ and learned that they are Florida East state plane measures in US Survey units. This isn’t a common measure for location, so I converted them into longitude and latitude. From there, I was able to use a function using our latitude and longitude that gave us data on each home. I was then able to obtain the address and zip code of each individual house from the dataset. There were 29 rows that the function didn’t return a zip code for, so I removed those columns completely to stay consistent with the rest of the data rows. This is all shown in python script – house\_features.py.

<p>Now, after our previous section I learned about some features with high positive correlation I can use to engineer a new variable to help predict price.  The variables are shown below:



```js
{
  dataset ['land\_to\_building'] = dataset ['parcel\_size'] / dataset ['home\_size'] 
  dataset ['cbdDist\_to\_landBuilding'] = dataset ['cbd\_dist'] / dataset ['land\_to\_building']
  dataset ['homesize\_to\_beds'] = dataset ['home\_size'] / dataset ['beds']
},
```

<p>I used ratio calculated variables strictly to avoid the multicollinearity problem when used in mainly linear regression models. A pre-defined method function in python called Variance Inflation Factor can also be used to determine the strength of the correlation between various independent models. If our variables from our features are less than a VIF score of 10, it can be used.


## Train and Test Split with Cross Validation

<p>For supervised machine learning problems, there are some tools used to prevent/minimize overfitting. For example, with linear regression, I usually fit the model on a training set to make predictions for the test set (the data that wasn’t trained). To further break this down, I split the data into two subsets: training and testing data to help make predictions on the test set. I do this using the “Scikit-Learn Library” and I use a 90/10 split meaning 90% of the data is used to train to make predictions for our test set, the 10% section. The only problem from only using the train-test split is if it wasn’t random and our subsets have certain data that the other subsets don’t. This could lead to overfitting, so I solve this problem by including k-folds cross validation. As shown in figure 3 below, it basically splits the data into k different folds and trains on k – 1 of those folds holding the last fold for test data. It then averages the model against all the folds and then creates a final-best model. 



| ![image](https://user-images.githubusercontent.com/77422313/162085770-a4c00218-600b-4770-b28b-1bb83ddab09f.png) | 
|:--:| 
| *Figure 3 - K-Folds CV* |

  
## Regression Model Implementations


<p>For only our linear regression models I used three different methods including scaling, a combinations function, and a polynomial features function. I scaled the data to standardize the variables for polynomial or interaction terms used, to avoid multicollinearity. A combination’s function was implemented to try all possible combinations of feature variables to find the best and lowest MSE score. To find the best model for regression I used sklearn’s “Polynomial Features” function which creates an interaction between the feature variables as well as raises each variable to the selected power, in my case 3. I set the Polynomial function to raise the variables to the 2nd with respect to the combinations function.

```js
{
  x_scaled = preprocessing.scale(x)
  x_scaled = pd.DataFrame(x_scaled, columns=('home_size', 'parcel_size', 'beds',\
              'age', 'pool', 'year', 'cbd_dist', 'E_stateplane', 'N_stateplane',\
                  'lattitude', 'longitutde', 'zipcode', 'land_to_building',\
                      'cbdDist_to_landBuilding'))


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
      lasso_cv_scores = cross_validate(Lasso(alpha=5.00), poly_x, Y, cv=k, scoring=('neg_mean_squared_error'))
      ridge_cv_scores = cross_validate(Ridge(alpha=5.75), poly_x, Y, cv=k, scoring=('neg_mean_squared_error'))

      ols_mse[str(combo_list)] = np.mean(ols_cv_scores['test_score'])
      lasso_mse[str(combo_list)] = np.mean(lasso_cv_scores['test_score'])
      ridge_mse[str(combo_list)] = np.mean(ridge_cv_scores['test_score'])
},
```



### A. Ordinary Least Squares

<p>From the scikit-learn website, I can verify OLS is used as the regression method. It also states the function definition as follows: Linear Regression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

<p>Using “﻿neg\_mean\_squared\_error” as our scoring, the model with the lowest MSE consisted of the variables ﻿'home\_size', 'year', 'cbd\_dist', 'E\_stateplane', 'longitutde', 'cbdDist\_to\_landBuilding'. The test mean squared error for our best linear regression model was ﻿7312.977.


### B. Lasso

<p>Least Absolute Shrinkage and Selection Operator (LASSO) is a machine learning regression algorithm that is quite like linear regression except it does have the capabilities to shrink the coefficients to zero to avoid overfitting. From the scikit-learn website, Lasso’s ability to regularize and shrink the coefficients allows it to be used for variable selection which in turn improves the prediction accuracy. To control for shrinkage applied to the coefficients to get a more parsimonious model, Lasso uses a tuning parameter, λ.  If λ=0, it is equal to the linear regression model, and as λ increases, the coefficients shrink and the ones that are equal zero are eliminated. 

<p>Using λ = 0.15 and “﻿neg\_mean\_squared\_error” as our scoring, I observe that the best model for Lasso is worse than our linear regression model, with an MSE of ﻿9191.795 and variables ﻿'home\_size', 'year', 'cbd\_dist', 'E\_stateplane', 'lattitude', 'cbdDist\_to\_landBuilding'.  The lambda value was set close to zero which is why the two models look similar, showing a lack of shrinkage possibly. 


### C. Ridge

<p>Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of the coefficients. From the scikit-learn website, the ridge coefficients minimize a penalized residual sum of squares, and the complexity parameter (α >= 0) controls the amount of shrinkage. The larger the value of the complexity parameter, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.

<p>Using α = 10 and “﻿neg\_mean\_squared\_error” as our scoring, I observe that the best model for Ridge is slightly worse than our linear regression model, with an MSE of ﻿7647.505 and variables ﻿'home\_size', 'year', 'cbd\_dist', 'E\_stateplane', 'N\_stateplane', 'cbdDist\_to\_landBuilding'


## D. Gradient Boosting XR

<p>XGradient Boosting is a machine learning regression algorithm that is a method of creating an ensemble of individual models and is used for regression and classification purposes. In order to understand this method, I read an article by Arthur Mello called XGBoost: theory and practice. An understanding of decision trees is necessary. Decision trees are simply a way of visualizing outcomes in a branching structure. A decision tree consists of a root node. It represents the population that is being analyzed and is further branched out into the various features known as the decision nodes, which split into further nodes. Additionally, the leaf node is a sub-node that does not split into further nodes. XGradient boosting is a technique that creates an ensemble of decision trees, with each decision tree improving on the performance of the previous one. It is known to combine the weak learners into stronger ones, meaning that each new tree that is built improves on the error of the previous one. XGradient Boosting also uses hyperparameter tuning which chooses a set of optimal parameters for learning algorithms and provides us with high performance models. 

Parameters used and their meanings: (https://xgboost.readthedocs.io/en/stable/parameter.html)

|Parameters|Description|
| :- | :- |
|max\_depth [default=6]|Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. |
|<p>subsample [default=1]</p><p></p>|Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting|
|min\_child\_weight [default=1]|<p>Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min\_child\_weight, then the building process will give up further partitioning. The larger min\_child\_weight is, the more conservative the algorithm will be.</p><p></p>|
|<p>colsample\_bytree, </p><p>colsample\_bylevel, </p><p>colsample\_bynode [default=1]</p><p></p>|<p>All colsample\_by\* parameters have a range of (0, 1], the default value of 1, and specify the fraction of columns to be subsampled.</p><p></p><p>colsample\_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.</p><p></p><p>colsample\_bylevel is the subsample ratio of columns for each level</p>|
|lambda [default=1, alias: reg\_lambda]|L2 regularization term on weights. Increasing this value will make model more conservative.|
|alpha [default=0, alias: reg\_alpha]|L1 regularization term on weights. Increasing this value will make model more conservative.|
|learning\_rate [default=0.3]|Step size shrinkage used in update to prevents overfitting. After each boosting step, I can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.|
|gamma [default=0, alias: min\_split\_loss]|Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.|


```js
{
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

},
```




I also used a randomized search cross validation pre-built function from the sklearn module. It tries random combinations of a range of values (I defined the number of iterations to 10). It is good at testing a wide range of values and normally it reaches a very good combination very fast (recommended for big datasets with parameter tuning). After feeding the variables into the regression model I observe that the best model after training/testing/tuning on the dataset for XGradient Boosting, was an MSE of ﻿2279.02 with an R2 of ﻿86.38% (With respect to the entire dataset). The high value of R2 shows that XGradient Boosting is suitable for large datasets as it creates a large ensemble of trees, with each tree correcting on the error of the previous, thus making the model accurate. 




## Validation of Best Model

Given a validation set I was able to re-estimate and fit the best model on the original data set and obtain predictions for the validation set.

||**Linear - OLS Regression**|**Lasso Regression**|**Ridge Regression**|**XGBoost**|
| :-: | :-: | :-: | :-: | :-: |
|**Test Set**|<p>MSE = ﻿ 7312.977</p><p></p>|<p>MSE = ﻿9191.795</p><p></p>|<p>MSE = ﻿7647.505</p><p></p>|<p>MSE = 2279.02</p><p>R2 = ﻿86.38%</p>|
|**Validation Set**||||<p>MSE = ﻿2499.51</p><p>R2 = 84.02%</p>|

Table 1 - MSE and R2 for the best model

## Conclusion

After using three regression algorithms to predict the house price, the model obtained by XGBoost proved to have been the best model. With a large dataset, XGBoost was suitable as it was able to create an ensemble of decision trees using the various features to best predict the house prices. To further show XGBoost’s performance, I included some visualizations below that show feature importance, predicted vs. actual data plots, and the decision tree on features.

![image](https://user-images.githubusercontent.com/77422313/162086076-c0913f06-e129-4e64-b9a5-da4d8782d7c6.png)

![image](https://user-images.githubusercontent.com/77422313/162086096-41426627-a1b3-47d1-a6e9-023efa66d237.png)

![image](https://user-images.githubusercontent.com/77422313/162086124-f29f96d7-025f-47c2-bdc5-b026cd6d04ed.png)



## Citations

Lee, Wei-Meng. “Statistics in Python - Collinearity and Multicollinearity.” *Medium*, Towards Data Science, 11 Dec. 2021, <https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f#:~:text=VIF%20allows%20you%20to%20determine,linear%20dependencies%20with%20other%20predictors>. 

“Train/Test Split and Cross Validation in Python.” *Towards Data Science*, https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6b61beca4b6. 

Editor, Minitab Blog. “When Is It Crucial to Standardize the Variables in a Regression Model?” *Minitab Blog*, <https://blog.minitab.com/en/adventures-in-statistics-2/when-is-it-crucial-to-standardize-the-variables-in-a-regression-model#:~:text=You%20should%20standardize%20the%20variables,produce%20excessive%20amounts%20of%20multicollinearity>. 

“Sklearn.linear\_model.Linearregression.” *Scikit*, https://scikit-learn.org/stable/modules/generated/sklearn.linear\_model.LinearRegression.html#:~:text=Ordinary%20least%20squares%20Linear%20Regression,predicted%20by%20the%20linear%20approximation.&text=Whether%20to%20calculate%20the%20intercept%20for%20this%20model. 

“XGBoost Parameters¶.” *XGBoost Parameters - Xgboost 1.5.2 Documentation*, https://xgboost.readthedocs.io/en/stable/parameter.html. 

Mello, Arthur. “XGBoost: Theory and Practice.” *Medium*, Towards Data Science, 17 Aug. 2020, https://towardsdatascience.com/xgboost-theory-and-practice-fb8912930ad6. 

