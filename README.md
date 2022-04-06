# HousePricePrediction

[Research Report .docx](https://github.com/Sawmonabo/HousePricePrediction/files/8430904/Research.Report.docx)


**HOUSE PRICE PREDICTION USING MACHINE LEARNING TECHNIQUES**
Prepared by: Sawmon Abossedgh



**Contents**

1. Introduction
1. Data Preparation and Description Using SQLite
1. Reading and Visualizing the Dataset on Python 
1. Feature Engineering
1. Train and Test Split w/Cross Validation
1. Regression Model Implementations
   * Liner Regression – OLS
   * Lasso/Ridge
   * Gradient Boosting XR
1. Validation
1. Visualizations / Conclusions
1. Citations






**Introduction**

The purpose of this project is to research and apply the various machine learning regression algorithms to undertake different models with training and testing process to identify a hedonic property price model that can be used to predict residential property prices in Orlando as a function of the physical and locational attributes of the properties. The data is first preprocessed and is then analyzed to summarize the main characteristics of the variables such as their correlation or any observable patterns. These variables are then fed into the different machine learning regression algorithms with the data being split into training and testing sets to estimate a model to predict house prices accurately. The best model will be identified by the measure of MSE (mean squared error) and its accuracy to the validation set. 


**Data Preparation and Description Using SQLite**

We are given files with the names prices.csv, characteristics.csv, and locations.csv. The files contain information on 10,000 residential property sales in Orange County, Florida. The first file contains a property identifier (pid), the most recent sales price of the property measured in thousands of dollars (price), and the year that the sale occurred (year). The second file contains a property identifier (pid), property-type code (tid), and numerous property characteristics. The property characteristics include the square feet of heated area (home\_size), the square feet of the parcel of land (parcel\_size), the number of bedrooms (beds), the age of the home in years (age), and an indicator variable (pool) that takes a value of 1 if the property has a pool and a value of 0 if it does not. The third file contains a property-type code (tid), a variable indicating the distance in meters to the central business district in downtown Orlando (cbd\_dist), and coordinates identifying the geographic location of each property (x\_coord and y\_coord). 

In SQLite, I created a database called Sales.db. Next, I made three tables within the database called Prices, Characteristics, and Locations and populated the tables with the data contained in the prices.csv, characteristics.csv, and locations.csv files. Once all data is uploaded into SQLite, I then joined the tables and exported a single file called sales.csv that contains the sale prices and dates from the Prices table, the property characteristics from the Characteristics table, and the location variables from the Locations table. 


**Reading and Visualizing the Dataset on Python.**

Once I contained a single csv file with the data I required, I read the data into Python as a dataframe. To ensure replicability of the results and to be able to compare the results later, I set the sample seed at 1234. Then I familiarized myself starting by generating summary statistics for our dataset using its function attributes:

- dataset.info() - visualized our datatypes from each of the variables from our dataset. Containing 9999 entries, 10 columns, and data types of integers and floats.
- ﻿dataset.describe() – gave us descriptive statistics summary on each variable from our dataset containing statistical measures of the count, mean, std, min, max, and percentiles of 25-50-75%.

With the visual packages python provides we can use histograms, a correlation matrix, and scatter plots to see correlation between variables. More specifically I wanted to see how our feature/predictor variables are correlated with our target variable, price. For instance, in Figure 1 below, we can see how all the predictor variables are shaped with respect to price. In Figure 2 below, we can see the correlation represented as a value between (+/-) 1, where positive one shows the highest positive correlation, negative one shows the highest negative correlation, and zero representing no correlation.









*Figure 2 - Correlation Heat Map*

From our heat map in Figure 2, we can see there are many correlated variables. We notice that variables ‘home\_size’ to ‘beds’ have the highest positive correlation, and ‘age’ to ‘cbd\_dist’ have the highest negative correlation which from a practical standpoint makes sense. Most importantly, we need to see the correlation between the predictor variables and our target, price. Looking at the first column of our heat map in Figure 2, we see ‘home\_size’ being the highest positive correlation.


**Feature Engineering**

Before I started making new variables I reverse engineered the variables ‘(x/y)\_coord’ and learned that they are Florida East state plane measures in US Survey units. This isn’t a common measure for location, so we converted them into longitude and latitude. From there, we were able to use a function using our latitude and longitude that gave us data on each home. We were then able to obtain the address and zip code of each individual house from the dataset. There were 29 rows that the function didn’t return a zip code for, so we removed those columns completely to stay consistent with the rest of the data rows. This is all shown in python script – house\_features.py.

Now, after our previous section we learned about some features with high positive correlation we can use to engineer a new variable to help predict price.  The variables are shown below:

1. ﻿ dataset ['land\_to\_building'] = dataset ['parcel\_size'] / dataset ['home\_size'] 
1. dataset ['cbdDist\_to\_landBuilding'] = dataset ['cbd\_dist'] / dataset ['land\_to\_building']
1. dataset ['homesize\_to\_beds'] = dataset ['home\_size'] / dataset ['beds']

We used ratio calculated variables strictly to avoid the multicollinearity problem when used in mainly linear regression models. A pre-defined method function in python called Variance Inflation Factor can also be used to determine the strength of the correlation between various independent models. If our variables from our features are less than a VIF score of 10, it can be used.


**Train and Test Split w/ K-Folds Cross Validation**

For supervised machine learning problems, there are some tools used to prevent/minimize overfitting. For example, with linear regression, we usually fit the model on a training set to make predictions for the test set (the data that wasn’t trained). To further break this down, I split the data into two subsets: training and testing data to help make predictions on the test set. I do this using the “Scikit-Learn Library” and I use a 90/10 split meaning 90% of the data is used to train to make predictions for our test set, the 10% section. The only problem from only using the train-test split is if it wasn’t random and our subsets have certain data that the other subsets don’t. This could lead to overfitting, so I solve this problem by including k-folds cross validation. As shown in figure 3 below, it basically splits the data into k different folds and trains on k – 1 of those folds holding the last fold for test data. It then averages the model against all the folds and then creates a final-best model. 


*Figure 3 - K-Folds CV*


**Regression Model Implementations**

For only our linear regression models I used three different methods including scaling, a combinations function, and a polynomial features function. I scaled the data to standardize the variables for polynomial or interaction terms used, to avoid multicollinearity. A combination’s function was implemented to try all possible combinations of feature variables to find the best and lowest MSE score. To find the best model for regression I used sklearn’s “Polynomial Features” function which creates an interaction between the feature variables as well as raises each variable to the selected power, in my case 3. We set our Polynomial function to raise the variables to the 2nd with respect to the combinations function.


*A.*    *Ordinary Least Squares – Linear*

From the scikit-learn website, we can verify OLS is used as the regression method. It also states the function definition as follows: Linear Regression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

Using “﻿neg\_mean\_squared\_error” as our scoring, the model with the lowest MSE consisted of the variables ﻿'home\_size', 'year', 'cbd\_dist', 'E\_stateplane', 'longitutde', 'cbdDist\_to\_landBuilding'. The test mean squared error for our best linear regression model was ﻿7312.977.


*B.*	*Lasso* 

Least Absolute Shrinkage and Selection Operator (LASSO) is a machine learning regression algorithm that is quite like linear regression except it does have the capabilities to shrink the coefficients to zero to avoid overfitting. From the scikit-learn website, Lasso’s ability to regularize and shrink the coefficients allows it to be used for variable selection which in turn improves the prediction accuracy. To control for shrinkage applied to the coefficients to get a more parsimonious model, Lasso uses a tuning parameter, λ.  If λ=0, it is equal to the linear regression model, and as λ increases, the coefficients shrink and the ones that are equal zero are eliminated. 

Using λ = 0.15 and “﻿neg\_mean\_squared\_error” as our scoring, we observe that the best model for Lasso is worse than our linear regression model, with an MSE of ﻿9191.795 and variables ﻿'home\_size', 'year', 'cbd\_dist', 'E\_stateplane', 'lattitude', 'cbdDist\_to\_landBuilding'.  The lambda value was set close to zero which is why the two models look similar, showing a lack of shrinkage possibly. 


*C.*	*Ridge* 

Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of the coefficients. From the scikit-learn website, the ridge coefficients minimize a penalized residual sum of squares, and the complexity parameter (α >= 0) controls the amount of shrinkage. The larger the value of the complexity parameter, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.

Using α = 10 and “﻿neg\_mean\_squared\_error” as our scoring, we observe that the best model for Ridge is slightly worse than our linear regression model, with an MSE of ﻿7647.505 and variables ﻿'home\_size', 'year', 'cbd\_dist', 'E\_stateplane', 'N\_stateplane', 'cbdDist\_to\_landBuilding'


*D.*    *XGradient Boosting*

XGradient Boosting is a machine learning regression algorithm that is a method of creating an ensemble of individual models and is used for regression and classification purposes. In order to understand this method, I read an article by Arthur Mello called XGBoost: theory and practice. An understanding of decision trees is necessary. Decision trees are simply a way of visualizing outcomes in a branching structure. A decision tree consists of a root node, it represents the population that is being analyzed and is further branched out into the various features known as the decision nodes, which split into further nodes. Additionally, the leaf node is a sub-node that does not split into further nodes. XGradient boosting is a technique that creates an ensemble of decision trees, with each decision tree improving on the performance of the previous one. It is known to combine the weak learners into stronger ones, meaning that each new tree that is built improves on the error of the previous one. XGradient Boosting also uses hyperparameter tuning which chooses a set of optimal parameters for learning algorithms and provides us with high performance models. 

Parameters used and their meanings: (https://xgboost.readthedocs.io/en/stable/parameter.html)

|Parameters|Description|
| :- | :- |
|max\_depth [default=6]|Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. |
|<p>subsample [default=1]</p><p></p>|Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting|
|min\_child\_weight [default=1]|<p>Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min\_child\_weight, then the building process will give up further partitioning. The larger min\_child\_weight is, the more conservative the algorithm will be.</p><p></p>|
|<p>colsample\_bytree, </p><p>colsample\_bylevel, </p><p>colsample\_bynode [default=1]</p><p></p>|<p>All colsample\_by\* parameters have a range of (0, 1], the default value of 1, and specify the fraction of columns to be subsampled.</p><p></p><p>colsample\_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.</p><p></p><p>colsample\_bylevel is the subsample ratio of columns for each level</p>|
|lambda [default=1, alias: reg\_lambda]|L2 regularization term on weights. Increasing this value will make model more conservative.|
|alpha [default=0, alias: reg\_alpha]|L1 regularization term on weights. Increasing this value will make model more conservative.|
|learning\_rate [default=0.3]|Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.|
|gamma [default=0, alias: min\_split\_loss]|Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.|


We also used a randomized search cross validation pre-built function from the sklearn module. It tries random combinations of a range of values (we defined the number of iterations to 10). It is good at testing a wide range of values and normally it reaches a very good combination very fast (recommended for big datasets with parameter tuning). After feeding the variables into the regression model we observe that the best model after training/testing/tuning on the dataset for XGradient Boosting, was an MSE of ﻿2279.02 with an R2 of 0 ﻿86.38% (With respect to the entire dataset). The high value of R2 shows that XGradient Boosting is suitable for large datasets as it creates a large ensemble of trees, with each tree correcting on the error of the previous, thus making the model accurate. 




**Validation of Models**

Given a validation set we were able to re-estimate and fit our models on the original data set and obtain predictions for the validation set.

||**Linear - OLS Regression**|**Lasso Regression**|**Ridge Regression**|**XGBoost**|
| :-: | :-: | :-: | :-: | :-: |
|**Test Set**|<p>MSE = ﻿ 7312.977</p><p></p>|<p>MSE = ﻿9191.795</p><p></p>|<p>MSE = ﻿7647.505</p><p></p>|<p>MSE = 2279.02</p><p>R2 = ﻿86.38%</p>|
|**Validation Set**||||<p>MSE = ﻿2499.51</p><p>R2 = 84.02%</p>|

*Table 1 - MSE and* R2* for the best model

**Conclusion**

After using three regression algorithms to predict the house price, the model obtained by XGBoost proved to have been the best model. With a large dataset, XGBoost was suitable as it was able to create an ensemble of decision trees using the various features to best predict the house prices. To further show XGBoost’s performance, I included some visualizations below that show feature importance, predicted vs. actual data plots, and the decision tree on features.





















**Citations**

Lee, Wei-Meng. “Statistics in Python - Collinearity and Multicollinearity.” *Medium*, Towards Data Science, 11 Dec. 2021, <https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f#:~:text=VIF%20allows%20you%20to%20determine,linear%20dependencies%20with%20other%20predictors>. 

“Train/Test Split and Cross Validation in Python.” *Towards Data Science*, https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6b61beca4b6. 

Editor, Minitab Blog. “When Is It Crucial to Standardize the Variables in a Regression Model?” *Minitab Blog*, <https://blog.minitab.com/en/adventures-in-statistics-2/when-is-it-crucial-to-standardize-the-variables-in-a-regression-model#:~:text=You%20should%20standardize%20the%20variables,produce%20excessive%20amounts%20of%20multicollinearity>. 

“Sklearn.linear\_model.Linearregression.” *Scikit*, https://scikit-learn.org/stable/modules/generated/sklearn.linear\_model.LinearRegression.html#:~:text=Ordinary%20least%20squares%20Linear%20Regression,predicted%20by%20the%20linear%20approximation.&text=Whether%20to%20calculate%20the%20intercept%20for%20this%20model. 

“XGBoost Parameters¶.” *XGBoost Parameters - Xgboost 1.5.2 Documentation*, https://xgboost.readthedocs.io/en/stable/parameter.html. 

Mello, Arthur. “XGBoost: Theory and Practice.” *Medium*, Towards Data Science, 17 Aug. 2020, https://towardsdatascience.com/xgboost-theory-and-practice-fb8912930ad6. 

