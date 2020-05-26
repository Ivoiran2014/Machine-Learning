#Importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("50_Startups.csv")

#Creating a matrix for non dependant and dependant variable
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Encoding categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])

#OneHotEncoder with ColumnTransformer
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('hot_encoder', OneHotEncoder(), [3])],\
                                      remainder='passthrough')

X = columnTransformer.fit_transform(X)

#Avoiding the dummy variable trap
X = X[:,1:]

#Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)

#Fitting the Multiple Linear Regression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the testing set result
y_pred = regressor.predict(X_test)


#Building an optiomal model using Backward Elimination

#Importing Stat Model Library
import statsmodels.api as sm
#Stat Model don't take in consideration the constant , so we need to add it 
X = np.append(arr = np.ones((50,1)), values = X, axis = 1)
X_opt = np.array(X[:,[0,1,2,3,4,5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(X[:,[0,1,3,4,5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(X[:,[0,3,4,5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(X[:,[0,3,5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(X[:,[0,3]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Fiting and drawing with the best model
best_feature = X_opt[:,1]

#Separate data into training and testing 
from sklearn.model_selection import train_test_split
bX_train,bX_test,by_train,by_test = train_test_split(best_feature, y,\
                  test_size = 0.2,random_state = 0)
      
bX_train = bX_train.reshape(-1,1)  
bX_test = bX_test.reshape(-1,1)    
by_train = by_train.reshape(-1,1)  
by_test = by_test.reshape(-1,1)

#Training our Data 
from sklearn.linear_model import LinearRegression
regressor_best = LinearRegression()
regressor_best.fit(bX_train,by_train)
#Predicting
by_pred = regressor_best.predict(bX_test)

#Drawing the Training set 
plt.scatter(bX_train,y_train,color='red')
#Drawing the regression line
plt.plot(bX_train,regressor_best.predict(bX_train),color='blue')
plt.title("Profit vs R&D Spend")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.show()






