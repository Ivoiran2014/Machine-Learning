#Importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Salary_Data.csv")

#Creating a matrix for non dependant and dependant variable
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 1/3, random_state = 0)

#Fitting the Simple Linear Rgeression to the training data
#Base in the Math , drawing the line that minimize the sum squared error
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set 
#Base in Math , we are giving x value and geting a y from the line we get by SSE
y_pred = regressor.predict(X_test)


#Drawing the Simple Linear Regression on Training set

#Scatter the training point on the graph
plt.scatter(X_train,y_train,color='red') 
#Draw the linear regression line itself
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()


#Drawing the Simple Linear Regression on Testing set

#Scatter the testing point on the graph
plt.scatter(X_test,y_test,color='red') 
#Draw the linear regression line line itself
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Testing Set)')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()