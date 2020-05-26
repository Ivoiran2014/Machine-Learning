#Importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Position_Salaries.csv")

#Creating a matrix for non dependant and dependant variable
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Split the data into training and testing set
#Due to the lack of info we wouldn't need to split the data

#Fitting the Simple Linear regression to the data
from sklearn.linear_model import LinearRegression
simple_linear_reg = LinearRegression()
simple_linear_reg.fit(X,y)

#Fitting the Polynomial Regression to the data
#Creating the matrix X with polynomial values
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
complex_linear_reg = LinearRegression()
complex_linear_reg.fit(X_poly,y)


#Visualizing the linear regression model
plt.scatter(X,y,color='red')
plt.plot(X, simple_linear_reg.predict(X), color='blue')
plt.title("Salary base on Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#Visualizing the polynomial regression model
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid, complex_linear_reg.predict(poly_reg.fit_transform(X_grid)), color='green')
plt.title("Salary base on Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()