#Importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

#Creating a matrix for non dependant and dependant variable
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

#Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 0.25, random_state = 0)


#Feature scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Fitting The Classifier to the training data
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train,y_train)

#Predictiding the test set result
y_pred = classifier.predict(X_test)


#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Visualizing training data

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
#Retrieving the min and max value
age_min_value = X_set[:,0].min() - 1
age_max_value = X_set[:,0].max() + 1
salary_min_value = X_set[:,1].min() - 1
salary_max_value = X_set[:,1].max() + 1
#Merging the x and y value into a grid
X1,X2 = np.meshgrid(np.arange(start = age_min_value, stop = age_max_value , step = 0.01),
            np.arange(start = salary_min_value, stop = salary_max_value , step = 0.01))

#Drawing the contour line between the 2 prediction
#Predicting wether the pixel belong to one or the other class 
#And we apply a color to the pixel
plt.contourf(X1, X2 ,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))

#We plot the limit of the age and salary
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
#Scatter the training data 
for i, j in enumerate(np.unique(y_set)):
      temp_c = ListedColormap(('red', 'green'))(i)
      color_2d = np.atleast_2d(temp_c)
      
      plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = color_2d,
                  label = j)
      
plt.title("KNN (Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


#Visualizing testing data

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
#Retrieving the min and max value
age_min_value = X_set[:,0].min() - 1
age_max_value = X_set[:,0].max() + 1
salary_min_value = X_set[:,1].min() - 1
salary_max_value = X_set[:,1].max() + 1
#Merging the x and y value into a grid
X1,X2 = np.meshgrid(np.arange(start = age_min_value, stop = age_max_value , step = 0.01),
            np.arange(start = salary_min_value, stop = salary_max_value , step = 0.01))

#Drawing the contour line between the 2 prediction
#Predicting wether the pixel belong to one or the other class 
#And we apply a color to the pixel
plt.contourf(X1, X2 ,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))

#We plot the limit of the age and salary
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
#Scatter the testing data 
for i, j in enumerate(np.unique(y_set)):
      temp_c = ListedColormap(('red', 'green'))(i)
      color_2d = np.atleast_2d(temp_c)
      
      plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = color_2d,
                  label = j)
      
plt.title("KNN (Testing Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()



