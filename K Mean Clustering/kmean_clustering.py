
#Importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Mall_Customers.csv")

#Creating a matrix for non dependant and dependant variable
X = dataset.iloc[:,[3,4]].values

#Visualising the Data
x_axis = X[:,0]
y_axis = X[:,1]

plt.scatter(x_axis,y_axis)
plt.title("Mall Customer Data")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()

from sklearn.cluster import KMeans
#Within-Cluster-Sum-of-Squares use to choose the appopriate number of cluster
wcss = []
for i in range(1, 11):
      kmeans = KMeans(n_clusters= i , init='k-means++', n_init=10, max_iter=300,
                      random_state = 0)
      kmeans.fit(X)
      wcss.append(kmeans.inertia_)
      
#Ploting the Elbow Graph
plt.plot(range(1,11),wcss)
plt.title("Elbow Graph")
plt.xlabel("Number of cluster")
plt.ylabel("WCSS")
plt.show()


#Creating the appopriate Clusterting
kmeans = kmeans = KMeans(n_clusters= 5 , init='k-means++', n_init=10, max_iter=300,
                      random_state = 0)
y_pred = kmeans.fit_predict(X)

#Visualising the clusetring 
plt.scatter(X[y_pred == 0,0],X[y_pred == 0,1], s = 100 , c = 'red',label = 'Careful')
plt.scatter(X[y_pred == 1,0],X[y_pred == 1,1], s = 100 , c = 'green',
            label='Expenses')
plt.scatter(X[y_pred == 2,0],X[y_pred == 2,1], s = 100 , c = 'blue',label = 'Ideal')
plt.scatter(X[y_pred == 3,0],X[y_pred == 3,1], s = 100 , c = 'cyan',
            label = 'Protective')
plt.scatter(X[y_pred == 4,0],X[y_pred == 4,1], s = 100 , c = 'magenta',
            label = 'Conservative')
#Visualising the centroid
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300 ,
            c = 'yellow')
plt.title("Mall Customer Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()














