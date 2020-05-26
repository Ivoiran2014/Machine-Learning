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

#Using Dendogram to find the optimal number of cluster
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X , method = 'ward'))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidian Distances")
plt.show()

#Fitting the Hierachical Clustering to the data 
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='ward')
y_cluster = hc.fit_predict(X)

#Visualising the Hierachical cluster
plt.scatter(X[y_cluster == 0,0],X[y_cluster == 0,1], s = 100, c = 'red',label='Careful')
plt.scatter(X[y_cluster == 1,0],X[y_cluster == 1,1], s = 100, c = 'green',
            label='Expenses')
plt.scatter(X[y_cluster == 2,0],X[y_cluster == 2,1], s = 100, c = 'blue',label = 'Ideal')
plt.scatter(X[y_cluster == 3,0],X[y_cluster == 3,1], s = 100, c = 'cyan',
            label = 'Protective')
plt.scatter(X[y_cluster == 4,0],X[y_cluster == 4,1], s = 100, c = 'magenta',
            label = 'Conservative')
plt.title("Hierachical Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()








