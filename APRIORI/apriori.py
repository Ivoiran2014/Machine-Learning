#Importing library
#%reset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions = []
#Transforming the dataframe into a list of list
for i in range(0,7501):
      transactions.append([str(dataset.values[i,j]) for j in range(0,20)])


#Training the Apriori to the dataset
from apyori import apriori
rules = apriori(transactions,min_support = 0.003, min_confidence = 0.2 , 
                min_lift = 3, min_length = 2)

result = list(rules)