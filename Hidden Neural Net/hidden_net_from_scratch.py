
#%reset
#Creating a 1 hidden layer neural net with Numpy

#The goal is to predict wether a team will win,do the fan are happy and 
#Do 1 player get depressed after the match ?

#input
import numpy as np
information = np.random.randint(1,10,size = 3)

#The Features(Number of fan,Win Stats, Number of goal score) Layer weight
feature_weight = np.random.rand(3,3)

#The Hidden Layer weight
hidden_weight = np.random.rand(3,3)

weights = [feature_weight,hidden_weight]

#The Prediction phase 
def multi_output_neural_network(vector,matrix):
      output = [0,0,0]
      #Predict
      for i in range(len(vector)):
            output[i] = matrix[i].dot(vector)
      return output


def hidden_net_prediction(imput,weights):
      #Made the hidden layer prediction
      hid_pred = imput.dot(weights[0])
      #Use the hidden layer prediction to predict 
      pred = hid_pred.dot(weights[1])
      return pred

prediction = hidden_net_prediction(information, weights)
print(prediction)







