import numpy as np
from typing import Literal
_Losses = Literal["AbE", "SqE", "MAbE", "MSqE"]

class Loss:
    def __init__(self, neuron) -> None:
        """
        neuron is of type Neuron
        """
        self.neuron = neuron

    def AbE(self, data: np.array):
        """ 
        Absolute Error 
        data = (X,Y)
        X: single input array
        Y: single label corresponding to X
        """
        
        return np.abs(self.neuron.calculate(data[0])[0] - data[1])
    
    def SqE(self, data: np.array):
        """ 
        Square Error 
        data = (X,Y)
        X: single input array
        Y: single label corresponding to X
        """
        return np.sqrt((self.neuron.calculate(data[0])[0]- data[1])** 2)

    def MAbE(self,data: np.array):
        """ 
        Mean Absolute Error 
        data = (X,Y)
        X: the set of inputs (an array of records)
        Y: the set of corresponding labels
        """
        X = data[:,0]
        Y = data[:,1]
        loss = 0
        for i in range(self.neuron.num_features):
            loss += np.abs(self.neuron.calculate(X[i])[0] - Y[i])

        return loss/len(Y)
    
    def MSqE(self, data: np.array):
        """ 
        Mean Square root Error 
        data = (X,Y)
        X: the set of inputs (an array of records)
        Y: the set of corresponding labels
        """
        X = data[:,0]
        Y = data[:,1]
        loss = 0
        for i in range(self.neuron.num_features):
            loss += (self.neuron.calculate(X[i])[0] - Y[i])**2

        return loss/len(Y)
        
    