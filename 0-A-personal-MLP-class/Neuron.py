import numpy as np
from Activation import Activation
from Loss import Loss


class Neuron:
    def __init__(self, num_features: int ,**kwargs) -> None:
        """
        weights: np.array
        bios: float
        act_func: activation function
        """
        self.num_features = num_features
        self.weights = kwargs.get('weights', np.random.uniform(0,1, size=num_features))
        self.bios = kwargs.get('bios', 0 )
        act_tmp = kwargs.get('act_func', None)
       
        # print(act_tmp)
        
        if act_tmp:
            self.active = act_tmp
        else:
            act_tmp = Activation()
            self.active = act_tmp.linear
        
        self.Loss = Loss(self)
        
    
    def calculate(self, X: np.array):
        """ 
        X: inputs of the perceptron
        """
        res = 0
        for i in range(self.num_features):
            res += self.weights[i] * X[i]
        res += self.bios
        return res , self.active(res)
        
