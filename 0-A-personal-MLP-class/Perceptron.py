

import numpy as np
from Neuron import Neuron
from typing import Literal, get_args
from Optimizer import Optimizer
from Loss import _Losses


class Perceptron(Neuron):
    def __init__(self, num_featuress: int ,**kwargss) -> None:
        super().__init__(num_features= num_featuress, **kwargss)
        self.optimize = Optimizer()

    def BGD(self, data: np.array, alpha: float = 0.1,delta=0.001, epoch=4, lss_fun='MAbE'):
        """ 
        batch gradient descent

        data: X |Y 
        last element of each record is the label of that record
        """
        options = get_args(_Losses)
        assert lss_fun in options, f" '{lss_fun}' is not in {options}"

        new_weights = self.weights
        grad = self.optimize.grad
        
        tmp_neuron = Neuron(num_features=self.num_features, 
                            active=self.active,
                            weights= new_weights,
                            bios= self.bios)
        for i in range(epoch):
            tmp_neuron. weights = new_weights
            new_weights -= alpha * grad(nueron= tmp_neuron,
                                        loss_fun= lss_fun,
                                        data= data,
                                        delta=delta)
        self.weights = new_weights
        return new_weights
    
    def SGD(self, data: np.array, alpha: float = 0.1,delta=0.001, epoch=4, lss_fun='AbE'):
        """ 
        stochastic gradient descient
        
        data: X |Y
        last element of each record is the label of that record
        """
        rnd_ind = np.random.randint(0,np.size(data[:,1]))
        
        options = get_args( _Losses)
        assert lss_fun in options, f" '{lss_fun}' is not in {options}"

        new_weights = self.weights
        grad = self.optimize.grad
        
        tmp_neuron = Neuron(num_features=self.num_features, 
                            active=self.active,
                            weights= new_weights,
                            bios= self.bios)
        for i in range(epoch):
            tmp_neuron. weights = new_weights
            new_weights -= alpha * grad(nueron= tmp_neuron,
                                        loss_fun= lss_fun,
                                        data= data[rnd_ind],
                                        delta=delta)
        self.weights = new_weights
        return new_weights

        
# TODO: complete the mini-Batch GD
    def mBGD(self, data: np.array, alpha: float = 0.1,delta=0.001, epoch=4, lss_fun='MAbE'):
        print("mBGD")
