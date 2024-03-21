import numpy as np
from Perceptron import Perceptron
from Activation import Activation
from Neuron import Neuron
# from Optimizer import Optimizer


from typing import Literal, get_args
from Loss import _Losses
class Layer:
    def __init__(self, n_neuron: int, num_features: int ,**kwargs) -> None:
        """
        num_features:
        weightss:
        bioses:
        n_neuron:
        act_func:

        """
        self.num_features = num_features
        self.weightss = kwargs.get('weights', np.random.uniform(0,1, size=(n_neuron,num_features)))
        self.bioses = kwargs.get('bios', np.zeros(n_neuron) )
        self.n_neuron = n_neuron
        self.Losses = []
        act_tmp = kwargs.get('act_func', None)
        if act_tmp:
            self.active = act_tmp
        else:
            act_tmp = Activation()
            self.active = act_tmp.linear
        self.perceptrons = []
        # print(self.active)
        for i in range(n_neuron):
            p = Perceptron(num_featuress=num_features, weights =self.weightss[i], act_func = self.active, bios=self.bioses[i])
            self.perceptrons.append(p)

    def calculate(self, data):
        outputs = []
        for i in range(self.n_neuron):
            #TODO: why the output is duplicated?
            # print(i,self.perceptrons[i].active)
            output = self.perceptrons[i].calculate(data)
            outputs.append(output)
        
        return outputs
    # TODO: `loss` function should compute loss of the layer on a given set of data and a given loss_function from the class Loss
    
    def loss(data: np.array, lss_func):
        pass
        # lss_func = lss_func
        # for percept in self.perceptrons:
        #     l = percept.Loss
    def BGD(self, data: np.array, alpha: float = 0.1,delta=0.001, epoch=4, lss_fun='MAbE'):
        """ 
        batch gradient descent

        data: X |Y 
        last element of each record is the label of that record
        """
        options = get_args(_Losses)
        assert lss_fun in options, f" '{lss_fun}' is not in {options}"
        
        for perceptron in self.perceptrons:
            new_weights = perceptron.weights
            grad = perceptron.optimize.grad
            
            tmp_neuron = Neuron(num_features=perceptron.num_features, 
                                active=perceptron.active,
                                weights= new_weights,
                                bios= perceptron.bios)
            for i in range(epoch):
                tmp_neuron.weights = new_weights
                new_weights -= alpha * grad(nueron= tmp_neuron,
                                            loss_fun= lss_fun,
                                            data= data,
                                            delta=delta)
            perceptron.weights = new_weights
