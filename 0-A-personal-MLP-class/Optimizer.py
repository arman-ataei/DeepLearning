import numpy as np
from Neuron import Neuron
from typing import Literal, get_args
from Loss import _Losses

class Optimizer:
    def __init__(self)  -> None:
        pass
    def grad(self,
             nueron : Neuron,
             data: np.array,
             loss_fun: _Losses="MAbE",
             delta: float = 0.00001,
             ):
        """ 
        tg_loss: targeted loss function that you wnat to calculate it's gradient
        weights: weights of the perceptron
        
        record: X |Y 
        last element of each record is the label of that record
        bios: 
        delta: 

        output: (Loss(w+delta) - Loss(w))/delta
        """
        options = get_args(_Losses)
        assert loss_fun in options, f" '{loss_fun}' is not in {options}"
        
        d_neuron = Neuron(num_features=nueron.num_features, 
                          active=nueron.active,
                          weights= nueron.weights + delta,
                          bios= nueron.bios)
        
        f1 = getattr(d_neuron.Loss, loss_fun)
        f2 = getattr(nueron.Loss, loss_fun)
        return (f1(data= data )-f2(data=data))/delta