import numpy as np
from typing import Literal
_Activations = Literal["linear", "binary", "sigmoid", "tanh" ,"relu"]
class Activation:
    def __init__(self) -> None:
        pass
    def linear(self, x):
        return x

    def binary(self, x):
        if x > 0:
            return 1

        if x <=0 :
            return 0

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def relu(self, x):
        return np.max([x,0])
    