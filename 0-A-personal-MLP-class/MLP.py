import numpy as np
from typing import Literal, get_args


from Layer import Layer
from Activation import _Activations, Activation
class MLP:
    def __init__(self) -> None:
        self.activation_function = Activation()
        self.Layers = []
        
    def add(self, n_neuron: int, n_features: int, **kwargs):
        """
        n_features: int
        n_neuron: int
        weightss:[n_layers,n_neuron,n_features]
        bioses: [n_layers]
        act_func:
        """
        weightss = kwargs.get('weights', np.random.uniform(0,1, size=(n_neuron,n_features)))
        bioses = kwargs.get('bios', np.zeros(n_neuron) )

        
        active = kwargs.get('act_func', None)
        activations = get_args(_Activations)
        assert active in activations, f" '{active}' is not in {activations}"
        active = getattr(self.activation_function, active)
        
        layer = Layer(n_neuron=n_neuron, num_features = n_features,weightss = weightss, bios=bioses, act_func=active )
        self.Layers.append(layer)
    def calculate(self, features: np.array):
        n_layers = len(self.Layers)

        for i in range(n_layers):
            self.output = [out[0] for out in self.Layers[i].calculate(data=features)]
            features = self.output
        
        return self.output