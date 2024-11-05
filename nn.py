import random
from engine import Value

"""
creates a Neuron object with a certain number of weights and one bias
"""

class Module():
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []
    
class Neuron(Module):

    # nin: number of inputs
    # w: nin number of weights for nin number of inputs
    # b: bias for neuron
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    # return value when Neuron(x) is called
    # x: input values
    def __call__(self, x):

        # pairing each weight up with corresponding input
        # w * x + b
        # tanh(w * x + b)
        paired = zip(self.w, x)
        activationVal = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return activationVal.relu() if self.nonlin else activationVal
    
    # returns a list of all the parameters for this Neuron
    def parameters(self):
        return self.w + [self.b]

"""
creates a Layer object with a certain number Neurons that have a certain amount of inputs
number of inputs is how many neurons there are in the layer before
"""  
class Layer(Module):
    # nin: nin number of inputs to each Neuron in layer
    # nout: number of Neurons in layer
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    
    # return value when Layer(x) is called
    # x: input values
    def __call__(self, x):
        # returns output of each neuron in the layer
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    # return parameters of each neuron in the layer
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
        # same code as above--above is just cleaner
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params

"""
creates a multilayer perceptron model with a set of inputs and a list of output Neurons
length of list is how many layers there are in the model
"""    
class MLP(Module):

    # nin: number of inputs
    # list of output Neurons (length of list is how many layers there are)
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    # return value when MLP(x) is called
    # x: input values
    def __call__(self, x):
        # returns each layer sequentially
        for layer in self.layers:
            x = layer(x)
        return x   
    
    # return all parameters for each node in the MLP
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

