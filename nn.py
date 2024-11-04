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



# TESTING
# making a binary classifier neural network with 3 inputs and 3 layers of 4, 4, 1, neurons respectively
"""
def binaryClassifier():

    input values, MLP, training data
    x = [2.0, 3.0, -1.0]    # input values
    n = MLP(3, [4, 4, 1])   # creating a MLP with 3 inputs and 3 layers of 4, 4, 1, neurons respectively
    n.parameters()          # getting all parameters (weights, biases) for all the neurons

    # training data
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0]
]
    # ground truth for testing data
    ys = [1.0, -1.0, -1.0, 1.0]                                        
    

    training
    iterations = 1000
    for k in range(iterations):
        # forward pass
        ypred = [n(x) for x in xs]                                         # running forward propagation to get predictions
        loss = sum([(yout - ygt)** 2 for ygt, yout in zip(ys, ypred)])     # finding loss b/w ys and y pred using mean squared loss function and summing all losses together

        # backward pass
        loss.backward()                                                    # running backpropagation to get gradients of all the neurons relative to the summed loss

        # gradient descent
        for p in n.parameters():
            # print('data: ', p.data)
            # print('grad:', p.grad)
            p.data += -0.01 * p.grad    # it is '+= -0.05' relationship because the cost function is mean squared error, cost will always be positive meaning target loss is 0
        print('loss{}: {}'.format(k, loss.data))
    
    print([n(x) for x in xs])

binaryClassifier()
"""