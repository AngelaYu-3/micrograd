from nn import Module, Neuron, Layer, MLP


# TESTING
# making a binary classifier neural network with 3 inputs and 3 layers of 4, 4, 1, neurons respectively
"""
input values, MLP, training data
"""
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
    

"""
training
"""
iterations = 100
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