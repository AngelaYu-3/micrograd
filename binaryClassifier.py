import random
import numpy as np
import matplotlib.pyplot as plt
from nn import Module, Neuron, Layer, MLP
from engine import Value
from sklearn.datasets import make_moons

"""
creating a random dataset for testing
"""
x, y = make_moons(n_samples=100, noise=0.1)
y = y*2 - 1
# plt.figure(figsize=(5, 5))
plt.scatter(x[:,0], x[:,1], c=y, s=20, cmap='jet')
plt.title('Input Data')
# plt.show()


"""
neural network with 16 inputs and 2 layers of 16 neurons, and 1 neuron respectively
"""
model = MLP(2, [16, 16, 1])


"""
loss function--this loss function uses batches
"""
def loss(batch_size=None):
    # if batch_size not given, use entire dataset for training
    if batch_size is None:
        xb, yb = x, y

    # if batch_size given, select random subset of data to form a mini-batch
    else:
        # create a random permutation of indices from 0 to x.shape[0]-1 (number of samples in x)
        # [:batch_size] selects first batch_size number of random indices from this permutation
        ri = np.random.permutation(x.shape[0])[:batch_size]
        xb, yb = x[ri], y[ri]

    # convert inputs to Value objects
    inputs = [list(map(Value, xrow)) for xrow in xb]

    # forward propagation to get predicted scores
    scores = list(map(model, inputs))

    # svm "max=margin" loss (hinge loss)
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]

    # compute average data loss--total data loss is sum of individual losses averaged out by number of elements in batch
    data_loss = sum(losses) * (1.0 / len(losses))

    # L2 regularization (weight decay) aplied to prevent the model from overfitting
    # term sum((p*P for p in model.parameters())) computes sum of ssquare of model's parameters
    # alpha is the regularization strength and added to total loss to penalize large weights
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))

    # total loss is the sum of data loss (hinge loss) and regularization loss
    total_loss = data_loss + reg_loss

    # calculate accuracy of prediction
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]

    # return total loss and accuracy of the batch--ratio of correct predictions to total predictions in the mini batch
    return total_loss, (sum(accuracy)) / (len(accuracy))


"""
backward propagation and optimization
"""
iterations = 100
for i in range(iterations):

    # forward
    total_loss, accuracy = loss(10)

    # backward
    model.zero_grad()
    total_loss.backward()

    # update parameters
    learning_rate = 1.0 - 0.9*i/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    print(f"step {i} loss {total_loss.data}, accuracy {accuracy*100}%")


"""
visualizing decision boundrary
"""
# grid setup
# h controls resolution of grid, smaller values give a more detailed decision boundary while larger values make it coarser
# calculations of x_min, x_max, y_min, y_max ensure that plot extends slightly beyond range of dataset
h = 0.25
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

# creating grid of points
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# flattening the grid--flatten 2D xx and yy arrays into 1D arrays making it easier to feed into model
Xmesh = np.c_[xx.ravel(), yy.ravel()]

# model predictions on the grid
# transform each row of Xmesh into list of Value objects
# map(model, inputs) applies model to each input to obtain scores (predictions) for each grid point
inputs = [list(map(Value, xrow)) for xrow in Xmesh]
scores = list(map(model, inputs))

# convert scores to binary labels (classification)
Z = np.array([s.data > 0 for s in scores])
Z = Z.reshape(xx.shape)

# plot decision boundary
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

# plot data points
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)

# set plot limits
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title('Binary Classification Predictions')
plt.show()


"""
simple binary classifier
making a binary classifier neural network with 3 inputs and 3 layers of 4, 4, 1, neurons respectively

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
# ground truth for training data
ys = [1.0, -1.0, -1.0, 1.0]  
    

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
"""