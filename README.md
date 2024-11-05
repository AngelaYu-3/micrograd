# micrograd

A small autograd engine inspired by [PyTorch Autograd Engine](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/). Implements backpropagation over a dynamically built DAG (directed acyclical graph) and small neural networks library with a PyTorch-like API. DAG only operates over scalar values (meaning we chop up each neuron into all of its individual operations). This autograd engine is powerful enough to build deep neural networks implementing binary classification. 

This project is for educational purposes and follows the course [Neural Networks: Zero to Hero by Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1). 

___

***Source code is commented to demonstrate deep, independent understanding of core concepts as well as for ease of future read-throughs. Questions included below are also demonstrations of own independent learnings.***

***Project allows for further applications of core concepts studied in [Neural Networks and Deep Learning by DeepLearning.AI and Andrew Ng](https://www.coursera.org/account/accomplishments/verify/ZJKF2ULGZVMS)***

## Example Usage: Building a Binary Classifier

```bash
# TESTING
# making a binary classifier neural network with 3 inputs and 3 layers of 4, 4, 1, neurons respectively
def binaryClassifier():
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
```

## Questions

**Q: Why are non-linear activation functions important, and how does using non linear activation functions allow
for modeling of more complex patterns?**

A: Non-linear functions are representive enough to approximate any function (think of Taylor series). High-level conceptual idea of training a NN is finding a function that maps inputs to outputs with the lowest loss. Thus, non-linear functions are needed to allow for "flexibility" in finding the correct weights or "coefficients" for the function that maps inputs to outputs.
___
**Q: When you get various derivatives of loss function with respect to weights (meaning knowing how loss changes as weights vary), how do you use those derivatives to update weights? (a += learning_rate * dL/da) Why do you add 'learning_rate * dL/da' to 'a'?**

A: Refer to the diagram below--same concept applies in higher dimensions (more weights)

[click here to see image](images/question2_diagram.pdf)


## License
MIT

