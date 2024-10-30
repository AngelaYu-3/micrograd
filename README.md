# micrograd

A small autograd engine inspired by [PyTorch Autograd Engine](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/). Implements backpropagation over a dynamically built DAG (directed acyclical graph) and small neural networks library with a PyTorch-like API. DAG only operates over scalar values (meaning we chop up each neuron into all of its individual operations). This autograd engine is powerful enough to build deep neural networks implementing binary classification. 

This source code is commented to demonstrate deeper understanding of core concepts and for ease of future understanding. Further implementing and practicing backpropagation core concepts studied in [Neural Networks and Deep Learning by DeepLearning.AI and Andrew Ng](https://www.coursera.org/account/accomplishments/verify/ZJKF2ULGZVMS) 

This project is for educational purposes and follows the course [Neural Networks: Zero to Hero by Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1).

## Installation
test
## Example usage
test
## Questions

**Q: Why are non-linear activation functions important, and how does using non linear activation functions allow
for modeling of more complex patterns?**

A: Non-linear functions are representive enough to approximate any function (think of Taylor series). High-level conceptual idea of training a NN is finding a function that maps inputs to outputs with the lowest loss. Thus, non-linear functions are needed to allow for "flexibility" in finding the correct weights or "coefficients" for the function that maps inputs to outputs.
___
**Q: When you get various derivatives of loss function with respect to weights (meaning knowing how loss changes as weights vary), how do you use those derivatives to update weights? (a += learning_rate * dL/da) Why do you add 'learning_rate * dL/da' to 'a'?**

A: Refer to the diagram below--same concept applies in higher dimensions (more weights)

[click here to see image](images/question2_diagram.pdf)


"""
## License
MIT

