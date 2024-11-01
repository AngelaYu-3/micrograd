
import math

"""
stores a single scalar value and 
solves for its gradient through backward propagation 
of an expression graph of mathematical operations 

(addition, subtraction, multiplication, division, exponential)
"""
class Value:

    # data: a single scalar value 
    # grad: stores the derivative of L with respect to scalar value
    # backward: function that propagates the gradient from output to input (chain rule going backwards through graph)
    # prev: set of tuples that stores what two values produce other values to keep expression graphs
    # op: stores what operation is acted upon the two values stored in children
    # label: labels each node corresponding to variable which holds scalar value
    def __init__(self, data, children=(), op='', label=''):
        self.data = data
        self.grad = 0
        self.backward = lambda: None
        self.prev = set(children)
        self.op = op
        self.label = label
    
    # adding two Value objects
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def backward():
            # for reasoning, think about chain rule going backwards for addition
            # print('out_plus: ', out.grad)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out.backward = backward()

        return out
    
    # multiplying two Value objects
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def backward():

            # print('out_mult: ', out.grad)
            # for reasoning, think about chain rule going backwards for addition
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out.backward = backward()

        return out
    
    # calculates self ^ other, other is always an int or a float NOT a Value object
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f'**{other}')

        def backward():
            self.grad += (other * self.data**(other - 1)) * out.grad
        out.backward = backward()

        return out
    
    # calculates e^Value
    def pow(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def backward():
            self.grad += out.data * out.grad
        out._backward = backward()

        return out

    # tanh activation function
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def backward():
            # used derivative of tanh
            self.grad += (1 - t**2) * out.grad
        out.backward = backward()
            
        return out
    
    # relu activation function
    def relu(self):
        x = self.data
        r = 0 if x < 0 else x
        out = Value(r, (self, ), 'relu')

        def backward():
            # used derivative of relu
            self.grad += self.grad + (out.data > 0) * out.grad
        out.backward = backward()

        return out
    
    # use topological sort to order all of the children of the graph from left to right to do back propagation in the correct order
    def backward_prop(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
                #print(topo)
        build_topo(self)

        # go one variable at a time and apply the chain rule (backward function) to get the variable's gradient (back propagation on topological sorted graph)
        # need to set self.grad to 1 because loss changes by 1 when loss varies (dL/dL), L is the very last node
        topo[len(topo) - 1].grad = 1.0
        print(topo[len(topo) - 1])
        print(topo[len(topo) - 1].data)
        print(topo[len(topo) - 1].grad)
        # print(reversed(topo))
        for v in reversed(topo):
            # print(v.grad)
            # print(type(v))
            # print(type(v))
            v.backward
            # print(v.grad)


    # negates self Value object
    def __neg__(self):
        return self * -1
    
    # if other + self is called where other is NOT a Value object
    def __radd__(self, other):
        return self + other
    
    # if self - other is called
    def __sub__(self, other):
        return self + (-other)
    
    # if other - self is called where other is NOT a Value object
    def __rsub__(self, other):
        return self + (-other)
    
    # if other * self is called where other is NOT a Value object
    def __rmul__(self, other):
        return self * other 

    # calculates self / other
    def __truediv__(self, other):
        return self * other**-1
    
    # if other / self is called where other is NOT a Value object
    def __rtruediv__(self, other):
        return self * other**-1
    
    # string representation of Value object
    def __repr__(self):
        return f"Value(data={self.data})"




# TESTING
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L.grad = 1.0


# ***backpropagation is just recursive computation of chain rule backwards through the graph***

# backpropagation will calculate the derivatives of L with respect to all the variables (a, b, c, d, e, f) using the chain rule
# we want to find these derivatives because they are useful in training where L is loss function 
# and some variables are weights (gradient descent)
# These derivatives of L (loss) with respect to weights demonstrates how a change in that variable changes the loss

# why does chain rule works? example below: 
# z: car    y: bicycle    x: walking man
# car travels twice as fast as a bicycle (dz/dy = 2)    bicycle is four times as fast as a walking man (dy/dx = 4)
# how fast does the car travel relative to the man? (2 * 4 = 8)  (dz/dx = dz/dy * dy/dx = 2 * 4 = 8)

# for above tests, generated expression graph after forward and backward pass looks like this
"""
                          
a [data: -3  grad:  6] -->         c [data: 10 grad: -2] -->        f [data: -2 grad:  4]  -->  (*) --> L [data: -8  grad: 1]
                                                          (+) -->  d [data: 4  grad: -2]  -->
                         (*) --> e [data: -6  grad: -2] -->
b [data: 2   grad: -4] -->
"""

# brute force solving L with respect to a (dL/da), how does loss change when a varies (slope of loss over a)
def lol():
    h = 0.0001

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L1 = L.data

    a = Value(2.0 + h, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L2 = L.data

    #print((L2 - L1) / h)
