
class Value:

    # data: a single scalar value 
    # children: tuple that stores what two values produce other values to keep expression graphs
    # op: stores what operation is acted upon the two values stored in children
    # label: labels each node corresponding to variable which holds scalar value
    # grade: stores the derivative of L with respect to scalar value
    def __init__(self, data, children=(), op='', label=''):
        self.data = data
        self.grad = 0
        self.prev = set(children)
        self.op = op
        self.label = label
    
    # adding two Value objects
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    # multiplying two Value objects
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), 'x')
        return out

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


# for above tests, generated expression graph forward pass looks like this

# backpropagation will calculate the derivatives of L with respect to all the variables (a, b, c, d, e, f)
# we want to find these derivatives because they are useful in training where L is loss function 
# and some variables are weights (gradient descent)
# These derivatives of L (loss) with respect to weights demonstrates how a change in that variable changes the loss
"""
FORWARD PROPAGATION
                          
a [data: -3  grad: 0] -->         c [data: 10 grad: 0] -->        f [data: -2 grad: 0] -->  (*) --> L [data: -8  grad: 1]
                                                          (+) --> d [data: 4  grad: 0]  -->
                         (*) --> e [data: -6  grad: 0] -->
b [data: 2   grad: 0]  -->
"""

# brute force solving L with respect to a (dL/da)
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

    print((L2 - L1) / h)
    
lol()