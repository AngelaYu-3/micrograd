from graphviz import Digraph

class Value:

    # data: a single scalar value 
    # children: tuple that stores what two values produce other values to keep expression graphs
    # op: stores what operation is acted upon the two values stored in children
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.prev = set(children)
        self.op = op
    
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
    
    # visualizing expression graphs
    #def trace(root):
        #dot = Digraph(format='svg', graph_attr={'rankdir'})



# testing
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a*b + c
print(d.op)