import torch

# building neural network that looks like this:
"""
x1 --w1->
           tanh((x1 * w1) + (x2 * w2) + b)  --> output
x2 --w2->
"""
x1 = torch.Tensor([2.0]).double()        ; x1.requires_grad = True    
x2 = torch.Tensor([0.0]).double()        ; x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()       ; w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()        ; w2.requires_grad = True
b = torch.Tensor([6.8813735870195432])   ; b.requires_grad = True
n = x1 * w1 + x2 * w2 + b
o = torch.tanh(n)

print(o.item())
print(o)
o.backward()

print('---')
print('x1', x1.grad.item())
print('w1', w1.grad.item())
print('x2', x2.grad.item())
print('w2', w2.grad.item())
