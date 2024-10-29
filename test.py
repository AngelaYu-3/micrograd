import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 3*x**2 - 4*x + 5

x = 3.0
# finding value at f(x)
print(f(x))

# plotting f(x) from -5 to 5 with increments of 0.25
xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
#plt.show()

# finding slope at f(x)
h = 0.000000001
print((f(x + h) - f(x)) / h)


# more complex example
h = 0.000000001

# inputs
a = 2.0
b = -3.0
c = 10.0

# derivatives with respect to h
d1 = a*b + c
a += h
d2 = a* b + c
print('da/dh: ', (d2 - d1) / h)

d1 = a*b + c
b += h
d2 = a* b + c
print('db/dh: ', (d2 - d1) / h)

d1 = a*b + c
c += h
d2 = a* b + c
print('dc/dh: ', (d2 - d1) / h)






