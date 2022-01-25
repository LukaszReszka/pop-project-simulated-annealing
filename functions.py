import numpy as np
from numpy import sin
from numpy import cos
from numpy import pi
from numpy import sqrt
from cec17_functions import cec17_test_func

def func0(x: list[float]) -> float:
    return x[0] ** 2

def func1(x): # a few, same local optimum
    x = x[0]
    return ((x-1) ** 2) * ((x+1) ** 3) * (5*x+2) ** 5

def func2(x): # a few different local optimum
    return x[0] ** 2 + x[1] ** 2

def func3(x):
    # x^4 - 5x^2 - 3x 
    return x**4 - 5*x**2 - 3*x

# three dimensional functions 
def rosenbrock_f(x):
    return (1-x[0]) ** 2 + 100*(x[1]-x[0]**2)**2

def shubert_f(x):
    x1, x2 = x
    sum1 = 0
    sum2 = 0
    for i in range(1,6):
        sum1 = sum1 + (i* cos(((i+1)*x1) +i))
        sum2 = sum2 + (i* cos(((i+1)*x2) +i))
    return sum1 * sum2

def bird_f(x):
    x, y = x
    return np.sin(x)*(np.exp(1-np.cos(y))**2)+np.cos(y)*(np.exp(1-np.sin(x))**2)+(x-y)**2

def cec_func(x0):
    # x: Solution vector
    x = x0
    # nx: Number of dimensions
    nx = len(x0)
    # mx: Number of objective functions
    mx = 1
    # func_num: Function number
    func_num = 8
    # Pointer for the calculated fitness
    f = [0]
    cec17_test_func(x, f, nx, mx, func_num)
    result = f[0]
    return result

