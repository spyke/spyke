"""Fits a sum of Gaussians model to a spike.
Adapted from http://www.scipy.org/Cookbook/FittingData"""

import numpy as np
import scipy.optimize
from pylab import figure, plot
import time

def gaussian(mu, sigma, x):
    return np.exp(- ((x-mu)**2 / (2*sigma**2)) )

g = gaussian

def model(p, x):
    """Sum of two Gaussians"""
    return p[0]*g(p[1], p[2], x) + p[3]*g(p[4], p[5], x)

def cost(p, x, y):
    """Distance to the target function"""
    return model(p, x) - y

x = np.arange(87)
y = np.array([  0.69486117,   8.16924953,  10.17962551,   6.11466599,
         1.31948435,  -0.14410365,   0.52754396,  -1.19760346,
        -6.5304656 , -12.67904949, -17.30953789, -21.16654778,
       -27.42707253, -35.58847809, -39.90871048, -36.30270767,
       -30.48548698, -28.91375732, -28.71337509, -22.10368347,
       -10.01594257,   0.77593756,   6.38806391,   9.77489376,
        13.51230049,  18.76480675,  24.56304169,  29.13644218,
        29.13644218,  25.91305542,  25.22989464,  31.09949875,
        38.20145798,  39.93480301,  36.9852066 ,  34.95500946,
        34.4129982 ,  32.2089653 ,  28.85248566,  26.9906559 ,
        27.04725838,  26.68888474,  25.23055649,  23.87973785,
        25.07825661,  27.58584404,  27.02534294,  19.5314827 ,
         8.8014183 ,   0.38737282,  -4.01576424,  -7.24346924,
       -10.06681824, -11.75206089, -11.93491077, -13.00849819,
       -17.08987427, -24.19359589, -30.66723061, -32.69648361,
       -29.18977356, -23.64463043, -20.0399704 , -20.32796288,
       -22.92987823, -24.34780502, -21.20810699, -14.53381634,
        -9.78805828, -10.76048088, -14.7164259 , -16.65529442,
       -16.18955612, -16.59147072, -19.17337418, -21.50229836,
       -21.18888855, -17.96524811, -14.06570435, -11.56622982,
       -10.79750252, -10.52579212, -10.45510769, -10.73945427,
       -10.3060379 ,  -7.99611521,  -4.38621092], dtype=np.float32)
p0 = [-10, 7, 2, 10, 20, 4] # initial parameter guess
t0 = time.clock()
for i in xrange(100):
    p, success = scipy.optimize.leastsq(cost, p0, args=(x, y))
t1 = time.clock()
print '%.3f sec' % (t1-t0)

figure()
plot(x, y, 'k.-')
plot(p[0]*g(p[1], p[2], x) + p[3]*g(p[4], p[5], x), 'r-')

'''
class Parameter(object):
    def __init__(self, value):
            self.value = value

    def set(self, value):
            self.value = value

    def __call__(self):
            return self.value


def fit(function, parameters, y, x = None):
    def f(params):
        i = 0
        for p in parameters:
                p.set(params[i])
                i += 1
        return y - function(x)

    if x == None:
        x = np.arange(y.shape[0])
    p = [param() for param in parameters]
    optimize.leastsq(f, p)


# giving initial parameters
mu = Parameter(7)
sigma = Parameter(3)
height = Parameter(5)

# define your function:
def f(x): return height() * exp(-((x-mu())/sigma())**2)

# fit! (given that data is an array with the data to fit)

fit(f, [mu, sigma, height], data)
'''
