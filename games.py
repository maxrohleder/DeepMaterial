import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector
from scipy.optimize import least_squares

np.random.seed(1908)

A = np.random.rand(5, 5)
B = np.random.rand(5, 5)
C = np.random.rand(5, 5)

D = np.ones(5,5)

def func(c):
    return c[0]*A + c[1]*B + c[2]*C

def loss(erg):
    return