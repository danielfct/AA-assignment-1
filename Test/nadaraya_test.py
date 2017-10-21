# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:10:43 2017

@author: Andrea
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2*np.cos(x/2) + x**2/5 + 3

def gaussiank(u):
    k=np.e**(-0.5*u**2)/np.sqrt(2*np.pi)
    return k

def nad_wat(K, h, X, Y, x):
    num = 0
    den = 0
    for ix in range(len(X)):
        yf = Y[ix]
        u = (x-X[ix])/h
        k = K(u)
        den = den + k
        num = num + yf * k
    return num/den


xs= np.random.rand(200) * 10
ys= f(xs) + 2*np.random.randn(*xs.shape)
grid= np.r_[0:10:512j]

new_grid= np.zeros(*grid.shape)
for i in range(len(grid)):
    new_grid[i]= nad_wat(gaussiank, 0.5, xs, ys, grid[i])

first_grid= np.zeros(*grid.shape)
for i in range(len(grid)):
    first_grid[i]= nad_wat(gaussiank, 1, xs, ys, grid[i])
    
second_grid= np.zeros(*grid.shape)
for i in range(len(grid)):
    second_grid[i]= nad_wat(gaussiank, 0.01, xs, ys, grid[i])

plt.plot(grid, f(grid), 'r--', label= 'Reference')
plt.plot(xs, ys, 'o', alpha= 0.5, label= 'Data')
plt.plot(grid, new_grid, 'g', label= 'Fitted with 0.5')
plt.plot(grid, first_grid, 'b', label = 'Fitted with 1')
plt.plot(grid, second_grid, 'y', label= 'Fitted with 0.01')
plt.legend(loc= 'best')