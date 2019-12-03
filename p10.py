import math
import numpy as np
from scipy import linalg

def lowess(x,y):
    
    n = len(x)
    yest = np.zeros(n)
    w = np.array([np.exp(- (x - x[i])**2/(2*0.4*0.4)) for i in range(n)])   
    
    for i in range(n):
        weights = w[:, i]
              
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        
        theta = linalg.solve(A, b)
        
        yest[i] = theta[0] + theta[1] * x[i] 
        
    return yest 

x = np.linspace(0,2*math.pi,100)
y = np.sin(x) + 0.2*np.random.randn(100)
yest = lowess(x,y)

import matplotlib.pyplot as pl
pl.plot(x,y)
pl.plot(x,yest)
pl.show() 
