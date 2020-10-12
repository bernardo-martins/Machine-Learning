# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 22:56:02 2020

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt

def createModel(dimensions):

    mat = np.loadtxt('polydata.csv', delimiter =';')
    x = mat[:,0]
    y = mat[:,1]
    coefs = np.polyfit(x,y,dimensions)
    
    pxs = np.linspace(0, max(x), 100)
    poly = np.polyval(coefs, pxs)
    
    
    plt.figure(figsize=(12,8))
    plt.plot(x,y,'or')
    plt.plot(pxs, poly, '-')
    plt.axis([0, max(x), -1.5, 1.5])
    plt.title('Degree: 2')
    plt.savefig(str(dimensions)+'testplot.png')
    plt.close();
    
    
    
def createAltModel(dimensions):
    matrix = np.loadtxt('polydata.csv', delimiter = ';')
    x = matrix[:,0]
    y = matrix[:,1]
    coefs = np.polyfit(x,y,dimensions) #non-linear transformation that is going to up our data dimensions
    pxs = np.linspace(0, max(x), 0.1)
    poly = np.polyval(coefs, pxs)
    
    plt.figure(figsize = (12,8))
    plt.plot(x,y,'or')
    plt.plot(pxs, poly, '-')
    plt.show()
    plt.close()
    
for i in range(1,7):
    createAltModel(i)