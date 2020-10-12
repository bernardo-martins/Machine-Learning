# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:43:03 2020

@author: ASUS
"""

#Msun = (v^2*r)/G

import math as m
import numpy as np
import matplotlib.pyplot as plt

def calculate():
    G = m.pow(6.67, -10)
    v = []  
    lines = open('planets.csv').readlines()
    for line in lines[1:]:
        splitted = line.split(',')
        vcalc = (2*m.pi*float(splitted[1]))/(float(splitted[2]))
        msun = ((vcalc*vcalc)*float(splitted[2]))/G
        v.append(msun)
    return v

res = calculate()

def plotIt():
    lines = open('planets.csv').readlines()
    x = [float(line.split(',')[1]) for line in lines[1:]]
    y = [float(line.split(',')[2]) for line in lines[1:]]
    coefs = np.polyfit(x,y,2)
    
    linspace = np.linspace(0, max(x),500)    
    
    val = np.polyval(coefs, linspace)
    plt.plot(x,y,'x')
    plt.plot(linspace, val, '-')
    plt.show()
    

plotIt()


    

