# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:37:32 2020

@author: ASUS
"""
import csv
import numpy as np
import random as r
import matplotlib.pyplot as plt

def drawPrediction(filename, dimensions):
    data = np.loadtxt(filename, delimiter=';')
    x = data[:,0]
    y = data[:,1]
    coefs = np.polyfit(x,y,dimensions);
    
    lin = np.linspace(-max(x), max(x), 100)
    poly = np.polyval(coefs, lin)
    
    plt.plot(x,y, 'x')
    plt.plot(lin, poly, '-')
    plt.axis([0, max(x), -max(y), max(y)])
    plt.show()
    plt.close()
    
    return

def makeRawFile(filename):
    with open(filename, 'w',newline='') as file:
        writer = csv.writer(file, delimiter=';');
        r.seed(1)
        for i in range(1,100):
            writer.writerow([r.randint(1,10)*r.uniform(0.0,0.8), r.randint(3,15)*r.uniform(0.0,0.8)])
    file.close();

def drawPredict1(filename, dimensions):
    data = np.loadtxt(filename, delimiter=';')
    x=data[:,0]
    y=data[:,1]
    coefs = np.polyfit(x,y,dimensions)
    
    linSpace = np.linspace(-max(x), max(x), 100)
    
    px = np.polyval(coefs, linSpace) #adapt the coefficients to the linearspace created
    
    plt.plot(x,y,'x')
    plt.plot(px, coefs, '-')
    plt.show()
    return
    
    
def drawPredict(filename, dimensions):
    data = np.loadtxt(filename, delimiter = ';')
    x = data[:,0]
    y = data[:,1]
    coefs = np.polyfit(x, y, dimensions)
    linspaceX = np.linspace(-max(x), max(x), 100)
    poly = np.polyval(coefs, linspaceX)
    plt.plot(x,y,'x')
    plt.plot(linspaceX, poly,'-')
    plt.show()
    
    

drawPredict("rawData.csv", 3)