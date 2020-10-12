# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:17:16 2020

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt

#standardized data is weird

data = np.loadtxt('bluegills.txt', delimiter = '\t', skiprows = 1)

def random_split(data, test_points):
    ranks = np.arange(data.shape[0])
    np.random.shuffle(ranks)
    train = data[ranks>=test_points,:]
    test = data[ranks<test_points,:]
    return (train, test)

def standardize(data):
    scaleY = np.max(data, axis = 0)
    data = data/scaleY
    return data

def mean_square_error(data, coefs):
    pred = np.polyval(coefs, data[:,0])
    error = np.mean((data[:,1]-pred)**2)
    return error

def findMinError(valid, test, maxDegree):
    best_err = 10000000
    for degree in range(1,maxDegree):
        coefs = np.polyfit(train[:,0], train[:,1], degree)
        print(coefs)
        print('-------')
        valid_error = mean_square_error(valid, coefs)
        drawPlot(coefs, test)
        if valid_error < best_err:
            best_err = valid_error
            best_coef = coefs
            best_degree = degree
    return best_degree, best_coef

def drawPlot(coefs, test):
    linsp = np.linspace(-4, 2, 20) #must be the same size as the 
    poly = np.polyval(coefs, test[:,0])
    #poly = np.polyval(coefs, linsp)
    plt.plot(linsp, poly, '-')

def showPlot(test):
    plt.plot(test[:,0], test[:,1], 'o')
    plt.axis([-4, 2, -3, 3])
    plt.show()
    plt.close()
    
#data_shuffled = random_split(data, 10)
data_stand = standardize(data)

train, temp = random_split(data_stand, len(data_stand)/2)
valid, test = random_split(temp, len(temp)/2)

(best_degree, best_coefs) = findMinError(valid, test, 6)

    
showPlot(test)

#select the best polynomial from degree 1 to 6





