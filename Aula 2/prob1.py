# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:48:27 2020

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('bluegills.txt', delimiter = '\t', skiprows = 1)

def random_split(data, test_points):
    ranks = np.arange(data.shape[0])
    np.random.shuffle(ranks)
    train = data[ranks>=test_points, : ]
    test = data[ranks<test_points, : ]
    return (train, test)

def mean_squared_error(coefs, data):
    PredYs = np.polyval(coefs, data[:,0])
    error = np.mean((data[:,1]-PredYs)**2) 
    return error

def stddata(data):
    means = np.mean(data, 0)
    stds = np.std(data, 0)
    data = (data-means)/stds
    return data

def plotSets(data, tp):
    plt.plot(data[:,0], data[:,1], tp)

def showPlot(data, coefs):
    pxs = np.linspace(-3,2,100)
    py = np.polyval(coefs, pxs)
    plt.plot(pxs, py, '-')
    return

def closePlot():
    plt.figure(1, figsize=(12, 8), frameon=True)
    plt.title('Blue gill size')
    plt.legend(loc="center right")
    plt.show()
    plt.close()
    return


def calculateBestValue(train, valid,test,  max_degree):
    best_error = 1000000
    for degree in range(1, max_degree):
        coefs = np.polyfit(train[:,0], train[:,1], degree)
        error = mean_squared_error(coefs, valid)
        showPlot(test, coefs)
        if(error < best_error):
            best_coefs = np.copy(coefs)
            best_error = error
            best_degree = degree
    plotSets(train, 'og')
    plotSets(test, 'ob')
    plotSets(valid, 'or')
    closePlot()
        


(train_data, sample) = random_split(data, int(data.shape[0]/2))
(valid_data, test_data) = random_split(sample, int(sample.shape[0]/2))

train_data = stddata(train_data)
valid_data = stddata(valid_data)
test_data = stddata(test_data)

calculateBestValue(train_data, valid_data, test, 6)



    