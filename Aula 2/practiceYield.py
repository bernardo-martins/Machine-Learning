# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:22:14 2020

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt

def shuffle_data(data, trshld):
    ranks = np.arange(data.shape[0])
    np.random.shuffle(ranks)
    train = data[ranks>=trshld, :]
    test = data[ranks<=trshld, :]
    return (train, test)

def open_file(filename):
    data = np.loadtxt(filename, delimiter='\t', skiprows=1)
    return data

def std_data(data):
    mean = np.mean(data, 0)
    std = np.std(data, 0)
    data = (data - mean)/std
    return data

def plotSets(train, valid, test):
    plt.plot(train[:,0], train[:,1], 'ob')
    plt.plot(valid[:,0], valid[:,1], 'og')
    plt.plot(test[:,0], test[:,1], 'or')

def drawPlot(coefs, data, error, degree):
    x = np.linspace(min(data[:,0])-1, max(data[:,0])+1,100)
    predYs = np.polyval(coefs, x)
    lbl = "{g:d}/{v:6.3f}".format(g = degree, v = error)
    plt.axis([min(data[:,0])-1.0,max(data[:,0])+1.0,min(data[:,1])-1.0,max(data[:,1])+1.0])
    plt.plot(x,predYs, label=lbl)
    return

def calculate_mean_square_error(predYs, realValue):
    error = np.mean((predYs-realValue)**2)
    return error
    
def closePlots():
    plt.legend(loc = "center right")
    plt.show()
    plt.close()

def find_best_solution(train, valid, test, max_degree):
    best_error = 100000
    for deg in [1,2,3]:
        plt.figure(1, figsize=(12, 8), frameon=True)
        coefs = np.polyfit(train[:,0], train[:,1], deg)
        predYs = np.polyval(coefs, valid[:,0])
        error = calculate_mean_square_error(predYs, valid[:,1])
        drawPlot(coefs, test, error, deg)
        if error<best_error:
            best_error = error
            best_coefs = coefs
    plotSets(train, valid, test)
    closePlots()
    return (best_error, best_coefs)
    
    


data = open_file('yield.txt')
data = std_data(data)
(train, sample) = shuffle_data(data, int(data.shape[0]/2))
(valid, test) = shuffle_data(sample, int(sample.shape[0]/2))
find_best_solution(train, valid, test, 9)


