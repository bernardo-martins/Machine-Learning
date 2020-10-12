# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:30:21 2020

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

def shuffle_data(data, thrsld):
    ranks = np.arange(data.shape[0])
    np.random.shuffle(ranks)
    train = data[ranks>=thrsld, :]
    test = data[ranks<thrsld, :]
    return (train, test)

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
    plt.show()
    plt.close()

def std_data(data):
    mean = np.mean(data, 0)
    std = np.std(data, 0)
    data = (data - mean)/std
    return data

def open_file(filename):
    data = np.loadtxt(filename, delimiter='\t', skiprows=1)
    return data

def plotSolution(solver):
    px = np.linspace(0.1, 10)
    px = px.reshape(-1,1)
    predys = solver.predict(px)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(px, predys, '-')
    return

def find_best_solution(train, valid, test, max_degree):
    best_error = 100000
    linspace = np.linspace(0.01, 10)
    for lmbd in linspace:
        plt.figure(1, figsize=(12, 8), frameon=True)
        solver = Ridge(alpha = lmbd, solver = 'cholesky', tol=0.001)
        solver.fit(train[:,:-1], train[:,-1])
        #error = calculate_mean_square_error(predYs, valid[:,1])
        #valid_err = np.mean((predYs-valid[:,-1])**2)
        plotSolution(solver)
        #errorv.append(valid_err)
        #lmbdv.append(lmbd)
    #plt.xlabel('lambda')
    #plt.ylabel('error')
    #plt.plot(lmbdv, errorv, 'o')
    #plt.axis([-0.1, 12, -2.0, 2.0])
    #closePlots()
    return best_error

#data = open_file('bluegills.txt')
#data = std_data(data)
#(train, sample) = shuffle_data(data, int(data.shape[0]/2))
#(valid, test) = shuffle_data(sample, int(sample.shape[0]/2))
#find_best_solution(train, valid, test, 8)
x = [[1,2,3,4,5]]
y = np.ones((x.shape[0], x.shape[0]+1))
print(y)