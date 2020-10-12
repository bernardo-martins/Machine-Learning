# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:37:34 2020

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt

def random_split(data,test_points):
    """return two matrices splitting the data at random
    """
    ranks = np.arange(data.shape[0])
    np.random.shuffle(ranks)
    train = data[ranks>=test_points,:]
    test = data[ranks<test_points,:]
    return train,test

def mean_square_error(data,coefs):
    """Return mean squared error
	   X on first column, Y on second column
    """
    PredYs = np.polyval(coefs,data[:,0])
    error = np.mean((PredYs-data[:,1])**2)
    return error

def plotSets(TrainSet,ValidSet,TestSet):
    plt.plot(TrainSet[:,0],TrainSet[:,1],'or')
    plt.plot(ValidSet[:,0],ValidSet[:,1],'og')
    plt.plot(TestSet[:,0],TestSet[:,1],'ob')
    
def plotcurve(Xs,coefs):
    PredYs=np.polyval(coefs,Xs)
    plt.plot(Xs,PredYs,'-')

def plotcurveB(Xs,coefs,labl):
    PredYs=np.polyval(coefs,Xs)
    plt.plot(Xs,PredYs,labl)
    
def stddata(File):
    #load and standardize data
    data = np.loadtxt(File,delimiter='\t',skiprows=1)
    means=np.mean(data,0)
    stds=np.std(data,0)
    data=(data-means)/stds
    return data
    #Ys = data[:,[-1]]
    #Xs = data[:,:-1]
    #means = np.mean(Xs,0)
    #stdevs = np.std(Xs,0)
    #Xs = (Xs-means)/stdevs

def TrainValidTestSets(data):
    train, temp = random_split(data, int(data.shape[0]/2))
    valid, test = random_split(temp, int(temp.shape[0]/2))
    return train,valid,test

def getTrainValidErrors(trainSet,validSet,coefs):
    train_error = mean_square_error(trainSet,coefs)
    valid_error = mean_square_error(validSet,coefs)
    return train_error,valid_error

def Tut2Prob1(File):
    data=stddata(File)
    trainSet,validSet,testSet = TrainValidTestSets(data)
    pxs = np.linspace(min(data[:,0]),max(data[:,0]),100)
    plt.figure(1, figsize=(12, 8), frameon=True)
    plotSets(trainSet,validSet,testSet)
    best_err = 10000000 # very large number
    bestcoefs=[]
    bestdegree=0
    listLegends=[]
    for degree in range(1,7):
        coefs = np.polyfit(trainSet[:,0],trainSet[:,1],degree)
        train_error,valid_error= getTrainValidErrors(trainSet,validSet,coefs)
        labl="{g:d}/{t:6.3f}/{v:6.3f}".format(g=degree,t=train_error,v=valid_error)
        plotcurveB(pxs,coefs,labl)
        if valid_error < best_err:
            best_err = valid_error
            bestcoefs = np.copy(coefs)
            bestdegree = degree
    test_error = mean_square_error(testSet,bestcoefs)
    plt.title('Blue gill size')
    plt.legend(loc="center right")
    plt.show()
    return bestdegree, test_error

Tut2Prob1('bluegills.txt')