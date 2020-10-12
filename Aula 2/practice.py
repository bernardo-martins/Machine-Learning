# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 09:35:55 2020

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt

def split_data(data, index):
    ranks = np.arange(data.shape[0])
    np.random.shuffle(ranks)
    test = data[ranks>=index,:]
    train = data[ranks<index,:]
    return (test, train)
    
def normalize_data(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    data = (data-mean)/std
    return data

def open_file(filename):
    data = np.loadtxt('bluegills.txt', delimiter = '\t', skiprows = 1)
    return data

def mean_square_error(correct_y, pred_y):
    error = np.mean((correct_y-pred_y)**2) 
    return error

def plot_set(train, valid, test):
    plt.plot(train[:,0], train[:,1], 'ob')
    plt.plot(valid[:,0], valid[:,1], 'og')
    plt.plot(test[:,0], test[:,1], 'or')
    
def show_and_close():
    plt.figure(1, figsize=(12, 8), frameon=True)
    plt.title('Blue gill size')
    plt.legend(loc="center right")
    plt.show()
    plt.close()
    
def draw_plot(coefs, test, lbl):
    px = np.linspace(min(test[:,0])-1, max(test[:,0])+1, 100)
    predy = np.polyval(coefs, px)
    plt.plot(px,predy, '-', label = lbl)
    return

def calculateBestSolution(train, valid, test, max_degree):
    best_error = 1000000
    for degree in range(1, max_degree):
        coefs = np.polyfit(train[:,0], train[:,1], degree)
        pred_y = np.polyval(coefs, valid[:,0])
        valid_error = mean_square_error(valid[:,1], pred_y)
        draw_plot(coefs, test, "{g:d}/{e:6.3f}".format(g = degree, e = valid_error))
        if(valid_error<best_error):
            best_error = valid_error
            best_coefs = coefs
            best_degree = degree
    plot_set(train, valid, test)        
    show_and_close()
    return (best_error, best_degree, best_coefs)
    
data = open_file('bluegills.txt')
(train, sample) = split_data(data, int(data.shape[0]/2))
(valid, test) = split_data(sample, int(sample.shape[0]/2))

train = normalize_data(train)
valid = normalize_data(valid)
test = normalize_data(test)

print(calculateBestSolution(train, valid, test, 7))
