# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:40:27 2020

@author: ASUS
"""

import numpy as np
from funcoes import poly_16features, create_plot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


def load_expand_standard_startifysplitkfold_data(filename):#expand, normalize and standardize data
    data = np.loadtxt(filename, delimiter = ',')
    
    np.random.shuffle(data)
    
    Xs = data[:, 1:]
    Ys = data[:, 0]   
    
    means = np.mean(Xs, axis = 0)
    stdev = np.std(Xs, axis = 0)
    Xs = ((Xs-means) / stdev)
    
    Xs = poly_16features(Xs) #(90,16) 
    
    final = np.insert(Xs, 0, Ys, axis = 1)   #insert ys along Xs in the columns starting with 0
        
    x = final[:,1:]
    y = final[:,0]
    
    return (y, x)

def calc_fold(feats, X, Y, train_ix, valid_ix, C = 1e12):
    reg = LogisticRegression(C = C, tol = 1e-10)
        
    reg.fit(X[train_ix, :feats],Y[train_ix])
    prob = reg.predict_proba(X[:, :feats])[:,1]
    
    squares = (prob-Y)**2
        
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix]) 
    

def logistic_regression_with_cross_validation(filename):
    
    (y, x) = load_expand_standard_startifysplitkfold_data(filename)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, stratify = y)
    
    nFolds = 5
    kf = StratifiedKFold(n_splits = nFolds)
    errors = []
    errors_per_c = []
    best_f_error = 100000
    bestFeats = 100000

    for feats in range(1,17):
                t_error = v_error = 0
                for train_ix, valid_ix in kf.split(y_train, y_train):
                    t, v = calc_fold(feats, x_train, y_train, train_ix, valid_ix)
                    t_error += t
                    v_error += v
                
                norm_verr = v_error/nFolds
                norm_terr = t_error/nFolds
                if(best_f_error>norm_verr):
                    best_f_error = norm_verr
                    bestFeats = feats
    
    bestC = 1
    vecc = [2**(i) for i in range(0,21)]
    
    for c in vecc:
                t_error = v_error = 0
                for train_ix, valid_ix in kf.split(y_train, y_train):
                    t, v = calc_fold(bestFeats, x_train, y_train, train_ix, valid_ix, c)
                    t_error += t
                    v_error += v
                
                norm_verr = v_error/nFolds
                norm_terr = t_error/nFolds
                errors.append((norm_verr, norm_terr))
                errors_per_c.append((norm_verr, c))
    bestC = min(errors_per_c, key = lambda item:item[0])[1]
    
    create_plot(x_train, y_train, x_test, y_test, bestFeats, bestC)
     
    return 
    
logistic_regression_with_cross_validation('data.txt')


