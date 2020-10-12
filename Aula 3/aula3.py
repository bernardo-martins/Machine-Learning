# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 09:18:47 2020

@author: ASUS
"""

import funcoes as f
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
# insert at 1, 0 is the script path (or '' in REPL)

def load_txt(filename):
    
    data = np.loadtxt('data.txt', delimiter=',')
    #shuffle
    np.random.shuffle(data)
    Ys = data[:,0]
    Xs = data[:, 1:]
    Xs = f.poly_16features(Xs)
    means = np.mean(Xs, axis = 0)
    stdevs = np.std(Xs, axis = 0)
    Xs = (Xs-means)/stdevs #normalization
    
    return Xs, Ys

def calc_fold(feats, X,Y, train_ix,valid_ix,C=1e12):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix,:feats],Y[train_ix])
    prob = reg.predict_proba(X[:,:feats])[:,1]
    squares = (prob-Y)**2
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix])

def plotErrors(featx, tra, vaa, folds):
    
    plt.plot(np.linspace(min(featx), max(featx), 140), tra, '-')
    plt.plot(np.linspace(min(featx), max(featx), 140), vaa, '-')
    
    return

def Tut3Prob1():
    (Xs,Ys) = load_txt('data.txt')
    X_r,X_t,Y_r,Y_t = train_test_split(Xs, Ys, test_size=0.33, stratify = Ys)
    folds = 10
    kf = StratifiedKFold(n_splits=folds)
    (bestfeats, besttra, bestvaa) = (10000,10000,10000)
    featx = []
    tra = []
    vaa = []
    for feats in range(2,16):
        featx.append(feats)
        tr_err = va_err = 0
        for tr_ix,va_ix in kf.split(Y_r,Y_r):
            r,v = calc_fold(feats,X_r,Y_r,tr_ix,va_ix)
            tr_err += r
            va_err += v
            tra.append(tr_err)
            vaa.append(va_err)
            if va_err<bestvaa:
                bestvaa = va_err
                bestfeats = feats
                besttra = tr_err
    f.create_plot(X_r, Y_r, X_t, Y_t, bestfeats, 1e5 )
    print('-----', )
    print(feats,':', tr_err/folds,va_err/folds)
    return

Tut3Prob1()