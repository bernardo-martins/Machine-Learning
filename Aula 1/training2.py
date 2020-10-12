# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:26:06 2020

@author: ASUS
"""

import csv
import numpy as np
import random as r
import matplotlib.pyplot as plt

#saber o periodo do planeta
def getOrbitalPeriod(planet):
    lines = open("planets.csv").readlines()
    for line in lines:
        res = line.split(',')
        if(res[0] == planet):
            return res[1]
#create a function that returns all orbital radius and period
def getDataOrganized(file_name):
    lines= open(file_name).readlines()
    rows = []
    for line in lines[1:]:
        res = line.split(',')
        rows.append((float(res[1]), float(res[2])))
    
    return np.array(rows)
        
print(getDataOrganized("planets.csv"))
