# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:58:35 2020

@author: ASUS
"""
#criar uma funçao que permita obter o orbital period (segunda coluna) a partir do nome do planeta
def getOrbitalPeriod(planet):
    fil = open('planets.csv')
    lines= fil.readlines()
    for line in lines:
        parts = line.split(',')
        if(parts[0] == planet):
            return float(parts[2])
    return -1

