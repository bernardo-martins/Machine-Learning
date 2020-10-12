# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:46:13 2020

@author: ASUS
"""

import matplotlib.pyplot as plt
#meter os dados todos de um ficheiro numa matriz
def load_planet_data(file_name):
    rows = []
    lines = open(file_name).readlines()
    for line in lines[1:]:
        parts = line.split(',')
        rows.append((float(parts[1]), float(parts[2]))) #vai adicionando os tuplos (x,y) a rows
        
    return numpy.array(rows); #cria um array com as rows

    
data = load_planet_data('planets.csv')
#mostrar tudo num gráfico a semelhança do que foi feito em 'Exercicios'
plt.figure()
plt.plot(data[:,0], data[:,1],'x')
plt.show()
plt.savefig('planetsplot.png')
plt.close()