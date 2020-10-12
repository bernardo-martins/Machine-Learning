# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:13:47 2020

@author: ASUS
"""

import numpy
#criar uma funçao que va buscar o ficheiro excel dos planetas e 
def load_planet_data(file_name):
    rows = []
    lines = open(file_name).readlines()
    for line in lines[1:]:
        parts = line.split(',')
        rows.append((float(parts[1]), float(parts[2]))) #vai adicionando os tuplos (x,y) a rows
        
    return numpy.array(rows); #cria um array com as rows

    
data = load_planet_data('planets.csv')

print(data[:,0]) #apresenta todas as linhas da primeira coluna apenas
print("--------------")
print(data[0,:]) #apresenta todas as colunas da primeira linha
print("--------------")
print(data[:,[0]]) #quero a primeira coluna num array
print("--------------")
print(data[:,0]>1) #verifica se os elementos da primeira coluna sao maiores do que 1
print("--------------")
print(data[data[:,0]>1,:]) #verifica os planetas em que a distancia ao sol é maior que 1


#Select the rows corresponding to planets with orbital periods greater than 10 years
data[data[:,1]>10,:]

#How many planets have orbital periods greater than 10 years
len(data[data[:,1]>10,:])
numpy.sum(data[:,1]>10) #visto que True-1 e False-0 então ele faz a soma dos que são True

#Select the orbital periods of the planets whose orbital periods in years
# are greater than twice theorbital radius in AU. 
data[data[:,1]>2*data[:,0], 1]




