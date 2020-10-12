# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:15:16 2020

@author: ASUS
"""

from pylab import *

t = arange(0.0, 2.0, 0.001)
s = sin(4*pi*t)*cos(2*pi*t)*sin(pi*t)

plot(t,s)

show();