#!/usr/bin/env python3

####################################################################
# FILE DESCRIPTION
####################################################################
# CSCI 5822 Assignment 9
#   3/7/2018
# ------------------------------------------------------------------
# Gaussian Processes and Time-Series

####################################################################
#  IMPORTS
####################################################################
from math import log, exp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.colors
import pprint
import random
from scipy.stats import gamma
from scipy.stats import norm
from numpy.linalg import inv

import pdb

####################################################################
#       VARIABLES AND CONSTANTS
####################################################################
random.seed(12)

a1 = .2
a2 = .2
sigma = 2

####################################################################
#  FUNCTION DEFINITIONS
####################################################################
res = {
    'y1' : [0],
    'y2' : [0],
    'z' : [0]
}

####################################################################
#  MAIN
###################################################################

u1 = norm()
u2 = norm()
u3 = norm()
#u2 = norm(loc=0, scale=sigma^2)
#u3 = norm(loc=0, scale=sigma^2)

def get_y1():
    a = res['y1']
    return a[len(a)-1]

def get_y2():
    a = res['y2']
    return a[len(a)-1]

for i in range(100):
    y2 = get_y1() + u1.rvs(size=1)[0]
    y1 = a1*get_y1() + a2*get_y2() + u2.rvs(size=1)[0]
    z = y1 + u3.rvs(size=1)[0]

    res['y1'].append(y1)
    res['y2'].append(y2)
    res['z'].append(z)

    #print(y1)
    #print(y2)
    print(z)

plt.plot(res['z'])
plt.show()

