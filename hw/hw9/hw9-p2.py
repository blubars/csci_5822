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

modes = [
    [0.5,0.5],
    [0.07,0.9],
    [0.1,0.1] ]

colors = ['b', 'r', 'g']

sigma = 2

####################################################################
#  FUNCTION DEFINITIONS
####################################################################
class LinDynSys:
    def __init__(self, init=0):
        self.u1 = norm()
        self.u2 = norm()
        self.u3 = norm()
        #u2 = norm(loc=0, scale=sigma^2)
        #u3 = norm(loc=0, scale=sigma^2)
        self.cur_mode = 0
        self.res = {'y1': [init], 'y2': [init], 
                    'z': [init], 'm': [self.cur_mode]}

    def draw_uniform(self):
        return random.choice([0,1,2])

    def get_y1(self):
        a = self.res['y1']
        return a[len(a)-1]

    def get_y2(self):
        a = self.res['y2']
        return a[len(a)-1]

    def set_mode(self, mode):
        self.cur_mode = mode

    def get_mode(self, mode=None):
        # if mode is None, return current mode.
        if mode is None:
            mode = self.cur_mode
        m = modes[mode]
        return m[0], m[1]

    def run(self, num_steps, mode=None, mode_trans_p=0):
        for i in range(num_steps):
            # draw new mode?
            if random.random() < mode_trans_p:
                new_mode = self.draw_uniform()
                self.set_mode(new_mode)
            a1, a2 = self.get_mode(mode)
            # run system time step
            y2 = self.get_y1() + self.u1.rvs(size=1)[0]
            y1 = a1*self.get_y1() + a2*self.get_y2() + self.u2.rvs(size=1)[0]
            z = y1 + self.u3.rvs(size=1)[0]
            # save results
            self.res['y1'].append(y1)
            self.res['y2'].append(y2)
            self.res['z'].append(z)
            self.res['m'].append(self.cur_mode)

    def reset(self):
        self.res['y1'] = [0]
        self.res['y2'] = [0]
        self.res['z'] = [0]
        self.res['m'] = [0]
        self.set_mode(0)

    def print_trans_matrix(self):
        a = np.zeros((3,3))
        prev_m = 0
        #print(self.res['m'])
        for m in self.res['m']:
            a[prev_m, m] += 1
            prev_m = m
        print(a)


    def plot(self, title=None):
        if title:
            plt.title(title)
        prev_m = self.res['m'][0]
        start = 0
        for i,m in enumerate(self.res['m']):
            if prev_m != m:
                # transition
                X = [x for x in range(start,i,1)]
                Y = [self.res['z'][x] for x in X]
                c = colors[prev_m]
                plt.plot(X, Y, c=c)
                start = max(i-1,0)
                prev_m = m
        # once more, at the end
        X = [x for x in range(start,len(self.res['z']),1)]
        Y = [self.res['z'][x] for x in X]
        c = colors[prev_m]
        plt.plot(X, Y, c=c)
        #plt.plot(self.res['z'], c=c)

        plt.show()

####################################################################
#  MAIN
###################################################################
def main():
    ds = LinDynSys(init=0)
    #for mode in [0,1,2]:
    #    ds.reset()
    #    ds.run(500, mode=mode)
    #    ds.plot()
    ds.run(500, mode=0)
    ds.plot("Mode 0 Dynamical System")
    ds.reset()
    ds.run(1500, mode_trans_p=0.02)
    ds.print_trans_matrix()
    ds.plot()

if __name__ == "__main__":
    main()

