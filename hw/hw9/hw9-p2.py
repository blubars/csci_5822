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
    [1, 0],
    [0.07,0.9],
    [0.1,0.1] ]

colors = ['b', 'r', 'g']

SIGMA = 0.25

####################################################################
#  FUNCTION DEFINITIONS
####################################################################
class LinDynSys:
    def __init__(self, init=0, sigma=SIGMA):
        #self.u1 = norm()
        #self.u2 = norm()
        #self.u3 = norm()
        #u2 = norm(loc=0, scale=sigma^2)
        #u3 = norm(loc=0, scale=sigma^2)
        self.cur_mode = 0
        self.sigma = sigma
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
            #y2 = self.get_y1() + self.u1.rvs(size=1)[0]
            y2 = self.get_y1() + np.random.normal(0, self.sigma)
            y1 = a1*self.get_y1() + a2*self.get_y2() + np.random.normal(0, self.sigma)
            z = y1 + np.random.normal(0, self.sigma)
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
        #print(self.res['y1'][0:10])
        #print(self.res['y2'][0:10])
        #X = [x for x in range(len(self.res['z']))]
        #Y = [self.res['y1'][x] for x in X]
        #plt.plot(X, Y, c="black")
        #Y = [self.res['y2'][x] for x in X]
        #plt.plot(X, Y, c="black")
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
        
    def plot_kalman(self):
        X = [x for x in range(len(self.res['z']))]
        Y = [self.res['z'][x] for x in X]
        plt.plot(X, Y, c="black", label="Observations")
        Y = np.array([self.x_hat[x] for x in X])
        s = np.array([self.p[x] for x in X])
        s = np.sqrt(s) * 2

        plt.fill_between(X, Y-s, Y+s, facecolor="silver")
        plt.plot(X, Y, c="g", label="Filter state (y1)")
        plt.legend()
        plt.show()

    def run_kalman_filter_1d(self, steps):
        Z = self.res['z']
        a1, a2 = self.get_mode(self.cur_mode)
        F = 1 #max(a1, a2)
        Q = 0.04 #self.sigma
        H = 1
        R = 0.2 #1 #self.sigma
        self.x_hat = np.zeros(steps+1)
        self.p = np.zeros(steps+1)
        self.p[0] = 1
        for i in range(1, steps+1, 1):
            self.kalman_filter_1d(Z[i], i, F, Q, H, R)
            
    def run_kalman_filter(self, steps):
        Z = self.res['z']
        #F = np.array([[self.a1, self.a2],[1, 0]])
        #Q = np.array([[self.sigma**2, 0], [0, self.sigma**2]])
        #H = np.array([[0, 1]])
        #R = self.sigma**2
        a1, a2 = self.get_mode(self.cur_mode)
        F = 1
        Q = self.sigma
        H = 1
        R = 1 #self.sigma
        #self.x_hat = np.zeros((steps, 2))
        self.x_hat = np.zeros(steps+1)
        self.p = np.zeros(steps+1)
        for i in range(1, steps+1, 1):
            self.kalman_filter_1d(Z[i], i, F, Q, H, R)

    def kalman_filter(self, z, k, F, Q, H, R):
        # prediction update
        x_hat = F.dot(self.x_hat[k - 1])         # mean of predicted X_k
        p_prior = F.dot(self.p[k - 1]).dot(F.T) + Q  # predicted (prior) estimate covariance
        # measurement update
        y_k = z_k - H.dot(x_hat)           # residual measurement
        S = H.dot(P).dot(H.T) + R             # residual covariance
        K = p_prior.dot(H.T).dot(S.inv)       # optimal kalman gain
        x_post = x_hat + K.dot(y_k)        # update a posterior estimate
        p_post = (np.identity(2) - K.dot(H)).dot(p_prior)  # estimate covariance
        
        # store
        self.x_hat[k] = x_post
        self.p[k] = p_post
        
    def kalman_filter_1d(self, z, k, F, Q, H, R):
        # prediction update
        x_hat = F * self.x_hat[k - 1]         # mean of predicted X_k
        p_prior = F * self.p[k - 1] * F + Q  # predicted (prior) estimate covariance
        # measurement update
        y_k = z - H * x_hat           # residual measurement
        S = H * p_prior * H + R       # residual covariance
        K = p_prior * H * (1/S)       # optimal kalman gain
        print("K:{}, S:{}, p_prior:{}".format(K,S,p_prior)) 
        x_post = x_hat + K * y_k        # update a posterior estimate
        p_post = (1 - K * H) * p_prior  # estimate covariance
        # store filter results
        self.x_hat[k] = x_post
        self.p[k] = p_post


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
    ds.run_kalman_filter_1d(500)
    ds.plot_kalman()
    ds.reset()
    ds.run(1500, mode_trans_p=0.02)
    ds.print_trans_matrix()
    ds.plot()
    
 

if __name__ == "__main__":
    main()

