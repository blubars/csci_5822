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
import os
import sys
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
from scipy.stats import multivariate_normal
from numpy.linalg import inv
from bayes_opt import BayesianOptimization

import pdb

####################################################################
#       VARIABLES AND CONSTANTS
####################################################################
random.seed(12)

F_RANGE = [0, 10]

####################################################################
#  FUNCTION DEFINITIONS
####################################################################

def p1_function(x1, x2):
    X = np.array([x1, x2])
    y = 0
    y = x1 + 2*x2
    rv1 = multivariate_normal([3, 3], [[1.0, 0.3], [0.3, 0.5]])
    rv2 = multivariate_normal([6, 7], [[1.0, 0.5], [0.5, 1.0]])
    y += rv1.pdf(X)*80
    y += rv2.pdf(X)*100
    return y

def add_noise(y):
    noise = norm.rvs(size=1)
    return y + noise[0]

def plot_heatmap(X, Y, Z, filename=None):
    #plt.imshow(Z, cmap='hot', extent=[-2.5, 2.5, 2.5, -2.5])
    plt.imshow(Z, cmap='hot', extent=[X[0]-0.5, X[len(X)-1]+0.5, Y[len(Y)-1]+0.5, Y[0]-0.5])
    plt.xticks(X)
    plt.yticks(Y)
    plt.colorbar()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

## THIS FUNC CODE ADAPTED FROM BAYESOPT PACKAGE EXAMPLES:
## https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation%20vs%20exploration.ipynb
def plot_bayes_opt_contour(f, bo, title="BayesOpt", xlabel="X1", ylabel="X2", fname=None):
    plt.gcf().clear()
    #pdb.set_trace()
    #x1s = [x["x1"] for x in bo.res["all"]["params"]]
    #x2s = [x["x2"] for x in bo.res["all"]["params"]]
    #ys = bo.res["all"]["values"]

    X = np.linspace(0, 10, 21)
    Y = np.linspace(0, 10, 21)
    xv, yv = np.meshgrid(X, Y)
    z = [[xx, yy] for xx,yy in zip(np.ravel(xv), np.ravel(yv))]
    z_mean = bo.gp.predict(z)
    Z = z_mean.reshape(np.shape(xv))
    #mean, sigma = bo.gp.predict(np.arange(F_RANGE[1]).reshape(-1, 1), return_std=True)
    #plt.figure(figsize=(16, 9))
    cs = plt.contour(X, Y, Z, 15)
    plt.clabel(cs, inline=1, fontsize=9)
    norm = matplotlib.colors.Normalize(vmin=cs.vmin, vmax=cs.vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
    sm.set_array([])
    plt.colorbar(sm, ticks=cs.levels, shrink=0.8, extend='both')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(bo.X[:,0].flatten(), bo.X[:,1].flatten(), c="red", s=50, zorder=10)
    if fname:
        plt.savefig(fname)
    else:
        plt.show()

def plot_contour(func, title, xlabel, ylabel, vmin=None, fname=None):
    X = np.linspace(0, 10, 41)
    Y = np.linspace(0, 10, 41)
    xv, yv = np.meshgrid(X, Y)
    z = [func(xx,yy) for xx,yy in zip(np.ravel(xv), np.ravel(yv))]
    Z = np.array(z).reshape(np.shape(xv))
    cs = plt.contour(X, Y, Z, 20)
    plt.clabel(cs, inline=1, fontsize=9)
    norm = matplotlib.colors.Normalize(vmin=cs.vmin, vmax=cs.vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
    sm.set_array([])
    plt.colorbar(sm, ticks=cs.levels, shrink=0.8, extend='both')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    if fname:
        plt.savefig(fname)
    else:
        plt.show()

class GaussProc:
    def __init__(self, data_x=None, data_y = None):
        this.data_x = data_x
        this.data_y = data_y

    def add_affinity_func(func):
        this.affinity = func

    def add_data(X, Y):
        if not data_x:
            data_x = [X]
        else:
            data_x.append(X)
            data_y.append(Y)

    def get_posterior_mean(self, in_x):
        a = affinity(in_x, self.data_x)
        inv = inv(affinity(data_x, self.data_x))
        return a.dot(inv).dot(data_y)

    def get_posterior_var(self, in_x):
        a = affinity(in_x, in_x)
        b = affinity(in_x, data_x)
        c = affinity(data_x, in_x)
        inv = inv(affinity(data_x, data_x))
        return a - b.dot(inv).dot(inv).dot(c)

def f1(x1,x2):
    return add_noise(p1_function(x1,x2))

####################################################################
#  MAIN
###################################################################
if __name__ == "__main__":
    run = None
    fname = None
    if len(sys.argv) > 1:
        run = sys.argv[1]

    # acquistion functions to try
    acq_f = ["ucb", "ei"]
    # parameters for acq functions
    acq_p = [
    # upper confidence bound: 0 = exploitation, inf = expl.
                [10, 20],
    # expected improvement
                [1e-4, 0.1]]

    #plot_contour(f1, "Function", "X1", "X2")

    if run:
        try:
            os.mkdir(str(run))
        except FileExistsError:
            pass

    #gp = GaussProc()
    #gp.add_data(X, np.array(Y))

    for afi, af in enumerate(acq_f):
        for ap in acq_p[afi]:
            if run:
                fname = str(run) + '/bo' +str(run) + '_' + af + '_' + str(ap) + '.png'
            bo = BayesianOptimization(f1,
                   {'x1': (0, 10), 'x2': (0, 10)})
            gp_params = {'kernel':None, 'alpha':1e-5}
            bo.maximize(init_points=4, n_iter=8, xi=ap, kappa=ap, acq=af, **gp_params)

            # Finally, we take a look at the final results.
            print(bo.res['max'])
            print(bo.res['all'])

            plot_bayes_opt_contour(p1_function, bo, fname=fname)

