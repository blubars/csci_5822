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
from matplotlib import gridspec
import pprint
import random
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import multivariate_normal
from numpy.linalg import inv
from bayes_opt import BayesianOptimization
from bayes_opt.helpers import acq_max
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, PairwiseKernel

import pdb

####################################################################
#       VARIABLES AND CONSTANTS
####################################################################
random.seed(12)

F_RANGE = [0, 10]
MAX_ITERATIONS = 12
TRUE_MAX = np.array([6.00, 7.00])

####################################################################
#  FUNCTION DEFINITIONS
####################################################################

def p1_function(x1, x2):
    X = np.array([x1, x2])
    #y = 0.5*x1 + 1*x2
    #y = -((x1-5)*(x2-5))*.2 + 5
    y = -((x1-4)**2 + (x2-6)**2) * .1 + 6
    rv1 = multivariate_normal([3, 3], [[2.0, 0.6], [0.6, 1.0]])
    rv2 = multivariate_normal([6, 7], [[1.0, -0.5], [-0.5, 1.0]])
    y += rv1.pdf(X)*60
    y += rv2.pdf(X)*80
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

def gp_posterior(bo, X):
    #bo.gp.fit(bo.X, bo.Y)
    mean, sigma = bo.gp.predict(X, return_std=True)
    return mean, sigma

## THIS FUNC CODE ADAPTED FROM BAYESOPT PACKAGE EXAMPLES:
## https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation%20vs%20exploration.ipynb
def plot_bayes_opt_contour(f, bo, title="BayesOpt", xlabel="X1", ylabel="X2", fname=None):
    #plt.gcf().clear()
    #pdb.set_trace()
    #x1s = [x["x1"] for x in bo.res["all"]["params"]]
    #x2s = [x["x2"] for x in bo.res["all"]["params"]]
    #ys = bo.res["all"]["values"]
    fig = plt.figure(figsize=(10,4))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    fig.suptitle(title, size=10)
    #gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(top=0.9)

    X = np.linspace(0, 10, 21)
    Y = np.linspace(0, 10, 21)
    xv, yv = np.meshgrid(X, Y)
    z = [[xx, yy] for xx,yy in zip(np.ravel(xv), np.ravel(yv))]
    z_mean, z_sigma = gp_posterior(bo, z)
    Z = z_mean.reshape(np.shape(xv))
    #Zconf = z_mean + z_sigma
    Zconf = z_sigma
    Zconf = Zconf.reshape(np.shape(xv))
    #mean, sigma = bo.gp.predict(np.arange(F_RANGE[1]).reshape(-1, 1), return_std=True)
    #plt.figure(figsize=(16, 9))
    cs1 = ax1.contour(X, Y, Z, 15)
    ax1.clabel(cs1, inline=1, fontsize=9)
    norm = matplotlib.colors.Normalize(vmin=cs1.vmin, vmax=cs1.vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cs1.cmap)
    sm.set_array([])
    fig.colorbar(sm, ticks=cs1.levels, shrink=0.8, extend='both', ax=ax1)
    ax1.set_title("Posterior")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.scatter(bo.X[:,0].flatten(), bo.X[:,1].flatten(), c="red", s=40, zorder=10)

    cs = ax2.contour(X, Y, Zconf, 15)
    ax2.clabel(cs, inline=1, fontsize=9)
    norm = matplotlib.colors.Normalize(vmin=cs.vmin, vmax=cs.vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cs.cmap)
    sm.set_array([])
    fig.colorbar(sm, ticks=cs.levels, shrink=0.8, extend='both', ax=ax2)
    ax2.set_title("Sigma")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.scatter(bo.X[:,0].flatten(), bo.X[:,1].flatten(), c="red", s=40, zorder=10)

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
    cs = plt.contour(X, Y, Z, 10)
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
    #return p1_function(x1,x2)
    return add_noise(p1_function(x1,x2))

####################################################################
#  MAIN
###################################################################
def main():
    run = None
    fname = None
    if len(sys.argv) > 1:
        run = sys.argv[1]

    # acquistion functions to try
    #acq_f = ["ei", "ucb"]
    acq_f = ["ucb"]
    # parameters for acq functions
    acq_p = [
    # expected improvement
                #[0.001, 0.01],
    # upper confidence bound: 0 = exploitation, inf = expl.
                [5, 10]]

    #plot_contour(f1, "Function", "X1", "X2")

    if run:
        try:
            os.mkdir(str(run))
        except FileExistsError:
            pass

    for afi, af in enumerate(acq_f):
        for ap in acq_p[afi]:
            title = ""
            if run:
                fname = str(run) + '/bo' +str(run) + '_' + af + '_' + str(ap) + '.png'
                title = "(Acq func=" + af + ", hyperparam=" + str(ap) + ')'

            # KERNELS: convariance
            # constant: adjust mean; RBF; WhiteKernel: estimate noise
            #kernel = ConstantKernel(constant_value=1e-5, constant_value_bounds=(1e-05, 100.0)) + ConstantKernel(constant_value=1e-5, constant_value_bounds=(1e-05, 10.0)) * RBF(length_scale=1.0, length_scale_bounds=(1e-05,10)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-05,1.0))
            #kernel = PairwiseKernel(gamma=1.0, gamma_bounds=(1e-5,10.0), metric="rbf")
            #kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-05, 10.0)) * RBF(length_scale=1.0, length_scale_bounds=(1e-05,10)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-05,1.0))
            kernel = ConstantKernel(constant_value=5.0, constant_value_bounds=(1e-05, 10.0)) * \
                        RBF(length_scale=1.0, length_scale_bounds=(0.5,2)) + \
                        WhiteKernel(noise_level=1e-05, noise_level_bounds=(1e-05,1.0))
            #kernel = ConstantKernel(constant_value_bounds=(1e-05, 10.0)) + RBF(length_scale=1.0, length_scale_bounds=(1e-05,10)) + WhiteKernel(noise_level=1e-05, noise_level_bounds=(1e-05,10.0))

            bo = BayesianOptimization(f1,
                   {'x1': (0, 10), 'x2': (0, 10)})
            #gp_params = {'kernel':kernel, 'alpha':0.1}
            gp_params = {'kernel':kernel}
            bo.maximize(init_points=4, n_iter=4, xi=ap, kappa=ap, acq=af, **gp_params)
            iters = 4
            for i in range(4, MAX_ITERATIONS, 1):
                # stop if we found the max
                maxp = bo.res['max']['max_params']
                maxa = np.array([maxp['x1'], maxp['x2']])
                if np.isclose(maxa, TRUE_MAX, atol=0.1).all():
                    break
                #prev_point = utility
                #y_max = bo.space.Y.max()
                #x_max = acq_max(ac=bo.util.utility,
                #            gp=bo.gp,
                #            y_max=y_max,
                #            bounds=bo.space.bounds,
                #            random_state=bo.random_state,
                #            **bo._acqkw)
                #print("MAX: {}".format(x_max))
                bo.maximize(init_points=0, n_iter=1, xi=ap, kappa=ap, acq=af)
                # converged if draws 2 points in same place
                iters += 1
            print("CONVERGED IN {} ITERATIONS".format(iters))
            maxp = bo.res['max']['max_params']
            maxa = np.array([maxp['x1'], maxp['x2']])
            print("MAX FOUND: {}\n".format(maxa))

            print("KERNEL PARAMS:")
            print(bo.gp.kernel_)
            print("\n")

            # Finally, we take a look at the final results.
            print(bo.res['max'])
            print(bo.res['all'])

            plot_bayes_opt_contour(p1_function, bo, fname=fname, title=title)

if __name__ == "__main__":
    main()


