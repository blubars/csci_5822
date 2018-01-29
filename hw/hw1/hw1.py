#!/usr/bin/env python3

####################################################################
# FILE DESCRIPTION
####################################################################
# CSCI 5822 Assignment 1
#   1/28/2018
# ------------------------------------------------------------------
# The goal of this assignment is to give you a concrete 
# understanding of the Tenenbaum (1999) work by implementing and 
# experimenting with a small scale version of the model applied 
# to learning concepts in two-dimensional spaces.  The further 
# goal is to get hands-on experience representing and manipulating 
# probability distributions and using Bayes' rule.

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

import pdb

####################################################################
#       VARIABLES AND CONSTANTS
####################################################################
HYP_SPACE = [x for x in range(1, 11)]


####################################################################
#  FUNCTION DEFINITIONS
####################################################################
def is_consistent(ex, i):
    # return TRUE if example is consistent with i,
    # else return FALSE
    x, y = ex
    if (x <= i) and (x >= -i) and (y <= i) and (y >= -i):
        return True
    else:
        return False

def get_expected_size_prior_pmf(sigma):
    priors = [exp((2*x / sigma) * -2) for x in HYP_SPACE]
    return priors

def get_posterior_pmf(prior, X):
    # P(H|X)
    # we will get one probability for each hypothesis. 
    lik = get_likelihood_pmf(X)
    return normalize([p*l for p,l in zip(prior, lik)])

def get_posterior(y, X, prior):
    # P(y \in C | X) = sum_{h} P(X|h)*P(h)*P(y /in C|h)
    #                = sum_{y fits h} posterior
    prob = 0
    pmf = get_posterior_pmf(prior, X)
    for i,post in zip(HYP_SPACE, pmf):
        if is_consistent(y, i):
            prob += post
    return prob

def get_likelihood_pmf(X):
    return [get_likelihood_instance(X, i) for i in HYP_SPACE]

def get_likelihood_instance(X, i):
    # p(X | h) = { 1/|h|^n if all examples consistent with h
    #              0       if not }
    # here, |h| = size(h) = s_i * s_i = s_i^2 = (2*i)^2
    size = (2*i)**2
    for x in X:
        if not is_consistent(x, i):
            return 0
    return 1 / (size**len(X))

def normalize(distr):
    # sum, then divide each element by sum
    tot = 0
    for x in distr:
        tot += x
    return [x / tot for x in distr]

def plot_bar(stats, title=None, xlabel=None, ylabel=None):
    plt.bar(HYP_SPACE, stats)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()

def plot_contour(func, examples, title, xlabel, ylabel, vmin=None):
    #X = np.linspace(-10, 10, 21)
    #Y = np.linspace(-10, 10, 21)
    X = np.linspace(-10, 10, 81)
    Y = np.linspace(-10, 10, 81)
    xv, yv = np.meshgrid(X, Y)
    z = [func((xx,yy)) for xx,yy in zip(np.ravel(xv), np.ravel(yv))]
    Z = np.array(z).reshape(np.shape(xv))
    #im = plt.imshow(Z, interpolation='bilinear', 
    #                cmap=cm.gray)
    if vmin:
        cs = plt.contour(X, Y, Z, vmin=vmin, vmax=0)
    else:
        cs = plt.contour(X, Y, Z, vmax=0)

    norm = matplotlib.colors.Normalize(vmin=cs.vmin, vmax=cs.vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
    sm.set_array([])
    plt.colorbar(sm, ticks=cs.levels, shrink=0.8, extend='both')

    ex_x = [a for a,b in examples]
    ex_y = [b for a,b in examples]
    plt.scatter(ex_x, ex_y, marker='.', zorder=10, color='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return vmin

def print_nums(stats):
    print('[', end='')
    for i,x in enumerate(stats):
        if i == len(stats)-1:
            end = "]"
        else:
            end = ", "
        print("{0:.4f}".format(x), end=end)

####################################################################
#  MAIN
####################################################################
if __name__ == "__main__":
    # Task 1: bar graph of prior P(H)
    # calculate prior
    p6 = normalize(get_expected_size_prior_pmf(6))
    p12 = normalize(get_expected_size_prior_pmf(12))
    print("Prior, sigma=6:")
    print_nums(p6)
    print("Prior, sigma=12:")
    print_nums(p12)
    #plot_bar(p6, "Prior probability mass, sigma=6", "h_i", "Probability")
    #plot_bar(p12, "Prior probability mass, sigma=12", "h_i", "Probability")

    # Task 2:
    # Given one observation, X = {(1.5, 0.5)}, compute the 
    # posterior P(H|X) with 𝜎 = 12. You will get one probability 
    # for each possible hypothesis. Display your result either 
    # as a bar graph or a list of probabilities. Use Tenenbaum's 
    # Size Principle as the likelihood function.
    X = [(1.5, 0.5)]
    post = get_posterior_pmf(p12, X)
    #plot_bar(post, "Posterior PMF for X=[(1.5, 0.5)], sigma=12", "h_i", "Probability")

    # Task 3
    p10 = normalize(get_expected_size_prior_pmf(10))
    def meshfunc(y):
        return log(get_posterior(y, X, p10))

    #plot_contour(meshfunc, X, "Log of the probability of (x,y) in concept", "x", "y")

    # Task 4: Repeat Task 3 for X = {(4.5, 2.5)}.
    X = [(4.5, 2.5)]
    #plot_contour(meshfunc, X, "Log of the probability of (x,y) in concept", "x", "y")

    # Task 5: Compute generalization predictions, P(y|X), over 
    # the whole input space for 𝜎 = 30 and three different sets 
    # of input examples: X = {(2.2, -.2)}, X = {(2.2, -.2), 
    # (.5, .5)}, and X = {(2.2, -.2), (.5, .5), (1.5, 1)}. 
    # Describe how the posterior is changing as new examples 
    # are added, and explain why this occurs.
    p30 = normalize(get_expected_size_prior_pmf(30))
    X1 = [(2.2, -.2)]
    X2 = [(2.2, -.2), (.5, .5)]
    X3 = [(2.2, -.2), (.5, .5), (1.5, 1)]
    X = X1
    plot_contour(meshfunc, X, "Log probability of (x,y) in concept, 1 ex", "x", "y")
    X = X2
    plot_contour(meshfunc, X, "Log probability of (x,y) in concept, 2 exs", "x", "y")
    X = X3
    plot_contour(meshfunc, X, "Log probability of (x,y) in concept, 3 exs", "x", "y")



