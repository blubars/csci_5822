#!/usr/bin/env python3

####################################################################
# FILE DESCRIPTION
####################################################################
# CSCI 5822 Assignment 5
#   3/7/2018
# ------------------------------------------------------------------
# This assignment is to get experience running various approximate
# inference schemes using sampling, as well as to better understand
# Bayes nets and continuous probability densities.

####################################################################
#  IMPORTS
####################################################################
from math import log, exp
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.mlab as mlab
#import matplotlib.colors
import pprint
import random
from scipy.stats import gamma
from scipy.stats import norm

import pdb

####################################################################
#       VARIABLES AND CONSTANTS
####################################################################
random.seed(12)

NUM_SAMPLES = 100000

univ_dict = {1:"CU", 0:"Metro"}
major_dict = {1:"CompSci", 0:"Business"}

####################################################################
#  FUNCTION DEFINITIONS
####################################################################

####################################################################
#  PART 1
###################################################################
def sample_intelligence():
    # normal distribution: mean=100, SD=15
    x = random.gauss(100, 15)
    return x

def sample_university(intelligence):
    # modeled by sigmoid function: P(U=CU|I) = 1 / (1 + e^[-(I-100)/5])
    p = 1 / (1 + exp(-(intelligence - 100) / 5))
    # get uniform sample [0,1]. if below p, return 1, else 0
    x = random.random()
    if x < p:
        return 1
    else:
        return 0

def sample_major(intelligence):
    # modeled by sigmoid function: P(U=CU|I) = 1 / (1 + e^[-(I-100)/5])
    p = 1 / (1 + exp(-(intelligence - 110) / 5))
    # get uniform sample [0,1]. if below p, return 1, else 0
    x = random.random()
    if x < p:
        return 1
    else:
        return 0

def sample_salary(i, u, m):
    # Salary ~ Gamma(0.1 * Intelligence + (Major==compsci) + 3 * (University==ucolo), 5)
    a = 0.1 * i + (m == 1) + 3 * (u == 1)
    b = 5
    #dist = gamma(a, scale=b)
    return gamma.rvs(a, scale=b)

def p_i(i):
    # probability of I=i
    return norm.pdf(i, loc=100, scale=15)

def p_u_given_i(u, i):
    # P(U=u|I=i)
    p = 1 / (1 + exp(-(i - 100) / 5))
    if u == 1:
        # CU
        return p
    else:
        return 1 - p

def p_m_given_i(m, i):
    # P(M=m|I=i)
    p = 1 / (1 + exp(-(i - 110) / 5))
    if m == 1:
        # CS
        return p
    else:
        return 1 - p

def p_s_given_imu(s, i, m, u):
    # P(S=s|I=i,U=u,M=m)
    a = 0.1 * i + (m == 1) + 3 * (u == 1)
    b = 5
    return gamma.pdf(s, a, scale=b)

def RV_to_string(rv, value):
    print(rv + "=")
    if rv == "I":
        print(value)
    elif rv == "U":
        print(univ_dict(value))
    elif rv == "M":
        print(major_dict(value))
    else:
        print('$' + str(value) + "k")

def sample_unif():
    i = random.random()*100 + 50
    u = random.random()
    if u < 0.5:
        u = 0
    else:
        u = 1
    m = random.random()
    if m < 0.5:
        m = 0
    else:
        m = 1
    return i, u, m

def estimate_p_u_m_given_s(s, n_samples=NUM_SAMPLES, plot=False):
    print("Estimating P(U,M|S={}) with {} samples:".format(s,n_samples))
    samples = []
    weights = np.zeros(n_samples)
    for run in range(n_samples):
        i = sample_intelligence()
        u = sample_university(i)
        m = sample_major(i)
        q = p_i(i) * p_u_given_i(u, i) * p_m_given_i(m, i)
        p = q * p_s_given_imu(s, i, m, u)
        w = p / q

        # version using uniform sampling
        #i, u, m = sample_unif()
        #p = p_i(i) * p_u_given_i(u, i) * p_m_given_i(m, i)* p_s_given_imu(s, i, m, u)
        #w = p

        samples.append((i,u,m,s))
        weights[run] = w
    # normalize weights
    w_sum = np.sum(weights)
    weights /= w_sum
    Z = np.zeros((2,2))
    for samp, w in zip(samples, weights):
        #print("u:{}, m:{}, w:{}".format(u,m,w))
        u = samp[1]
        m = samp[2]
        Z[u,m] += w
    #plt.contourf([0,1],[0,1],Z)
    #print(Z)
    for u in range(2):
        for m in range(2):
            print("[{},\t{}]:\t{}".format(
                univ_dict[u], major_dict[m], Z[u,m]))
    if plot:
        plt.imshow(Z, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

####################################################################
#  PART 2
###################################################################
class GausRV:
    def __init__(self):
        self.p = 0


####################################################################
#  MAIN
###################################################################
if __name__ == "__main__":
    # PART 1: ------------------------------------------
    #   estimate posterior: P(U,M|S) = P(U,I,M)/P(S)
    #   ~ P(I)P(U|I)P(M|I)P(S=s|I,U,M)
    #   sample I, U, M; 
    #   use to estimate posterior w/ importance sampling
    if 0:
        estimate_p_u_m_given_s(120)
        estimate_p_u_m_given_s(60)
        estimate_p_u_m_given_s(20)

    # PART 2: ------------------------------------------
    if 1:
        num_bins = 32
        burn_in = 100
        burn_in 
        for



