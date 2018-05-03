#!/usr/bin/env python3

####################################################################
# FILE DESCRIPTION
####################################################################
# CSCI 5822 Assignment 0: Naive Bayes
#    01/23/2018
#    Brian Lubars
# ------------------------------------------------------------------

####################################################################
#  IMPORTS
####################################################################
import argparse
import random
from collections import defaultdict
from collections import namedtuple
from math import log

import pdb

####################################################################
#  GLOBAL VARS & CONSTANTS
####################################################################
FILENAME = "titanic.txt"

CLASS = ['1st', '2nd', '3rd', 'crew']
AGE = ['adult', 'child']
GENDER = ['male', 'female']
SURVIVE = ['yes', 'no']

joint_counts = [[[[0 for a in SURVIVE] for b in GENDER] for c in AGE] for d in CLASS]
total_count = 0

def make_joints(entries):
    c_dict = {y:x for x,y in enumerate(CLASS)}
    a_dict = {y:x for x,y in enumerate(AGE)}
    g_dict = {y:x for x,y in enumerate(GENDER)}
    s_dict = {y:x for x,y in enumerate(SURVIVE)}
    global total_count
    for entry in entries:
        if entry.isspace():
            continue
        c,a,g,s = entry.split()
        joint_counts[c_dict[c]][a_dict[a]][g_dict[g]][s_dict[s]] += 1
        total_count += 1
    
def print_joints():
    for ci, c in enumerate(CLASS):
        for ai, a in enumerate(AGE):
            for gi, g in enumerate(GENDER):
                for si, s in enumerate(SURVIVE):
                    print("{}\t{}\t{}\t{}:\t{}".format(c,a,g,s, get_joint(ci,ai,gi,si)))
    print("Total entries: {}".format(total_count))
    
def get_joint(cls, age, gen, surv):
    return joint_counts[cls][age][gen][surv]
    
# empirical probability of death, from joint probs
def get_cond_prob_death_emp(cls, age, gen):
    surv, die = joint_counts[cls][age][gen]
    if die == 0:
        return 0
    else:
        return die / (surv + die)

# unconditional prob of death
def get_prob_death():
    death_cnt = 0
    for ci, c in enumerate(CLASS):
        for ai, a in enumerate(AGE):
            for gi, g in enumerate(GENDER):
                death_cnt += get_joint(ci, ai, gi, 1)
    return death_cnt / total_count
    
# conditional probabilities: marginalize over joints
def get_nb_cond_prob(rand_var, value, survive=1):
    joint = 0
    if rand_var is CLASS:
        # marginalize over other vars
        for ai, a in enumerate(AGE):
            for gi, g in enumerate(GENDER):
                joint += get_joint(value, ai, gi, survive)
    elif rand_var is AGE:
        # marginalize over other vars
        for ci, c in enumerate(CLASS):
            for gi, g in enumerate(GENDER):
                joint += get_joint(ci, value, gi, survive)
    elif rand_var is GENDER:
        # marginalize over other vars
        for ci, c in enumerate(CLASS):
            for ai, a in enumerate(AGE):
                joint += get_joint(ci, ai, value, survive)
    else:
        print("ERROR!")
    if survive == 1:
        denom_prob = get_prob_death()
    else:
        denom_prob = 1 - get_prob_death()
    return (joint / total_count) / denom_prob
    
def print_nb_cond_prob(rand_var, survive=1):
    if rand_var is CLASS:
        s1 = "Class"
    elif rand_var is GENDER:
        s1 = "Gender"
    else: 
        s1 = "Age"
    if survive == 1:
        s2 = "death"
    else:
        s2 = "survive"
    print("\nPr({}|{}):".format(s1, s2))
    for x in rand_var:
        print("{}".format(x), end='\t')
    print("")
    sum = 0
    for xi, x in enumerate(rand_var):
        p = get_nb_cond_prob(rand_var, xi, survive)
        print("{}".format(p), end='\t')
        sum += p
    print("\nsum:{}".format(sum))
    
def get_cond_prob_death_nb(cls, gen, age):
    rand_vars = [CLASS, GENDER, AGE]
    values = [cls, gen, age]
    # P(death|x) = PROD_i P(x_i|death) * P(death) /
    #     [PROD_i P(x_i|death)*P(death) + PROD_i P(x_i|surv)*P(surv)
    # death probs
    cond_death_probs = [get_nb_cond_prob(XX, xx, 1) for XX, xx in zip(rand_vars, values)]
    p_x_giv_death = 1
    for p in cond_death_probs:
        p_x_giv_death = p_x_giv_death * p
    # survival probs
    cond_surv_probs = [get_nb_cond_prob(XX, xx, 0) for XX, xx in zip(rand_vars, values)]
    p_x_giv_surv = 1
    for p in cond_surv_probs:
        p_x_giv_surv = p_x_giv_surv * p
    p_death = get_prob_death()
    
    # multiply likelihood by priors
    p_x_and_death = p_x_giv_death * p_death
    p_x_and_surv = p_x_giv_surv * (1 - p_death)
    
    # return final probability
    # sanity check
    #print((p_x_and_death + p_x_and_surv))
    return p_x_and_death / (p_x_and_death + p_x_and_surv)
    
    
if __name__ == "__main__":
    f = open(FILENAME, 'r')
    make_joints(list(f))
    f.close()
    
    # task 0: joint prob table
    print_joints()
    
    # task 1: Pr(death | gender, age, class)
    for ci, c in enumerate(CLASS):
        for gi, g in enumerate(GENDER):
            for ai, a in enumerate(AGE):
                p = get_cond_prob_death_emp(ci, ai, gi)
                print("{}\t{}\t{}:\t{}".format(c, g, a, p))
                
    # task 2: naive bayes classifier
    vars = [CLASS, GENDER, AGE]
    #print(get_nb_cond_prob(CLASS, 0))
    for v in vars:
        print_nb_cond_prob(v)
        
    for v in vars:
        print_nb_cond_prob(v, 0)
    
    print("\nProb death:{}\n".format(get_prob_death()))
    
    # task 2, cond't: Pr(death | gender, age, class) w/ Naive Bayes
    print("Conditional probability table: P(death|X) using Naive Bayes")
    for ci, c in enumerate(CLASS):
        for gi, g in enumerate(GENDER):
            for ai, a in enumerate(AGE):
                p = get_cond_prob_death_nb(ci, gi, ai)
                print("{}\t{}\t{}:\t{}".format(c, g, a, p))

