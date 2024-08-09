# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.

from os import stat
from joblib import parallel_backend
import scipy.stats as stats
import math
import numpy as np
# from poibin import PoiBin
import time
# from numba import vectorize, njit, guvectorize,jit, cuda
# import computeamplification_HP_GDP


import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy as sp
import copy
# from gdp_to_dp import gdp_resolve_eps


# This document contains 4 computations: 2 empirical and 2 theoretical.
# 1. Empirical analysis
# 2. Theoretical analysis

def resolve_t(eps, mu, sigma, delta, n, w):
    # print(w)
    def t_func(t):
        i = np.array([k for k in range(0,n-1)])
        # wi_me = stats.norm.cdf((i+0.5-mu)/sigma)
        wi = stats.binom.pmf(i,n,mu/n)
        pi_top = stats.binom.pmf(np.maximum(np.floor(i+1-(i+1)/(t+1)),0) ,i , 0.5)
        pi_bottom = stats.binom.pmf(np.maximum(np.floor(i-(i+1)/(t+1)),0), i, 0.5)
        l = - np.sum(wi * pi_top)/np.sum(wi * pi_bottom)
        f =  (1-delta) * (-2*w + (1-2*w) * l - delta + np.exp(eps))
        if np.isnan(f):
            return 10000
        return f #,l,-2*w + (1-2*w) * l 
    # for xi in np.linspace(-100,20,10):
    #     yi= t_func(xi)
    #     print(xi,yi)
    # print(yi)
    a=1e-2
    b=100
    fa=t_func(a)
    fb=t_func(b)
    num=0
    while fb < 0 and num<100:
        b += 50
        num+=1
        fb=t_func(b) 
    if fb<0:
        # print('fb<0', fb)
        return -1 # false!
    num=0
    while fa > 0 and num < 100:
        # print(fa,fb)
        a += 0.05
        num += 1
        fa=t_func(a)
    if fa>0:
        # print('fa>0', fa)
        return 0.005
    num = 0
    while a<=b and num<100:
        x0=(a+b)/2
        fx0=t_func(x0)  
        # print(fa,fb,fx0)
        if fx0<1e-5 and fx0>=0:
            # print('t:',x0,' diff:', fx0,' diff<10e-10')
            return x0
        if fa*fx0<0:
            b=x0
            fb=fx0
            # print('解在左侧,a:',a,'  b:',b,'  x0:',x0)
        elif fb*fx0<0:
            a=x0
            fa=fx0
            # print('解在右侧,a:',a,'  b:',b,'  x0:',x0)
        num += 1
    if t_func(x0) < 0:
        return b
    return x0

# given eps calculate delta corollary 4.3
def delta_eps_func(eps, d1, mu, sigma, n, w):
    t = resolve_t(eps, mu, sigma, d1, n, w)
    # print('t:', t)
    i = np.array([k for k in range(0,n-1)])
    # wi = stats.norm.cdf((i+0.5-mu)/sigma)
    wi = stats.binom.pmf(i,n,mu/n)
    Fleft = stats.binom.cdf(np.maximum(np.floor(i-(i+1)/(t+1)),0), i , 0.5)
    Fright = stats.binom.cdf(np.maximum(np.floor(i+1-(i+1)/(t+1)),0), i , 0.5)
    f_delta = (-np.exp(eps)+ (1-d1)*2*w + d1) * np.sum(wi * Fleft) + (1-d1)*(1-2*w) * np.sum(wi * Fright)
    # print(eps, f_delta, Fleft, Fright)
    return f_delta #,l,-2*w + (1-2*w) * l 

# given delta calculate eps
def get_eps(epsupper, delta_s, delta_1, mu, sigma, n, w):
    a=0
    b=epsupper
    fa = delta_s - delta_eps_func(a, delta_1, mu, sigma, n, w)
    fb= delta_s - delta_eps_func(b, delta_1, mu, sigma, n, w)
    num=0
    while a<=b and num<100:
        x0 = (a+b)/2
        fx0=delta_s - delta_eps_func(x0, delta_1, mu, sigma, n, w)
        # print(fa,fb,fx0)
        if fx0<1e-12 and fx0>=0:
            # print('epsilon:',x0,' diff:', fx0,' diff<10e-10')
            return x0
        if fa*fx0<0:
            b=x0
            fb=fx0
            # print('解在左侧,a:',a,'  b:',b,'  x0:',x0)
        elif fb*fx0<0:
            a=x0
            fa=fx0
            # print('解在右侧,a:',a,'  b:',b,'  x0:',x0)
        num += 1
    fx0=delta_s - delta_eps_func(x0, delta_1, mu, sigma, n, w)
    if fx0 < 0:
        return b
    return x0    

# #if UL=1 then produces upper bound, else produces lower bound.
def numericalanalysis(n, epsorig, deltaorig, delta, num_iterations, step, upperbound, mech, C):
    '''
    Empirically computes the privacy guarantee of achieved by shuffling n eps0-DP local reports.
    num_iterations = number of steps of binary search, the larger this is, the more accurate the result
    If upperbound=True then this produces an upper bound on the true shuffled eps, and if upperbound=False then it produces a lower bound.
    '''
    # start = time.time()
    e1_idx = np.argmax(epsorig)
    pij_expectation, sigma, gamma, p1 = probHP(epsorig, deltaorig, e1_idx, mech, C)
    eps1 = epsorig[e1_idx]
    delta1 = deltaorig[e1_idx]
    # #mu-GDP version --CAO
    # mu = math.sqrt(2/ (pij_expectation- p1))
    # #transfer to DP
    # eps_central = gdp_resolve_eps(mu, delta)
    # return eps_central

    #fDP mixture
    eps = get_eps(eps1, delta, delta1, pij_expectation, sigma, n, p1)
    # delta = delta_eps_func(eps, delta1, pij_expectation, sigma, n, p1)
    return eps

# given eps, return delta
def numericalanalysis_delta(n, epsorig, deltaorig, eps, num_iterations, step, upperbound, mech, C):
    '''
    Empirically computes the privacy guarantee of achieved by shuffling n eps0-DP local reports.
    num_iterations = number of steps of binary search, the larger this is, the more accurate the result
    If upperbound=True then this produces an upper bound on the true shuffled eps, and if upperbound=False then it produces a lower bound.
    '''
    # start = time.time()
    e1_idx = np.argmax(epsorig)
    pij_expectation, sigma, gamma, p1 = probHP(epsorig, deltaorig, e1_idx, mech, C)
    eps1 = epsorig[e1_idx]
    delta1 = deltaorig[e1_idx]
    #fDP mixture
    delta = delta_eps_func(eps, delta1, pij_expectation, sigma, n, p1)
    return delta

# ========== pij HP ===========

def probHP(ei_arr, di_arr, e1_idx, mech, C):  
    n = len(ei_arr)
    
    e1 = ei_arr[e1_idx]
    d1 = di_arr[e1_idx]
    mu = 0
    sigma = 0
    gamma = 0
    for i in range(len(ei_arr)):
        ei = ei_arr[i]
        di = di_arr[i]
        if i == e1_idx:
            continue # x2~xn, ei!=e1
        #aaai version
        # pij = ei/ej * (1-np.exp(-ej))/(1-np.exp(-ei)) * np.exp(-np.maximum(ej, ei)) / n 
        # mu += np.sum(pij) #aaai version
        # sigma += np.sum(pij*(1-pij)) #aaai version
        #cao version
        # pij = 2 / (np.exp(ei)+1)
        # stronger
        # pij = 2 / (np.exp(e1)+1)
        # p1 = 1/(1+np.exp(e1))
        #cikm version
        pij, p1 = compute_pij(ei, di, e1, d1, mech, C)
        # print(pij)
        mu += np.sum(pij) #aaai version
        sigma += np.sum(pij*(1-pij)) #aaai version
        sigma = np.sqrt(sigma)
    return mu, sigma, gamma, p1


def compute_pij(ei, di, ej, dj, mech, C):
    mu_i = C
    mu_j = 0
    roots = []

    eps_1 = ej
    eps_2 = ej
    eps_3 = ei
    x_range = 20
    x = np.linspace(-x_range,x_range+0.1,10001)
    if mech == 'laplacian':
        b1 = np.abs(mu_j-mu_i)/eps_1
        b2 = np.abs(mu_j-mu_i)/eps_2
        b3 = np.abs(mu_j-mu_i)/eps_3

        cdf_1 = sp.stats.laplace.cdf(x,loc=mu_j, scale = b1) #x10
        cdf_2 = sp.stats.laplace.cdf(x,loc=mu_i, scale = b2) #x11
        cdf_3 = sp.stats.laplace.cdf(x,loc=mu_j, scale = b3) #xi
    if mech == 'gaussian':
        sigma_1 = 2*np.log(1.25/dj) * C**2 / ej**2 
        sigma_2 = 2*np.log(1.25/dj) * C**2 / ej**2 
        sigma_3 = 2*np.log(1.25/di) * C**2 / ei**2 

        cdf_1 = sp.stats.norm.cdf(x,loc=mu_j, scale = sigma_1) #x10
        cdf_2 = sp.stats.norm.cdf(x,loc=mu_i, scale = sigma_2) #x11
        cdf_3 = sp.stats.norm.cdf(x,loc=mu_j, scale = sigma_3) #xi
    cdf_roll_1 = np.roll(cdf_1,1)
    pmf_1 = cdf_1[1:] - cdf_roll_1[1:]
    cdf_roll_2 = np.roll(cdf_2,1)
    pmf_2 = cdf_2[1:] - cdf_roll_2[1:]
    cdf_roll_3 = np.roll(cdf_3,1)
    pmf_3 = cdf_3[1:] - cdf_roll_3[1:]

    # plt.switch_backend('agg')
    # fig = plt.figure()
    # plt.plot(x[1:],pmf_1)
    # plt.plot(x[1:],pmf_2)
    # plt.plot(x[1:],pmf_3)
    # plt.legend(['1','2','3'])
    # plt.xlim(-3,3)
    # plt.show()
    # plt.savefig('./pdf1.png', dpi=600)
    # plt.close()

    x1 = np.where(np.logical_and(np.greater(pmf_1,pmf_3),np.greater(pmf_1,pmf_2)))[0]
    x2 = np.where(np.logical_and(np.greater(pmf_2,pmf_3),np.greater(pmf_2,pmf_1)))[0]
    # p10 = np.sum(pmf_3[x1[0]:x1[-1]+1])
    # p11 = np.sum(pmf_3[x2[0]:x2[-1]+1])

    if mech == 'laplacian':
        if len(x1)>0:
            #xi looks like x10
            if np.abs(x[x1[0]] + x_range) < 0.1:
                p10 = sp.stats.laplace.cdf(x[x1[-1]], loc=mu_j, scale = b3)
            elif np.abs(x[x1[-1]] - x_range) < 0.1:
                p10 = 1 - sp.stats.laplace.cdf(x[x1[0]], loc=mu_j, scale = b3)
            else:
                p10 = sp.stats.laplace.cdf(x[x1[-1]], loc=mu_j, scale = b3) - sp.stats.laplace.cdf(x[x1[0]], loc=mu_j, scale = b3)
        else:
            p10=1
        if len(x2)>0:
            #xi looks like x11
            if np.abs(x[x2[0]] + x_range) < 0.1:
                p11 = sp.stats.laplace.cdf(x[x2[0]], loc=mu_j, scale = b3)
            elif np.abs(x[x2[-1]] - x_range) < 0.1:
                p11 = 1 - sp.stats.laplace.cdf(x[x2[0]], loc=mu_j, scale = b3)
            else: 
                p11 = sp.stats.laplace.cdf(x[x2[-1]], loc=mu_j, scale = b3) - sp.stats.laplace.cdf(x[x2[0]], loc=mu_j, scale = b3)
        else:
            p11=1
        p1 = sp.stats.laplace.cdf(C/2, loc=C, scale = b1) # 1/(1+e)
    if mech == 'gaussian':
        if len(x1)>0:
            #xi looks like x10
            if np.abs(x[x1[0]] + x_range) < 0.1:
                p10 = sp.stats.norm.cdf(x[x1[-1]], loc=mu_j, scale = sigma_3) - dj/2
            elif np.abs(x[x1[-1]] - x_range) < 0.1:
                p10 = 1 - sp.stats.norm.cdf(x[x1[0]], loc=mu_j, scale = sigma_3) - dj/2
            else:
                p10 = sp.stats.norm.cdf(x[x1[-1]], loc=mu_j, scale = sigma_3) - sp.stats.norm.cdf(x[x1[0]], loc=mu_j, scale = sigma_3)
        else:
            p10 = 1
        if len(x2)>0:
            #xi looks like x11
            if np.abs(x[x2[0]] + x_range) < 0.1:
                p11 = sp.stats.norm.cdf(x[x2[0]], loc=mu_j, scale = sigma_3) - dj/2
            elif np.abs(x[x2[-1]] - x_range) < 0.1:
                p11 = 1 - sp.stats.norm.cdf(x[x2[0]], loc=mu_j, scale = sigma_3) - dj/2
            else: 
                p11 = sp.stats.norm.cdf(x[x2[-1]], loc=mu_j, scale = sigma_3) - sp.stats.norm.cdf(x[x2[0]], loc=mu_j, scale = sigma_3)
        else:
            p11=1
        p1 = sp.stats.norm.cdf(C/2, loc=C, scale = sigma_1) - dj/2
    # print(x[x1[0]],x[x1[-1]],x[x2[0]],x[x2[-1]])
    # print(p10,p11,min(p10,p11))
    # if min(p10,p11)*2 >1:
    #     print(print(x[x1[0]],x[x1[-1]],x[x2[0]],x[x2[-1]]), p10, p11)
    return min(p10,p11)*2, p1


# ei = 1
# ej = 1
# di = 10**(-10)
# dj = 10**(-10)
# # mech = "laplacian"
# mech = "gaussian"
# C = 0.1

# pij = compute_pij(ei,di,ej,dj,mech,C)
# print('eps:',ei,ej)
# print('HP:',pij)

# # print('HP RR:', (1+math.e**(-ej))/(1+math.e**ei))
# # print('HP RR:', (1+math.e**(-ei))/(1+math.e**ej))
# print('hiding:', 1/(0+math.e**ej) )
# print('stronger:', 2/(1+math.e**ej) )
# print('generalized:', 2/(1+math.e**ei) )
# # # # plot_pdf(ei,di,ej,dj,mech,C, roots)

'''
l=0.1
r=1
n=200
eps0 = np.random.uniform(l, r, n)
delta0 = np.array([1e-10]*n)
eps1 = r
delta1 = 1e-10
P = [0,0]
Q = [0,0]
num_0_1 = [0,0]
num_0_1_baseline = 0
for i in range(n):
    print(i)
    p0, p1 = compute_pij(eps0[i],delta0[i],eps1,delta1,mech,C)
    p = 1/(1+math.e**eps1) 
    num_0_1[0] += p0
    num_0_1[1] += p1
    num_0_1_baseline += p
P = copy.deepcopy(num_0_1)
Q = copy.deepcopy(num_0_1)
P[0] += 1
Q[1] += 1
print('p0!=p1',P, Q, P[0]/Q[0], P[1]/Q[1])

P = [min(num_0_1), min(num_0_1)]
Q = [min(num_0_1), min(num_0_1)]
P[0] += 1
Q[1] += 1
print('p0=p1', P, Q, P[0]/Q[0], P[1]/Q[1])

P = [num_0_1_baseline, num_0_1_baseline]
Q = [num_0_1_baseline, num_0_1_baseline]
P[0] += 1
Q[1] += 1
print('stronger', P, Q, P[0]/Q[0], P[1]/Q[1])
'''
# n=20000
# l=0.05
# r=1
# delta_l = 0
# delta = 1e-6
# num_iterations = 10
# step = 100
# upperbound=True
# # mech="laplacian"
# # delta_l=0
# mech="gaussian"
# delta_l=1e-8
# C=0.1
# epsorig = np.random.uniform(l, r, n)
# deltaorig = np.array([delta_l]*n)
# re = numericalanalysis(n, epsorig, deltaorig, delta, num_iterations, step, upperbound, mech, C)
# print(re)
# re_gdp = computeamplification_HP_GDP.numericalanalysis(n, epsorig, deltaorig, delta, num_iterations, step, upperbound, mech, C)
# print(re_gdp)