# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.

import scipy.stats as stats
import math
import numpy as np

# This document contains 4 computations: 2 empirical and 2 theoretical.
# 1. Empirical analysis
# 2. Theoretical analysis

# ========= SUPPORT FUNCTIONS ==========
# Algo4
# This function uses binary search to approximate the smallest eps such that deltacomp will output something smaller than delta (i.e. an algorithm is (eps, delta)-DP)
def binarysearch(deltacomp, delta, num_iterations, epsupper):
    '''
    binary search to find min epsilon such that deltacomp(epsilon)<delta
    deltacomp = function that takes epsilon as input and outputs delta
    num_iterations = number of iterations, accuracy is 2^(-num_iterations)*epsupper
    epsupper = upper bound for epsilon. You should be sure that deltacomp(epsupper)<delta.
    '''
    llim = 0
    rlim = epsupper
    for t in range(num_iterations):
        mideps = (rlim + llim) / 2
        delta_for_mideps = deltacomp(mideps, delta)
        if delta_for_mideps < delta:
            llim = llim
            rlim = mideps
        else:
            llim = mideps
            rlim = rlim
    return rlim

# ================/EXACT EMPIRICAL ANALYSIS WITH STEPS - SAMPLING EMPIRICAL/==============

#This a subroutine in the main algorithm5.
def onestep(c, eps, eps0, pminusq):
    '''
    onestep computes the e^(eps)-divergence between p=alpha*Bin(c,0.5)+(1-alpha)*(Bin(c,1/2)+1) and q=alpha*(Bin(c,0.5)+1)+(1-alpha)*Bin(c,1/2), where alpha=e^(eps)/(1+e^(eps))
    if pminusq=True then computes D_(e^eps)(p|q), else computes D_(e^eps)(q|p)
    '''
    alpha = math.exp(eps0) / (math.exp(eps0) + 1) #q，depends on R(x1)
    effeps = math.log(((math.exp(eps) + 1) * alpha - 1) / ((1 + math.exp(eps)) * alpha - math.exp(eps)))
    if pminusq == True:
        beta = 1 / (math.exp(effeps) + 1)
    else:
        beta = 1 / (math.exp(-effeps) + 1)
    cutoff = beta * (c + 1)
    pconditionedonc = (alpha * stats.binom.cdf(cutoff, c, 0.5) + (1 - alpha) * stats.binom.cdf(cutoff - 1, c, 0.5))
    qconditionedonc = ((1 - alpha) * stats.binom.cdf(cutoff, c, 0.5) + alpha * stats.binom.cdf(cutoff - 1, c, 0.5))
    if pminusq == True:
        return (pconditionedonc - math.exp(eps) * qconditionedonc)
    else:
        return ((1 - qconditionedonc) - math.exp(eps) * (1 - pconditionedonc))


def deltacomp(n, eps0, eps, deltaupper, step, upperbound = True):
    '''
    Let C=Bin(n-1, e^(-eps0)) and A=Bin(c,1/2) and B=Bin(c,1/2)+1 and alpha=e^(eps0)/(e^(eps0)+1)
    p samples from A w.p. alpha and B otherwise
    q samples from B w.p. alpha and A otherwise
    deltacomp attempts to find the smallest delta such P and Q are (eps,delta)-indistinguishable, or outputs deltaupper if P and Q are not (eps, deltaupper)-indistinguishable.
    If upperbound=True then this produces an upper bound on the true delta (except if it exceeds deltaupper), and if upperbound=False then it produces a lower bound.
    '''
    deltap = 0  # this keeps track of int max{0, p(x)-q(x)} dx
    deltaq = 0  # this keeps track of int max{0, q(x)-p(x)} dx
    probused = 0  # To increase efficiency, we're only to search over a subset of the c values.
    # This will keep track of what probability mass we have covered so far.

    # p = math.exp(-eps0) #hiding among clones FMT21
    p = 2/(math.exp(eps0)+1) #stronger FV23
    expectation = (n-1)*p
    # Now, we are going to iterate over the n/2, n/2-step, n/2+step, n/2-2*steps, ...
    for B in range(1, int(np.ceil(n/step)), 1):
        for s in range(2):
            if s == 0:
                if B==1:
                    upperc = int(np.ceil(expectation+B*step))  # This is stepping up by "step". 从c的期望附近开始查找
                    lowerc = upperc - step
                else:
                    upperc = int(np.ceil(expectation + B * step))  # This is stepping up by "step".
                    lowerc = upperc - step + 1
                if lowerc>n-1: #判断lowerc是否超过最大实验成功次数n-1，没有超过就可以计算
                    inscope = False
                else:
                    inscope = True
                    upperc = min(upperc, n-1)
            if s == 1:
                lowerc = int(np.ceil(expectation-B*step))
                upperc = lowerc + step - 1
                if upperc<0:
                    inscope = False
                else:
                    inscope = True
                    lowerc = max(0, lowerc)

            if inscope == True: #lowerc是合法的，求C的积分
                cdfinterval = stats.binom.cdf(upperc, n - 1, p) - stats.binom.cdf(lowerc, n - 1, p) + stats.binom.pmf(lowerc, n - 1, p) # why又个pdf在这？
            # This is the probability mass in the interval (in Bin(n-1, p))

                if max(deltap, deltaq) > deltaupper:
                    return deltaupper

                if 1 - probused < deltap and 1 - probused < deltaq:
                    if upperbound == True:
                        return max(deltap + 1 - probused, deltaq + 1 - probused)
                    else:
                        return max(deltap, deltaq)

                else:
                    deltap_upperc = onestep(upperc, eps, eps0, True) #calculate deltaP deltaQ
                    deltap_lowerc = onestep(lowerc, eps, eps0, True)
                    deltaq_upperc = onestep(upperc, eps, eps0, False)
                    deltaq_lowerc = onestep(lowerc, eps, eps0, False)

                    if upperbound == True:
                        # compute the maximum contribution to delta in the segment.
                        # The max occurs at the end points of the interval due to monotonicity
                        deltapadd = max(deltap_upperc, deltap_lowerc)
                        deltaqadd = max(deltaq_upperc, deltaq_upperc)
                    else:
                        deltapadd = min(deltap_upperc, deltap_lowerc)
                        deltaqadd = min(deltaq_upperc, deltaq_lowerc)

                    deltap = deltap + cdfinterval * deltapadd
                    deltaq = deltaq + cdfinterval * deltaqadd

                probused = probused + cdfinterval  # updates the mass of C covered so far

    return max(deltap, deltaq)


# #if UL=1 then produces upper bound, else produces lower bound.
def numericalanalysis_step(n, epsorig, delta, num_iterations, step, upperbound, clip_bound=None):
    '''
    Empirically computes the privacy guarantee of achieved by shuffling n eps0-DP local reports.
    num_iterations = number of steps of binary search, the larger this is, the more accurate the result
    If upperbound=True then this produces an upper bound on the true shuffled eps, and if upperbound=False then it produces a lower bound.
    '''
    # in order to speed things up a bit, we start the search for epsilon off at the theoretical upper bound.
    # if epsorig < math.log(n / (16 * math.log(4 / delta))):#hiding
    if epsorig < math.log(n / (8 * math.log(2 / delta))):#stronger
        # checks if this is a valid parameter regime for the theoretical analysis.
        # If yes, uses the theoretical upper bound as a starting point for binary search
        epsupper = closedformanalysis(n, epsorig, delta)
    else:
        epsupper = epsorig

    def deltacompinst(eps, delta):
        return deltacomp(n, epsorig, eps, delta, step, upperbound)

    return binarysearch(deltacompinst, delta, num_iterations, epsupper)

def numericalanalysis(n, epsorig, deltaorig, delta, num_iterations, step, upperbound, clip_bound=None):
    def delta_func(delta_s):
        # eps = numericalanalysis_step(n, epsorig, delta, num_iterations, step, upperbound, clip_bound)
        eps = numericalanalysis_step(n, epsorig, delta_s, num_iterations, step, upperbound, clip_bound)
        return (delta - delta_s - (np.exp(eps)+1)*(1+np.exp(-epsorig)/2) * n * deltaorig, eps)
    a=1e-20
    b=delta
    fa,_ = delta_func(a)
    fb,_ = delta_func(b)
    num=0
    # if fa < 0:
    #     print('fail to bound')
    #     return epsorig
    while a<=b and num<100:
        x0 = (a+b)/2
        fx0,eps = delta_func(x0)
        # print(fa,fb,fx0)
        if fx0<1e-12 and fx0>=0:
            # print('epsilon:',x0,' diff:', fx0,' diff<10e-10')
            return eps
        if fa*fx0<0:
            b=x0
            fb=fx0
            # print('解在左侧,a:',a,'  b:',b,'  x0:',x0)
        elif fb*fx0<0:
            a=x0
            fa=fx0
            # print('解在右侧,a:',a,'  b:',b,'  x0:',x0)
        num += 1
    fx0,eps = delta_func(x0)
    if fx0 < 0:
        print('fail to bound')
        return epsorig
    return eps    
    
def numericalanalysis_delta(n, epsorig, deltaorig, eps_s, num_iterations, step, upperbound, clip_bound=None):
    def eps_func(delta_s):
        eps = numericalanalysis_step(n, epsorig, delta_s, num_iterations, step, upperbound, clip_bound)
        return (eps_s-eps, delta_s + (np.exp(eps)+1)*(1+np.exp(-epsorig)/2) * n * deltaorig)
    a=1e-20
    b=1.5
    fa,_ = eps_func(a)
    fb,_ = eps_func(b)
    num=0
    # if fa < 0:
    #     print('fail to bound')
    #     return epsorig
    while a<=b and num<100:
        x0 = (a+b)/2
        fx0,delta_final = eps_func(x0)
        # print(fa,fb,fx0)
        if fx0<1e-6 and fx0>=0:
            # print('epsilon:',x0,' diff:', fx0,' diff<10e-10')
            return delta_final
        if fa*fx0<0:
            b=x0
            fb=fx0
            # print('解在左侧,a:',a,'  b:',b,'  x0:',x0)
        elif fb*fx0<0:
            a=x0
            fa=fx0
            # print('解在右侧,a:',a,'  b:',b,'  x0:',x0)
        num += 1
    eps_diff, delta_final = eps_func(b)
    if eps_diff < 0:
        print('fail to bound')
        return 1
    return delta_final 
# ===========/THEORY/========
def closedformanalysis(n, epsorig, delta):
    '''
    Theoretical computation the privacy guarantee of achieved by shuffling n eps0-DP local reports.
    '''
    # if epsorig > math.log(n / (16 * math.log(4 / delta))): #hiding
    if epsorig > math.log(n / (8 * math.log(2 / delta))): #stronger
        print("This is not a valid parameter regime for this analysis")
        return epsorig
    else:
        # stronger 
        a = 4 * (2 * math.log(4 / delta)) ** 0.5 / (n*(math.exp(epsorig)+1)) ** (1 / 2)
        c = 4 / n
        b = math.exp(epsorig) - 1
        return math.log(1 + b * (a+c))        
        # hiding
        # a = 8 * (math.exp(epsorig) * math.log(4 / delta)) ** (1 / 2) / (n) ** (1 / 2)
        # c = 8 * math.exp(epsorig) / n
        # e = math.log(1 + a + c)
        # b = 1 - math.exp(-epsorig)
        # d = (1 + math.exp(-epsorig - e))
        # return math.log(1 + (b / d) * (a + c))

# n=10000
# num_iterations = 10
# step = 100
# upperbound=True
# e = 3
# d = 10**(-10)
# d_final = 10**(-5)
# # mech = "laplacian"
# mech = "gaussian"
# C = 1

# # pij, roots = compute_pij(ei,di,ej,dj,mech,C)
# # print('eps:',ei,ej)
# # print('HP:',pij)


# # delta_s = ds + (1+np.exp(ej))*(1+np.exp(-ej)/2)*n*dj #hiding
# # re = numericalanalysis(n, e, d, d_final, num_iterations, step, upperbound, C)


# eps_final=0.1
# re = numericalanalysis_delta(n, e, d, eps_final, num_iterations, step, upperbound, C)
# print(re)
# print("hiding delta:",d_final)
# print('stronger:', 2/(1+math.e**e) / 2)
# print('generalized:', 2/(1+math.e**e) / 2)
# # plot_pdf(ei,di,ej,dj,mech,C, roots)

