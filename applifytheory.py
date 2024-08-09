# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
# import EoN.privAmp.computeamplification_HP_numerical as CA_HP
import computeamplification_HP_fDP as CA_HP_fDP
import computeamplification_perS as CA_perS
import computeamplification as CA_uniS
import computeamplification_GDP as CA_GDP
import computeamplification_approx as CA_approax
import numpy as np
from scipy.special import comb, gamma

############### for perS
#number of iterations of binary search. The higher T is, the more accurate the result
num_iterations = 10
# #This is a parameter of the empirical analysis computation that can be tuned for efficiency. The larger step is, the less accurate the result, but more efficient the algorithm.
step = 100

n = 1 * 10**4
# epsorig = np.random.uniform(1,1, n)
epsorig = np.array([1]*n)
delta = 10**(-5)

eps_max = np.max(epsorig)

class Clones:
    """Base class for "privacy amplification by shuffling" bounds."""

    def __init__(self, name='BoundBase', num_interations=10, step=100, mech=None, clip_bound=None, pure_on=True):
        self.name = name
        self.num_interations = num_interations
        self.step = step
        self.mech = mech
        self.clip_bound = clip_bound
        self.pure_on = pure_on
        print(self.mech)

    def get_name(self, with_mech=False):
        return self.name

################## pure DP ##############################
class UniS(Clones):
    """Implement the bound from Clones et al. [FV'23]"""

    def __init__(self, name='FV'):
        super(UniS, self).__init__(name=name)
        # The constants in the bound are only valid for a certain parameter regime
        
    def get_eps(self, eps, n, delta):
        eps_max = np.max(eps[0])
        try:
            numerical_upperbound = CA_uniS.numericalanalysis(n, eps_max, delta, self.num_interations, self.step, True)
        except AssertionError:
            return eps_max #np.nan
        return numerical_upperbound

    
class PerS(Clones):
    """Implement the bound from Liu et al. [LZX'23]"""

    def __init__(self, name='LZX'):
        super(PerS, self).__init__(name=name)
        # The constants in the bound are only valid for a certain parameter regime
        
    def get_eps(self, eps, n, delta):
        try:
            numerical_upperbound = CA_perS.numericalanalysis(n, eps[0], delta, self.num_interations, self.step, True)
        except AssertionError:
            return np.max(eps) #np.nan
        return numerical_upperbound


class RDP(Clones):
    """Implement the bound from Erlignsson et al. [SODA'19]"""

    def __init__(self, name='GDDTK'):
        super(RDP, self).__init__(name=name)
        # The constants in the bound are only valid for a certain parameter regime
    def get_eps(self, eps, n, delta):
        eps = np.max(eps[0])
        dp_upperbound_min = eps
        try:
            n_bar = int((n-1)/(2*np.exp(eps)) + 1)
            for lambd in range(2,5000,50):
                sum = 0
                for i in range(2, lambd+1, 1):   
                    sum += comb(lambd, i) * i * gamma(i/2.0) * ((np.exp(2*eps)-1)**2/(2*np.exp(2*eps)*n_bar))**(i/2.0)
                rdp_upperbound = 1 / (lambd-1) * np.log(1+ comb(lambd, 2)* ((np.exp(eps)-1)**2)/(n_bar*np.exp(eps)) + sum \
                    + np.exp(eps*lambd-(n-1)/(8*np.exp(eps)))) # upperbound1
                dp_upperbound = self.rdp2dp(rdp_upperbound, lambd, delta)
                if dp_upperbound_min > dp_upperbound:
                    dp_upperbound_min = dp_upperbound
        except AssertionError:
            return eps
       
        return dp_upperbound_min
    
    def rdp2dp(self, rdp_e, lambd, delta):
        return rdp_e + (np.log(1/delta)+(lambd-1)*np.log(1-1/lambd)-np.log(lambd))/(lambd-1)


################## approax DP ##############################
class General_GDP(Clones):
    """Implement the bound from Chen et al. [CCC'24]"""

    # def __init__(self, name='HP'):
    #     super(HP, self).__init__(name=name)
        # The constants in the bound are only valid for a certain parameter regime
    def __init__(self, name='CCC', num_interations=10, step=100, mech=None, clip_bound=None, pure_on=True):
        self.name = name
        self.num_interations = num_interations
        self.step = step
        self.mech = mech
        self.clip_bound = clip_bound
        self.pure_on = pure_on
        
    def get_eps(self, eps, n, delta):
        try:
            eps_local = eps[0]
            delta_local = eps[1]
            if self.pure_on:
                delta_local *= 0
            numerical_upperbound = CA_GDP.numericalanalysis(n, eps_local, delta_local, delta, self.num_interations, self.step, True, self.mech, self.clip_bound)
        except AssertionError:
            return np.max(eps) #np.nan
        return numerical_upperbound 

    def get_delta(self, eps, n, eps_s):
        try:
            eps_local = eps[0]
            delta_local = eps[1]
            if self.pure_on:
                delta_local *= 0
            numerical_upperbound = CA_GDP.numericalanalysis_delta(n, eps_local, delta_local, eps_s, self.num_interations, self.step, True, self.mech, self.clip_bound)
        except AssertionError:
            return np.max(eps) #np.nan
        return numerical_upperbound 

class UniS_approax(Clones):
    """Implement the bound from Chen et al. [FV'23]"""

    # def __init__(self, name='HP'):
    #     super(HP, self).__init__(name=name)
        # The constants in the bound are only valid for a certain parameter regime
    def __init__(self, name='FV', num_interations=10, step=100, mech=None, clip_bound=None):
        self.name = name
        self.num_interations = num_interations
        self.step = step
        self.mech = mech
        self.clip_bound = clip_bound
        
    def get_eps(self, eps, n, delta):
        try:
            eps_idx = np.argmax(eps[0])
            eps_local = eps[0][eps_idx]
            delta_local = eps[1][eps_idx]
            numerical_upperbound = CA_approax.numericalanalysis(n, eps_local, delta_local, delta, self.num_interations, self.step, True, self.clip_bound)
        except AssertionError:
            return np.max(eps) #np.nan
        return numerical_upperbound 

    def get_delta(self, eps, n, eps_s):
        try:
            eps_idx = np.argmax(eps[0])
            eps_local = eps[0][eps_idx]
            delta_local = eps[1][eps_idx]
            numerical_upperbound = CA_approax.numericalanalysis_delta(n, eps_local, delta_local, eps_s, self.num_interations, self.step, True, self.clip_bound)
        except AssertionError:
            return np.max(eps) #np.nan
        return numerical_upperbound 

################# Ours #####################
class HP_fDP(Clones):
    """Implement the bound from Liu et al. [HP]"""
    def __init__(self, name='Ours', num_interations=10, step=100, mech=None, clip_bound=None, pure_on=True):
        self.name = name
        self.num_interations = num_interations
        self.step = step
        self.mech = mech
        self.clip_bound = clip_bound
        self.pure_on = pure_on
        
    def get_eps(self, eps, n, delta):
        try:
            eps_local = eps[0]
            delta_local = eps[1]
            if self.pure_on:
                delta_local *= 0
            numerical_upperbound = CA_HP_fDP.numericalanalysis(n, eps_local, delta_local, delta, self.num_interations, self.step, True, self.mech, self.clip_bound)
        except AssertionError:
            return np.max(eps) #np.nan
        return numerical_upperbound 
    
    def get_delta(self, eps, n, eps_s):
        try:
            eps_local = eps[0]
            delta_local = eps[1]
            if self.pure_on:
                delta_local *= 0
            numerical_upperbound = CA_HP_fDP.numericalanalysis_delta(n, eps_local, delta_local, eps_s, self.num_interations, self.step, True, self.mech, self.clip_bound)
        except AssertionError:
            return np.max(eps) #np.nan
        return numerical_upperbound 