from math import sqrt, log, exp
from scipy.optimize import root_scalar
from scipy.stats import binom
import numpy as np


class ShuffleAmplificationBound:
    """Base class for "privacy amplification by shuffling" bounds."""

    def __init__(self, name='BoundBase', tol=None):
        """Parameters:
            name (str): Name of the bound
            tol (float): Error tolerance for optimization routines
        """
        self.name = name
        # Set up a default tolerance for optimization even if none is specified
        if tol is None:
            self.tol_opt = 1e-12
        else:
            self.tol_opt = tol
        # Tolerance for delta must be larger than optimization tolerance
        self.tol_delta = 10*self.tol_opt

    def get_name(self, with_mech=True):
        return self.name

    def get_delta(self, eps, eps0, n):
        """This function returns delta after shuffling for given parameters:
            eps (float): Target epsilon after shuffling
            eps0 (float): Local DP guarantee of the mechanism being shuffled
            n (int): Number of randomizers being shuffled
        """
        raise NotImplementedError

    def threshold_delta(self, delta):
        """Truncates delta to reasonable parameters to avoid numerical artifacts"""
        # The ordering of the arguments is important to make sure NaN's are propagated
        return min(max(delta, self.tol_delta), 1.0)


class Erlingsson(ShuffleAmplificationBound):
    """Implement the bound from Erlignsson et al. [SODA'19]"""

    def __init__(self, name='EFMRTT', tol=None):
        super(Erlingsson, self).__init__(name=name, tol=tol)
        # The constants in the bound are only valid for a certain parameter regime
        self.max_eps0 = 0.5
        self.min_n = 1000
        self.max_delta = 0.01
        self.mech = None

    def check_ranges(self, eps=None, eps0=None, n=None, delta=None):
        """Check that a set of parameters is within the range of validity of the bound"""
        if eps0 is not None:
            assert eps0 <= self.max_eps0
            if eps is not None:
                assert eps <= eps0
        if n is not None:
            assert n >= self.min_n
        if delta is not None:
            assert delta <= self.max_delta

    def get_delta(self, eps, eps0, n):
        """Implement the bound delta(eps,eps0,n) in [EFMRTT'19]"""
        try:
            self.check_ranges(eps=eps, eps0=eps0, n=n)
            delta = exp(-n * (eps / (12 * eps0))**2)
            self.check_ranges(delta=delta)
        except AssertionError:
            return np.nan

        return self.threshold_delta(delta)

    def get_eps(self, eps0, n, delta):
        """Implement the bound eps(eps0,n,delta) in [EFMRTT'19]"""
        eps0 = np.max(eps0)
        try:
            self.check_ranges(eps0=eps0, n=n, delta=delta)
            eps = 12*eps0*sqrt(log(1/delta)/n)
            self.check_ranges(eps=eps, eps0=eps0)
        except AssertionError:
            return eps0

        return eps

    def get_eps0(self, eps, n, delta):
        """Implement the bound eps0(eps,n,delta) in [EFMRTT'19]"""
        try:
            self.check_ranges(eps=eps, n=n, delta=delta)
            eps0 = eps/(12*sqrt(log(1/delta)/n))
            self.check_ranges(eps=eps, eps0=eps0)
        except AssertionError:
            return np.nan

        return eps0


class NumericShuffleAmplificationBound(ShuffleAmplificationBound):
    """Base class for amplification bounds that are given in implicit form:
    F(eps,n,mechanism) <= delta
    This class implements the numerics necessary to recover eps and eps0 from implicit bounds.
    """

    def __init__(self, mechanism, name, tol=None):
        """Numeric bounds depend on properties of the mechanism"""
        super(NumericShuffleAmplificationBound, self).__init__(name=name, tol=tol)
        self.mechanism = mechanism

    def get_name(self, with_mech=True):
        # if with_mech:
        #     return '{}, {}'.format(self.name, self.mechanism.get_name())
        # return self.name
        # temp
        return self.name

    def get_delta(self, eps, eps0, n):
        """Getting delta is bound dependent"""
        raise NotImplementedError

    def get_eps(self, eps0, n, delta, min_eps=1e-6):
        """Find the minimum eps giving <= delta"""
        eps0 = np.max(eps0)
        assert eps0 >= min_eps
        # If this assert fails consider decreasing min_eps
        assert self.get_delta(min_eps, eps0, n) >= delta

        def f(x):
            return self.get_delta(x, eps0, n) - delta

        # Use numeric root finding
        sol = root_scalar(f, bracket=[min_eps, eps0], xtol=self.tol_opt)

        assert sol.converged
        eps = sol.root

        return eps

    def get_eps0(self, eps, n, delta, max_eps0=10):
        """Find the maximum eps0 giving <= delta"""

        assert eps <= max_eps0
        # If this assert fails consider increasing max_eps0
        assert self.get_delta(eps, max_eps0, n) >= delta

        def f(x):
            current_delta = self.get_delta(eps, x, n)
            return current_delta - delta

        # Use numeric root finding
        sol = root_scalar(f, bracket=[eps, max_eps0], xtol=self.tol_opt)

        assert sol.converged
        eps0 = sol.root

        return eps0


class Hoeffding(NumericShuffleAmplificationBound):
    """Numeric amplification bound based on Hoeffding's inequality"""

    def __init__(self, mechanism, name='BBGN', tol=None):
        super(Hoeffding, self).__init__(mechanism, name, tol=tol)
        self.mech=None

    def get_delta(self, eps, eps0, n):

        if eps >= eps0:
            return self.tol_delta

        self.mechanism.set_eps0(eps0)

        gamma_lb, gamma_ub = self.mechanism.get_gamma()
        a = exp(eps) - 1
        b = self.mechanism.get_range_l(eps)

        delta = 1/(gamma_lb*n)
        delta *= b**2 / (4*a)
        delta *= (1 - gamma_lb*(1-exp(-2 * a**2 / b**2)))**n

        return self.threshold_delta(delta)


class BennettExact(NumericShuffleAmplificationBound):
    """Numeric amplification bound based on Bennett's inequality"""

    def __init__(self, mechanism, name='Bennett', tol=None):
        super(BennettExact, self).__init__(mechanism, name, tol=tol)

    def get_delta(self, eps, eps0, n):

        if eps >= eps0:
            return self.tol_delta

        self.mechanism.set_eps0(eps0)

        gamma_lb, gamma_ub = self.mechanism.get_gamma()
        a = exp(eps) - 1
        b_plus = self.mechanism.get_max_l(eps)
        c = self.mechanism.get_var_l(eps)

        alpha = c / b_plus**2
        beta = a * b_plus / c
        #eta = a / b_plus
        eta = 1.0 / b_plus

        def phi(u):
            phi = (1 + u) * log(1 + u) - u
            if phi < 0:
                # If phi < 0 (due to numerical errors), u should be small
                # enough that we can use the Taylor approximation instead.
                phi = u**2
            return phi

        exp_coef = alpha * phi(beta)
        div_coef = eta * log(1 + beta)

        def expectation_l(m):
            #coefs = np.divide(np.exp(-m * exp_coef), m * div_coef)
            coefs = np.exp(-m * exp_coef) / div_coef
            return coefs

        delta = 1 / (gamma_lb * n)
        expectation_term = binom.expect(expectation_l, args=(n, gamma_lb), lb=1, tolerance=self.tol_opt, maxcount=100000)
        delta *= expectation_term

        return self.threshold_delta(delta)
