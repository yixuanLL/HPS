import numpy as np
import scipy.stats as stats


# given eps, resolve delta
def gdp_paper(mu, eps):
    delta = stats.norm.cdf(-eps/mu + mu/2) - np.e**eps * stats.norm.cdf(-eps/mu - mu/2)
    print('delta', delta)
# given delta, resolve eps
def gdp_resolve_eps(mu, delta):
    def eps_delta_func(x):
        return stats.norm.cdf(-x/mu + mu/2) - np.e**x * stats.norm.cdf(-x/mu - mu/2) - delta
    a=-10
    b=20
    fa=eps_delta_func(a)
    fb=eps_delta_func(b)
    num = 0
    while a<=b and num<10000:
        x0=(a+b)/2
        fx0=eps_delta_func(x0)   
        if abs(fx0)<10e-12:
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
    return x0



if __name__ == '__main__':
    mu = 0.005
    eps = 2
    delta = 1e-5
    gdp_paper(mu, eps)
    gdp_resolve_eps(mu, delta)


# from autodp
# def fdp(sigma):
#     params = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
#     # delta_list = [0,1e-8, 1e-6, 1e-4, 1e-2, 0.3, 0.5, 1]
#     delta_list = [1e-8]

#     # f-DP implementation
#     gm3 = GaussianMechanism(sigma, name='GM3', RDP_off=True, approxDP_off=True, fdp_off=False)

#     # direct approxdp implementation
#     agm = lambda x: get_eps_ana_gaussian(sigma, x)

#     eps_direct = np.array([agm(delta) for delta in delta_list])

#     # the fdp is converted by numerical methods from privacy profile.
#     eps_converted = np.array([gm3.get_approxDP(delta) for delta in delta_list])

#     print('eps_direct', eps_direct)
#     print('eps_converted', eps_converted)