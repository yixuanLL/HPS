from blanket.shuffleddp.mechanisms import *
from blanket.shuffleddp.amplification_bounds import *
import matplotlib.pyplot as plt
from applifytheory import *
import numpy as np


# delta_s = 10**(-6)
# delta_l = 10**(-8)
delta = 10**(-9)
# l=0.5
# r=2
l = 0.05
r = 0.5
delta_l = 10**(-9)
delta_comp=10**(-6)
# ns = np.geomspace(1000, 100000, num=20, dtype=int)
ns=[10000]
def plot_panel(xs, bounds):

    i=-1
    k=-1
    mi = ''
    lsi = '--'
    for b in bounds:
        print('theory,mech:', b.get_name(), b.mech)
        for dist in ['Uniform1', 'Gauss1', 'MixGauss1']:
            print('dist:', dist)
            ys = list()
            for x in xs:
                eps0 = gen_eps(l, r, x, dist)
                re = b.get_eps(eps0, x, delta)
                ys.append(re)
                print('dimension level:',x, '\t', re)  
                composition(re, delta, delta_comp, 1570, 7850)  

def composition(eps_before, delta_before, delta_after, b, d):
    eps_rs = eps_before*np.sqrt(2*d*np.log(1/delta_after))+d*eps_before*(np.exp(eps_before)-1)
    eps_ps =  eps_before*np.sqrt(4*b*np.log(1/delta_after))+2*b*eps_before*(np.exp(eps_before)-1)
    delta_rs = d*delta_before+delta_after
    delta_ps = 2*b*delta_before+delta_after
    print('fully composition d dim:', eps_rs, delta_rs)
    print('post sparsification b dim:', eps_ps, delta_ps)

def gen_eps(l, r, n, dist):
    if dist == 'Uniform1':
        eps0 = np.random.uniform(l, r, n)
        delta0 = np.array([delta_l]*n)
        return (eps0, delta0)
    #for aaai version
    elif dist == 'Gauss1':
        eps0 = np.random.normal(0.1, 1, n)
        eps0 = np.maximum(eps0, l)
        eps0 = np.minimum(eps0, r)
        delta0 = np.array([delta_l]*n)
        return (eps0, delta0)
    elif dist == 'MixGauss1':
        step = int(n*0.5)
        eps_low = np.random.normal(0.1, 1, step)
        eps_high = np.random.normal(0.5, 1, n-step)
        eps0 = np.concatenate((eps_low, eps_high))
        eps0 = np.maximum(eps0, l)
        eps0 = np.minimum(eps0, r)
        delta0 = np.array([delta_l]*n)
        return (eps0, delta0)
    # for cikm version
    # elif dist == 'Gauss1':
    #     eps0 = np.random.normal(r*0.75, 0.5, n)
    #     eps0 = np.maximum(eps0, l)
    #     eps0 = np.minimum(eps0, r)
    #     delta0 = np.array([delta_l]*n)
    #     return (eps0, delta0)
    # elif dist == 'MixGauss1':
    #     step = int(n*0.5)
    #     eps_low = np.random.normal(r*0.5, 0.2, step)
    #     eps_high = np.random.normal(r*0.8, 0.2, n-step)
    #     eps0 = np.concatenate((eps_low, eps_high))
    #     eps0 = np.maximum(eps0, l)
    #     eps0 = np.minimum(eps0, r)
    #     delta0 = np.array([delta_l]*n)
    #     return (eps0, delta0)
    elif dist == 'Single':
        eps0 = np.array([l]*n)
        delta0 = np.array([delta_l]*n)
        return (eps0, delta0)
    elif dist == 'Step':
        step = int(n*0.5)
        eps_low = np.array([l]*step)
        eps_high = np.array([r]*(n-step))
        eps0 = np.concatenate((eps_low, eps_high))
        delta0 = np.array([delta_l]*n)
        return (eps0, delta0)
    else:
        return 0


clip_bound = 0.1
# pure DP
pure_bounds = [
            HP_fDP(mech="laplacian", clip_bound=clip_bound, pure_on=True)
          ]
appox_bounds =   [
            HP_fDP(mech="gaussian", clip_bound=clip_bound, pure_on=False)
          ]  
bound_list =[pure_bounds, appox_bounds]
# bound_list =[pure_bounds, []]
# bound_list =[[], appox_bounds]

# for thesis Random response
pure_bounds = [
            # Hoeffding(RRMechanism()),
            Hoeffding(LDPMechanism()),
            RDP(),
            UniS(),
            # General_GDP(pure_on=True),
            # PerS_RR(),
            PerS()
            # HP_fDP(mech="RR", clip_bound=clip_bound, pure_on=True),
            # HP_fDP(mech="laplacian", clip_bound=clip_bound, pure_on=True)
          ]
appox_bounds =   [
            UniS_approax(),
            # General_GDP(pure_on=False),
            # HP_fDP(mech="gaussian", clip_bound=clip_bound, pure_on=False)
          ]  
bound_list =[appox_bounds]


i=1
plt.switch_backend('agg')
for bounds in bound_list: 
    i += 1
    if len(bounds)==0:
        continue

    print(bounds)

    ## calculate bounds
    plot_panel(ns, bounds)

    if i==1:
        path_name = "pure"
    if i==2: ##approx
        path_name = "approx"
    # plt.xlabel('$n$',fontsize=14)
    # plt.ylabel('$\\varepsilon^s$',fontsize=14)
    # plt.title(title_txt,fontsize=14)
    # plt.xscale('log')
    # plt.xticks(size=14)
    # plt.yticks(size=14)
    # plt.legend(fontsize=14, loc="upper right")
    # # plt.yscale('log')
    # plt.show()
    # path = './epsilon1_ch_lap_'+ path_name + '.pdf'
    # plt.savefig(path)
    # print('----'+path+'----')
    # plt.close()

# composition(0.05, delta_l, delta_comp, 1570, 7850)

# delta=10**(-9)
# epsilon=0.5
# noise_multiplier = np.sqrt(2*np.log(1/delta))/epsilon
# print(noise_multiplier)



     
