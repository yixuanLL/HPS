from blanket.shuffleddp.mechanisms import *
from blanket.shuffleddp.amplification_bounds import *
import matplotlib.pyplot as plt
from applifytheory import *
import numpy as np
import time


# delta_s = 10**(-6)
# delta_l = 10**(-8)

l=0.5
r=2
# l = 0.2
# r = 1
delta_l = 10**(-10)
# ns = np.geomspace(1000, 100000, num=20, dtype=int)
ns=[10000]
def plot_panel(ns, bounds):
    fig = plt.figure()
    ls = ['--', ':', '-.', '--','--','--','--', '-', '-', '-']
    m = ['', '', '', '','', '', '', '', '', '']
    c = ['dodgerblue', 'blueviolet', 'darkcyan', 'yellowgreen', 'gold', 'lightcoral','khaki', 'salmon', 'orange','yellowgreen', 'r', 'orange','slategrey']
    ours_c = ['r', 'orange'] #, 'yellowgreen']
    ours_m = ['o', 'p', '*']
    i=-1
    k=-1
    mi = ''
    lsi = '--'
    for b in bounds:
        print('theory,mech:', b.get_name(), b.mech)
        for dist in ['Uniform2', 'Gauss2', 'Step2']:
            print('dist:', dist)
            ys = list()
            for n in ns:
                user_idx = [uid for uid in range(0, n, 500)]
                eps0 = gen_eps(l, r, n, dist)
                eps0 = np.sort(eps0)
                # print(eps0)
                start = time.time()
                for canary_idx in user_idx:
                    re = b.get_eps(eps0, n, delta, canary_idx)
                    ys.append(re)  
                    # print(canary_idx, eps0[canary_idx], re)  
                end = time.time()
                print("time:", np.round(end-start,1))
                i+=1       
                # plt.plot(user_idx, ys, label="DP", linestyle=lsi, marker=mi, color=c[i], markevery=50)   
                # plt.plot(user_idx, ys, label="LDP", linestyle=lsi, marker=mi, color=c[i+1], markevery=50) 
                # plt.legend(loc='upper right')

                eps0_sample = eps0[0][user_idx]
                plt.plot(eps0_sample, ys, label="LDP-DP epsilon", linestyle="-", marker=mi, color=c[0], markevery=50) 
                plt.plot(eps0_sample, ys/eps0_sample, label="DP/LDP amp ratio", linestyle="--", marker=mi, color=c[1], markevery=50) 
                print("n:",n)
                print("eps0:", eps0_sample)
                print("ys:", ys)
                plt.legend()

                plt.xlabel('$\\varepsilon^l$',fontsize=14)
                plt.ylabel('$\\varepsilon^s$',fontsize=14)
                title_txt = '$\epsilon^l_i \in$ [{}, {}]'.format(l,r)
                plt.title(title_txt,fontsize=14)
                # plt.xscale('log')
                plt.xticks(size=14)
                plt.yticks(size=14)
                plt.legend(fontsize=14, loc="upper right")
                # plt.yscale('log')
                plt.show()
                path = './pdp_'+b.mech+'_'+ dist + '_'+ str(n) + '.pdf'
                plt.savefig(path)
                print('----'+path+'----')
                plt.close()


def gen_eps(l, r, n, dist):
    if dist == 'Uniform2':
        eps0 = np.random.uniform(l, r, n)
        delta0 = np.array([delta_l]*n)
        return (eps0, delta0)
    elif dist == 'Gauss2':
        eps0 = np.random.normal(r*0.75, 0.5, n)
        eps0 = np.maximum(eps0, l)
        eps0 = np.minimum(eps0, r)
        delta0 = np.array([delta_l]*n)
        return (eps0, delta0)
    elif dist == 'MixGauss':
        step = int(n*0.5)
        eps_low = np.random.normal(r*0.5, 0.2, step)
        eps_high = np.random.normal(r*0.8, 0.2, n-step)
        eps0 = np.concatenate((eps_low, eps_high))
        eps0 = np.maximum(eps0, l)
        eps0 = np.minimum(eps0, r)
        delta0 = np.array([delta_l]*n)
        return (eps0, delta0)
    elif dist == 'Single':
        eps0 = np.array([l]*n)
        delta0 = np.array([delta_l]*n)
        return (eps0, delta0)
    elif dist == 'Step2':
        step = int(n*0.5)
        eps_low = np.array([l]*step)
        eps_high = np.array([r]*(n-step))
        eps0 = np.concatenate((eps_low, eps_high))
        delta0 = np.array([delta_l]*n)
        return (eps0, delta0)
    else:
        return 0


clip_bound = 0.1

pure_bounds = [
            # Hoeffding(LaplaceMechanism()),
            # RDP(),
            # UniS(),
            # General_GDP(pure_on=True),
            # PerS(),
            HP_fDP_PDP(mech="laplacian", clip_bound=clip_bound, pure_on=True)
          ]
appox_bounds =   [
            # UniS_approax(),
            # General_GDP(pure_on=False),
            HP_fDP_PDP(mech="gaussian", clip_bound=clip_bound, pure_on=False)
          ]  
# bound_list =[pure_bounds, appox_bounds]
bound_list =[pure_bounds, appox_bounds]

i=0
plt.switch_backend('agg')
for bounds in bound_list: 
    i += 1
    if len(bounds)==0:
        continue

    print(bounds)

    ## calculate bounds
    plot_panel(ns, bounds)




     
