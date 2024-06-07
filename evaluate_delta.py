from blanket.shuffleddp.mechanisms import *
from blanket.shuffleddp.amplification_bounds import *
import matplotlib.pyplot as plt
from applifytheory import *
import numpy as np


# delta_s = 10**(-6)
# delta_l = 10**(-8)

l=0.5
r=2
delta_l = 10**(-10)
# ns = np.geomspace(1000, 100000, num=20, dtype=int)
ns=[10000]
eps_s_list = [0.01, 0.03, 0.05, 0.08, 0.1]
def plot_panel(xs, bounds):
    fig = plt.figure()
    ls = ['--', ':', '-.', '--','--','--','--', '-', '-', '-']
    m = ['', '', '', '','', '', '', '', '', '']
    c = ['slategrey', 'dodgerblue', 'blueviolet', 'darkcyan', 'yellowgreen', 'gold', 'lightcoral','khaki', 'salmon', 'orange','yellowgreen', 'r', 'orange']
    ours_c = ['r', 'orange'] #, 'yellowgreen']
    ours_m = ['o', 'p', '*']
    i=-1
    k=-1
    mi = ''
    lsi = '--'
    for dist in ['Uniform']: #, 'Gauss']: #, 'MixGauss']:
        print('dist:', dist)
        for b in bounds:
        # print('theory,mech:', b.get_name(), b.mech)

            for x in xs:
                ys = list()
                for eps_s in eps_s_list:
                    eps0 = gen_eps(l, r, x, dist)
                    re = b.get_delta(eps0, x, eps_s)
                    ys.append(re)
                print(b.get_name(),'n={}, delta_s={}'.format(x, ys)) 

            # i+=1       
            # if b.get_name() not in [ "Ours", "LZX", "CCC"]: 
            #     plt.plot(xs, ys, label=b.get_name(), linestyle=lsi, marker=mi, color=c[i], markevery=5)   
            #     break  
            # else:
            #     k += 1
            #     ci = c[i]

            #     # color = ours_c[k%3]
            #     if b.get_name() in ["Ours"]:
            #         lsi = '-'
            #         ci = ours_c[k%2]
            #         mi = ours_m[k%2]
            #         # dist = dist +' '+ b.mech
            #     me = 5
            #     if dist == 'MixGauss':
            #         me = 3
            #     plt.plot(xs, ys, label=b.get_name()+' '+dist, linestyle=lsi, marker=mi, color=ci, markevery=me) 
    plt.legend(loc='upper right')


def gen_eps(l, r, n, dist):
    if dist == 'Uniform':
        eps0 = np.random.uniform(l, r, n)
        delta0 = np.array([delta_l]*n)
        return (eps0, delta0)
    elif dist == 'Gauss':
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
            Hoeffding(LaplaceMechanism()),
            # RDP(),
            UniS(),
            General_GDP(pure_on=True),
            PerS(),
            HP_fDP(mech="laplacian", clip_bound=clip_bound, pure_on=True)
          ]
appox_bounds =   [
            UniS_approax(),
            General_GDP(pure_on=False),
            HP_fDP(mech="gaussian", clip_bound=clip_bound, pure_on=False)
          ]  
# bound_list =[pure_bounds, appox_bounds]
bound_list =[[], appox_bounds]
i=0
# plt.switch_backend('agg')
for bounds in bound_list: 
    i += 1
    if len(bounds)==0:
        continue

    print(bounds)

    ## calculate bounds
    plot_panel(ns, bounds)

    # if i==1:
    #     plt.ylim(0,0.6)
    #     title_txt = '$\\varepsilon^l_i \\in [{}, {}]'.format(l,r)
    # if i==2: ##approx
    #     plt.ylim(0,0.45)
    #     title_txt = '$\\varepsilon^l_i \\in [{}, {}], \\delta^l = 10^{}$'.format(l,r,'{-%d}' % np.log10(1/delta_l))
    # plt.xlabel('$n$')
    # plt.ylabel('$\\varepsilon^c$')
    # plt.title(title_txt)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.show()
    # path = './epsilon2_fdp'+str(i)+'.pdf'
    # plt.savefig(path)
    # print('----'+path+'----')
    # plt.close()



     
