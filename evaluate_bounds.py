from blanket.shuffleddp.mechanisms import *
from blanket.shuffleddp.amplification_bounds import *
import matplotlib.pyplot as plt
from applifytheory import *
import numpy as np


# delta_s = 10**(-6)
# delta_l = 10**(-8)

# l=0.5
# r=2
l = 0.05
r = 1
delta_l = 10**(-10)
ns = np.geomspace(1000, 100000, num=20, dtype=int)
# ns=[1000,1000]
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
    for b in bounds:
        print('theory,mech:', b.get_name(), b.mech)
        for dist in ['Uniform1', 'Gauss1']: #, 'MixGauss']:
            print('dist:', dist)
            ys = list()
            for x in xs:
                eps0 = gen_eps(l, r, x, dist)
                re = b.get_eps(eps0, x, delta)
                ys.append(re)
                print(x, '\t', re)     
            i+=1       
            if b.get_name() not in [ "Ours", "LZX", "CCC"]: 
                plt.plot(xs, ys, label=b.get_name(), linestyle=lsi, marker=mi, color=c[i], markevery=5)   
                break  
            else:
                k += 1
                ci = c[i]

                # color = ours_c[k%3]
                if b.get_name() in ["Ours"]:
                    lsi = '-'
                    ci = ours_c[k%2]
                    mi = ours_m[k%2]
                    # dist = dist +' '+ b.mech
                me = 5
                if dist == 'MixGauss':
                    me = 3
                plt.plot(xs, ys, label=b.get_name()+' '+dist, linestyle=lsi, marker=mi, color=ci, markevery=me) 
    plt.legend(loc='upper right')


def gen_eps(l, r, n, dist):
    if dist == 'Uniform1':
        eps0 = np.random.uniform(l, r, n)
        delta0 = np.array([delta_l]*n)
        return (eps0, delta0)
    elif dist == 'Gauss1':
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
            RDP(),
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
bound_list =[pure_bounds, appox_bounds]
# bound_list =[pure_bounds, []]
# bound_list =[[], appox_bounds]
i=0
plt.switch_backend('agg')
for bounds in bound_list: 
    i += 1
    if len(bounds)==0:
        continue

    print(bounds)

    ## calculate bounds
    plot_panel(ns, bounds)

    if i==1:
        # plt.ylim(0,0.6)# 0.5-2
        title_txt = '$\epsilon^l_i \in$ [{}, {}]'.format(l,r)
        path_name = "pure"
    if i==2: ##approx
        # plt.ylim(0,0.45)# 0.5-2
        plt.ylim(0,0.18)# 0.05-1
        title_txt = '$\epsilon^l_i \in [{}, {}], \delta^l = 10^{}$'.format(l,r,'{-%d}' % np.log10(1/delta_l))
        path_name = "approx"
    plt.xlabel('$n$',fontsize=14)
    plt.ylabel('$\\varepsilon^s$',fontsize=14)
    plt.title(title_txt,fontsize=14)
    plt.xscale('log')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.legend(fontsize=14, loc="upper right")
    # plt.yscale('log')
    plt.show()
    path = './epsilon1_'+ path_name + '.pdf'
    plt.savefig(path)
    print('----'+path+'----')
    plt.close()



     
