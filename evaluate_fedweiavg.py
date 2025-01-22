from blanket.shuffleddp.mechanisms import *
from blanket.shuffleddp.amplification_bounds import *
import matplotlib.pyplot as plt
from applifytheory import *
import numpy as np
import time


# delta_s = 10**(-6)
# delta_l = 10**(-8)

l=0.1
r=0.5
# l = 0.2
# r = 1
# delta_l = 10**(-10)
delta_l = 10**(-5)

# ns = np.geomspace(1000, 100000, num=20, dtype=int)
ns=[1000]
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
        for dist in ['Uniform1']:
            print('dist:', dist)
            y_shuffled = list()
            y_sampled_eps = list()
            y_sampled_delta = list()
            for n in ns:
                delta0 = np.array([delta_l]*n)
                user_idx = [uid for uid in range(0, n, 10)]
                user_idx.append(n-1)
                user_idx.append(n-50)
                # generate epsilon from noise multiplier
                noise_multiplier = gen_noise(l, r, n, dist)
                eps0 = noise_to_eps(noise_multiplier, delta0)
                q = FedWeight(noise_multiplier, scale_coef, clip_bound)
                # eps0 = np.sort(eps0)
                # print(eps0)
                start = time.time()
                for canary_idx in user_idx:
                    # shuffle
                    shuffled_eps = b.get_eps((eps0, delta0), n, delta, canary_idx)
                    y_shuffled.append(shuffled_eps)
                    # subsample
                    sampled_eps, sampled_delta = subsample_amp(shuffled_eps, delta, q[canary_idx])
                    y_sampled_eps.append(sampled_eps)  
                    y_sampled_delta.append(sampled_delta)
                    # print(canary_idx, eps0[canary_idx], re)  
                end = time.time()
                print("time:", np.round(end-start,1))
                i+=1       
                # plt.plot(user_idx, ys, label="DP", linestyle=lsi, marker=mi, color=c[i], markevery=50)   
                # plt.plot(user_idx, ys, label="LDP", linestyle=lsi, marker=mi, color=c[i+1], markevery=50) 
                # plt.legend(loc='upper right')

                eps0_sample = eps0[user_idx]
                res_matrix = np.vstack((eps0_sample, y_shuffled))
                res_matrix = np.vstack((res_matrix, y_sampled_eps))
                res_matrix = np.vstack((res_matrix, y_sampled_delta))
                res_matrix = res_matrix.T[np.argsort(res_matrix.T[:,0])].T
                plt.plot(res_matrix[0], res_matrix[1], label="LDP-ShuffleDP epsilon", linestyle="-", marker=mi, color=c[0], markevery=50) 
                plt.plot(res_matrix[0], res_matrix[1]/res_matrix[0], label="SDP/LDP amp ratio", linestyle="--", marker=mi, color=c[1], markevery=50) 

                plt.plot(res_matrix[0], res_matrix[2], label="LDP-ShuffleDP-Subsample epsilon", linestyle="-", marker=mi, color=c[3], markevery=50) 
                plt.plot(res_matrix[0], res_matrix[2]/res_matrix[0], label="SSDP/LDP amp ratio", linestyle="-.", marker=mi, color=c[4], markevery=50) 

                # plt.plot(res_matrix[0], res_matrix[3], label="LDP-ShuffleDP-Subsample delta", linestyle="-", marker=mi, color=c[5], markevery=50) 
                # plt.plot(res_matrix[0], res_matrix[3]/delta_l, label="SSDP/LDP delta amp ratio", linestyle="--", marker=mi, color=c[6], markevery=50) 
                print("n:",n)
                print("eps0:", res_matrix[0])
                print("y_shuffled:", res_matrix[1])
                print("y_sampled_eps:", res_matrix[2])
                print("q", q[user_idx])
                print("y_sampled_delta:", y_sampled_delta)
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
                path = './pdp_fedwei_'+b.mech+'_'+ dist + '_'+ str(n) + '.pdf'
                plt.savefig(path)
                print('----'+path+'----')
                plt.close()

# for the algorithm "FedSampAvg", calculate the weights and sample ratio                
def FedWeight(noise_multiplier, scale_coef, clip_bound):
    noise_sigma = noise_multiplier * clip_bound
    per_inver_var = 1 / noise_sigma**2 #1/zi
    per_sample_ratio = per_inver_var / np.sum(per_inver_var) * scale_coef
    per_sample_ratio = np.clip(per_sample_ratio, 0, 1)
    return per_sample_ratio

# subsampling amplification effect with personalized qi, ref: 2018 balle Privacy Amplification by Subsampling, theorem 8
def subsample_amp(eps_before, delta_before, q):
    eps_after = np.log(1+q*(np.e ** eps_before - 1))
    delta_after = delta_before * q
    return eps_after,delta_after

# for guassian mechanism, given sigma, calculate epsilon. ref: 2016 DPSGD
def noise_to_eps(noise_multiplier, delta):
    eps = np.sqrt(2 * np.log(1.25/delta)) / noise_multiplier
    return eps

def gen_noise(l_nm, r_nm, num_clients, dist):
    if dist == 'Uniform1':
        print('unifrom left, right,', l_nm, r_nm)
        noise_multiplier = np.random.uniform(l_nm, r_nm, num_clients)
    elif dist == 'Step1':
        print('step 90% left, 10% right,', l_nm, r_nm)
        noise_multiplier = np.array([l_nm]*int(num_clients*0.9)+[r_nm]*(num_clients-int(num_clients*0.9)))
    elif dist == 'Guass1':
        print('Guassian mean is ', l_nm)
        noise_multiplier = np.random.normal(l_nm, 1.0, num_clients)
        np.clip(noise_multiplier, 0.05)
    else:
        noise_multiplier = [0]
    noise_multiplier = np.sort(noise_multiplier)[::-1]
    return noise_multiplier


clip_bound = 1
scale_coef = 300
delta=1e-4

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
bound_list =[appox_bounds]

i=0
plt.switch_backend('agg')
for bounds in bound_list: 
    i += 1
    if len(bounds)==0:
        continue

    print(bounds)

    ## calculate bounds
    plot_panel(ns, bounds)




     
