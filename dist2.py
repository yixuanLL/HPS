import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
def compute_pij(ej, dj, mech, C):
    mu_i = C
    mu_j = 0
    roots = []

    eps_1 = ej
    
    x_range = 3
    x = np.linspace(-x_range,x_range+0.1,10001)
    if mech == 'laplacian':
        b1 = np.abs(mu_j-mu_i)/eps_1
        cdf_1 = stats.laplace.cdf(x,loc=mu_j, scale = b1) #x10
    if mech == 'gaussian':
        sigma_1 = 2*np.log(1.25/dj) * C**2 / ej**2 
        print(sigma_1)
        cdf_1 = stats.norm.cdf(x,loc=mu_j, scale = sigma_1) #x10


    plt.switch_backend('agg')
    # fig = plt.figure()
    plt.figure(figsize=(8, 5))
    # plt.plot(x[1:],pmf_1)
    # plt.plot(x[1:],pmf_2)
    # plt.plot(x[1:],pmf_3)
    plt.plot(x,stats.norm.pdf(x,loc=mu_j, scale = sigma_1), label='$R(x_1)$')
    plt.plot(x,stats.norm.pdf(x,loc=mu_i, scale = sigma_1), label='$R(x\'_1)$')
    x1 = (mu_i+mu_j)/2
    y = np.linspace(0,0.6,300)

    xp = np.linspace(-3, x1,300)
    plt.fill_between(xp, stats.norm.pdf(xp, loc=mu_i, scale = sigma_1), xp*0, color='g', alpha=0.3, label="$R(x_1)$ max")
    xp = np.linspace(x1, 3, 300)
    plt.fill_between(xp, stats.norm.pdf(xp, loc=mu_j, scale = sigma_1), xp*0, color='orange', alpha=0.3, label="$R(x\'_1)$ max")
    # plt.xlabel(fontsize=14)
    # plt.ylabel(fontsize=14)
    plt.legend(fontsize=14)
    plt.ylim(0,0.7)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.show()

    plt.savefig('./dist2.png', dpi=600, bbox_inches='tight')
    plt.close()

    return 0


ei = 5
ej = 5
di = 10**(-10)
dj = 10**(-10)
# mech = "laplacian"
mech = "gaussian"
C = 0.6

compute_pij(ei,di,mech,C)
print('eps:',ei,ej)
