from blanket.shuffleddp.mechanisms import *
from blanket.shuffleddp.amplification_bounds import *
import matplotlib.pyplot as plt
# from applifytheory import *
import numpy as np
import scipy as sp
import math
# from computeamplification_HP_fDP import compute_pij

def compute_pij(ei, di, ej, dj, mech, C, loci):
    mu_i = C
    mu_j = 0
    roots = []

    eps_1 = ej
    eps_2 = ej
    eps_3 = ei
    x_range = 10
    x = np.linspace(-x_range,x_range+0.1,10001)
    if mech == 'laplacian':
        b1 = np.abs(mu_j-mu_i)/eps_1
        b2 = np.abs(mu_j-mu_i)/eps_2
        b3 = np.abs(mu_j-mu_i)/eps_3

        cdf_1 = sp.stats.laplace.cdf(x,loc=mu_j, scale = b1) #x10
        cdf_2 = sp.stats.laplace.cdf(x,loc=mu_i, scale = b2) #x11
        cdf_3 = sp.stats.laplace.cdf(x,loc=loci, scale = b3) #xi
    if mech == 'gaussian':
        sigma_1 = 2*np.log(1.25/dj) * C**2 / ej**2 
        sigma_2 = 2*np.log(1.25/dj) * C**2 / ej**2 
        sigma_3 = 2*np.log(1.25/di) * C**2 / ei**2 

        cdf_1 = sp.stats.norm.cdf(x,loc=mu_j, scale = sigma_1) #x10
        cdf_2 = sp.stats.norm.cdf(x,loc=mu_i, scale = sigma_2) #x11
        cdf_3 = sp.stats.norm.cdf(x,loc=loci, scale = sigma_3) #xi
    cdf_roll_1 = np.roll(cdf_1,1)
    pmf_1 = cdf_1[1:] - cdf_roll_1[1:]
    cdf_roll_2 = np.roll(cdf_2,1)
    pmf_2 = cdf_2[1:] - cdf_roll_2[1:]
    cdf_roll_3 = np.roll(cdf_3,1)
    pmf_3 = cdf_3[1:] - cdf_roll_3[1:]

    # plt.switch_backend('agg')
    # fig = plt.figure()
    # plt.plot(x[1:],pmf_1)
    # plt.plot(x[1:],pmf_2)
    # plt.plot(x[1:],pmf_3)
    # plt.legend(['1','2','3'])
    # plt.xlim(-3,3)
    # plt.show()
    # plt.savefig('./pdf1.png', dpi=600)
    # plt.close()

    x1 = np.where(np.logical_and(np.greater(pmf_1,pmf_3),np.greater(pmf_1,pmf_2)))[0]
    x2 = np.where(np.logical_and(np.greater(pmf_2,pmf_3),np.greater(pmf_2,pmf_1)))[0]
    # p10 = np.sum(pmf_3[x1[0]:x1[-1]+1])
    # p11 = np.sum(pmf_3[x2[0]:x2[-1]+1])

    if mech == 'laplacian':
        p1 = sp.stats.laplace.cdf(C/2, loc=C, scale = b1) # 1/(1+e)
        if len(x1)>0:
            #xi looks like x10
            if np.abs(x[x1[0]] + x_range) < 0.1:
                p10 = sp.stats.laplace.cdf(x[x1[-1]], loc=loci, scale = b3)
            elif np.abs(x[x1[-1]] - x_range) < 0.1:
                p10 = 1 - sp.stats.laplace.cdf(x[x1[0]], loc=loci, scale = b3)
            else:
                p10 = sp.stats.laplace.cdf(x[x1[-1]], loc=loci, scale = b3) - sp.stats.laplace.cdf(x[x1[0]], loc=loci, scale = b3)
        else:
            p10=1-p1
        if len(x2)>0:
            #xi looks like x11
            if np.abs(x[x2[0]] + x_range) < 0.1:
                p11 = sp.stats.laplace.cdf(x[x2[0]], loc=loci, scale = b3)
            elif np.abs(x[x2[-1]] - x_range) < 0.1:
                p11 = 1 - sp.stats.laplace.cdf(x[x2[0]], loc=loci, scale = b3)
            else: 
                p11 = sp.stats.laplace.cdf(x[x2[-1]], loc=loci, scale = b3) - sp.stats.laplace.cdf(x[x2[0]], loc=loci, scale = b3)
        else:
            p11=1-p1
        
    if mech == 'gaussian':
        p1 = sp.stats.norm.cdf(C/2, loc=C, scale = sigma_1) - dj/2
        if len(x1)>0:
            #xi looks like x10
            if np.abs(x[x1[0]] + x_range) < 0.001:
                p10 = sp.stats.norm.cdf(x[x1[-1]], loc=loci, scale = sigma_3) - dj/2
            elif np.abs(x[x1[-1]] - x_range) < 0.001:
                p10 = 1 - sp.stats.norm.cdf(x[x1[0]], loc=loci, scale = sigma_3) - dj/2
            else:
                p10 = sp.stats.norm.cdf(x[x1[-1]], loc=loci, scale = sigma_3) - sp.stats.norm.cdf(x[x1[0]], loc=loci, scale = sigma_3)
        else:
            p10 = 1-p1
        if len(x2)>0:
            #xi looks like x11
            if np.abs(x[x2[0]] + x_range) < 0.001:
                p11 = sp.stats.norm.cdf(x[x2[0]], loc=loci, scale = sigma_3) - dj/2
            elif np.abs(x[x2[-1]] - x_range) < 0.001:
                p11 = 1 - sp.stats.norm.cdf(x[x2[0]], loc=loci, scale = sigma_3) - dj/2
            else: 
                p11 = sp.stats.norm.cdf(x[x2[-1]], loc=loci, scale = sigma_3) - sp.stats.norm.cdf(x[x2[0]], loc=loci, scale = sigma_3)
        else:
            p11=1-p1
        
    # print(x[x1[0]],x[x1[-1]],x[x2[0]],x[x2[-1]])å
    # print(p10,p11,min(p10,p11))
    # if min(p10,p11)*2 >1:
    #     print(print(x[x1[0]],x[x1[-1]],x[x2[0]],x[x2[-1]]), p10, p11)
    return p10, p11, p1


eps1 = 3
n = 501
eps = np.linspace(1e-3, eps1*2-0.05, n)
delta = 1e-10
# mech = "gaussian"
mech = "laplacian"
C=1
p10, p11, p1 = compute_pij(1.9,delta,2,delta,mech,C,0.97)
print(p10,p11, p1)
p_ours = []
p_stronger = []
p_fv = (1-delta)/(1+math.e**eps1)
plt.switch_backend('agg')
for i in range(n):

    p0, p1, _ = compute_pij(eps[i],delta,eps1,delta,mech,C,0)
    # print(i, p0, p1)
    print(i)
    p_ours.append(min(p0,p1))
    p_stronger.append(p_fv)

plt.plot(eps, p_ours, label='$p_i$ Ours', color='seagreen', linewidth=1.5)
plt.plot(eps, p_stronger, label='$p_i= \\frac{1-\delta_1}{1+e^{\epsilon_1}}$', color='steelblue', alpha=0.5, linestyle='-.', linewidth=1.5)
plt.grid(color='grey', 
         linestyle='--', 
         linewidth=1, 
         alpha=0.3, 
         which="major")

plt.xlabel('$\epsilon_i$',fontsize=14)
plt.ylabel('$p_i$',fontsize=14)
plt.legend(fontsize=14)
plt.show()
plt.xticks(size=14)
plt.yticks(size=14)
path = './p.pdf'
plt.savefig(path, bbox_inches='tight')
plt.close()

p0list = []
p1list = []
p0list1 = []
p1list1 = []
loci = np.linspace(0,C,n)
eps = 3
eps1 = 3
for i in range(1,n-1,1):
    p0, p1, _ = compute_pij(eps,delta,eps1,delta,mech,C, loci[i])

    p0list.append(p0)
    p1list.append(p1)
    p0, p1, _ = compute_pij(2,delta,eps1,delta,mech,C, loci[i])
    print(i, loci[i], p0, p1)
    p0list1.append(p0)
    p1list1.append(p1)
plt.plot(p0list, p1list, label='$\epsilon_1=3$, $\epsilon_i=3$', color='seagreen', linewidth=1.5)
plt.plot(p0list1, p1list1, label='$\epsilon_1=3$, $\epsilon_i=2$', color='orange', linewidth=1.5)
plt.grid(color='grey', 
         linestyle='--', 
         linewidth=1, 
         alpha=0.3, 
         which="major")
# plt.plot(p0list_stronger, p1list_stronger, label='FV', color='orange')
plt.xlabel('$p_i^0$',fontsize=14)
plt.ylabel('$p_i^1$',fontsize=14)
plt.xlim(0.1,0.52)
plt.ylim(0.1,0.52)
plt.xticks(size=14)
plt.yticks(size=14)
plt.legend(fontsize=14)
plt.show()
path = './p0_p1.pdf'
plt.savefig(path, bbox_inches='tight')
plt.close()
    
