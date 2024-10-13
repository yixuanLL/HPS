import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
def compute_pij(ei, di, ej, dj, mech, C):
    mu_i = C
    mu_j = 0
    roots = []

    eps_1 = ej
    eps_2 = ej
    eps_3 = ei
    x_range = 3
    x = np.linspace(-x_range,x_range+0.1,10001)
    if mech == 'laplacian':
        b1 = np.abs(mu_j-mu_i)/eps_1
        b2 = np.abs(mu_j-mu_i)/eps_2
        b3 = np.abs(mu_j-mu_i)/eps_3
        
        cdf_1 = stats.laplace.cdf(x,loc=mu_j, scale = b1) #x10
        cdf_2 = stats.laplace.cdf(x,loc=mu_i, scale = b2) #x11
        cdf_3 = stats.laplace.cdf(x,loc=mu_j+C/4, scale = b3) #xi
    if mech == 'gaussian':
        sigma_1 = 2*np.log(1.25/dj) * C**2 / ej**2 
        sigma_2 = 2*np.log(1.25/dj) * C**2 / ej**2 
        sigma_3 = 2*np.log(1.25/di) * C**2 / ei**2 
        print(sigma_1)

        cdf_1 = stats.norm.cdf(x,loc=mu_j, scale = sigma_1) #x10
        cdf_2 = stats.norm.cdf(x,loc=mu_i, scale = sigma_2) #x11
        cdf_3 = stats.norm.cdf(x,loc=mu_j+C/4, scale = sigma_3) #xi
    cdf_roll_1 = np.roll(cdf_1,1)
    pmf_1 = cdf_1[1:] - cdf_roll_1[1:]
    cdf_roll_2 = np.roll(cdf_2,1)
    pmf_2 = cdf_2[1:] - cdf_roll_2[1:]
    cdf_roll_3 = np.roll(cdf_3,1)
    pmf_3 = cdf_3[1:] - cdf_roll_3[1:]


    plt.switch_backend('agg')
    # fig = plt.figure()
    plt.figure(figsize=(8, 5))
    # plt.plot(x[1:],pmf_1)
    # plt.plot(x[1:],pmf_2)
    # plt.plot(x[1:],pmf_3)
    plt.plot(x,stats.norm.pdf(x,loc=mu_j, scale = sigma_1), label='$R(x_1^0)$')
    plt.plot(x,stats.norm.pdf(x,loc=mu_i, scale = sigma_2), label='$R(x_1^1)$')
    plt.plot(x,stats.norm.pdf(x,loc=mu_j+C/4, scale = sigma_3), label='$R(x_i)$')
    # plt.plot(x,stats.norm.pdf(x,loc=mu_j, scale = sigma_1), label='$R(x_1)$')
    # plt.plot(x,stats.norm.pdf(x,loc=mu_i, scale = sigma_2), label='$R(x\'_1)$')
    # plt.plot(x,stats.norm.pdf(x,loc=mu_j+C/4, scale = sigma_3), label='$R(x_i)$')
    xl = [(mu_i+mu_j)/2]*300
    y = np.linspace(0,0.6,300)
    plt.plot(xl,y,linewidth=0.6, linestyle='-.', color='grey')
    x1 = np.where(np.logical_and(np.greater(pmf_1,pmf_3),np.greater(pmf_1,pmf_2)))[0]
    x2 = np.where(np.logical_and(np.greater(pmf_2,pmf_3),np.greater(pmf_2,pmf_1)))[0]


    if mech == 'laplacian':
        if len(x1)>0:
            #xi looks like x10
            if np.abs(x[x1[0]] + x_range) < 0.1:
                p10 = stats.laplace.cdf(x[x1[-1]], loc=mu_j+C/4, scale = b3)
            elif np.abs(x[x1[-1]] - x_range) < 0.1:
                p10 = 1 - stats.laplace.cdf(x[x1[0]], loc=mu_j+C/4, scale = b3)
            else:
                p10 = stats.laplace.cdf(x[x1[-1]], loc=mu_j+C/4, scale = b3) - stats.laplace.cdf(x[x1[0]], loc=mu_j+C/4, scale = b3)
        else:
            p10=1
        if len(x2)>0:
            #xi looks like x11
            if np.abs(x[x2[0]] + x_range) < 0.1:
                p11 = stats.laplace.cdf(x[x2[0]], loc=mu_j+C/4, scale = b3)
            elif np.abs(x[x2[-1]] - x_range) < 0.1:
                p11 = 1 - stats.laplace.cdf(x[x2[0]], loc=mu_j+C/4, scale = b3)
            else: 
                p11 = stats.laplace.cdf(x[x2[-1]], loc=mu_j+C/4, scale = b3) - stats.laplace.cdf(x[x2[0]], loc=mu_j+C/4, scale = b3)
        else:
            p11=1
    if mech == 'gaussian':
        if len(x1)>0:
            #xi looks like x10
            if np.abs(x[x1[0]] + x_range) < 0.1:
                p10 = stats.norm.cdf(x[x1[-1]], loc=mu_j+C/4, scale = sigma_3)
            elif np.abs(x[x1[-1]] - x_range) < 0.1:
                p10 = 1 - stats.norm.cdf(x[x1[0]], loc=mu_j+C/4, scale = sigma_3)
            else:
                p10 = stats.norm.cdf(x[x1[-1]], loc=mu_j+C/4, scale = sigma_3) - stats.norm.cdf(x[x1[0]], loc=mu_j+C/4, scale = sigma_3)
        else:
            p10 = 1
        if len(x2)>0:
            #xi looks like x11
            if np.abs(x[x2[0]] + x_range) < 0.1:
                p11 = stats.norm.cdf(x[x2[0]], loc=mu_j+C/4, scale = sigma_3)
            elif np.abs(x[x2[-1]] - x_range) < 0.1:
                p11 = 1 - stats.norm.cdf(x[x2[0]], loc=mu_j+C/4, scale = sigma_3)
            else: 
                p11 = stats.norm.cdf(x[x2[-1]], loc=mu_j+C/4, scale = sigma_3) - stats.norm.cdf(x[x2[0]], loc=mu_j+C/4, scale = sigma_3)
        else:
            p11=1
    xp = x[x1[0]: x1[-1]]
    print(x1[-1], x1[0])
    plt.fill_between(xp, stats.norm.pdf(xp, loc=mu_j+C/4, scale = sigma_3), xp*0, color='g', alpha=0.3, label="$R(x_1^0)$ max")
    # plt.fill_between(xp, stats.norm.pdf(xp, loc=mu_j+C/4, scale = sigma_3), xp*0, color='g', alpha=0.3, label="$R(x_1)$ max")
    xp = x[x2[0]: x2[-1]]
    print(x2[-1], x2[0])
    plt.fill_between(xp, stats.norm.pdf(xp, loc=mu_j+C/4, scale = sigma_3), xp*0, color='orange', alpha=0.3, label="$R(x_1^1)$ max")
    # plt.fill_between(xp, stats.norm.pdf(xp, loc=mu_j+C/4, scale = sigma_3), xp*0, color='orange', alpha=0.3, label="$R(x\'_1)$ max")
    # plt.xlabel(fontsize=14)
    # plt.ylabel(fontsize=14)
    plt.legend(fontsize=14)
    plt.ylim(0,0.7)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.show()

    plt.savefig('./dist.png', dpi=600, bbox_inches='tight')
    plt.close()

    return min(p10,p11)*2


ei = 4
ej = 5
di = 10**(-10)
dj = 10**(-10)
# mech = "laplacian"
mech = "gaussian"
C = 0.6

pij = compute_pij(ei,di,ej,dj,mech,C)
print('eps:',ei,ej)
print('HP:',pij)

# print('HP RR:', (1+math.e**(-ej))/(1+math.e**ei))
# print('HP RR:', (1+math.e**(-ei))/(1+math.e**ej))
print('hiding:', 1/(0+math.e**ej) )
print('stronger:', 2/(1+math.e**ej) )
print('generalized:', 2/(1+math.e**ei) )
# # # plot_pdf(ei,di,ej,dj,mech,C, roots)