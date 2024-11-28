import numpy as np

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

delta_l=0
n=1000
eps0,_=gen_eps(0.05, 1, n, 'Gauss2')    
fenmu = np.sum(eps0**2)
g = 1
for i in range(0,n,100):
    e = eps0[i]
    sigma = 1/e
    args = [g**2+sigma**2, -2*g**2-2*sigma**2, 2*sigma**2+g**2, 0, -e**2/fenmu]
    r = np.roots(args)
    idx = np.where(np.iscomplex(r)==False)
    # print(idx, r[idx])
    idx_pos = np.where(r[idx]>0)
    print(e, r[idx][idx_pos])