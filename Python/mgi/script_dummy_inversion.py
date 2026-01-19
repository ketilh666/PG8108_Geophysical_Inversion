# -*- coding: utf-8 -*-
"""
Test script for the Bayesian inversion module.

Simple synt test data are created inside the module

Programmed: Ketil Hokstad 23. Spetember 2020

Created on Wed Sep 23 08:43:48 2020
@author: kehok
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

# KetilH stuff
import inversion as inv

#-----------------------------------------------
#   Make some bullshit data
#-----------------------------------------------

d2r = np.pi/180.0

nd = 60
amps, ampc  = 1.0, 1.0 

pp_true = 5 + 350*rnd.rand(nd) # Numbers between 5 and 355   
data = [amps*np.sin(d2r*pp_true), ampc*np.cos(d2r*pp_true)]
nm = len(data)

# Make some noise
relnoise = 0.10 
noise = [amps*relnoise*rnd.rand(nd), ampc*relnoise*rnd.rand(nd)]

for jj in range(nm):
    data[jj] = data[jj] + noise[jj]

#------------------------------------------------
#   Prior (very wide)
#------------------------------------------------

na = 361
aa = np.linspace(0,360,na)

mu_pri  = 180.0
sig_pri = 360.0
pri = inv.Prior(mu=mu_pri, sig=sig_pri, aa=aa, kplot=False, verbose=1)

#-------------------------------------------------
#    Forward models for the inversion
#-------------------------------------------------

synt = [amps*np.sin(d2r*aa), amps*np.cos(d2r*aa)]

#-------------------------------------------------
#    Construct the likelihood object
#-------------------------------------------------

mu_err  = [0,0]
sig_err = [3*amps*relnoise, 3*ampc*relnoise]
mlh = inv.Likelihood(data=data, synt=synt, aa=aa, sig=sig_err)

#-------------------------------------------------
#   Run inversion
#-------------------------------------------------

kkk = 0 
post = mlh.bayes_inv(pri, kkk=kkk, verbose=1, kplot=True)

#-------------------------------------------------
#    Plot result
#-------------------------------------------------

fig, axs = plt.subplots(2,2, figsize=(16,12))

ax = axs.ravel()[0]
ax.scatter(pp_true, data[0], c='g', marker='o', label='data 0')
ax.scatter(pp_true, data[1], c='b', marker='o', label='data 1')
ax.set_xlabel('pp_true [-]')
ax.set_ylabel('data [-]')
ax.legend()

ax = axs.ravel()[1]
uv = np.max(pp_true)*np.array([0,1], dtype=float)
ax.plot(uv,uv,'k-')
ax.scatter(pp_true, post.map, c='b', marker='o', label='map')
ax.scatter(pp_true, post.mu,  c='r', marker='o', label='mu_post')
ax.set_xlabel('pp_true [-]')
ax.set_ylabel('pp_estimated [-]')
ax.legend()

ax = axs.ravel()[2]
ax.plot(post.aa, post.mlh_list[0], 'g-', label='mlh_0')
ax.plot(post.aa, post.mlh_list[1], 'b-', label='mlh_1')
ax.plot(post.aa, post.pdf_post, 'r-', label='post')
ax.scatter(pp_true[post.kkk], 0, c='k', marker='x', label='true')
ax.set_xlabel('pp [-]')
ax.set_ylabel('pdf [-]')
ax.legend()

ax = axs.ravel()[3]
jnd = np.argsort(pp_true)
ax.errorbar(pp_true[jnd], post.mu[jnd], yerr=post.sig[jnd], 
                    marker='o', c='b', label='post') 
ax.scatter(pp_true[jnd],pp_true[jnd], c='r', marker='o', label='true')
ax.legend()

fig.savefig('png/Dummy_Inversion.png') 

plt.show(block=False)







