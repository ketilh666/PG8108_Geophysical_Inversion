# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:03:21 2022

@author: kehok
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

# KetilH stuff
import saturation as sat

#-------------------------------
#   Read logs
#-------------------------------


kblocked = True
if kblocked:
    fname = 'data/Wells_Blocked_Regular.pkl'
    blk = '_blocked'
else:
    fname = 'data/Wells_Lasfiles.pkl'
    blk = ''

with open(fname, 'rb') as fid: well_list = pickle.load(fid)

#----------------------------------------
#  Forward modeling parameters
#----------------------------------------

# oil/gas ratio
soil = 0.5

swmin, swmax, nsw = 0.01, 1, 100
sw = np.linspace(swmin, swmax, nsw)

#----------------------------------------
#   Stat.stuff
#----------------------------------------

# Noise mean and variance
sig_err = {'log_rt': 1.0, 'vp': 100.0, 'rhob': 50.0}

# Prior mean and variance
mu_pri, sig_pri = 0.7, 0.5

#----------------------------------------
#   Rock phys inversion. Compute saturation 
#   given log_rt, vp, rho, vcl, phi
#----------------------------------------

#inv_keys = ['vp']
#inv_keys = ['log_rt', 'vp']
inv_keys = ['log_rt', 'vp', 'rhob']

print('inv_keys: {}'.format(inv_keys))

mgi_list = [None for wl in well_list]
for kk, well in enumerate(well_list):

    mgi_list[kk] = sat.sat_inversion(well, sig_err, inv_keys, mu_pri, sig_pri, 
                                     sw, soil, kplot=False, verbose=1)
    
    # Remove predictions above Top Stø 
    kt = well['tops']['name'].index('Top Stø')
    ind = well['tvd'] < well['tops']['tvd'][kt]
    mgi_list[kk]['post_mu'][ind]  = np.nan
    mgi_list[kk]['post_map'][ind] = np.nan
    mgi_list[kk]['post_sig'][ind] = np.nan
    
#------------------------------
#   PLot results; 3 wells
#------------------------------

key_list  = ['phi', 'vcl', 'log_rt', 'vp', 'rhob', 'sw']
unit_list = ['[-]', '[-]', '[ohmm]', '[m/s]', '[kg/m3]', '[-]']

for kk, well in enumerate(well_list):
    
    fig, axs = plt.subplots(1,6, figsize=(16, 8))
    
    # PLot well tops
    for jj, key in enumerate(key_list):
        ntops = len(well['tops']['tvd'])
        ax = axs.ravel()[jj]
        for ii in range(ntops):
            tvd = well['tops']['tvd'][ii]*np.ones(2, dtype=float)
            par = np.array([np.nanmin(well[key]), np.nanmax(well[key])])
            if jj==5: par=np.array([0, 1.0])
            ax.plot(par, tvd, 'k')
            if jj==5: ax.text(1, tvd[0], well['tops']['name'][ii])

    # plot measured logs
    for jj, key in enumerate(key_list):        
        ax = axs.ravel()[jj]
        ax.plot(well[key], well['tvd'], 'b', label='measured')
        ax.set_xlabel('{} {}'.format(key, unit_list[jj]))
        ax.set_title(key)
        ax.invert_yaxis()
        if jj==0: ax.set_ylabel('tvd [m]')
        if jj==5: ax.set_xlim(0, 1)
        
    # PLot synt logs
    jj = 5
    ax = axs.ravel()[jj]
    dd = mgi_list[kk]
    ax.plot(dd['post_mu'], dd['tvd'], 'g', label='post_mu') 
    ax.plot(dd['post_map'], dd['tvd'], 'r', label='post_map') 
    ax.plot(dd['post_map']+dd['post_sig'], dd['tvd'], 'r:') 
    ax.plot(dd['post_map']-dd['post_sig'], dd['tvd'], 'r:') 
    ax.legend()
        
    fig.tight_layout(pad=1.0)
    fig.suptitle(well['name'])

    png_name = 'png/Rock_Physics_Inversion{}_{}.png'.format(blk,kk)
    fig.savefig(png_name)

plt.show(block=False)

    
    
    
    
    


