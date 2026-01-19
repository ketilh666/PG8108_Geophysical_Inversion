# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:03:21 2022

@author: kehok
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

# KetilH stuff
import lfp as lfp 

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

#------------------------------
#    Rockphys modeling
#------------------------------

nwell = len(well_list)
Tg = 38.0 # Thermal gradient, oC/km

for well in well_list:
    
    phi = well['phi']  # Total porosity
    So  = well['so']   # Oil saturation
    Sg  = well['sg']   # Gas saturation
    vcl = well['vcl']       # Clay fraction
    
    # Rockphysics modeling
    zzz = well['tvd'] - well['zsf'] # Depth below mudline
    log_rt = lfp.archie(So, Sg, phi, zzz)
    vp, vs, rho = lfp.gassmann(So, Sg, phi, vcl, zzz, Tg=Tg)
    
    # Store synt logs with the other log curves
    well['synt_log_rt'] = log_rt.copy()
    well['synt_vp']  = vp.copy()
    well['synt_vs']  = vs.copy()
    well['synt_rhob'] = rho.copy()
    
#------------------------------
#   PLot logs; 3 wells
#------------------------------

key_list  = ['phi', 'vcl', 'log_rt', 'vp', 'rhob', 'sw']
unit_list = ['[-]', '[-]', '[ohmm]', '[m/s]', '[kg/m3]', '[-]']
key_synt  = ['synt_log_rt', 'synt_vp', 'synt_rhob']

for kk, well in enumerate(well_list):
    
    print(kk, well['name'], well['tvd'].shape)
    
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
    for jj, key in enumerate(key_synt):
        ax = axs.ravel()[jj+2]
        ax.plot(well[key], well['tvd'], 'r', label='modeled') 
        ax.legend()
        
    fig.tight_layout(pad=1.0)
    fig.suptitle(well['name'])
    png_name = 'png/Rock_Physics_Calibration{}_{}.png'.format(blk,kk)
    fig.savefig(png_name)

plt.show(block=False)

    
    
    
    
    


