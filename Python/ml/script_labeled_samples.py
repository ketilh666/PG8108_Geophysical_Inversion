# -*- coding: utf-8 -*-
"""
Create labeled samples using Archi and Gassmann equations
calibrated to Castberg wells, for train&test of ML model

Created on Fri Oct 22 13:04:10 2021

@author: kehok@equinor.com
"""

#---------------------------------------------------------------
#   Rock physics modeling using IDDP2 MGI_GT result
#---------------------------------------------------------------

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import pandas as pd

# Ketil's stuff
import lfp as lfp

#---------------------------------------------------------------
#  Forward modeling ranges
#---------------------------------------------------------------

# Thermal gradient
Tg = 38e-3

# oil/gas ratio
soil = 0.5

# Water saturation
swmin, swmax, nsw = 0.02, 1, 50
sw1 = np.linspace(swmin, swmax, nsw)

# Porosity
phimin, phimax, nphi = 0.05, 0.25, 5
phi1 = np.linspace(phimin, phimax, nphi)

# Clay fraction
vclmin, vclmax, nvcl = 0.2, 0.6, 3
vcl1 = np.linspace(vclmin, vclmax, nvcl)

# Depth
zmin, zmax, nz = 600, 2500, 20 
zzz1 = np.linspace(zmin, zmax, nz)

# Arrays for making labeled samples
[gsw, gphi, gvcl, gzzz] = np.meshgrid(sw1, phi1, vcl1, zzz1)
sw  = gsw.ravel()
phi = gphi.ravel()
vcl = gvcl.ravel()
zzz = gzzz.ravel()

#-----------------------------------------------------------------
#   Create labeled samples
#-----------------------------------------------------------------

so = soil*(1-sw)
sg = (1-soil)*(1-sw)
log_rt = lfp.archie(so, sg, phi, zzz)
vp, vs, rhob = lfp.gassmann(so, sg, phi, vcl, zzz, Tg=Tg)
ai = 1e-6*rhob*vp

#--------------------------------------
#   Add some noise
#--------------------------------------

noise = 10 # Percentage of noise to add
ns = sw.shape[0]
if noise > 0:
    rn0 = 1e-2*noise
    log_rt = (1.0 + rn0*(2*rnd.rand(ns)-1))*log_rt
    rhob   = (1.0 + rn0*(2*rnd.rand(ns)-1))*rhob
    vp     = (1.0 + rn0*(2*rnd.rand(ns)-1))*vp
    vs     = (1.0 + rn0*(2*rnd.rand(ns)-1))*vs
    ai    = (1.0 + rn0*(2*rnd.rand(ns)-1))*ai

#-----------------------------------------------------------------
#  Store in a dataframe
#-----------------------------------------------------------------

targ = ['sw']
aux_list  = ['phi', 'vcl']
feat_list = ['log_rt', 'vp', 'rhob', 'ai']
idd  = ['zzz']

cols = targ + aux_list + feat_list + idd
data = np.array([sw, phi, vcl, log_rt, vp, rhob, ai, zzz]).T
df = pd.DataFrame(columns=cols, data=data)

# Save to Excel file
fname = 'data/Train_and_Test_Data_noise' + str(noise) + '.xlsx'
with pd.ExcelWriter(fname) as fid:
    df.to_excel(fid, index=False)

#-----------------------------------------
#   Plot train&test data
#-----------------------------------------  

plot_list = targ + aux_list + feat_list
unit_list = ['[-]', '[-]', '[-]', '[ohmm]', '[m/s]', '[kg/m3]', '[kg/sm2]']
            
# Open a figure for plotting
fig, axs = plt.subplots(1,len(plot_list), figsize=(18,10))

for jj in range(len(plot_list)):
    ax = axs.ravel()[jj]
    key = plot_list[jj]
    ax.scatter(df[key], df['zzz'], marker='.')
    ax.set_ylabel('zbsf [m]')
    ax.set_xlabel('{} {}'.format(key, unit_list[jj]))
    ax.set_title(key)
    ax.invert_yaxis()

fig.tight_layout(pad=2.0)
fig.savefig('png/LFP_Labeled_Samples.png')

plt.show(block=False)