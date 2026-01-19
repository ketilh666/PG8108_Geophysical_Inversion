# -*- coding: utf-8 -*-
"""
Purpose: 
Run regression ML  

Created on Fri Apr 17 09:23:37 2020
@author: kehok@equinor.com
"""
#------------------------------------------------------
#  Imports
#------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.ensemble as ens
import pickle

# KetilH stuff: Geostatistical regression type ML
import greg as greg 

#---------------------------------------------------
# Read input data
#---------------------------------------------------

# Samples: features and target
smp_file = 'data/Train_and_Test_Data_noise5.xlsx'
df_smp_in = pd.read_excel(pd.ExcelFile(smp_file))

# Make som fake x,y coordinates (for plotting only)
df_smp_in[['x', 'y']] = df_smp_in[['sw', 'phi']]

# Features on grid for prediction:
well_file = 'data/Wells_Blocked_Regular.pkl'
with open(well_file, 'rb') as fid: well_list = pickle.load(fid)

#----------------------------------------
#   Set job parameters
#---------------------------------------

targ = df_smp_in.columns.to_list()[0]
feat_list_all = ['phi', 'vcl', 'log_rt', 'vp', 'rhob', 'ai', 'zzz']
feat_unit_all = ['[-]', '[-]', '[ohmm]', '[m/s]', '[kg/m3]', '[kg/sm2]', 'm']

krun = 2
if   krun == 0:
    ind_feat_use = [0, 1, 2, 3, 4, 6]  # Use vp and density
elif krun == 1:
    ind_feat_use = [0, 1, 2, 5, 6]     # Use acoustic impedance
elif krun == 2:
    ind_feat_use = [1, 2, 3, 4, 6]     # Use vp and density, no porosity
elif krun == 3:
    ind_feat_use = [1, 2, 5, 6]        # Use acoustic impedance, no porosity

# Feature list for current run
feat_list = [feat_list_all[jj] for jj in ind_feat_use]
feat_unit = [feat_unit_all[jj] for jj in ind_feat_use]

# ML pars:
test_size = 0.20

# Column keys and units:
key_x, unit_x = 'x', ' [dummy]'
key_y, unit_y = 'y', ' [dummy]'
key_t, unit_t = 'sw', ' [-]'

#------------------------------------------------------------
#  Thrash some sample data?
#------------------------------------------------------------

df_smp = df_smp_in[[key_x, key_y, key_t] + feat_list]

#------------------------------------------------------------
#   Train ML models for mean and varaince:
#     o reg_mu : model for estimating mean
#     o reg_sig: model for estimating variance
#------------------------------------------------------------

# Init regressor objects:
n_est = 100
reg_mu  = ens.RandomForestRegressor(n_estimators=n_est)
reg_sig = ens.RandomForestRegressor(n_estimators=n_est)

# Cross validation train/test
reg_mu, reg_sig, test_mu, test_sig = greg.fit_cv(reg_mu, reg_sig, df_smp,
                                     key_t, feat_list, verbose=1)

#------------------------------------------------------------
#   Predict target from well logs
#------------------------------------------------------------

nwell = len(well_list)
pred_list = [None for jj in range(nwell)]
for jj, well in enumerate(well_list):
    
    zzz = well['tvd'] - well['zsf']
    data = np.array([well['phi'], well['vcl'], well['log_rt'], 
                     well['vp'], well['rhob'], well['ai'], zzz]).T
    df_grd_in = pd.DataFrame(columns=feat_list_all, data=data)
        
    # Get the features to use in ML
    df_grd = df_grd_in[feat_list]

    pred_list[jj] = greg.predict_ml(reg_mu, reg_sig, df_grd, 
                                    key_t, feat_list, verbose=1)
    
    # Mute above Top reservoir
    kt = well['tops']['name'].index('Top Stø')
    ind = well['tvd'] < well['tops']['tvd'][kt]
    pred_list[jj].loc[ind, 'mu']  = np.nan
    pred_list[jj].loc[ind, 'sig']  = np.nan
    
#------------------------------------------------------------
#   ML performance plots
#------------------------------------------------------------

# Just for plotting labels: get the regressor name
reg_name, srun = str(reg_mu.__str__).split()[3], ''

# Plot correlation heatmap:
title = 'Feature correlation'
fig_cr = greg.plot_correl(df_smp, key_t, feat_list, 
                          key_x=key_x, key_y=key_y, title=title)

# Plot cross-validation scores:
scores, title = ['r2', 'ev', 'mase'], srun + reg_name
fig_sc = greg.plot_scores(test_mu, test_sig, scores, title=title)

# PLot feature importance:
title_mu, title_sig = srun + 'CV1 mean', srun + 'CV2 variance'
fig_f1 = greg.plot_feat_importance(reg_mu.feature_importances_, 
                                  feat_name=feat_list, title=title_mu)
fig_f2 = greg.plot_feat_importance(reg_sig.feature_importances_, 
                                  feat_name=feat_list, title=title_sig)

# Plot confusion:
kbest = test_mu['kbest']
title = srun + 'Confusion'
fig_cf = greg.plot_confusion(test_mu['pred'][kbest], test_mu['targ'], 
                                    title=title, key_t=key_t, unit_t=unit_t)

# Save train&test figs to png files
fig_cr.savefig('png/LFP_run' + str(krun) + '_Feature_Correlation.png')
fig_sc.savefig('png/LFP_run' + str(krun) + '_CV_Scores.png')
fig_cf.savefig('png/LFP_run' + str(krun) + '_Confusion.png')
fig_f1.savefig('png/LFP_run' + str(krun) + '_FeatureImportance_CV1.png')
fig_f2.savefig('png/LFP_run' + str(krun) + '_FeatureImportance_CV2.png')

#------------------------------
#   Plot ML predictions
#------------------------------

key_list  = ['phi', 'vcl', 'log_rt', 'vp', 'rhob', 'sw']
unit_list = ['[-]', '[-]', '[ohmm]', '[m/s]', '[kg/m3]', '[-]']

for kk, well in enumerate(well_list):
    
    fig_sw, axs = plt.subplots(1,6, figsize=(16, 8))

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
        
    # PLot predicted saturation
    jj = 5
    ax = axs.ravel()[jj]
    mu_sw, sig_sw = pred_list[kk]['mu'], pred_list[kk]['sig']
    ax.plot(mu_sw, well['tvd'], 'r', label='pred_mu') 
    ax.plot(mu_sw - sig_sw, well['tvd'], 'r:') 
    ax.plot(mu_sw + sig_sw, well['tvd'], 'r:') 
    ax.legend()
        
    fig_sw.tight_layout(pad=1.0)
    fig_sw.suptitle(well['name'])
    png_name = 'png/LFP_run' + str(krun) + '_sw_ML_Prediction_' + str(kk) + '.png'
    fig_sw.savefig(png_name)

plt.show(block=False)
