# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 07:38:02 2019

@author: kehok
"""

#------------------------------------------------------------------------
#    Test some ML stuff on Bent HIll ODP data (Leg 139 and 169)
#------------------------------------------------------------------------

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, but is otherwise unused.
import matplotlib.pyplot as plt

# SciKit Learn stuff (install with pip install sklearn)
import sklearn.linear_model as lin
import sklearn.ensemble as ens
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fixing some plotting issues 
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#----------------------------------------
#   Set job parameters
#---------------------------------------

n_feat = 2
n_est = 100

kscl_data  = False
kplot_data = False

in_file =  'data/ODPdata_BentHill_allHoles.xlsx'
with pd.ExcelFile(in_file) as fid:
    df_in = pd.read_excel(fid,'Data')

# Lumpt sulfide fractions into one total vSMS
df_in['noise'] = 0.1*np.random.random_sample((df_in.shape[0],1))
df_in['vSMS'] = df_in['vPyr'] + df_in['vPoh'] + df_in['vChp'] + df_in['vSph'] + df_in['noise']

#-----------------------------------------------------------------
#   Invetigate some ML stuff in sklearn
#-----------------------------------------------------------------

# Target and features
targ = 'vSMS'

# Use 2 or 3 features for train&test?
if n_feat == 2:
    feat_list = ['rhob','vp']
    ind = (df_in['vp'].notnull()) & (df_in['rhob'].notnull()) 
else:
    feat_list = ['rhob','vp', 'TC']
    ind = (df_in['vp'].notnull()) & (df_in['rhob'].notnull()) & (df_in['TC'].notnull())

# Get rid of the Nans
df_ml = df_in[ind]
print('feat_list = {}'.format(feat_list))
print('n_samp = {}'.format(df_ml.index.size))

# Split in test and train data: X=features, y=targets
trainX0, testX0, trainy, testy = train_test_split(df_ml[feat_list], df_ml[targ], 
                                    test_size=0.33, shuffle=True, random_state=42)
    
#---------------------------------------------    
# Optional: Scale the data to standardize
#---------------------------------------------
   
if kscl_data:
    # Standard scaler
    scaler = StandardScaler()
    scaler.fit(trainX)    
    trainX = scaler.transform(trainX0)    
    testX  = scaler.transform(testX0)
else:
    # Do nothing
    trainX, testX = trainX0, testX0    


#------------------------------------------------------------------
#   Initialize list of regressor objects
#------------------------------------------------------------------

reg_list = [
       lin.LinearRegression(),
       ens.RandomForestRegressor(n_estimators=n_est),
       ens.ExtraTreesRegressor(n_estimators=n_est),
       ens.GradientBoostingRegressor(n_estimators=n_est),
       ]

#-----------------------------------------------------------------
#   Train and test all regressors
#-----------------------------------------------------------------

# Initialize List of dicts to gather results
tst_list = [{} for reg in reg_list]

# Loop over regressor objects
for jj, reg in enumerate(reg_list):
    
    # Train
    reg.fit(trainX, trainy)

    # Test
    tst_list[jj]['pred']  = reg.predict(testX)
    
    # R2 scores
    tst_list[jj]['r2_train'] = reg.score(trainX, trainy)
    tst_list[jj]['r2_test']  = reg.score(testX, testy)

    # Only decission trees give feature importance
    try:
        tst_list[jj]['featimp'] = reg.feature_importances_
    except:
        tst_list[jj]['featimp'] = [None for feat in feat_list]
        
    # Regressor name (for plot headers)
    tst_list[jj]['name'] = str(reg.__str__).split()[3]

#---------------------------------------------------------------
#   Make som plots
#---------------------------------------------------------------

# PLot confusion
nreg = len(tst_list)
fig, axs = plt.subplots(1,nreg, figsize=(3.5*nreg, 4))

# 45 degree line
x45 = y45 = np.array([0, 0.8])

for jj, tst in enumerate(tst_list):

    ax = axs.ravel()[jj]
    ax.plot(x45, y45, 'k-')
    ax.scatter(testy, tst['pred'], c='r')
    ax.set_title(tst['name'])
    ax.set_xlabel('vSMS true [-]')
    ax.set_ylabel('vSMS pred [-]')
    ax.set_aspect('equal')

fig.suptitle('Confusion. Held-out test samples (n_feat = {})'.format(n_feat))
fig.savefig('png/BentHill_confusion_' + str(n_feat) + '.png')

# PLot train and test scores
names = [tst['name'] for tst in tst_list]
r2_train = [tst['r2_train'] for tst in tst_list]
r2_test  = [tst['r2_test'] for tst in tst_list]
fig, ax = plt.subplots(1)
ax.plot(names, r2_train,'g-o', label='train') 
ax.plot(names, r2_test,'b-o', label='test')
ax.set_ylim(0.5, 1)
ax.legend()
ax.set_xticklabels(names, rotation=40)    
ax.set_ylabel('R2 score [-]')
ax.set_title('Train&Test scores (n_feat = {})'.format(n_feat))
fig.savefig('png/BentHill_scores_' + str(n_feat) + '.png')

#-----------------------------------------------
#  PLot input data?
#-----------------------------------------------

if kplot_data:

    # Define cathegorical colors:
    litho = df_in['litho'].unique()
    col = {'Sediment':'g', 'Mafic':'b', 
           'SulfideMassive':'r', 'SulfideClastic': 'y', 'SulfideSediment': 'm'}
    
    fig, axs = plt.subplots(1,2, figsize=(12,5.5) )
    
    # PLot density vs TC
    ax = axs.ravel()[0]
    for cl in litho:
        wrk = df_in[df_in['litho']==cl] 
        ax.scatter(wrk['rhob'],wrk['TC'],marker='o', c=col[cl], label=cl)
    ax.legend()
    ax.set_xlim(1000,5000)
    ax.set_ylim(0,16)
    ax.set_xlabel('Density [kg/m3]')
    ax.set_ylabel('Thermal conductivity [W/mk]')
    
    # PLot density vs vp
    ax = axs.ravel()[1]
    for cl in litho:
        wrk = df_in[df_in['litho']==cl] 
        ax.scatter(wrk['rhob'],wrk['vp'],marker='o', c=col[cl], label=cl)
    ax.legend()
    ax.set_xlim(1000,5000)
    ax.set_ylim(1000,6500)
    ax.set_xlabel('Density [kg/m3]')
    ax.set_ylabel('P-wave velocity [m/s]')
    
    fig.suptitle('Bent Hill ODP Leg 139&169')
    fig.savefig('png/BentHill_dens_TC_and_vp.png')
    
    #-----------------------------------------------
    #   Make a 3D plot
    #-----------------------------------------------
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pp=ax.scatter(df_in['rhob'],df_in['TC'],df_in['vp'],c=df_in['vSMS'],marker='o')
    ax.set_xlabel('Density [kg/m3]')
    ax.set_ylabel('TC [W/mK]')
    ax.set_zlabel('vp [m/s]')
    fig.suptitle('Bent Hill: Density, TC, vp')
    cb=fig.colorbar(pp)
    cb.set_label('vSMS')
    fig.savefig('png/BentHill_3D_vSMS.png')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for cl in litho:
        wrk = df_in[df_in['litho']==cl] 
        ax.scatter(wrk['rhob'],wrk['TC'], wrk['vp'], marker='o', 
                   c=col[cl], label=cl)
    ax.legend()
    ax.set_xlabel('Density [kg/m3]')
    ax.set_ylabel('TC [W/mK]')
    ax.set_zlabel('vp [m/s]')
    fig.suptitle('Bent Hill: Density, TC, vp')
    fig.savefig('png/BentHill_3D_dens_TC_vp.png')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for cl in litho:
        wrk = df_in[df_in['litho']==cl] 
        ax.scatter(wrk['rhob'],wrk['TC'], wrk['porosity'], marker='o', 
                   c=col[cl], label=cl)
    ax.legend()
    ax.set_xlabel('Density [kg/m3]')
    ax.set_ylabel('TC [W/mK]')
    ax.set_zlabel('Porosity [-]')
    fig.suptitle('Bent Hill: Density, TC, porosity')
    fig.savefig('png/BentHill_3D_dens_TC_porosity.png')

plt.show(block=False)