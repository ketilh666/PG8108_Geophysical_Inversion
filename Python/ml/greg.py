# -*- coding: utf-8 -*-
"""
Regression-type machine learning

Created on Mon Oct 12 14:23:16 2020
@author: kehok@equinor.com
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd
import numpy.random as rnd
from scipy.interpolate import griddata

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.cluster import k_means

#------------------------------------------------------------------
#  Fit ML regressors for a set of target and feature samples
#------------------------------------------------------------------

def fit_cv(reg_mu, reg_sig, df_smp, key_t, feat_list, **kwargs):
    
    """ Fit ML regressors by cross validation 
        for a set of target and feature samples.
    
    Two estimators are computed by CV:
        1. ML model for the mean
        2. ML model for the variance
            
    Parameters
    ----------
    reg_mu: initialized regressor object
        Regressor for estimating the mean
    reg_sig: initialized regressor object
        Regressor for estimating the variance
    df_smp: pd.DataFrame
        Target and feature samples for train and test
        Columns of the dataframe are assumed to be:
        [lon, lat, target, features] (in any order)
    key_t: str
        Target key (column name in df)
    feat_list: list of str. Features to use in ML train&test
        
    **kwargs
    --------
    test_size: float, optional (default is 0.2)
        Fraction of samples to be used for test
    n_cv: int, optional (deafult is 5)
        Cross validation fold
    qc_cv: bool, optional (default False)
    key_x: str, optional (default 'lon')
        Label for the x-coordinate/longitude
    key_y: str, optional (default 'lat')
        Label for the y-coordinate/latitude
    verbose: int, optional (default 0)
        Print shit?
    kplot: bool, optional (default False)
        Make som e QC plots?
        
    Returns
    -------
    reg_mu: regressor object
        Model for estimating the mean, best model from CV1
    reg_sig: regressor object
        Model for estimating the variance, best model from CV2
    test_mu: dict
        Regressor for estimating the mean, best from CV
    test_sig: dict
        Test prediction and score for variance
        
    Programmed: KetilH 15. October 2020
    """
    
    # Get the kwargs:
    test_size = kwargs.get('test_size', 0.2) 
    n_cv = kwargs.get('n_cv', 5) 
    use_all = kwargs.get('use_all', True)
    qc_cv = kwargs.get('qc_cv', False)

    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude    
    key_c = 'clu' # for internal use
    
    verbose = kwargs.get('verbose', 0)
    kplot = kwargs.get('kplot', False)

    if verbose:
        print('greg.fit_cv:')
        print(' o feat_list = {}'.format(feat_list))
        print(' o key_t = {}'.format(key_t))
        print(' o test_size = {}'.format(test_size))
        print(' o n_cv = {}'.format(n_cv))
        print(' o use_all = {}'.format(use_all))

    # Random splitting:  X=feature_matrix, y=target_vector
    trainX0, testX, trainy0, testy = train_test_split(df_smp[feat_list], 
                               df_smp[key_t], test_size=test_size, 
                               shuffle=True, random_state=0)
            
    test_size2 = 0.5 # always
    trainX1, trainX2, trainy1, trainy2 = train_test_split(trainX0, trainy0,
                               test_size=test_size2, shuffle=True, random_state=0)   
        
    # Use all for mean model:
    if use_all:
        trainX1, trainy1 = trainX0, trainy0
                
    ####  Cross-validation 1: Train mean model 
        
    # n-fold crossvalidation
    scoring = ['r2', 'explained_variance', 'neg_mean_squared_error']
    cv_mu = cross_validate(reg_mu,trainX1,trainy1, cv=n_cv, 
                           scoring=scoring,
                           return_train_score=True,
                           return_estimator=True)    

    # Test CV models for mean on held out data
    keys_mu = ['pred', 'r2', 'ev', 'mase']
    test_mu = {key: [np.nan for ii in range(n_cv)] for key in keys_mu}
    test_mu['targ'] = np.array(testy)      # Same for all
    for jj in range(n_cv):
        test_mu['pred'][jj] = cv_mu['estimator'][jj].predict(testX)
        test_mu['r2'][jj] = r2_score(testy, test_mu['pred'][jj])
        test_mu['ev'][jj] = explained_variance_score(testy, test_mu['pred'][jj])  
        test_mu['mase'][jj] = mase_score(testy, test_mu['pred'][jj])

    # Best model for mean:
    ind = np.array(test_mu['r2']) <= cv_mu['train_r2']   # To avoid underfitting
    kbest_mu = np.argmax(np.array(test_mu['r2'])[ind])
    reg_mu = cv_mu['estimator'][kbest_mu]
    test_mu['kbest'] = kbest_mu
    test_mu['feat_list'] = feat_list
    
    # Return full CV dict for QC?
    if qc_cv:
        cv_mu.pop('estimator') # No need to store all
        test_mu['cv'] = cv_mu
    

    ####  Cross-validation 2: Train variance model 
    
    # Using the best estimator to compute mean for Step2
    trainy2_se = (trainy2 - reg_mu.predict(trainX2))**2
    cv_sig = cross_validate(reg_sig,trainX2,trainy2_se, cv=n_cv, 
                            scoring=scoring,
                            return_train_score=True,
                            return_estimator=True)    
   
    # Test CV models for variance on held out data
    keys_sig = ['pred', 'r2', 'ev', 'mase']
    test_sig = {key: [np.nan for ii in range(n_cv)] for key in keys_sig}
    testy_se = (testy - reg_mu.predict(testX))**2   # square error
    testy_rmse = np.sqrt(testy_se)                  # linear error
    test_sig['targ'] = testy_rmse                   # same for all
    for jj in range(n_cv):
        try:
            test_sig['pred'][jj] = np.sqrt(cv_sig['estimator'][jj].predict(testX))
            test_sig['r2'][jj] = r2_score(testy_rmse, test_sig['pred'][jj])
            test_sig['ev'][jj] = explained_variance_score(testy_rmse, test_sig['pred'][jj])    
        except:
            test_sig['pred'][jj] = testy_rmse
            test_sig['r2'][jj] = -1
            test_sig['ev'][jj] = -1          
        test_sig['mase'][jj] = mase_score(testy_rmse, test_sig['pred'][jj])
            
    # Best model for variance:
    ind = np.array(test_sig['r2']) <= cv_sig['train_r2']   # To avoid underfitting
    kbest_sig = np.argmax(np.array(test_sig['r2'])[ind])
    #kbest_sig = np.argmax(test_sig['r2'])
    reg_sig = cv_sig['estimator'][kbest_sig]
    test_sig['kbest'] = kbest_sig
    test_sig['feat_list'] = feat_list
 
    # Return full CV dict for QC?
    if qc_cv:
        cv_sig.pop('estimator') # No need to store all
        test_sig['cv'] = cv_sig

    # Return estimators and score stats:
    return reg_mu, reg_sig, test_mu, test_sig

#------------------------------------------------------------------
#  Fit ML regressors for a set of target and feature samples
#------------------------------------------------------------------

def predict_ml(reg_mu, reg_sig, df_grd, key_t, feat_list, **kwargs):
    
    """ Predict target map from a set of featiure maps.
    
    Two basic predictions are performed:
        1. Predict the target mean
        2. Predict the target  variance
    The prediction is done using two ML models, one for each step..
    
    Subsequently the mean and variance can be used to compute 
    the P10/P50/P90 estimates
    
    Parameters
    ----------
    reg_mu: initialized regressor object
        Regressor for estimating the mean
    reg_sig: initialized regressor object
        Regressor for estimating the variance
    df_grd: pd.DataFrame
        Feature maps for prediction of target
        [lon, lat, features] (in any order)
    key_t: str
        Target key (column root name in output df)
    feat_list: list of str. Features to use in ML prediction
        
    **kwargs
    --------
    key_x: str, optional (default 'lon')
        Label for the x-coordinate/longitude
    key_y: str, optional (default 'lat')
        Label for the y-coordinate/latitude
    verbose: int, optional (default 0)
        Print shit?
    kplot: bool, optional (default False)
        Make som e QC plots?
        
    Returns
    -------
    pred_maps: pd.DataFrame
        Target maps predicted by ML models
        
    Programmed: KetilH 16. October 2020
    """
    
    # Get the kwargs:
    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude    
    
    verbose = kwargs.get('verbose', 0)

    if verbose:
        print('greg.predict_ml:')
        print(' o key_t = %s' %key_t)

    # Get the list of features
    perc_list = ['P90','P50','P10','mu','sig']
    
    # Predict the mean
    pred_mu   = reg_mu.predict(df_grd[feat_list])
    
    # Predict the variance squared
    pred_sig = np.sqrt(reg_sig.predict(df_grd[feat_list]))
    
    # Output pd.DatFrame:
    pred_maps = pd.DataFrame(columns=[key_x, key_y] + perc_list)

    # Output maps:
    sf = 1.28 # To compute P10 and P90
    pred_maps[perc_list[0]] = pred_mu - sf*pred_sig
    pred_maps[perc_list[1]] = pred_mu
    pred_maps[perc_list[2]] = pred_mu + sf*pred_sig
    pred_maps[perc_list[3]] = pred_mu
    pred_maps[perc_list[4]] = pred_sig
    
#    # Clip on zero:
#    for ii, key in enumerate(perc_list):
#        pred_maps[key] = np.clip(pred_maps[key], a_min=0, a_max=None)
    
    return pred_maps
    
#----------------------------------------------------------
#    MASE score
#----------------------------------------------------------

def mase_score(y_true, y_pred):
    """ Compute Mean Absolute Scaled Error (MASE) score  """

    nn = y_true.shape[0]
    rnum = y_true - y_pred
    rden = y_true - np.mean(y_true)
    mase = np.sum(np.abs(rnum)/np.abs(rden))/np.float(nn)

    return mase

#----------------------------------------------------------
#    PLot functions
#------------------------------------------------------------

def plot_correl(df_smp, key_t, feat_list, **kwargs):
    """ Plot some data analytics """
    
    # Get the kwargs
    key_x = kwargs.get('key_x', 'lon') # Column key for x or longitude 
    key_y = kwargs.get('key_y', 'lat') # Column key for y or latitude
    title = kwargs.get('title', 'Feature correlation') # Plot title
    
    # Pearson correlation:
    pear_list = [key_t] + feat_list
    n_pear = len(pear_list)
    rr_pear = df_smp[pear_list].corr()

    # Make a heatmap
    fig = plt.figure(figsize=(12,10))
    ax = sns.heatmap(rr_pear, annot=True,
                     xticklabels=pear_list, yticklabels=pear_list,
                     cmap=cm.Reds, vmin=-1, vmax=1, square=True)
    ax.set_yticklabels(pear_list, rotation=0)
    ax.set_title(title, fontsize=14)
    bot, top = plt.ylim()
    plt.ylim(bot+0.5,top-0.5)    
        
    return fig

def plot_feat_importance(feat_imp, **kwargs):
    """ Plot feature importance barchart """

    # Get kwargs
    nf = feat_imp.shape[0]
    feat_name = kwargs.get('feat_name', ['feat_'+str(ii)  for ii in range(nf)])
    title = kwargs.get('title', 'Relative importance')

    # Sort to get indices:
    ind = np.argsort(feat_imp)

    # Plot barchart
    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    ax.barh(np.array(feat_name)[ind], feat_imp[ind])
    ax.set_xlabel('Relative importance [-]')
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    return fig

def plot_confusion(pred, targ, **kwargs):
    """ Plot confusion """

    # Get kwargs
    key_t = kwargs.get('key_t', 'target')
    unit_t = kwargs.get('unit_t', '[-]')
    title = kwargs.get('title', 'Confusion')

    # 45 degree line
    vmin = np.min([np.min(pred), np.min(targ)])
    vmax = np.max([np.max(pred), np.max(targ)])
    a45 = np.array([vmin, vmax],dtype=float)

    # Make plot
    fig, ax = plt.subplots(1,1, figsize=(6,5))
    fig.tight_layout(pad=4.0)
    ax.plot(a45,a45,'k-')
    ax.scatter(pred, targ,s=8, c='r',marker='o')
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal','box')
    ax.set_xlim(vmin,vmax)
    ax.set_ylim(vmin,vmax)
    ax.set_xlabel('Predicted ' + key_t  + unit_t); 
    ax.set_ylabel('Observesd ' + key_t  + unit_t); 
    plt.tight_layout()

    return fig

def plot_scores(test_mu, test_sig, score_list, **kwargs):
    """ Plot test scores from cross validation """
    
    title  = kwargs.get('title', '') # Plot title

    fig, ax = plt.subplots(1, len(score_list), figsize=(10,4))
    fig.suptitle(title)
    fig.tight_layout(pad=4.0)
    
    for ii, ai in enumerate(ax):
        score = score_list[ii]
        indx = [jj for jj in range(len(test_mu[score]))]
    
        ai.scatter(indx, test_mu[score], label='CV1')
        ai.scatter(indx, test_sig[score], label='CV2')
        ai.legend()

        ai.set_title(score)
        ai.set_xlabel('CV iter [-]')
        ai.set_ylabel('Score [-]')

    return fig
