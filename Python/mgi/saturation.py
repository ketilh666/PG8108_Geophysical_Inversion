# -*- coding: utf-8 -*-
"""
Inversion of resistivity, P-wave veocity and (optionally) density 
for water saturation

Functions defined
-----------------    

Programmed: Ketil Hokstad, 20. April 2022

Created on Wed Apr 20 07:14:28 2022
@author: kehok@equinor.com
"""

# Standard Python stuff
import matplotlib.pyplot as plt
import numpy as np  

# KetilH stuff
import lfp  as lfp
import inversion as inv

#------------------------------------------------------
#   MGI LFP inversion function
#------------------------------------------------------
def sat_inversion(dd, sig_err, inv_keys, mu_pri, sig_pri, sw, soil, **kwargs):
    """ Inversion of log resistivity, P-wave velocity, density for saturation.
    
    Model parameters inverted:
        o log_rh [ohmm], log10 resistivity
        o vp [m/s]
        o rhob [kg/m3], bulk density (optional)
        o AI = rhob*vp, acoustic impedance (alternative to vp)
    
    Parameters
    ----------
    dd: dict
        Data and headers
        dict_keys(['name', 'depth', 'tvd', 'zsf', 'phi', 'vcl', 
                   'log_rt', 'vp', 'vs', 'rhob', 
                   'sw', 'so', 'sg', 'tops', 'ai'])
    sig_err_in: dict
        Error variance 
    inv_keys: list
        list of keys in data dict to beinverted, e.g. inv_list = ['log_rh','vp', 'rhob']
    mu_pri: float:
        Gaussian prior mean
    sig_pri: float:
        Gaussian prior variance
        
    **kwargs:
    verbose: int, optional (default 0)
        Print shit?
    kplot: bool, optional (default False)
        
    Returns
    -------
    mgi: dict
        Posterior mean, MAP and variance
    
    Programmed: KetilH 20. April 2022
    """
    
    # Get the kwargs:
    verbose = kwargs.get('verbose', 0)   # Print shit? 
    kplot = kwargs.get('kplot', False)   # Plot or not?

    Tg = kwargs.get('Tg', 38e-3) # Thermal gradient [oC/m]
    if Tg>1.0: Tg = 1e-3*Tg      # if Tg given in oC/km
    
    # Oil and gas saturation
    so = soil*(1-sw)
    sg = (1-soil)*(1-sw)
    
    # Prior distribution (once and for all)
    pri = inv.Prior(mu=mu_pri, sig=sig_pri, aa=sw, kplot=kplot, verbose=verbose)
    
    mgi = {'tvd': dd['tvd'], 'name': dd['name']}
    # Allocate arrays for inversion output
    mgi['post_mu']  = np.zeros_like(dd['tvd'])
    mgi['post_map'] = np.zeros_like(dd['tvd'])
    mgi['post_sig'] = np.zeros_like(dd['tvd'])
    
    w1 = np.ones(1) # data needs to ba alist of numpy arrays
    nd = dd['tvd'].shape[0]
    
    if verbose>0:
        print('sat_inversion: {}'.format(dd['name']))

    for jj in range(nd):
        
        # Deterministic parameters
        zzz = dd['tvd'][jj] - dd['zsf'] # Depth below mudline
        phi = dd['phi'][jj]
        vcl = dd['vcl'][jj]
 
        fw = {}
        fw['log_rt'] = lfp.archie(so, sg, phi, zzz)
        fw['vp'], fw['vs'], fw['rhob'] = lfp.gassmann(so, sg, phi, vcl, zzz, Tg=Tg)
        fw['ai'] = 1e-6*fw['vp']*fw['rhob']

        data = [dd[key][jj]*w1 for key in inv_keys]
        synt = [fw[key] for key in inv_keys]
        sig  = [sig_err[key] for key in inv_keys]
        
        mlh = inv.Likelihood(data=data, synt=synt, aa=sw, sig=sig)
        post = mlh.bayes_inv(pri)
        
        mgi['post_mu'][jj]  = post.mu[0]
        mgi['post_map'][jj] = post.map[0]
        mgi['post_sig'][jj] = post.sig[0]

    
    return mgi
    




