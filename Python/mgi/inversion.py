# -*- coding: utf-8 -*-
""" Objects and functions for core MGI Bayesian inversion.

The core objects and functions are implemented such that they are
applicable to any variety of MGI with one-parameter estimation

TODO: Another core module is needed for two-parameter estimation

Programmed: Ketil Hokstad, 22. September 2020

Created on Tue Sep 22 13:05:35 2020
@author: kehok@equinor.com
"""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.interpolate as interp
from scipy.stats import norm

class Posterior:
    """ Object for MAP, posterior mean and variance.
    
    Parameters
    ----------
    nd:  int. Prior mean
    verbose: int=0, optional. Print shit if verbose>0
    
    Returns
    -------
    self: object
        
    Example
    -------
    >>> post = Posterior(nd=nd, kkk=1)
    
    Programmed: KetilH, 22. September 2020    
    """
    
    def __init__(self,**kwargs):
        
        # Get kwargs:
        self.nd  = kwargs.get('nd',0)          # No of data points
        self.kkk = kwargs.get('kkk',-1)        # Save pdf of sample kkk for QC
        self.verbose = kwargs.get('verbose',0) # Print shit? 

        # Initialize arrays:
        self.map = np.zeros(self.nd, dtype=float)
        self.mu  = np.zeros(self.nd, dtype=float)
        self.sig = np.zeros(self.nd, dtype=float)
       
        # Print?
        if self.verbose > 1:
            print('Posterior.__init__ (v0.1):')
            print(' o nd  = %d' %self.nd)
            print(' o kkk = %d' %self.kkk)


#----------------------------------------------------------------
#   Object for the prior part
#----------------------------------------------------------------

class Prior:
    """ Object for prior pdf. Assuming Gaussian prior. 
    
    Parameters
    ----------
    mu: float
        prior mean
    sig: float
        prior variance
    aa: array like, float, optional
       independent variable for pdf(a) computation
    verbose: int=0, optional
        Print shit if verbose>0
    kplot: bool
        Plot the pdf?
    
    Returns
    -------
    self: object
        
    Example
    -------
    >>> pri = Prior(mu=mu_pri, sig=sig_pri, aa=aa)

    Programmed: KetilH, 22. September 2020
    """
    
    def __init__(self,**kwargs):
        
        # Get optionals:
        self.mu  = kwargs.get('mu',0) # No of geophysical parameters
        self.sig = kwargs.get('sig',1) # No of geophysical parameters
        self.aa  = kwargs.get('aa', np.linspace(0,1,101))
        self.na  = self.aa.shape[0]
        self.verbose = kwargs.get('verbose',0) 
        self.kplot = kwargs.get('kplot',False)       
        
        # Gaissian prior:
        self.pdf = norm.pdf(self.aa, loc=self.mu, scale=self.sig)
                
        # Print?
        if self.verbose > 1:
            print('Prior.__init__ (v0.1):')
            print(' o mu  = %f' %self.mu)
            print(' o sig = %f' %self.sig)

        # QC plot?
        if self.kplot:
            plt.figure()
            plt.plot(self.aa, self.pdf, 'r-', label='Prior')
            xp = np.min(self.aa) + 0.01*(np.max(self.aa) - np.min(self.aa))
            plt.ylim(0, 1.1*np.max(self.pdf))
            zp1 = 0.95*np.max(self.pdf)
            zp2 = 0.90*np.max(self.pdf)
            plt.text(xp,zp1,'mu=%.2f' %(self.mu))
            plt.text(xp,zp2,'sig=%.2f' %(self.sig))
            plt.xlabel('arg [-]')
            plt.ylabel('prior pdf [-]')
            
#----------------------------------------------------------------
#   Object and methods for the likelihood part
#    o model parameters to be inverted
#    o forward modeling
#    o error variances
#----------------------------------------------------------------

class Likelihood:
    
    """ Object for the likelihood part.

    o model parameters to be inverted
    o forward modeling
    o error variances
    
    Parameters
    ----------
    
    **kwargs
    --------
    nm: int, optional (default is 1)
        Number of geophysical model parmeter types to be inverted
        e,g. nm=3 for inversion of rho, vp, vs
    data: list of np.arrays
        geophysical model parameter arrays to be inverted
    synt: list of np.arrays
        geophysical forward model arrays
    aa: array like, float, optional
       independent variable for forward modeling fp(a)
    mu: list of floats
        likelihood error means
    sig: list of floats, optional
        likelihood error variances
    verbose: int, optional
        Print shit?
    kplot: bool, optional
        QC plotting?
        
    Returns
    -------
    self: object
        
    Example
    -------
    >>> mlh = Likelihood(data=data, synt=synt, aa=aa, sig=sig_err)

    Programmed: KetilH, 22. September 2020
    """
       
    def __init__(self,**kwargs):
        
        # Get kwargs:        
        nm = kwargs.get('nm',1) # No of model pars (temporary)
        self.data = kwargs.get('data', [np.nan for jj in range(nm)])
        
        # Got fish?
        self.nm = len(self.data)        
        self.nd = self.data[0].shape[0]
        
        # Synt: List of length na vectors
        self.aa = kwargs.get('aa', [])
        self.synt = kwargs.get('synt', [np.nan for jj in range(nm)])                
        
        # Error mean and variance:
        self.mu  = kwargs.get('mu', [0.0 for jj in range(self.nm)]) # Error mean
        self.sig = kwargs.get('sig',[1.0 for jj in range(self.nm)]) # Error variance

        self.verbose = kwargs.get('verbose',0) # Print shit? 
        self.kplot = kwargs.get('kplot',False) # Plot or not?
        
        if self.verbose > 1:
            print('Likelihood.__init__ (v0.1):')
            print(' o nm = %d' %self.nm)
            print(' o na = %d' %self.na)
            
    #----------------------------------------------------------------
    #   Inversion method
    #----------------------------------------------------------------
            
    def bayes_inv(self, pri, **kwargs):   
        
        """ One-parameter Bayesian inversion method

        Method of a Likelihood object.
        Data to be inverted, and forward model for the statistical
        inversion is input via the initiation of a Likelihoood object.

        o Inversion core, applicable to many MGI varieties
        o Input is assumed to be a 1D vector of model parameters (reshape)
        o Common forward model applied to all model parameters (precomputed)
        o Computes MAP, posterior mean and variance
        
        Parameters
        ----------
        pri: Prior object 
             Usually a Gaussian prior distribution. 
             
        **kwargs
        --------
        kkk: int, optional (default -1) 
             Store the full pdf for sample kkk
            
        Returns
        -------
        post: Posterior object
            
        Example
        -------
        >>>   pri = Prior(mu=mu_pri, sig=sig_pri, aa=sw)
        >>>   mlh = Likelihood( data=data, synt=synt, aa=sw, sig=sig_err)
        >>>   post = mlh.bayes_inv(pri=pri, kkk=1)

        Programmed: KetilH, 22. September 2020
        """
       
        # Noisy or slient?
        kkk = kwargs.get('kkk',-1)
        verbose = kwargs.get('verbose', 0)   # Print shit? 
        kplot = kwargs.get('kplot', False) # Plot or not?
        
        if verbose>0:
            print('bayes_inv: kkk=%d' %kkk )

        # Get some parameters:
        nm = len(self.data)
        nd = self.data[0].shape[0]
        aa = self.aa
        na = self.aa.shape[0]
        da = aa[1] - aa[0] # for integration

        if nd<1:
            print('bayes_inv: got no data')
            return 

        # Intitalize a list for partial likelihoods
        mlh_list = [np.nan for ii in range(nm)]

        # Initialize Posterior object:
        post = Posterior(nd=nd, kkk=kkk)

        #-------------------------------------
        # Loop over all spatial locations
        #-------------------------------------
        
        for jj in range(nd):
            
            # Initialize the pdf
            pdf_mlh  = np.ones_like(self.aa)
            
            # Build up the likelihood ditribution:
            for ii in range(nm):                
                wrk = self.synt[ii] - self.data[ii][jj]
                mlh_list[ii] = norm.pdf(wrk, loc=0, scale=self.sig[ii])
                pdf_mlh = pdf_mlh*mlh_list[ii]
                
            # Posterior=lieklihood*prior:
            pdf_post = pdf_mlh*pri.pdf

            # Normalize:
            pdf_mlh  = pdf_mlh/np.sum(da*pdf_mlh)
            pdf_post = pdf_post/np.sum(da*pdf_post)

            # MAP, posterior mean and variance
            jmap = np.argmax(pdf_post)
            post.map[jj] = aa[jmap]
            post.mu[jj]  = np.sum(da*aa*pdf_post)
            post.sig[jj] = np.sum(da*((post.mu[jj]-aa)**2)*pdf_post)
            post.sig[jj] = np.sqrt(post.sig[jj])
            
            # Keep full pdf for item kkk for QC plotting: NB .copy() needed
            if jj == post.kkk:
                # Normalize partial likelihoods:
                for kk in range(nm):
                    mlh_list[kk] = mlh_list[kk]/np.sum(da*mlh_list[kk])
                # Store in output struct:
                post.mlh_list = mlh_list.copy()
                post.pdf_mlh = pdf_mlh.copy()
                post.pdf_post = pdf_post.copy()
                post.pdf_pri = pri.pdf.copy()
                post.aa = aa.copy()
            
                # QC plotting:
                if kplot:
                    plot_pdfs(aa, pdf_post, pdf_mlh, pri.pdf, mlh_list,kkk=kkk)
                    
        # Return Posterior object
        return post
            
#-------------------------------------------------
#   Function for QC plotting pdfs
#-------------------------------------------------

def plot_pdfs(aa, pdf_post, pdf_mlh, pdf_pri, mlh_list, **kwargs):
    
    """ Helper function for QC plotting all pdfs """
    
    kkk = kwargs.get('kkk',666)

    # Get a color table
    cols = cm.tab20.colors
    
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    
    nm = len(mlh_list)
    for jj in range(nm):
        ax.plot(aa, mlh_list[jj], color=cols[jj], label='mlh_'+str(jj))
        
    ax.plot(aa, pdf_pri, color=cols[nm+0], label='prior')
    ax.plot(aa, pdf_mlh, color=cols[nm+1], label='likelihood')
    ax.plot(aa, pdf_post, color=cols[nm+2], label='posterior')
    ax.set_xlabel('aa [-]')
    ax.set_ylabel('pdf [-]')
    ax.set_title('pdfs for sample %d' %kkk)
    ax.legend()

            
            
        
        
        
        
