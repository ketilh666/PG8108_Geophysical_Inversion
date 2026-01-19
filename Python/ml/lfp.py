# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:47:47 2022

@author: kehok@equinor.com
"""

import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------------------------
#   LFP functions
#   Archie, Waxman Smith and Gassmann
#   Adapted from Matlab
#---------------------------------------------------------------------

#-----------------------------------------------
#  Archie equation
#-----------------------------------------------

def archie(So, Sg, phi, zzz, **kwargs):
    """ Archie or Waxman-Smits model for resistivity

    Parameters
    ----------
        So  : array of floats. Oil saturation
        Sg  : array of floats. Gas saturation
        phi : array of floats. Porosity 
        zzz : array of floats. Depth below seabed
    All arrays must have same shape and size or scalar float
        
    **kwargs
    --------
        
    Returns
    -------
        log_rt: array of floats. Log10 resitivity
        
    Programmed: 
        Ketil Hokstad, March 2008
        KetilH 1. April 2022 (Python)
    """

    # Get the kwargs

    # Archie parameters (fixed, make kwargs later)
    turt = 0.9     # Turtousity factor
    n = 2.0        # Saturation exponent
    m = 1.8        # Porosity exponent
    r_water = 0.06 # Water resistivity, from LFP_RW
    
    # Water saturation
    Sw = 1 - So - Sg           # Water saturation
    Sw = np.maximum(Sw,1.0e-3) # Make sure we dont divide by zero  

    # Archie
    r_sat  = turt*r_water/(phi**m*Sw**n) # Resistivity of saturated rock
    log_rt = np.log10(r_sat)          # Log Vertical resistivity (log)
   
    return log_rt

#-----------------------------------------------
#  Gassmann equation
#-----------------------------------------------

def gassmann(So, Sg, phi, vcl, zzz, **kwargs):
    
    """ Gassmann model fro vp, vs, rho

    Parameters
    ----------
        So  : array of floats. Oil saturation
        Sg  : array of floats. Gas saturation
        phi : array of floats. Porosity 
        vcl : array of floats. Clay fraction.  
        zzz : array of floats. Depth below seabed
    All arrays must have same shape and size or scalar float
        
    **kwargs
    --------
    Tg: float. Thermal gradient (default Tg=38e-3 oC/m)
    patchy_sat: bool. optional (default False)
        
    Returns
    -------
        vp, vs, rho: array of floats. Log10 resitivity
        
    Programmed: 
        Ketil Hokstad, November 2008
        Torgeir Wiik, November 2018
        Sondre Olsson, January 2019
        Ketil Hokstad, March 2019
        KetilH 1. April 2022 (Python)
    """

    # kwargs
    patchy_sat = kwargs.get('patchy_sat', False)
    Tg = kwargs.get('Tg', 38.0e-3)

    # Tg given in oC/km?
    if Tg>1.0: Tg = 1e-3*Tg

    # Water saturation
    Sw = 1 - So - Sg           # Water saturation
    Sw = np.maximum(Sw,1.0e-3) # Make sure we dont divide by zero  

    # Quartz and clay properties
    rho_qz = 2640   # Density quartz. 
    K_qz   = 38e9   # Bulk modulus quartz.
    mu_qz  = 44e9   # Shear modulus quartz
    #K_cl   = 20.9e9 # Bulk modulus clay
    #mu_cl  = 6.85e9 # Shear modulus clay

    # Trickery
    biot = 0.60     # Biot coefficient from Mitchell and Soga [2005], Quartzitic sandstone
    
    rho_matrix = rho_qz  # Density matrix      
    mu_matrix  = mu_qz   # Shear modulus matrix    
    K_matrix   = K_qz    # Bulk modulus matrix
    
    # Fluid parameters
    K_gas     = 0.830e9  # Tuning
    K_oil     = 1.200e9  # Just guessing
    K_water   = 2.8806e9 # From LFP_KFLW
    rho_gas   = 147.9    # Gas density from LFP_RHOG
    rho_oil   = 800      # Oil density
    rho_water = 1042.9   # Water density from LFP_RHOW
    
    # Density
    rho_fluid = Sw*rho_water + So*rho_oil + Sg*rho_gas # Fluid density
    rho_sat = (1-phi)*rho_matrix + phi*rho_fluid # Density of saturated rock
    rho = rho_sat # Density returned by function
    
    # Han model for Vp and Vs (brine filled). 
    # Data from Han (1996)
    sig_han = np.array([5   ,10   ,20   ,30   ,40   ,100   ]) # Effective stress
    Ap_han  = np.array([5.26, 5.39, 5.49, 5.55, 5.59,  5.59])
    Bp_han  = np.array([7.08, 7.08, 6.94, 6.96, 6.93,  6.93])
    Cp_han  = np.array([2.02, 2.13, 2.17, 2.18, 2.18,  2.18])
    As_han  = np.array([3.16, 3.29, 3.39, 3.47, 3.52,  3.52])
    Bs_han  = np.array([4.77, 4.73, 4.73, 4.84, 4.92,  4.92])
    Cs_han  = np.array([1.64, 1.74, 1.81, 1.87, 1.89,  1.89])
    
    # Approximate effective stress:
    gz = 9.82      # Acceleration of gravity
    rho_avg = 2300 # Approx avg bulk density
    sig_eff = 1e-6*gz*(rho_avg-biot*rho_water)*zzz # Effective stress [MPa]
    
    # Interpolate:
    Ap = 1.00e3*np.interp(sig_eff, sig_han, Ap_han)
    Bp = 1.00e3*np.interp(sig_eff, sig_han, Bp_han)
    Cp = 1.00e3*np.interp(sig_eff, sig_han, Cp_han)
    As = 1.00e3*np.interp(sig_eff, sig_han, As_han)
    Bs = 1.00e3*np.interp(sig_eff, sig_han, Bs_han)
    Cs = 1.00e3*np.interp(sig_eff, sig_han, Cs_han)
    
    # Velocities from Han model
    vp_han =  Ap - Bp*phi - Cp*vcl
    vs_han =  As - Bs*phi - Cs*vcl
    
    # Water saturated density
    rho_han = (1-phi)*rho_matrix + phi*rho_water
    
    #-------------------------------------------------
    #   Gassmann fluid substitution around Gassmann
    #-------------------------------------------------   
    
    # Temperature effect on shear modulus
    aa = 18.8  # Tuning parameter
    stemp = (1-aa/(Tg*zzz))
    mu_dry=(1-biot)*stemp*mu_matrix # Shear modulus dry
    mu_dry = np.maximum(mu_dry, 0.0)
    
    # Bulk properties for brine filled rock (from Han model)
    Ks1 = rho_han*(vp_han**2 - (4/3)*vs_han**2)    
    Kf1 = K_water
    
    # Bulk modulus of HC 
    if patchy_sat:
        Kf2 = Sg*K_gas + So*K_oil + Sw*K_water      # Patchy saturation
    else:
        Kf2 = 1/(Sg/K_gas + So/K_oil + Sw/K_water)  # Uniform saturation
    
    # Fluid substitution:
    vs = np.sqrt(mu_dry/rho) # S-wave velocity
    a1 = Ks1/(K_matrix-Ks1)
    f1 = Kf1/(K_matrix-Kf1)
    f2 = Kf2/(K_matrix-Kf2)
    gg = (a1+(1/phi)*(f2-f1))
    Ks2 = (gg/(1+gg))*K_matrix
    
    vp = np.sqrt((Ks2 + (4/3)*mu_dry)/rho) # P-wave velocity
     
    # vp = vp_han
    # vs = vs_han
    
    return vp, vs, rho

#---------------------------------------------
#   Test script
#---------------------------------------------

if __name__ == '__main__':
    
    z0, nz, dz = 1000.0, 151, 10.0
    tvd = np.linspace(z0, z0+(nz-1)*dz, nz)
    
    phi0, gam = 0.30, 0.25e-3
    phi = phi0*np.exp(-gam*tvd)
    Sg  = np.zeros_like(tvd)
    So  = np.zeros_like(tvd)
    vcl = np.ones_like(tvd)
    cec = np.zeros_like(tvd)

    # make a HC zone
    iz0, iz1, iz2 = 100, 105, 115
    Sg[iz0:iz1]  = 0.9
    So[iz1:iz2]  = 0.9
    phi[iz0:iz2] = 0.20
    vcl[iz0:iz2] = 0.10
    
    # Compute
    patchy_sat = False
    vp, vs, rho = gassmann(So, Sg, phi, vcl, tvd, patchy_sat=patchy_sat)        
    log_rh_A  = archie(So, Sg, phi, tvd)
    #log_rh_WS = waxman_smits(So, Sg, phi, vcl, cec, tvd) # Wrong
        
    # Make a plot
    fig, axs = plt.subplots(1,5, figsize=(12,6))        

    ax = axs.ravel()[0]
    Sw = 1.0 - So - Sg
    ax.plot(Sw, tvd, 'b')
    ax.plot(So, tvd, 'g')
    ax.plot(Sg, tvd, 'r')
    ax.set_ylabel('Saturation [-]')
    ax.set_title('Saturation')

    ax = axs.ravel()[1]
    ax.plot(phi, tvd, 'r')
    ax.set_ylabel('Porosity [-]')
    ax.set_title('Porosity')

    ax = axs.ravel()[2]
    log_rh = log_rh_A
    ax.plot(log_rh, tvd, 'r')
    ax.set_ylabel('log_rh [ohmm]')
    ax.set_title('Resistivity')

    ax = axs.ravel()[3]
    ax.plot(vp, tvd, 'r')
    ax.plot(vs, tvd, 'b')
    ax.set_ylabel('vp and vs [m/s]')
    ax.set_title('P- and S-velocity')

    ax = axs.ravel()[4]
    ax.plot(rho, tvd, 'r')
    ax.set_ylabel('rho [kg/m3]')
    ax.set_title('Density')


    for ax in axs:
        ax.set_ylabel('Depth [mbsl]')
        ax.invert_yaxis()
        
    fig.tight_layout(pad=2.0)
    
    plt.show()
    
