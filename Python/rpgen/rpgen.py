# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:30:54 2024

@author: KEHOK
"""

import numpy as np
import matplotlib.pyplot as plt

def rpgen(strike, dip, rake, gamma, sigma, TKO, AZM, **kwargs):
    
    """
    Matlab header
    -------------
    %RPGEN Calculate radiation pattern using shear-tensile source model.
    %   rpgen(strike,dip,rake,gamma,sigma, TKO, AZM) calculates P-wave, S-wave,
    %   SH-wave and SV-wave radiation pattern using shear-tensile source model
    %   presented in [see references 1, 2, 3 for details]. All input angles 
    %   (strike, dip, rake of the fault, tensile angle gamma, takeoff angle 
    %   TKO and azimuth from the source to the observation point AZM) should 
    %   be in degrees. The takeoff angle is measure from bottom. The function 
    %   returns matrices of the same size as input TKO and AZM matrices. 
    %
    %   Input parameters:
    %
    %     strike, dip, rake: fault plane parameters (degrees).
    %     gamma:  tensile angle in degrees (0 degrees for pure shear faulting, 
    %             90 degrees for pure tensile opening).
    %     sigma:  Poisson's ratio.
    %     TKO:    matrix of takeoff angles for which to calculate the correspo-
    %             nding radiation pattern coefficients (degrees, the takeoff 
    %             angles are measured from bottom).
    %     AZM:    matrix of corresponding azimuths (in degrees) for which the 
    %             radiation pattern coefficients should be calculated.
    %
    %   Output parameters:
    %   
    %     Gp, Gs, Gsh, Gsv - P-wave, S-wave, SH-wave, and SV-wave radiation 
    %     pattern coefficients calculated for corresponding takeoff angles 
    %     and azimuths specified in TKO and AZM matrices.
    %
    %   References:
    %
    %     [1] Kwiatek, G. and Y. Ben-Zion (2013). Assessment of P and S wave 
    %         energy radiated from very small shear-tensile seismic events in 
    %         a deep South African mine. J. Geophys. Res. 118, 3630-3641, 
    %         doi: 10.1002/jgrb.50274
    %     [2] Ou, G.-B., 2008, Seismological Studies for Tensile Faults. 
    %         Terrestrial, Atmospheric and Oceanic Sciences 19, 463.
    %     [3] Vavryèuk, V., 2001. Inversion for parameters of tensile 
    %         earthquakes.” J. Geophys. Res. 106 (B8): 16339–16355. 
    %         doi: 10.1029/2001JB000372.
    
    %   Copyright 2012-2013 Grzegorz Kwiatek.
    %   $Revision: 1.3 $  $Date: 2013/09/15 $    
    
    Ported to Python: KetilH, 11.April 2024
    """
    
    P_only = kwargs.get('P_only', True)
        
    sin = np.sin
    cos = np.cos    
    pi  = np.pi
    
    strike = strike * pi/180
    dip = dip*pi/180
    rake = rake*pi/180
    gamma = gamma*pi/180
    TKO = TKO*pi/180
    AZM = AZM*pi/180

    # P-wave first motion amplitudes
    Gp = cos(TKO)*(cos(TKO)*(sin(gamma)*(2*cos(dip)**2 - (2*sigma)/(2*sigma - 1)) + sin(2*dip)*cos(gamma)*sin(rake)) - cos(AZM)*sin(TKO)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike)) + sin(AZM)*sin(TKO)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma))) + sin(AZM)*sin(TKO)*(cos(TKO)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma)) + cos(AZM)*sin(TKO)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) + sin(AZM)*sin(TKO)*(cos(gamma)*(sin(2*strike)*cos(rake)*sin(dip) - sin(2*dip)*cos(strike)**2*sin(rake)) - sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*cos(strike)**2*sin(dip)**2))) - cos(AZM)*sin(TKO)*(cos(TKO)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike)) - sin(AZM)*sin(TKO)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) + cos(AZM)*sin(TKO)*(cos(gamma)*(sin(2*dip)*sin(rake)*sin(strike)**2 + sin(2*strike)*cos(rake)*sin(dip)) + sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*sin(dip)**2*sin(strike)**2)));

    # S-wave first motion amplitudes
    if not P_only:
        Gs = ((sin(AZM)*sin(TKO)*(cos(AZM)*cos(TKO)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) - sin(TKO)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma)) + cos(TKO)*sin(AZM)*(cos(gamma)*(sin(2*strike)*cos(rake)*sin(dip) - sin(2*dip)*cos(strike)**2*sin(rake)) - sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*cos(strike)**2*sin(dip)**2))) - cos(TKO)*(sin(TKO)*(sin(gamma)*(2*cos(dip)**2 - (2*sigma)/(2*sigma - 1)) + sin(2*dip)*cos(gamma)*sin(rake)) + cos(AZM)*cos(TKO)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike)) - cos(TKO)*sin(AZM)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma))) + cos(AZM)*sin(TKO)*(sin(TKO)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike)) + cos(TKO)*sin(AZM)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) - cos(AZM)*cos(TKO)*(cos(gamma)*(sin(2*dip)*sin(rake)*sin(strike)**2 + sin(2*strike)*cos(rake)*sin(dip)) + sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*sin(dip)**2*sin(strike)**2))))**2 + (cos(TKO)*(cos(AZM)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma)) + sin(AZM)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike))) - sin(AZM)*sin(TKO)*(sin(AZM)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) - cos(AZM)*(cos(gamma)*(sin(2*strike)*cos(rake)*sin(dip) - sin(2*dip)*cos(strike)**2*sin(rake)) - sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*cos(strike)**2*sin(dip)**2))) + cos(AZM)*sin(TKO)*(sin(AZM)*(cos(gamma)*(sin(2*dip)*sin(rake)*sin(strike)**2 + sin(2*strike)*cos(rake)*sin(dip)) + sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*sin(dip)**2*sin(strike)**2)) + cos(AZM)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma))))**2)**(1/2);
    
        Gsh = cos(TKO)*(cos(AZM)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma)) + sin(AZM)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike))) - sin(AZM)*sin(TKO)*(sin(AZM)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) - cos(AZM)*(cos(gamma)*(sin(2*strike)*cos(rake)*sin(dip) - sin(2*dip)*cos(strike)**2*sin(rake)) - sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*cos(strike)**2*sin(dip)**2))) + cos(AZM)*sin(TKO)*(sin(AZM)*(cos(gamma)*(sin(2*dip)*sin(rake)*sin(strike)**2 + sin(2*strike)*cos(rake)*sin(dip)) + sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*sin(dip)**2*sin(strike)**2)) + cos(AZM)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)));
    
        Gsv = sin(AZM)*sin(TKO)*(cos(AZM)*cos(TKO)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) - sin(TKO)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma)) + cos(TKO)*sin(AZM)*(cos(gamma)*(sin(2*strike)*cos(rake)*sin(dip) - sin(2*dip)*cos(strike)**2*sin(rake)) - sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*cos(strike)**2*sin(dip)**2))) - cos(TKO)*(sin(TKO)*(sin(gamma)*(2*cos(dip)**2 - (2*sigma)/(2*sigma - 1)) + sin(2*dip)*cos(gamma)*sin(rake)) + cos(AZM)*cos(TKO)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike)) - cos(TKO)*sin(AZM)*(cos(gamma)*(cos(2*dip)*cos(strike)*sin(rake) - cos(dip)*cos(rake)*sin(strike)) - sin(2*dip)*cos(strike)*sin(gamma))) + cos(AZM)*sin(TKO)*(sin(TKO)*(cos(gamma)*(cos(2*dip)*sin(rake)*sin(strike) + cos(dip)*cos(rake)*cos(strike)) - sin(2*dip)*sin(gamma)*sin(strike)) + cos(TKO)*sin(AZM)*(cos(gamma)*(cos(2*strike)*cos(rake)*sin(dip) + (sin(2*dip)*sin(2*strike)*sin(rake))/2) - sin(2*strike)*sin(dip)**2*sin(gamma)) - cos(AZM)*cos(TKO)*(cos(gamma)*(sin(2*dip)*sin(rake)*sin(strike)**2 + sin(2*strike)*cos(rake)*sin(dip)) + sin(gamma)*((2*sigma)/(2*sigma - 1) - 2*sin(dip)**2*sin(strike)**2)));
  
    if P_only:
        return Gp
       
    else:
        return Gp, Gs, Gsh, Gsv 

#---------------------------
#  Test script
#---------------------------

if __name__ == '__main__':
    
    block = False
        
    # Strike, dip and rake
    strike = 0.0
    dip = 90.0
    rake_list = [0. ,45.0, -90.]
    
    # Tensile faulting angle
    gamma = 0.0
    
    # Poisson ration
    ps_rat = 1.8 # vp/vs-ratio
    nu = 0.5*(ps_rat**2-2)/(ps_rat**2-1)
    
    # Ray take off angle 
    tko= 30.
    
    # Azimuth range to model
    # azm_max = 180.0
    azm_max = 360.0
    azm = np.linspace(0, azm_max, 361)
    
    
    fig, axs = plt.subplots(2,2, figsize=(16,7))
    
    for rake in rake_list:
                
        Gp, Gs, Gsh, Gsv = rpgen(strike, dip, rake, gamma, nu, tko, azm, P_only=False)
        
        ax = axs.ravel()[0]
        ax.plot(azm, Gp, label=f'rake={rake}')
        
        ax = axs.ravel()[1]
        ax.plot(azm, Gs, label=f'rake={rake}')
    
        ax = axs.ravel()[2]
        ax.plot(azm, Gsh, label=f'rake={rake}')
    
        ax = axs.ravel()[3]
        ax.plot(azm, Gsv, label=f'rake={rake}')
        
    # Add titles and legends
    tit_list = ['Gp', 'Gs', 'Gsh', 'Gsv']
    for kk, ax in enumerate(axs.ravel()):
        ax.plot(azm, np.zeros_like(azm), 'k--') # plot zero line
        ax.legend()
        ax.set_xlabel('Azimuth [deg]')
        ax.set_ylabel('Amplitude [-]')
        ax.set_title(tit_list[kk])
        
    fig.suptitle(f'rpgen output: take-off polar angle tko={tko} deg, strike={strike} deg, dip={dip} deg')
    fig.tight_layout(pad=1.0)
    
    fig.savefig(f'rpgen_radiation_tko{int(tko)}.png')
    
    plt.show(block=block)
    
    