# -*- coding: utf-8 -*-
"""
Test script for magnetic inversion of simple anomaly

Created on Thu Jan  7 11:04:08 2021
@author: kehok@equinor.com
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

from magfunc import MapData
from magfunc import marq_leven
from magfunc import green, jacobi
from magfunc import plot_gauss_newton
from magfunc import load_test_model

#---------------------------------------------
# Run tests
#---------------------------------------------

gscl = 250. # sale of the problem (grid spacing)

# niter =  0                # Linear inversion if niter==0
# z_shift_ini = 0.0         # Shift of tru base in initial model
niter =  6               # Number of iterations in non-lin GN inversion
z_shift_ini = -0.2*gscl  # Shift of base source layer for initial inversion model 
lam = 1e-6                # Marquardt-Levenberg regularization parameter

to_nT, from_nT = 1.0e9, 1.0e-9

k_frst, k_last = 2,2
for ktest in range(k_frst,k_last+1):
    
    #-----------------------------------------
    #   Get the model
    #-----------------------------------------
    
    print('##### Test {} #####'.format(ktest))
    
    if   ktest == 0: inc, dec, zr = 90,   0,  -4*gscl
    elif ktest == 1: inc, dec, zr = 90,   0,  -8*gscl
    elif ktest == 2: inc, dec, zr = 90,   0, -12*gscl
    elif ktest == 3: inc, dec, zr = 90,   0, -16*gscl
    elif ktest == 4: inc, dec, zr = 75,   5,  -4*gscl # North Atlantic
    elif ktest == 5: inc, dec, zr = 45, -25,  -4*gscl # TAG-ish

    #------------------------------------------
    # Forward model synthetic data
    #------------------------------------------

    # Load the test model        
    model = load_test_model(gscl=gscl, inc=inc, dec=dec)    

    #  Data mesh
    nx, ny = 80+1, 64+1
    xh, yh = gscl*((nx-1)//2), gscl*((ny-1)//2)
    data = MapData(np.linspace(-xh, xh, nx), np.linspace(-yh, yh, ny), zr)
    print('Top  source layer : {} m'.format(model.z[0][0,0]))
    print('Base source layer : {} m'.format(model.z[1][0,0]))
    print('Recording altitude: {} m'.format(data.z[0][0,0]))

    # Vectors for the Green's function
    vr = np.vstack([data.gx.flatten(), data.gy.flatten(), data.z[0].flatten()]).T
    vm_1 = np.vstack([model.gx.flatten(), model.gy.flatten(), model.z[0].flatten()]).T
    vm_2 = np.vstack([model.gx.flatten(), model.gy.flatten(), model.z[1].flatten()]).T
    vt_e, vt_m = model.vt_e, model.vt_m
    
    # Compute Green's function 
    print('Make some synt data')
    eps = 1e-32                # Avoid division by zero
    ds = model.dx*model.dy     # Surface element
    AA = ds*green(vr, vm_1, vm_2, vt_e, vt_m, eps)
    
    # Make synt data
    mm = model.mag.reshape(model.nx*model.ny,1)
    dd = AA.dot(mm)

    #---------------------------------------------------------------------
    # Initialize stuff for inversion
    #---------------------------------------------------------------------

    # Initialize the inversion output objects
    synt  = MapData(data.x, data.y, data.z)
    inver = MapData(model.x, model.y, model.z)
    inver.vt_e, inver.vt_m = model.vt_e, model.vt_m 
    # Initial value for base of source layer
    inver.z[1] = inver.z[1] + z_shift_ini 
    
    # Initialize lists for gathering iterations
    magn_it = [None for ii in range(niter+1)]   # Inverted magnetization
    base_it = [None for ii in range(niter+1)]   # Inverted base source layer
    synt_it = [None for ii in range(niter+1)]   # Synt data from current model
    rank_it = [None for ii in range(niter+1)]   # Rank of pseudo inverse
    synt.rms_err = [None for ii in range(niter+1)]   # RMS error of current model
    
    # Compute once and for all:
    ds = inver.dx*inver.dy
    gx_flat, gy_flat = inver.gx.flatten(), inver.gy.flatten()
    vm_1 = np.vstack([gx_flat, gy_flat, inver.z[0].flatten()]).T
    vr   = np.vstack([data.gx.flatten(), data.gy.flatten(), data.z[0].flatten()]).T
    vt_e, vt_m = inver.vt_e, inver.vt_m

    #-------------------------------------------------------------
    #   Linear inversion: fixed z2 (base of source layer)
    #-------------------------------------------------------------

    # First iter is the linear inversion (initial value for M is zero)
    tic = time.perf_counter()
    it = 0
    print('Iteration {}: Linear inversion'.format(it))
    base_it[it] = inver.z[1].reshape(-1,1)
    vm_2 = np.vstack([gx_flat, gy_flat, inver.z[1].flatten()]).T
    LL = ds*green(vr, vm_1, vm_2, vt_e, vt_m, eps)
    magn_it[it], rank_it[it] = marq_leven(LL, dd, lam)
    base_it[it] = inver.z[1].reshape(-1,1) # Not updated, same is initial
     
    #-------------------------------------------------------------
    # Non-linear inversion: Joint update of magnetization and z2
    #-------------------------------------------------------------
    
    # Non-linear Gauss-Newton iterations
    nh = magn_it[0].shape[0]
    for it in range(niter):
        
        print('Iteration {}: Non-linear inversion'.format(it+1))
        # Compute data residual for current model:
        vm_2 = np.vstack([gx_flat, gy_flat, base_it[it].flatten()]).T
        LL = ds*green(vr, vm_1, vm_2, vt_e, vt_m, eps)
        synt_it[it] = LL.dot(magn_it[it])
        deld = dd - synt_it[it]
        synt.rms_err[it] = np.sqrt(np.sum(deld**2)/np.sum(dd**2))
        
        # Compute Jacobian matrix
        KK = ds*jacobi(vr, magn_it[it], vm_2, vt_e, vt_m, eps)
        JJ = np.hstack((LL, KK)) # The full Jacobian
        
        # Compute model update (mag and z2)
        delm, rank_it[it+1] = marq_leven(JJ, deld, lam)
        magn_it[it+1] = magn_it[it] + delm[:nh]
        base_it[it+1] = base_it[it] + delm[nh:] 

    #-------------------------------------------------------
    #  Data residual and rms error after last iteration
    #-------------------------------------------------------

    # Synt data and error after last iteration:
    it = niter
    vm_2 = np.vstack([gx_flat, gy_flat, base_it[it].flatten()]).T
    LL = ds*green(vr, vm_1, vm_2, vt_e, vt_m, eps)
    synt_it[it] = LL.dot(magn_it[it])
    deld = dd - synt_it[it]
    synt.rms_err[it] = np.sqrt(np.sum(deld**2)/np.sum(dd**2))

    # Timing
    toc = time.perf_counter()
    time_inv = toc - tic
    print('Time inversion: {} sec.'.format(time_inv))

    #-------------------------------------------------------
    # Reshape and plot last update
    #-------------------------------------------------------

    # The data
    data.tma = to_nT*dd.reshape(data.ny, data.nx)

    # Get the first and last iterations for plotting:
    inver.mag0 = magn_it[ 0].reshape(inver.ny, inver.nx)
    inver.magn = magn_it[-1].reshape(inver.ny, inver.nx)
    inver.zb0 = base_it[ 0].reshape(inver.ny, inver.nx)
    inver.zbn = base_it[-1].reshape(inver.ny, inver.nx)
    synt.tma0 = to_nT*synt_it[ 0].reshape(synt.ny, synt.nx) 
    synt.tma  = to_nT*synt_it[-1].reshape(synt.ny, synt.nx)
    
    #-------------------------------
    #  PLot results
    #-------------------------------
    
    head = 'Magnetic test {}:'.format(ktest)
    figs = plot_gauss_newton(inver, synt, data, head=head, interp='bicubic')
     
    # Dump plots to png files
    if niter == 0: prefix = 'lin' 
    else: prefix = 'gn'
    figs[0].savefig(prefix + '_test_{}'.format(ktest) + '_inversion.png')
    if niter>0: figs[1].savefig(prefix + '_test_{}'.format(ktest) + '_relerr.png')
    
    # PLot the Marquardt-Levenberg matrix
    fig, ax = plt.subplots(1,1)
    nm, ata = len(mm), LL.T.dot(LL) 
    mlm= ata + lam*np.diag(ata.diagonal())
    im = ax.imshow(mlm, origin='upper', cmap=cm.magma)
    cm.ScalarMappable.set_clim(im)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8); 
    ax.set_title('Test {} (iter {}): L^TL + lam*diag(L^TL)'.format(ktest, it))
    fig.savefig(prefix + '_test_{}'.format(ktest) + '_marq_leven_matrix.png')
        
    plt.show(block=False)
