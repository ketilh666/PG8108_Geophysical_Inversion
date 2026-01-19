# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:19:11 2020
@author: kehok
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#-----------------------
# Some constants
#-----------------------

mu0 = 4*np.pi*1e-7
d2r, r2d = np.pi/180, 180/np.pi

#--------------------------------------------------
# Lambda function to compute scalar product
#--------------------------------------------------

# Scalar product for 3-comp vectors
dot = lambda a, b: a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#--------------------------------------------------
#   Compute Greens function matrix
#--------------------------------------------------

def green(vr, vm_1, vm_2, vt_e, vt_m, eps):
    """ Compute the magnetic Green's function matrix.
 
    The function depends only on geometry, i.e. x, y, z of magnetic 
    anomaly and receiver, respectively. 
    
    Parameters
    ----------
    vr: float, array of 3C vectors, shape=(nr,3)
        Co-ordinates of the observation points
    vm_1 float, array of 3C vector, shape=(nm,3)
        Horizontal coordinates and top of anomaly points
    vm_2: float, array of 3C vector, shape=(nm,3)
        Horizontal coordinates and base of anomaly points
    vt_e: float, 3C vector, shape=3)
        Direction of earth magnetic background field (tangent vector)
    vt_m: float, 3C vector, shape=(3)
        Direction of magnetization, currently va=vt
    eps: float, stabilization
    
    Returns
    -------
    grn: float
        Aray of the magnetic Green's function matrix

    Programmed: 
        KetilH, 13. December 2017 (Matlab)
        KetilH,  9. December 2020  
        KetilH, 13. January  2021  
    """
            
    nr = vr.shape[0]
    nm = vm_2.shape[0]
    
    # Compute Green's function array
    grn = np.zeros([nr,nm], dtype=float)
    for jj in range(nr):       # Data space
        for ii in range(nm):   # Model space
    
            vp = vm_1[ii,:] - vr[jj,:]
            vq = vm_2[ii,:] - vr[jj,:]
        
            p1 = np.sqrt(dot(vp,vp) + eps)
            p2 = p1*p1  
            p3 = p1*p2
            
            q1 = np.sqrt(dot(vq,vq) + eps)
            q2 = q1*q1 
            q3 = q1*q2 
         
            pw1 = -(vt_m[2] + dot(vt_m,vp)/p1)*(vt_e[2] + dot(vt_e,vp)/p1)/((vp[2]+p1)**2) 
            pw2 =  (dot(vt_m,vt_e)/p1 - (dot(vt_m,vp))*(dot(vt_e,vp))/p3)/(vp[2]+p1)
            
            qw1 = -(vt_m[2] + dot(vt_m,vq)/q1)*(vt_e[2] + dot(vt_e,vq)/q1)/((vq[2]+q1)**2) 
            qw2 =  (dot(vt_m,vt_e)/q1 - (dot(vt_m,vq))*(dot(vt_e,vq))/q3)/(vq[2]+q1)
        
            rf  = mu0/(4*np.pi)
            grn[jj,ii] = rf*(qw1 + qw2 - pw1 - pw2)
    
    return grn

#--------------------------------------------------
#   Compute Jacobian matrix wrt z2
#--------------------------------------------------

# Define function
def jacobi(vr, smag, vm, vt_e, vt_m, eps):
    """ Compute magnetic Jacobian matrix wrt z2 for the current model.
 
    The Jacobian wrt base source layer z2 is  J = (dQ/dz2)*M
    
    Parameters
    ----------
    vr: float, array of 3C vectors, shape=(nr,3)
        Co-ordinates of the observation points
    smag: float, array of scalars, shape=nm
        Scalar magnetization of the anomaly
    vm: float, array of 3C vector, shape=(nm,3)
        Horizontal coordinates and base of anomaly points
    vt_e: float, 3C vector, shape=3)
        Direction of earth magnetic background field (tangent vector)
    vt_m: float, 3C vector, shape=(3)
        Direction of magnetization, currently va=vt
    eps: float, stabilization
    
    Returns
    -------
    jac_ij: float
        One element of the magnetic Jacobian wrt z2

    Programmed: 
        KetilH,  7. January 2020  
    """
    
    nr = vr.shape[0]
    nm = vm.shape[0]
    
    # Compute Jacobian matrix wrt z_base
    jac = np.zeros([nr,nm], dtype=float)
    for jj in range(nr):       # Data space
        for ii in range(nm):   # Model space

            vq = vm[ii,:] - vr[jj,:]
            
            q1 = np.sqrt(dot(vq,vq) + eps)
            q2 = q1*q1 
            q3 = q1*q2 
            q5 = q3*q2
            
            w1 = -dot(vt_m,vt_e)/q3
            w2 = 3*(dot(vt_m,vq))*(dot(vt_e,vq))/q5
        
            rf  = mu0/(4*np.pi)
            jac[jj,ii] = rf*(w1 + w2)*smag[ii]
    
    return jac

#------------------------------------------------------------------------
#   Marquardt-Levenberg solver
#------------------------------------------------------------------------

def marq_leven(AA, dd, lam):
    """Core Marquardt-Levenberg solution.
    
    Parameters
    ----------
    AA: float, matrix, shape=(nd,nm)
    dd: float, vector, shape=(nd)
    lam: float, for regularization
    
    Returns
    -------
    mm: float, vector, shape=(nm). Solution
    rank: float. Rank of the ATA matrix
    
    Programmed:
        KetilH, 13. December 2017 (Matlab)
        KetilH,  9. December 2020    
    """

    dd_pc = AA.T.dot(dd)
    ATA = AA.T.dot(AA)
    HH = np.diag(ATA.diagonal())
    mm, res, rank, s = np.linalg.lstsq(ATA+lam*HH, dd_pc, rcond=None)
    
    return mm, rank

#---------------------------------------------
#  Class for map data
#--------------------------------------------

class MapData:
    """Set up the geometry, including np.meshgrid(x,y)
    
    Parameters
    ----------
    x: array of floats, shape=(nx)
    y: array of floats, shape=(ny)
    z: list of float or array of floats, [shape=(ny, nx)]
    
    Returns
    -------
    self: object
    
    Programmed: 
        KetilH,  6. December 2020 
    """
    
    def __init__(self, x, y, z):
    
        self.nx, self.ny = len(x), len(y)
        self.dx, self.dy = x[1]-x[0], y[1]-y[0]
        
        self.x, self.y = x, y
        self.gx, self.gy = np.meshgrid(x, y) 
       
        # z must be alist
        if not isinstance(z,list): z=[z]
        
        # Map data (horizons)
        self.z = []
        for jj in range(len(z)):
            if   isinstance(z[jj], float): 
                self.z.append(z[jj]*np.ones_like(self.gx))
            elif isinstance(z[jj], int):
                self.z.append(float(z [jj])*np.ones_like(self.gx))
            else:
                self.z.append(z[jj])
        
    def __repr__(self):
        return str(self.__dict__)

#---------------------------------------------
#  Class for magnetic background field
#--------------------------------------------

class MagneticBackgroundField:
    """ Define Earth magnetic background field for a geographical 
    location of interest
    
    Parameters
    ----------
    b0, float
        Background field magnitude [nT]
    inc, float
        Inclination angle [deg]
    dec, float
        Declination angle [deg]
        
    Returns
    -------
    self: object
    
    Programmed: 
        KetilH, 12. December 2017 (Matlab)
        KetilH,  5. December 2020
    """
    
    def __init__(self, b0, inc, dec, **kwargs):
        
        self.b0 = b0 
        self.inc = inc
        self.dec = dec
        
        self.lon = kwargs.get('lon',63.429722222)
        self.lat = kwargs.get('lat',10.393333333)
        
        # Polar and azimuth angles:
        self.theta = (90.0 - self.inc) # polar angle with vertical
        self.phi   = self.dec      # azimuth
        
        # Tangent unit vector to earth magntic field
        tx = np.cos(d2r*inc)*np.cos(d2r*dec) 
        ty = np.cos(d2r*inc)*np.sin(d2r*dec)
        tz = np.sin(d2r*inc)
        self.vt = np.array([tx, ty, tz], dtype=float)
        
         # Normal unit vector in same vertical plane
        rn2 = (tx**2 + ty**2)*(ty)**2  + ((tx)**2 + (ty)**2)**2
        ux =  tx*ty/np.sqrt(rn2)
        uy =  ty*ty/np.sqrt(rn2)
        uz = -(tx**2 + ty**2)/np.sqrt(rn2)
        self.vn = np.array([ux, uy, uz], dtype=float)

        # b0 and h0 vectors:
        self.vb0 = self.b0*self.vt
        self.vh0 = self.vb0/mu0
        self.h0  = self.b0/mu0      # A/m
        
        self.label = 'Earth magnetic field b0 [SI]'
    
    def __repr__(self):
        return str(self.__dict__)

#----------------------------------------------------------------
# PLot inversion results
#----------------------------------------------------------------

def plot_gauss_newton(inver, synt, data, **kwargs):
    "PLot result from Gauss Newton"    
        
    # Backward compatibility
    try:    mag0 = inver.mag0
    except: mag0 = inver.mag
    try:    magn = inver.magn
    except: magn = inver.mag
    try:    zb0  = inver.zb0
    except: zb0  = inver.z[1]
    try:    zbn  = inver.zbn
    except: zbn  = inver.z[1]
    
    # Get the kwargs
    niter = kwargs.get('niter', len(synt.rms_err)-1)
    scl_up = 1.00
    mmin = kwargs.get('mmin', np.min(scl_up*magn)) # Model parameter range
    mmax = kwargs.get('mmax', np.max(scl_up*magn)) # Model parameter range
    zmin = kwargs.get('zmin', np.min(scl_up*zbn))  # Depth range
    zmax = kwargs.get('zmax', np.max(scl_up*zbn))  # Depth range
    prop_name = kwargs.get('prop_name','NRM')
    interp = kwargs.get('interp', 'none')
    scl = kwargs.get('scl', 1e-3)
    cmap = kwargs.get('cmap',cm.viridis)
    
    print(mmin, mmax)

    # Grav or mag?
    head = kwargs.get('head', 'Magnetic inversion')
    if   head.lower()[0:3] == 'gzz':
        anom = np.nan
        resid = np.nan
        alab, plab = 'gzz [Eo]', 'Density [kg/m3]'
    elif head.lower()[0] == 'g':
        anom = np.nan
        resid = np.nan
        alab, plab = 'gz [mGal]', 'Density [kg/m3]'
    else:
        anom  = data.tma
        resid = data.tma - synt.tma # Data residual
        alab, plab = 'TMA [nT]', 'NRM [A/m]'
        
    amin = kwargs.get('amin', np.min(1.05*anom)) # Data range
    amax = kwargs.get('amax', np.max(1.05*anom)) # Data range

    # figure list
    figs = []

    # plot the input data and inversion results
    if niter: fig, axs = plt.subplots(2,3,figsize=(12,8))
    else:     fig, axs = plt.subplots(1,3,figsize=(12,5))
    fig.suptitle(head)

    # z pos down
    xtnt  = [scl*data.y[0], scl*data.y[-1], scl*data.x[0], scl*data.x[-1]]
    xtnt2 = [scl*inver.y[0], scl*inver.y[-1], scl*inver.x[0], scl*inver.x[-1]]
    
    ax = axs.ravel()[0]
    im = ax.imshow(anom.T, origin='lower', extent=xtnt, 
                   cmap=cmap, interpolation=interp)        
    cm.ScalarMappable.set_clim(im,vmin=amin,vmax=amax)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(alab)
    ax.set_xlim(scl*inver.y[0], scl*inver.y[-1])
    ax.set_ylim(scl*inver.x[0], scl*inver.x[-1])
    ax.set_ylabel('northing x [km]')
    ax.set_xlabel('easting y [km]')
    ax.set_title('Data input')
 
    ax = axs.ravel()[1]
    im = ax.imshow(mag0.T, origin='lower', extent=xtnt2, vmin=mmin,vmax=mmax,
                   cmap=cmap, interpolation=interp)
    #cm.ScalarMappable.set_clim(im,)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(plab)
    ax.set_xlim(scl*inver.y[0], scl*inver.y[-1])
    ax.set_ylim(scl*inver.x[0], scl*inver.x[-1])
    ax.set_ylabel('northing x [km]')
    ax.set_xlabel('easting y [km]')
    ax.set_title('{} linear inversion'.format(prop_name))
    
    ax = axs.ravel()[2]
    im = ax.imshow(zb0.T, origin='lower', extent=xtnt2, 
                   cmap=cmap, interpolation=interp)
    cm.ScalarMappable.set_clim(im,vmin=zmin,vmax=zmax)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label('Base [m]')
    ax.set_xlim(scl*inver.y[0], scl*inver.y[-1])
    ax.set_ylim(scl*inver.x[0], scl*inver.x[-1])
    ax.set_ylabel('northing x [km]')
    ax.set_xlabel('easting y [km]')
    ax.set_title('z_base initial')

    # The rest is relevant only for iterative GN inversion:
    if niter:    

        ax = axs.ravel()[3]
        im = ax.imshow(resid.T, origin='lower', extent=xtnt, 
                       cmap=cmap, interpolation=interp)
        cm.ScalarMappable.set_clim(im,vmin=amin,vmax=amax)
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
        cb.set_label(alab)
        ax.set_xlim(scl*inver.y[0], scl*inver.y[-1])
        ax.set_ylim(scl*inver.x[0], scl*inver.x[-1])
        ax.set_ylabel('northing x [km]')
        ax.set_xlabel('easting y [km]')
        ax.set_title('Residual iter {}'.format(niter))
     
        ax = axs.ravel()[4]
        im = ax.imshow(inver.magn.T, origin='lower', extent=xtnt2, 
                       cmap=cmap, interpolation=interp)
        cm.ScalarMappable.set_clim(im,vmin=mmin,vmax=mmax)
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
        cb.set_label(plab)
        ax.set_xlim(scl*inver.y[0], scl*inver.y[-1])
        ax.set_ylim(scl*inver.x[0], scl*inver.x[-1])
        ax.set_ylabel('northing x [km]')
        ax.set_xlabel('easting y [km]')
        ax.set_title('{} GN iter {}'.format(prop_name, niter))
        
        ax = axs.ravel()[5]
        im = ax.imshow(inver.zbn.T, origin='lower', extent=xtnt2, 
                       cmap=cmap, interpolation=interp)
        cm.ScalarMappable.set_clim(im,vmin=zmin,vmax=zmax)
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
        cb.set_label('Base [m]')
        ax.set_xlim(scl*inver.y[0], scl*inver.y[-1])
        ax.set_ylim(scl*inver.x[0], scl*inver.x[-1])
        ax.set_ylabel('northing x [km]')
        ax.set_xlabel('easting y [km]')
        ax.set_title('z_base GN iter {}'.format(niter))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    figs.append(fig)

    # PLot the rms error
    if niter:
        it = [ii for ii in range(len(synt.rms_err))]
        figs.append(plt.figure())
        plt.plot(it[1:], synt.rms_err[1:],'r-')
        plt.xlabel('GN iteration [-]')
        plt.ylabel('Rel RMS misfit [-]')
        plt.suptitle('{} Rel RMS error'.format(head))    
                   
    return figs

#-----------------------------------------------------------------
#   Make a simple diffractor model for testing
#-----------------------------------------------------------------
        
def load_test_model(**kwargs):
    """Load a simple model for testing """
    
    # kwargs
    gscl = kwargs.get('gscl', 25.0)  # grid scale (grid size)
    inc = kwargs.get('inc', 90.0)  # Inclination
    dec = kwargs.get('dec',  0.0)  # Declination
    mag = kwargs.get('mag', 10.0)  # Magnetization anomaly
    rho = kwargs.get('rho',100.0)  # Density anomaly
    
    # Earth background field
    B0 = 1e-9*52000 # Back ground field in Tesla
    bgf = MagneticBackgroundField(B0, inc, dec)
    
    # Model mesh: Top of anomaly is at z=0
    dx, dy = 1.0*gscl, 1.0*gscl
    nx, ny = 12+1, 8+1
    xh, yh = gscl*((nx-1)//2), gscl*((ny-1)//2)
    xm, ym = np.linspace(-xh, xh, nx), np.linspace(-yh, yh, ny)
    model = MapData(xm, ym, [0, 2*gscl])
    
    # Location of the anomaly is at the center:
    ixc, iyc = (model.nx-1)//2, (model.ny-1)//2
    model.x0, model.y0 = model.x[ixc], model.y[iyc]
    
    # Magnetic anomaly
    model.mag = np.zeros_like(model.gx)
    model.mag[iyc, ixc] = mag
    model.vt = bgf.vt.copy()
    model.vt_e, model.vt_m = model.vt, model.vt

    # Density anomaly
    model.rho = np.zeros_like(model.gx)
    model.rho[iyc, ixc] = rho

    return model
