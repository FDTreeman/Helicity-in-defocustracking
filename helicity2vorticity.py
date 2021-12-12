'''
#-*- coding: utf-8 -*- 

Created on 2021年9月10日

@author: Treeman
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import linalg
import copy
from csaps import csaps,NdGridCubicSmoothingSpline
from scipy import interpolate





def particles2field(df,fps,binsx=10,min_particles=1):
    '''
    Parameters:
    df: The data obtained from GDPT.
    fps: The experiment video fps.
    binsx: The number of bins for X axis, default is 10. binsy,binsz defined by binsx.
    min_particles: Divides discontinuous points (noise points), default is 1.
    
    Returns:
    edges: list. A list of D arrays describing the bin edges for each dimension.
    rho: particles density.
    vx,vy,vz: the speed component of particles along x,y,z axis.
    '''
    rhop = np.array(df[['dat_X','dat_Y','dat_Z']])
    vxp = np.array(df['dat_DX'])*fps
    vyp = np.array(df['dat_DY'])*fps
    vzp = np.array(df['dat_DZ'])*fps
    
    Lx = 2000.
    Ly = 2000.

    binsy = int(binsx*Ly/Lx)
    binsz = int(binsx*Ly/Lx)

    rho,edges = np.histogramdd(rhop, bins=[binsx,binsy,binsz], 
                               range=None, weights=None, density=False)
    vx,_ = np.histogramdd(rhop, bins=[binsx,binsy,binsz], 
                               range=None, weights=vxp, density=False)
    vy,_ = np.histogramdd(rhop, bins=[binsx,binsy,binsz], 
                               range=None, weights=vyp, density=False)
    vz,_ = np.histogramdd(rhop, bins=[binsx,binsy,binsz], 
                               range=None, weights=vzp, density=False)

    vx = vx/rho
    vy = vy/rho
    vz = vz/rho
    vx[rho<min_particles]=0.
    vy[rho<min_particles]=0.
    vz[rho<min_particles]=0.
    
    return edges,rho,vx,vy,vz

def edges2centers(edges):
    
    '''
    find bins center.
    '''   
    
    centers = [0.5*(edge[1:]+edge[:-1]) for edge in edges]
    return centers

    

    

def field2vorticity(edges,rho,vx,vy,vz,smooth=None,method='csaps',rhomin=1,kwargs_griddata = {'method':'linear','fill_value':0.}):
    '''
    #csaps.NdGridCubicSmoothingSpline(
    #or csaps
    #see https://csaps.readthedocs.io/en/latest/api.html#object-oriented-api
    Aim to get helicity and smooth data
    
    Parameters:
    edge: list. A list of D arrays describing the bin edges for each dimension.
    rho: particles density.
    vx,vy,vz: the speed component of particles along x, y, z axis.
    smooth: [Optional] float
        Smoothing parameter in range [0, 1] where:
        0: The smoothing spline is the least-squares straight line fit
        
        1: The cubic spline interpolant with natural condition
    method: 'csaps','griddata', or 'rbf' are provided. 
    rhomin: Minimum particle density.
    kwargs_griddata: other kwargs for 'griddata' method.
    
    return:
    X: find bins center.
    rho(X): the particle density which find the center.
    [vx(X),vy(X),vz(X)]: velocity component of bins along x, y, z axis.
    [omegax,omegay,omegaz]: omega component of bins along x, y, z axis.
    h: helicity.
    
    
    '''
    X = edges2centers(edges)
    if method == 'csaps':
        rho = csaps(X,rho,smooth = smooth)
        vx = csaps(X,vx,smooth=smooth) #TODO: edit the csaps to enable weights=rho
        vy = csaps(X,vy, smooth=smooth)
        vz = csaps(X,vz,smooth=smooth)

        dvxdx = vx(X,nu=[1,0,0])
        dvxdy = vx(X,nu=[0,1,0])
        dvxdz = vx(X,nu=[0,0,1])
        dvydx = vy(X,nu=[1,0,0])
        dvydy = vy(X,nu=[0,1,0])
        dvydz = vy(X,nu=[0,0,1])
        dvzdx = vz(X,nu=[1,0,0])
        dvzdy = vz(X,nu=[0,1,0])
        dvzdz = vz(X,nu=[0,0,1])
        omegax = dvzdy-dvydz
        omegay = dvxdz-dvzdx
        omegaz = dvydx-dvxdy
        h = vx(X)*omegax+vy(X)*omegay+vz(X)*omegaz
        return X,rho(X),[vx(X),vy(X),vz(X)],[omegax,omegay,omegaz],h
    
    elif method == 'griddata':
        #fill the holes first using linear interpolation, possibly can also set nearest or cubic by changing griddata options
        #print(rho.shape)
        pointsx,pointsy,pointsz = np.meshgrid(*zip(X),indexing='ij')
        points = np.vstack([p.flatten() for p in [pointsx,pointsy,pointsz]]).T   #Npoints*3
        ind_ok = rho.flatten()>=rhomin
        rhoi,vxi,vyi,vzi = [interpolate.griddata(points[ind_ok,:], field.flatten()[ind_ok], points, **kwargs_griddata) 
                            for field in [rho,vx,vy,vz]]
        rhoi,vxi,vyi,vzi = [x.reshape(pointsx.shape) for x in [rhoi,vxi,vyi,vzi]]
        #print(rhoi.shape)
        return field2vorticity(edges,rhoi,vxi,vyi,vzi,smooth=smooth,method='csaps')
    
    elif method == 'rbf':
        #fill the holes first using linear interpolation, possibly can also set nearest or cubic by changing griddata options
        #print(rho.shape)
        pointsx,pointsy,pointsz = np.meshgrid(*zip(X),indexing='ij')
        points = np.vstack([p.flatten() for p in [pointsx,pointsy,pointsz]]).T   #Npoints*3
        ind_ok = rho.flatten()>=rhomin
        rhoi,vxi,vyi,vzi = [interpolate.RBFInterpolator(points[ind_ok,:], field.flatten()[ind_ok], 
                                                        neighbors=100, smoothing=0.0, kernel='thin_plate_spline', 
                                                        epsilon=None, degree=None)(points)
                            for field in [rho,vx,vy,vz]]
        rhoi,vxi,vyi,vzi = [x.reshape(pointsx.shape) for x in [rhoi,vxi,vyi,vzi]]
        #print(rhoi.shape)
        return field2vorticity(edges,rhoi,vxi,vyi,vzi,smooth=smooth,method='csaps')
        
    
    

def vectorxy2theta(df,X,omegax,omegay):
    '''
    Aim to get omegatheta
    
    Parameters:
    df: original data.
    X: find bins center.
    omegax: The X-axis component of omega
    omegay: The Y-axis component of omega
    
    return: Omegatheta
    
    
    '''
    Xp = np.array(df[['dat_X','dat_Y','dat_Z']])
    xc,yc,zc = np.mean(Xp,axis=0)
    xedge,yedge,zedge = X
    x,y,z = np.meshgrid(xedge,yedge,zedge,indexing='ij')
    R = np.sqrt((x-xc)**2+(y-yc)**2)
    #theta = np.arctan2((edgey[1:]-y0)/R,(edgex[1:]-x0)/R)
    omegatheta = -(y-yc)*omegax/R+(x-xc)*omegay/R
    return omegatheta


def field_error(df,edges,vx,vy,vz,fps):
    '''
    define rules for finding errors.
    
    Parameters:
    df: original data. 
    edge: list. A list of D arrays describing the bin edges for each dimension.
    vx,vy,vz: velocity component of bins along x, y, z axis.
    fps: The frame rate of the video
    
    return: 
    rms: 
    rms_normalized: 
    Cm: 
    
    
    '''
    X = edges2centers(edges)
    vn = np.sqrt(vx**2+vy**2+vz**2)
    vxp = np.array(df['dat_DX'])*fps
    vyp = np.array(df['dat_DY'])*fps
    vzp = np.array(df['dat_DZ'])*fps
    
    Xp = np.array(df[['dat_X','dat_Y','dat_Z']])
    #Xp = [Xpa[:10,i] for i in range(Xpa.shape[1])]
    df['dvx'] = vxp - interpolate.interpn(X, vx, Xp, method='linear',bounds_error=False,fill_value=0.)
    df['dvy'] = vyp - interpolate.interpn(X, vy, Xp, method='linear',bounds_error=False,fill_value=0.)
    df['dvz'] = vzp - interpolate.interpn(X, vz, Xp, method='linear',bounds_error=False,fill_value=0.)
    dvn2 = df['dvx']**2+df['dvy']**2+df['dvz']**2
    df['dvn2'] = dvn2
    rho,_ = np.histogramdd(Xp, bins=edges, 
                               range=None, weights=None, density=False)
    dvn2,_ = np.histogramdd(Xp, bins=edges, 
                               range=None, weights=dvn2, density=False) #sum of the error^2
    Cm,_ = np.histogramdd(Xp, bins=edges, 
                               range=None, weights=df['dat_Cm'], density=False)
    rms = np.sqrt(dvn2)/rho
    rms_normalized = np.sqrt(dvn2)/(rho*vn)
    Cm = Cm/rho
    
    return rms,rms_normalized,Cm
    

def cleanup(df,edges,rho,vx,vy,vz,fps,eps,plot=False):
    '''
    clean up errors and return the clean data.
    
    '''
    rms,rms_n,Cm = field_error(df,edges,vx,vy,vz,fps=fps)
    X = np.array([rms_n.flatten(),Cm.flatten()])
    ind_ok = np.all(np.isfinite(X)==1,axis=0)
    #print(X.shape)
    X = X[:,ind_ok]
    #print(X.shape)
    scaler = RobustScaler()
    cluster = DBSCAN(eps=eps)
    pipeline = make_pipeline(scaler,cluster)
    true_points_ok = pipeline.fit_predict(X.T)
    true_points = -np.ones_like(vx.flatten())
    true_points[ind_ok] = true_points_ok
    if plot:
        plt.figure()
        plt.scatter(rms_n.flatten(),Cm.flatten(),c=true_points)
        plt.xlabel('rms_n')
        plt.ylabel('Cm')
        plt.colorbar()
        #plt.xlim([-2*np.nanstd(rms_n)+np.nanmean(rms_n),2*np.nanstd(rms_n)+np.nanmean(rms_n)])
        #plt.ylim([-2*np.nanstd(Cm)+np.nanmean(Cm),2*np.nanstd(Cm)+np.nanmean(Cm)])
    true_points = np.reshape(true_points,vx.shape,order='C')
    rho_clean = copy.copy(rho)
    vx_clean = copy.copy(vx)
    vy_clean = copy.copy(vy)
    vz_clean = copy.copy(vz)
    for dat in [rho_clean,vx_clean,vy_clean,vz_clean]:
        dat[true_points!=0] = 0.
    return rho_clean,vx_clean,vy_clean,vz_clean


def calculate_helic(l,h1,rho,df,X1,X,omegax1,omegay1,vx1,vy1,vz1):

    '''
    calcuate amplitude in GDPT
    
    '''
    beta = 0.5
    #-------------------------A_GDPT-------------------------
    if l==-1:
        A1 = 3.75267847485909  # nm
        A_GDPT = beta*A1*np.sqrt(0.27/0.83)
    elif l==-5:
        A5 = 3.48470704027767  # nm
        A_GDPT =beta*A5*np.sqrt(0.19/0.82) 
    elif l==-10:
        A10 = 3.36429168629482  # nm
        A_GDPT = beta*A10*np.sqrt(0.23/0.76)
    elif l==-15:
        A15 = 3.22085156732772  # nm
        A_GDPT = beta*A15*np.sqrt(0.19/0.73)   
    elif l==-20:
        A20 = 3.22312285922789  # nm
        A_GDPT = beta*A20*np.sqrt(0.14/0.71)
    else:
        raise Exception("Invalid level! ")
    
    #---------------------------H0--------------------------   
    cl = 1910                 # m/s    
    cs = 3484                 # m/s    
    frequency =  20e6         # Hz 
    omega = 2*np.pi*frequency    
    kSAW = omega/cs
    kl0 = omega/cl 
    kz = np.sqrt(kl0**2-kSAW**2)    
    tanalpha = kz/kSAW    
    Vd = 2.0e-9              # m^3   
    b = 2.5
    
    #----------------------------H_star----------------------
    
    sin_alpha = kSAW/kl0
    con_alpha = np.sqrt(1-sin_alpha**2)
    tan_beta = sin_alpha/con_alpha**2
    
    param_a = l*b**2*omega**5/(2*cl**3)
    param_b = 0.25*Vd*tan_beta*(A_GDPT*1e-9)**4
    
    H_star = param_a*param_b
    
    
    #---------------------------Omega_star-------------------
    
    Omega_star=0.5*omega**3*b*tanalpha*(A_GDPT*1e-9)**2/cl**2
    
    #-------------------------helicity (H) -------------------------------
    H = h1 #total helicity m^4/s^2
    V = rho>=3 #droplet volume (from GDPT data of course!)
    
    #------------------------vorticity (OmegaTheta) --------------------- 
    omegatheta = vectorxy2theta(df,X1,omegax1,omegay1)
    OmegaTheta = omegatheta
    #-----------------------the average speed ---------------------------
    U = np.sqrt(vx1**2+vy1**2+vz1**2)
    for d in list(range(h1.ndim))[::-1]:
        H = np.trapz(H,axis=d,dx=(X[d][1]-X[d][0]))# integrating the H in the droplet
        V = np.trapz(V,axis=d,dx=(X[d][1]-X[d][0]))
        OmegaTheta = np.trapz(OmegaTheta,axis=d,dx=(X[d][1]-X[d][0]))
        U = np.trapz(U,axis=d,dx=(X[d][1]-X[d][0]))
    
        # U = #integral of U, then divide by droplet volume -> mean velocity
    
    vorticity = OmegaTheta/2e-9
    
    H_norm = H/H_star
    Omega_norm = vorticity/Omega_star
    average_speed = U/2e-9
    Volum_GDPT = round(V*1e9,2)
    return A_GDPT,H,vorticity,H_star, Omega_star,H_norm,Omega_norm,average_speed,Volum_GDPT



