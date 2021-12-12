'''
#-*- coding: utf-8 -*- 
Created on 2020年8月28日

@author: Treeman
'''
import numpy as np
import matplotlib.pyplot as plt
import copy

bins = 30
def add_polar(df,X0,Y0):
    '''
    Convert Cartesian coordinate system to cylindrical coordinate system, and get r, theta, vr, vtheta.
    
    Parameters:
    df: original data. 
    X0,Y0: the center of the original data.
    
    
    '''
    df['R'] = np.sqrt((df['dat_X']-X0)**2+(df['dat_Y']-Y0)**2)
    df['theta'] = np.arctan2((df['dat_Y']-Y0)/df['R'],(df['dat_X']-X0)/df['R'])#Y0)/df['R'])
    costheta = (df['dat_X']-X0)/df['R']
    sintheta = (df['dat_Y']-Y0)/df['R']
    df['vR'] = costheta*df['dat_DX']+sintheta*df['dat_DY'] #0.5
    df['vtheta'] = -sintheta*df['dat_DX']+costheta*df['dat_DY']
    df['vXY'] = np.sqrt(df['dat_DX']**2+df['dat_DY']**2)
    df['vRZ'] = np.sqrt(df['vR']**2+df['dat_DZ']**2)
def gram_schmidt(n):
    '''
    creates a new orthonormal basis starting fom n
    '''
    u0 = np.array(n)
    u1 = np.array([1,0,0]) - u0*u0.dot([1,0,0])/u0.dot(u0)
    u2 = np.array([0,1,0]) - u0*u0.dot([0,1,0])/u0.dot(u0) - u1*u1.dot([0,1,0])/u1.dot(u1)
    e0 = u0/np.sqrt(u0.dot(u0))
    e1 = u1/np.sqrt(u1.dot(u1))
    e2 = u2/np.sqrt(u2.dot(u2))
    return np.array([e1,e2,e0]).T #column vectors define the new basis

def project_data(df,n):
    '''
    projects the data in df along vector n.
    '''
    df2 = copy.copy(df)
    M = gram_schmidt(n)
    x,y,z = M.dot(df.loc[:,['dat_X','dat_Y','dat_Z']].T)
    dx,dy,dz = M.dot(df.loc[:,['dat_DX','dat_DY','dat_DZ']].T)
    df2.update({'dat_X':x,'dat_Y':y,'dat_Z':z,'dat_DX':dx,'dat_DY':dy,'dat_DZ':dz})
    return df2
    
 
def plotstreamingfield(df,keys,method ='normalize',rhomin=0.,vlimit=6000.,figsize=[8,5],fontsize=16,
                       cmap_font='Blues',cmap='hot_r',arrow_color='vy',fps=5,X0=1000,Y0=1000,
                       plt_point =False,title=False,viewer = 'azimuthal',scale=None):
    '''
    this function aims to visualize the azimuthal and axial streaming 
    
    this function has been symmetrized
        #X0 = 1400
        #Y0 = 1100
    Parameters: 
    df: data.
    keys: keys of df.
    method:two method are provide, normalize: normalize the velocity. scaling: log the velocity.
    rhomin: Minimum particle density.
    vlimit: limit the maximum speed to remove the outlier.
    figsize: the figure size.
    fontsize: the font size.
    cmap_font: the bins color.
    cmap: the arrows color map.
    arrow_color: the arrows color in different directions, several options are available: 'vx', 'vy', 'v', 'blue', 'red'.
    fps: The experiment video fps.
    X0,Y0: is the center point position.
    plt_point: plot the center point, True or False.
    title: the figure title, True or False.
    viewer: the view direction, 'azimuthal' or 'axial'.
    scale: quiver scale, float, optional
        Number of data units per arrow length unit, e.g., m/s per plot width; 
        a smaller scale parameter makes the arrow longer. Default is None.
        
    '''
    add_polar(df,X0,Y0)

    plt.figure(figsize=figsize)
    hx,_,_,_ = plt.hist2d(df[keys['x']],df[keys['y']],weights = df[keys['vx']],bins=bins)
    hy,_,_,_ = plt.hist2d(df[keys['x']],df[keys['y']],weights = df[keys['vy']],bins=bins)
    hz,_,_,_ = plt.hist2d(df[keys['x']],df[keys['y']],weights = df['vRZ'],bins=bins)
    rho,x,y,_ = plt.hist2d(df[keys['x']],df[keys['y']],bins=bins)
    hr,xr,yr,_ = plt.hist2d(df[keys['x']],df[keys['y']],weights = 1/(2*np.pi*df['R']),bins=bins)
    dx,dy,dxr,dyr = map(lambda x: np.unique(np.sort(x)),[x,y,xr,yr])
    dx = dx[1]-dx[0]
    dy = dy[1]-dy[0]
    dxr = dxr[1]-dxr[0]
    dyr = dyr[1]-dyr[0]
    
    if 'backgnd' in keys:
        if keys['backgnd']==1:
            plt.pcolormesh(x,y,rho.T/(dx*dy),cmap=cmap_font)#paticles number/um^2
        elif keys['backgnd']=='normalized_density':
            plt.pcolormesh(xr,yr,hr.T/(dxr*dyr),cmap=cmap_font)#paticles number/um^3
        else:
            plt.hist2d(df[keys['x']],df[keys['y']],weights = df[keys['backgnd']],bins=bins,cmap=cmap_font)

        cb=plt.colorbar()
        cb.ax.tick_params(labelsize=fontsize)
    x = 0.5*(x[1:]+x[:-1])
    y = 0.5*(y[1:]+y[:-1])
    x,y = np.meshgrid(x,y,indexing='ij')
    hx = fps*hx
    hy = fps*hy
    hz = hz/rho
    ind = rho>rhomin
    x1,y1,vx1,vy1,rho1,z1,r1= map(lambda z: z[ind],[x,y,hx,hy,rho,hz,hr])
    v1=np.sqrt(vx1**2+vy1**2)
    ind2 = v1<vlimit*rho1
    x,y,vx,vy,rho,_,_ = map(lambda z: z[ind2],[x1,y1,vx1,vy1,rho1,z1,r1])
    if method == 'normlize':
        vx = vx/rho
        vy = vy/rho 
    elif method == 'scaling':
        vx = np.log(vx)
        vy = np.log(vy) 
    else:
        raise Exception("Invalid mehtod!")  
    
    if title==True:
        plt.title('Velocity field. Max speed is {number:.{digits}f} um/s'.format(number=np.max(np.sqrt(vx**2+vy**2)), digits=0),
                  fontsize=fontsize)
    else:
        pass

    v=np.sqrt(vx**2+vy**2)

    if arrow_color=='vx':
        plt.quiver(x,y,vx,vy,vx,width=3e-3,cmap=cmap)
        vxmax = np.max(np.abs(vx))
        plt.clim(-vxmax,vxmax)
    elif arrow_color=='vy':
        vymax = np.max(np.abs(vy))
        plt.quiver(x,y,vx,vy,vy,width=3e-3,cmap=cmap)
        plt.clim(-vymax,vymax)
    elif arrow_color=='v':
        plt.quiver(x,y,vx,vy,v,width=3e-3,cmap=cmap)
    elif arrow_color=='blue':
        plt.quiver(x,y,vx,vy,color='b',width=3e-3,cmap=cmap,scale=scale)
    else:
        plt.quiver(x,y,vx,vy,color='r',width=3e-3,cmap=cmap,scale=scale)

    plt.axis('equal')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if viewer == 'axial':
        plt.xlabel('r ({a})'.format(a=r'$\mu$m'),fontsize=fontsize) 
        plt.ylabel('z ({b})'.format(b=r'$\mu$m'),fontsize=fontsize)
        print('Axial velocity field. Max speed is {number:.{digits}f} um/s'.format(number=np.max(np.sqrt(vx**2+vy**2)), digits=0))
    else:
        plt.xlabel('x ({a})'.format(a=r'$\mu$m'),fontsize=fontsize) 
        plt.ylabel('y ({b})'.format(b=r'$\mu$m'),fontsize=fontsize)
        print('Azimuthal velocity field. Max speed is {number:.{digits}f} um/s'.format(number=np.max(np.sqrt(vx**2+vy**2)), digits=0))


    if plt_point == True:
        plt.plot(X0,Y0,'og')
    else:
        pass



if __name__=='__main__':
    
    
    X0,Y0 = df[['dat_X','dat_Y']].mean()
    fontsize=14
    figsize=[8,6.5]
    keys = {'x':'dat_X',
               'y':'dat_Y',
               'vx':'dat_DX',
               'vy':'dat_DY',
               'backgnd':1}
    azimrho,azimv,azimv1,azimdf=plotstreamingfield(df,keys,method = 'normalize',rhomin=5.,vlimit=6000,
                figsize=figsize,fontsize=fontsize,cmap_font='Blues',cmap='seismic_r',arrow_color='red',
                fps=fps,X0=X0,Y0=Y0,plt_point =True,title=True,scale=None)#[8,5]
    
    keys = {'x':'R',
               'y':'dat_Z',
               'vx':'vR',
               'vy':'dat_DZ',
               'backgnd':'normalized_density'}
    axialrho,axialv,axialv1,axialdf=plotstreamingfield(df,keys,method = 'normalize',rhomin=5.,vlimit=600,
               figsize=figsize,fontsize=fontsize,cmap_font='Blues',cmap='seismic_r',arrow_color='vy',
               fps=fps,X0=X0,Y0=Y0,plt_point =False,title=True,viewer='axial',scale=None)#viewer represents the axis
        
