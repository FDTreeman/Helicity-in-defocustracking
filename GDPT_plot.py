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
    
 
def plotfield2helicity(df,keys,normalize = True,rhomin=0.,v0=0.,figsize=[8,5],fontsize=16,fontdict={'weight':'normal','size': 30},
                       cmap_font='Blues',cmap='hot_r',arrow_color='vy',scaling='none',fps=5,X0=1000,Y0=1000,
                       plt_point =False,title=False,viewer = 'azimuthal',scale=None):
    '''
    this function has been symmetrized
        #X0 = 1400
        #Y0 = 1100
    '''
    add_polar(df,X0,Y0)


    #keys = [x,y,vx,vy,backgnd]
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
            #plt.hist2d(df[keys['x']],df[keys['y']],bins=bins,cmap=cmap_font)
            plt.pcolormesh(x,y,rho.T/(dx*dy),cmap=cmap_font)#paticles number/um^2
        elif keys['backgnd']=='normalized_density':
            plt.pcolormesh(xr,yr,hr.T/(dxr*dyr),cmap=cmap_font)#paticles number/um^3
        else:
            plt.hist2d(df[keys['x']],df[keys['y']],weights = df[keys['backgnd']],bins=bins,cmap=cmap_font)
        #plt.colorbar()
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
    ind2 = v1<v0*rho1
    x,y,vx,vy,rho,_,_ = map(lambda z: z[ind2],[x1,y1,vx1,vy1,rho1,z1,r1])
    if scaling=='log':
        vx = np.log(vx)
        vy = np.log(vy)
    #ind3 = rho2<rhomax
    #x,y,vx,vy,rho = map(lambda z: z[ind3],[x2,y2,vx2,vy2,rho2])
    #x,y,vx,vy,rho = map(lambda z:z[ind],[x,y,hx,hy,rho])
    if normalize:
        vx = vx/rho
        vy = vy/rho
        #plt.title('velocity field. Max speed is {number:.{digits}f} um/s'.format(number=np.max(np.sqrt(vx**2+vy**2)), digits=0))
    if title==True:
        plt.title('Velocity field. Max speed is {number:.{digits}f} um/s'.format(number=np.max(np.sqrt(vx**2+vy**2)), digits=0),
                  fontsize=fontsize)
    else:
        pass
    #q = plt.quiver(x,y,vx,vy,color='r',width=3e-3)  
    v=np.sqrt(vx**2+vy**2)
    #q = plt.quiver(x,y,vx,vy,v,width=3e-3,cmap='hot_r')
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
    #plt.subplots_adjust(top=0.2,bottom=0.1,left=0.4,right=0.5,hspace=0,wspace=0)
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


    #plt.axis('off')
    #plt.xticks([])
    #plt.yticks([])
    if plt_point == True:
        plt.plot(X0,Y0,'og')
        #plt.savefig('w0_2.svg')
    else:
        pass
    #plt.savefig(filepath + '.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
    #plt.savefig(filepath + '.jpg', bbox_inches='tight', dpi=300, pad_inches=0.0)    
    #plt.savefig(filepath + '_tif.tif', bbox_inches='tight', dpi=300, pad_inches=0.0)
    #plt.savefig(filepath + '_jpg.jpg', bbox_inches='tight', dpi=300, pad_inches=0.0)
    #plt.savefig(filepath + '.eps', bbox_inches='tight', dpi=300, pad_inches=0.0) 
    #plt.show()

    return rho,v,v1,df


if __name__=='__main__':
    from GDPT_handling_vibration import handling_vibration
    from GDPT_import_data import import_data
    from pathlib import Path
    import os
    folder = 'H:/GDPTlab/GDPTlab_V1.2/data/NatCom/PDMS/w15_new/w15_IDT=1_2021.9.18_16mV_1_cut/'
    df = import_data(folder)
    df=handling_vibration(df,window=15)
    
    filename = 'F:/paper/Figures/7_test/GDPT/'   #savepath

    my_file = Path(filename)
    if my_file.is_dir()== False:
        os.mkdir(filename)
    else:
        pass
    filepath1=filename+'w15_azimuthal3'
    filepath2=filename+'w15_axial3' 
    keys = {'x':'dat_X',
               'y':'dat_Y',
               'vx':'dat_DX',
               'vy':'dat_DY',
               'backgnd':1}
    fontsize=28
    
    X0,Y0 = df[['dat_X','dat_Y']].mean()
    X0,Y0 =X0,Y0
    hxy0_1,v0_1,v01,dftest11=plot_field(df,keys,normalize = True,rhomin=5.,v0=600,
                                        figsize=[16,12.5],fontsize=fontsize,fontdict={'weight':'normal','size': fontsize},
                                        cmap_font='Blues',cmap='seismic_r',arrow_color='red',
                                        fps=20,X0=X0,Y0=Y0,plt_point =True,title=True,scale=None)#[8,5]
    #'cet_diverging_gkr_60_10_c40',cet_diverging_linear_bjr_30_55_c53,cet_cyclic_wrwbw_40_90_c42,seismic
    #https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html,https://colorcet.holoviz.org/user_guide/index.html
    
    
    #-------------------------W15_axial------------------------------------
    
    keys = {'x':'R',
               'y':'dat_Z',
               'vx':'vR',
               'vy':'dat_DZ',
               'backgnd':'normalized_density'}
    hxy0_2,v0_2,v02,dftest12=plot_field(df,keys,normalize = True,rhomin=5.,v0=60,
                                        figsize=[16,12.5],fontsize=fontsize,fontdict={'weight':'normal','size': fontsize},
                                        cmap_font='Blues',cmap='seismic_r',arrow_color='vy',
                                        fps=20,X0=X0,Y0=Y0,plt_point =False,title=True,viewer='axial',scale=None)
    
