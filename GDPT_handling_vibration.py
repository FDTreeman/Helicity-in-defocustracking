'''
#-*- coding: utf-8 -*- 
Created on 2020年8月28日

@author: Treeman
'''
import pandas as pd
import scipy.signal as signal

def handling_vibration(df,window=15.,method = 'moving_average'):
    '''
    heldling the problem of the particles rotation when particles move in the droplet
    
    '''
    
    grouped = df.groupby('dat_ID')
    out = []
    keys = ['dat_X','dat_Y','dat_Z','dat_DX','dat_DY','dat_DZ']
    for _, group in grouped:
        if method == 'sgolay':
            group[keys] = signal.savgol_filter(group.loc[:,keys], window,polyorder=3, axis=0,mode='nearest')
        else:
            group[keys] = group.loc[:,keys].rolling(window=window).mean()
        out.append(group)
    df = pd.concat(out)
    df = df.dropna()
    df['dat_Z'] = 1.45839*df['dat_Z']   #refractive index of gly    #1.45839  #90%   #1.4629  93%
    df['dat_DZ'] = 1.45839*df['dat_DZ'] #refractive index of gly    #1.45839  #90%     #1.4629  93%
    
    
    keys = ['dat_X','dat_Y','dat_Z','dat_DX','dat_DY','dat_DZ']
    for i in keys:
        df.loc[:,i] *= 1e-6             # convert unit from um to m
    
    print('the length of df after handling:',len(df))
    
    return df