'''
#-*- coding: utf-8 -*- 
Created on 2020年8月28日

@author: Treeman
'''

import pandas as pd
from scipy.io import loadmat

bins = 30

#import data

def folder(l,exp,filepath):
    '''
    l = [-1,-5,-10,-15,-20]
    exp = [1,2,3]
    filepath: the data filepath
    
    '''
    if l == -1:
        if exp == 1:
            filename = 'w1_exp1'
            eps = 0.9
        elif exp == 2:
            filename = 'w1_exp2'
            eps = 0.92 # or 0.93
        elif exp == 3:
            filename = 'w1_exp3'
            eps = 0.8
        else:
            raise Exception("Invalid level!")    
        folder = filepath + filename
        
        
    elif l == -5:
        if exp == 1:
            filename = 'w5_exp1'
            eps = 0.31
        elif exp == 2:
            filename = 'w5_exp2'
            eps = 0.1
        elif exp == 3:
            filename = 'w5_exp3'
            eps = 0.66
        else:
            raise Exception("Invalid level!")
        folder = filepath + filename
        
        
    elif l == -10:
    
        if exp == 1:
            filename = 'w10_exp1'
            eps = 0.41
        elif exp == 2:
            filename = 'w10_exp2'
            eps = 0.3
        elif exp == 3:
            filename = 'w10_exp3'
            eps = 0.4
        else:
            raise Exception("Invalid level!")   
        folder = filepath + filename
        
        
    elif l == -15:
        if exp == 1:
            filename = 'w15_exp1'
            eps = 0.31
        elif exp == 2:
            filename = 'w15_exp2'
            eps = 0.3
        elif exp == 3:
            filename = 'w15_exp3'
            eps = 0.31
        else:
            raise Exception("Invalid level!")  
        folder = filepath + filename
        
        
    elif l == -20:
        if exp == 1:
            filename = 'w20_exp1'
            eps = 0.31
        elif exp == 2:
            filename = 'w20_exp2'
            eps = 0.3
        elif exp == 3:
            filename = 'w20_exp3'
            eps = 0.26
        else:
            raise Exception("Invalid level!")
        folder = filepath + filename
        
        
    else:
        raise Exception("Invalid level!")

    return folder,eps


def import_data(folder):
    '''
    
    
    '''
    files = ['Cm','DX','DY','DZ','ID','In','X','Y','Z']
    data = {}
    for f in files:
        data.update(loadmat(folder+'/dat_'+f+'.mat'))
    
    df = {}
    for k,v in data.items():
        try:
            df[k] = v.flatten()
        except AttributeError:
            pass
    df = pd.DataFrame(df)
    print('the length of df before handling:',len(df))
    #df.head()
    return df
