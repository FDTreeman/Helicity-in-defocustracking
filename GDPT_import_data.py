'''
#-*- coding: utf-8 -*- 
Created on 2020年8月28日

@author: Treeman
'''

import pandas as pd
from scipy.io import loadmat

bins = 30

#import data

def folder(l,filepath):
    '''
    Here are providing two examples: w-1 and w-15, 
    so,l = [-1,-15]
    filepath: the data filepath
    
    '''
    if l == -1:
        filename = 'w1'
        eps = 0.92 # or 0.93
        folder = filepath + filename
    elif l == -15:
        filename = 'w15'
        eps = 0.31
        folder = filepath + filename
    else:
        raise Exception("Invalid level!")

    return folder,filename,eps


def import_data(folder):
    '''
    read data
    
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
