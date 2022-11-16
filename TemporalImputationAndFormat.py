
"""
Last updated Tue Apr 12 2022

@author: nsauthier@gmail.com

This script implement a two-step imputation of laboratory
1. Last value carried forward
2. Random imputation from a normal distribution between normal values

"""

import numpy as np
import pandas as pd
import time

def get_random(servacro, reference_range):
    # Generate a value between low and high normaly distributed as 95% are between low and high
    # Truncated value are returned to avoid lower than 0.
    while 1:
        temp = np.random.normal(reference_range.loc[servacro,'imputation_mean'], reference_range.loc[servacro,'imputation_std'])
        if reference_range.loc[servacro,'low'] <= temp <= reference_range.loc[servacro,'high']:
            
            return round(temp,1)

def TemporalImputationAndFormat(df_lab,ref_range):
    #Keeping only the last 6 day for imputation. ease computation
    t0 = time.time()
    print('Starting imputation...')
    df_lab = df_lab[df_lab.step_to_disch >=-18]
    
    list_pt = list(df_lab.dw_pid.unique())
    list_labs = list(ref_range.servacro.unique())
    list_time_step = [-9, -8, -7, -6, -5, -4, -3, -2, -1]
    
    #Reindex allows a quick way to creat NaN for timestep without a lab
    df_lab = df_lab.set_index(['dw_pid', 'servacro', 'step_to_disch'])

    # Dropping all unused cols to ease computation
    df_lab = df_lab.loc[:, 'lbres_ck']

    df_lab = df_lab.reindex(pd.MultiIndex.from_product([list_pt, list_labs, list_time_step],
     names=['dw_pid', 'servacro', 'step_to_disch']),fill_value=np.nan)
    
    #Timing for performance improvement.
    t1 = time.time()
    t = t1-t0
    print('Total reindexing time: {0:4.0f} min and {1:2.0f} seconds'.format(t//60, t%60))
    
    #First imputation step forward fill. Here is the bottleneck step.
    df_lab.update(df_lab.groupby(level=[0,1]).fillna(method='ffill'))
    df_lab = df_lab.to_frame().reset_index()
    
    #Timing for performance improvement.
    t2 = time.time()
    t = t2-t1
    print('Total forward fill time: {0:4.0f} min and {1:2.0f} seconds'.format(t//60, t%60))
    
    #Keeping only the last 3 day for random imputation and model as per protocol
    df_lab = df_lab[df_lab.step_to_disch >=-9]
    
    ref_range = ref_range.set_index('servacro')
    
    #Second imputation step get a random value
    df_lab['lbres_ck'] = df_lab.apply(lambda x: get_random(x['servacro'], ref_range) if np.isnan(x['lbres_ck']) else x['lbres_ck'], axis=1)    
    
    #Timing for performance improvement.
    t3 = time.time()
    t = t3-t2
    print('Total random imputation time: {0:4.0f} min and {1:2.0f} seconds'.format(t//60, t%60))

    return df_lab