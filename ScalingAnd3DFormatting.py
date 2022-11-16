"""
Last updated Tue Apr 12 2022

@author: nsauthier@gmail.com

This script take a tidy list of labs and scale it (mean=0, std=1) and format the data
in a 3D shape (pt, lab, time_step)

Needs a training in order to get the mean / std

TODO: format in order for the init to either learn of get the mean/std

"""

#import
import pandas as pd
import numpy as np
from random import shuffle

class ScalingAnd3DFormatting:
    def scaler_train(self, temporal_data):
        self.mean = temporal_data.groupby('servacro')["lbres_ck"].mean()
        self.std = temporal_data.groupby('servacro')["lbres_ck"].std()
        
        return self.mean, self.std
        
    def temporal_format(self, temporal_data):
        #takes a list of labs
        #return a formated 3d data of format
        
        def calcs(temporal, npt, ntime, nservacro):
            #first scale down
            temporal = np.divide(np.subtract(temporal[:,0], temporal[:,1]),temporal[:,2])
            #then reshape to 3d pt x time x lab
            temporal = temporal.reshape((npt, ntime, nservacro))
            return temporal
        
        n_pt = temporal_data.dw_pid.nunique()
        n_time = temporal_data.step_to_disch.nunique()
        n_servacro = temporal_data.servacro.nunique()
        
        mean_std = pd.merge(self.mean, self.std, left_index=True, right_index = True)
        mean_std.columns = ['mean','std']
        
        temporal_data = pd.merge(temporal_data, mean_std, left_on='servacro', right_index=True)
        temporal_data = temporal_data.sort_values(['dw_pid','step_to_disch','servacro'])[['lbres_ck','mean','std']].values
        
        self.formatted = calcs(temporal_data, n_pt, n_time, n_servacro)
        
        return self.formatted
    
    def last_val_format(self,temporal_data, static_data, static_columns):
        
        #Keeping only the last lab value
        last_time = temporal_data.step_to_disch.unique().max()
        last_val = temporal_data[temporal_data.step_to_disch == last_time]
        last_val = last_val.sort_values(['dw_pid','servacro'])['lbres_ck'].values
        
        #Reformatting
        n_pt = temporal_data.dw_pid.nunique()
        n_servacro = temporal_data.servacro.nunique()
        last_val = last_val.reshape((n_pt, n_servacro))
        
        stat_last_val = static_data.sort_values(by = 'dw_pid')[static_columns].values
        stat_last_val = stat_last_val.reshape((n_pt,len(static_columns)))
        
        return np.hstack((last_val,stat_last_val))