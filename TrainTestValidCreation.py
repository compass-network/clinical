"""
Last updated Tue Apr 12 2022

@author: nsauthier@gmail.com

this script updates a list of patients that are separated in train / test /valid
because of multiples iterations it keeps continuity and keep strict separation.

"""

#import
import pandas as pd
import numpy as np
from random import shuffle

class TrainTestValidCreation:
    #Shuffle data into an autoencoder train set, train set, valid and test set.
    #The shuffle is random, based on a frac for train/test/valid AND a pre-specified positiv fraction in those groupes
    #the remnant of the data is used to creat the autoencoder train set
    
    #Can load an old file to keep the train/test/valid separation to avoid contamination
    
    def __init__(self, file_name, output_dir,
                 frac_test = 0.2,frac_valid = 0.2,
                 frac_train = 0.6, frac_pos = 0.15):
    
        self.file_name = file_name #"urghosp_opera_2019-11-11"
        self.output_dir = output_dir #script_dir[0:script_dir.find('code')]+"output/clinical_data"
        self.frac_test = frac_test
        self.frac_valid = frac_valid
        self.frac_train = frac_train
        self.frac_pos = frac_pos

    def traintestvalid(self, new_list, load=True):
        #It takes into account that the patients have the same pdonor tag for all the dwpid
        #Because the randomization is by patient.
        if load:
            old_listing = pd.read_csv(self.output_dir+"/8_transplantqc_donors2020_ICUCARD/"+\
                                      self.file_name +"_listing.csv")
            #as a safeguard
            old_listing.to_csv(self.output_dir+"/8_transplantqc_donors2020_ICUCARD/"+\
                               self.file_name +"_listing_old.csv")
            #keeping only new dw_pids
            df_residual = new_list[~new_list.dw_pid.isin(list(old_listing.dw_pid.unique()))]
            
            if len(df_residual):
                #There is a residual so we need to reseparate the data: 
                ids_donors = list(df_residual[df_residual.pdonor == 1].dw_pid.unique())
                shuffle(ids_donors)
                
                #To avoid duplicates of dw_pids
                new_list_non_donors = df_residual[~df_residual.dw_pid.isin(ids_donors)]
                
                ids_nondonors = list(new_list_non_donors[new_list_non_donors.pdonor == 0].dw_pid.unique())
                shuffle(ids_nondonors)
                
                cut1 = round(self.frac_train*len(ids_donors))
                cut2 = cut1 + round(self.frac_valid*len(ids_donors))
                cut3 = round(self.frac_train*len(ids_donors) * ((1-self.frac_pos) / self.frac_pos))
                cut4 = cut3 + round(self.frac_valid*len(ids_donors) * ((1-self.frac_pos) / self.frac_pos))
                cut5 = cut4 + round(self.frac_test*len(ids_donors) * ((1-self.frac_pos) / self.frac_pos))
                
                #Get the pids
                new_train = ids_donors[0:cut1] + ids_nondonors[0:cut3]
                new_valid = ids_donors[cut1:cut2] + ids_nondonors[cut3:cut4]
                new_test = ids_donors[cut2:] + ids_nondonors[cut4:cut5]
                new_AE_train = ids_nondonors[cut5:]
                
                #Adding the patients from the old list
                new_train = new_train + list(old_listing[old_listing.subset == "train_set"].dw_pid.values)
                new_valid = new_valid + list(old_listing[old_listing.subset == "valid_set"].dw_pid.values)
                new_test = new_test + list(old_listing[old_listing.subset == "test_set"].dw_pid.values)
                new_AE_train = new_AE_train + list(old_listing[old_listing.subset == "AE_train_set"].dw_pid.values)
                
                #reshape
                new_train = new_list[new_list.dw_pid.isin(new_train)]
                new_train = new_train[['dw_pid', 'hospital', 'noadm','pdonor']]
                new_train['subset'] = "train_set"
                
                new_valid = new_list[new_list.dw_pid.isin(new_valid)]
                new_valid = new_valid[['dw_pid', 'hospital', 'noadm','pdonor']]
                new_valid['subset'] = "valid_set"
                
                new_test = new_list[new_list.dw_pid.isin(new_test)]
                new_test = new_test[['dw_pid', 'hospital', 'noadm','pdonor']]
                new_test['subset'] = "test_set"
                
                new_AE_train = new_list[new_list.dw_pid.isin(new_AE_train)]
                new_AE_train = new_AE_train[['dw_pid', 'hospital', 'noadm','pdonor']]
                new_AE_train['subset'] = "AE_train_set"
                
                
                return pd.concat([new_train, new_valid, new_test, new_AE_train])
                
            else:
                #No new patients, same separation
                print("No new data")
                return old_listing
            
        else: #New listing
            ids_donors = list(new_list[new_list.pdonor == 1].dw_pid.unique())
            shuffle(ids_donors)
            
            #To avoid duplicates of dw_pids
            new_list_non_donors = new_list[~new_list.dw_pid.isin(ids_donors)]
            
            ids_nondonors = list(new_list_non_donors[new_list_non_donors.pdonor == 0].dw_pid.unique())
            shuffle(ids_nondonors)
            
            cut1 = round(self.frac_train*len(ids_donors))
            cut2 = cut1 + round(self.frac_valid*len(ids_donors))
            cut3 = round(self.frac_train*len(ids_donors) * ((1-self.frac_pos) / self.frac_pos))
            cut4 = cut3 + round(self.frac_valid*len(ids_donors) * ((1-self.frac_pos) / self.frac_pos))
            cut5 = cut4 + round(self.frac_test*len(ids_donors) * ((1-self.frac_pos) / self.frac_pos))
            
            #Get the pids
            new_train = ids_donors[:cut1] + ids_nondonors[:cut3]
            new_valid = ids_donors[cut1:cut2] + ids_nondonors[cut3:cut4]
            new_test = ids_donors[cut2:] + ids_nondonors[cut4:cut5]
            new_AE_train = ids_nondonors[cut5:]
            
            #reshape
            new_train = new_list[new_list.dw_pid.isin(new_train)]
            new_train = new_train[['dw_pid', 'hospital', 'noadm', 'pdonor']]
            new_train['subset'] = "train_set"
            
            new_valid = new_list[new_list.dw_pid.isin(new_valid)]
            new_valid = new_valid[['dw_pid', 'hospital', 'noadm', 'pdonor']]
            new_valid['subset'] = "valid_set"
            
            new_test = new_list[new_list.dw_pid.isin(new_test)]
            new_test = new_test[['dw_pid', 'hospital', 'noadm', 'pdonor']]
            new_test['subset'] = "test_set"
            
            new_AE_train = new_list[new_list.dw_pid.isin(new_AE_train)]
            new_AE_train = new_AE_train[['dw_pid', 'hospital', 'noadm', 'pdonor']]
            new_AE_train['subset'] = "AE_train_set"
            
            return pd.concat([new_train, new_valid, new_test, new_AE_train])