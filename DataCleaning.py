
"""
Created Sept 2019
Last updated Tue Apr 12 2022

@author: nsauthier@gmail.com

  This script uses custom functions defined in DataCleaningFunctions
  in order to clean a list of labs extracted from  CITADEL.
  It outputs a cleaned file
  
  In its second part it cleand a static file.

  In the third part it clean and add a ICD10 file

"""

#import
import pandas as pd
import numpy as np
import time
from DataCleaningFunctions import rare_labs_patient_cleaning
from DataCleaningFunctions import clean_servacro, clean_longdesc, clean_values
from DataCleaningFunctions import  outliers_removal, special_cases, lab_names

class lab_data:
    def __init__(self, file_lab, file_save_name, output_dir):
        self.file_lab = file_lab #'urghosp_lab_results_opera_donors_2019-11-11'
        self.file_name = file_save_name #"urghosp_opera_2019-11-11"
        self.output_dir = output_dir #script_dir[0:script_dir.find('code')]+"output/clinical_data"
        self.rarelabscleaningpercent = 0.9

    def load(self):
        t0 = time.time()
        
        cols_to_load = ['dw_pid', 'resultdtm','specimencollectionmethodcd', 
        'servacro', 'longdesc', 'lbres_ck']

        dtype = {'dw_pid':'category',
                 'resultdtm':str,
                 'specimencollectionmethodcd':'category',
                 'specimentypecd':str,
                 'servacro':'category',
                 'longdesc':'category',
                 'lbres_ck':str}

        print('Start loading data.')

        self.df = pd.read_csv(self.output_dir +'/'+self.file_lab, 
          compression='gzip',
          usecols = cols_to_load,
          dtype = dtype)

        print('Loading done at {0:6.1f} seconds'.format(time.time()-t0))
        print('Total of {} laboratory lines loaded.'.format(len(self.df)))
        
        self.df.dropna(subset = ['lbres_ck'], inplace = True)
        self.df.servacro.cat.remove_unused_categories(inplace=True)

        print('Total of {} patients'.format(self.df.dw_pid.nunique()))
        
        self.df.reset_index(inplace=True)

        return self

    def clean(self):

        t0 = time.time()
        #servacro cleaning
        self.df['servacro'] = clean_servacro(self.df.servacro)
        self.df.dropna(subset = ['servacro'], inplace=True)
        self.df.servacro.cat.remove_unused_categories(inplace=True)
        t_servacro = time.time()
        print('Time servacro cleaning: {0:4.0f} seconds'.format(t_servacro-t0))
        
        #longdesc cleaning
        self.df.longdesc = clean_longdesc(self.df.longdesc)
        self.df.longdesc.cat.remove_unused_categories(inplace=True)
        t_longdesc = time.time()
        print('Time longdesc cleaning: {0:4.0f} seconds'.format(t_longdesc-t_servacro))
        
        #results cleaning
        self.df.lbres_ck = clean_values(self.df.lbres_ck)
        t_lbres_ck = time.time()
        print('Time values cleaning: {0:4.0f} seconds'.format(t_lbres_ck- t_longdesc))
        
        #special cleaning
        self.df = special_cases(self.df)
        t_special = time.time()
        print('Time special cleaning: {0:4.0f} seconds'.format(t_special - t_lbres_ck))
        
        self.df.dropna(subset = ['servacro','longdesc', 'lbres_ck']).reset_index(inplace=True)
        
        #conversion to numerical
        #Keeping non numerical for quality check. Mostly cancelled / not done yet labs
        self.not_float = self.df[~self.df.lbres_ck.str.contains('^-?\d+\.*\d*$', regex=True)]
        
        self.df = self.df[self.df.lbres_ck.str.contains('^-?\d+\.*\d*$', regex=True)]
        self.df.loc[:,'lbres_ck'] = self.df.lbres_ck.astype('float32')
        
        #Outliers removal
        self.df = outliers_removal(self.df)
        t_outliers = time.time()
        print('Time outliers + conversion to float: {0:4.0f} seconds'.format(t_outliers - t_special))
        print("Total of {} patients pre rare removal".format(self.df.dw_pid.nunique()))
        
        #Cleaning values that are either rare labs (present in less than a specified fraction)
        #Or patient that have labs value than a specified value
        self.clean_labs = rare_labs_patient_cleaning(self.df, self.rarelabscleaningpercent)
        t_rare = time.time()
        
        self.clean_labs.servacro.cat.remove_unused_categories(inplace=True)
        self.clean_labs.longdesc.cat.remove_unused_categories(inplace=True)

        #Change names to be more readables
        name_unknown = [lab for lab in self.clean_labs.servacro.cat.categories if not lab_names.get(lab)]
        if name_unknown:
            print('New labs, please add names for',name_unknown )
        self.clean_labs.servacro = self.clean_labs.servacro.cat.rename_categories(lab_names)
        self.clean_labs = self.clean_labs.drop(columns = 'longdesc')
        print('Time rare removal : {0:4.0f} seconds'.format(t_rare  - t_special))
        print("Total of {} patients post rare removal".format(self.clean_labs.dw_pid.nunique()))
        return self.clean_labs

        def export(self):
            self.clean_labs.to_csv(self.output_dir+'/'+
                self.file_name+"_labs_cleaned.csv.gz",
                compression = 'gzip',
                index=False
                )
            return self

class static_data:
    def __init__(self, file_static, file_save_name, output_dir, raw_radiology_file):
        self.file_static = file_static#'urghosp_patient_episodes_merged_transfered2019-11-22'
        self.file_name = file_save_name #"urghosp_opera_2019-11-11"
        self.output_dir = output_dir #script_dir[0:script_dir.find('code')]+"output/clinical_data"
        self.raw_radiology_file = raw_radiology_file
        self.rarelabscleaningpercent = 0.9
        self.timestep = 8 #hours
        
    def load(self):
        #load data        
        col_to_load = ['dw_pid','icu1_card2_mix3','dossier','noadm_mrg','hospital_mrg','nam', 'age', 
        'sexe', 'pdonor', 'donor_type','dtdeces','dhreadm_mrg', 
        'dhredeb', 'dhrefin', 'last_icuep','servcode_mrg']

        self.static = pd.read_csv(self.output_dir +"/"+self.file_static + ".csv",
            usecols = col_to_load, 
            parse_dates = ['dtdeces','dhreadm_mrg', 'dhredeb', 'dhrefin'])
        
        return self

    def clean(self):
        #remove aberrantes values
        self.static = self.static[self.static.age >=16]
        #rename columns
        self.static.rename(columns = {'dhreadm_mrg':'dhreadm',
          'noadm_mrg':'noadm',
          'hospital_mrg':'hospital',
          'servcode_mrg':'servcode'},inplace=True)
        ## Servcode = code of service before ICU cleaning
        
        #missing servcodes replaced by Unkown tag
        self.static.loc[self.static.servcode.isna(), 'servcode'] = 'UNKN'
        #numerci servocdes are associated to emergency room and replaced by URG
        self.static.loc[self.static.servcode.str.isnumeric(), 'servcode'] = 'URG'
        #Space removing
        self.static['servcode'] = self.static['servcode'].str.strip()
        # Some Servcodes are different because of different hospital, but are the same service
        servcode_merge={'ANES':'ANE','CARA':'CAR','CARB':'CAR','CARC':'CAR','CARD':'CAR','CARR':'CAR','CGHO': 'CGH0',
        'GASH': 'HEP', 'GASG': 'GAST', 'GAS': 'GAST','CGDO':'CGD0','HEMO':'HEMA','HEME':'HEMA',
        'HEM':'HEMA','NEU':'NEUR', 'NEP':'NEPH','ORT':'ORTH','ONGY':'GYNO','PNE':'PNEU','INT':'MINT',
        'NCH':'NCHI','RHU':'RHUM','RTH':'RHUM','PLA':'PLAS','RAD':'RADX','MIN':'MING','END':'ENDO',
        'GER':'GERI','ONC':'ONCO'
        }
        self.static['servcode'].replace(servcode_merge, inplace=True)
        
        ##Attributing a death in ICU flag for pdonor or dead within 24 hours of discharge
        self.static['death_in_icu'] = 0
        self.static.loc[self.static.pdonor == 1,'death_in_icu'] = 1
        self.static.loc[((self.static.dtdeces-self.static.dhrefin).map(lambda x:x.days*24 + x.seconds/3600) < 24),'death_in_icu'] =1  
        
        ##Keeping only the last episode
        self.static = self.static[self.static.last_icuep == 1]
        
        ##Removing short episod (less than 16 hours)
        pt_pre_short = self.static.dw_pid.nunique()
        self.static = self.static.loc[~((self.static.dhrefin - self.static.dhreadm).map(lambda x:x.days*24 + x.seconds/3600) < 16)]
        pt_post_short = self.static.dw_pid.nunique()
        print("Removing short episods (<16hours). Loss of {} patients. Remaining: {} patients".format(pt_pre_short-pt_post_short, pt_post_short))

        ## Adding scan and imaging data
        col_to_load = ['dw_pid', 'nam', 'dossier', 'ci_dossier', 'hospital',
        'date_heure_exam', 'statut', 'description', 'modalite',
        'type_niveau4']

        rad = pd.read_csv(self.output_dir + "/" + self.raw_radiology_file,
          compression='gzip',
          usecols = col_to_load
          )

        rad = rad[(rad.description.str.contains(r"c[ée]r[ée]brale?s?|neuro|t[êe]tes?", case=False)) &
        ((rad.description.str.contains(r'SCAN', case=False)) | (rad.modalite.isin(['Tomodensitométrie', '(film extérieur)']))) &
                        (rad.statut.isin(['Rapport Confirme', 'Ex. termine, Lecture exterieur'])) &  # Remove "en attente" etc.
                        (rad.type_niveau4.isna())]  #s remove angio and modification

        #Merge to get list of patient who had a head scan before ICU end of stay
        rad = rad.merge(self.static[['dw_pid','dhreadm','dhrefin']], on='dw_pid')
        rad['date_heure_exam'] = rad['date_heure_exam'].apply(pd.to_datetime)
        rad = rad[(rad.dhreadm <= rad.date_heure_exam)&
        (rad.date_heure_exam <= rad.dhrefin)]
        had_head_scan = rad.dw_pid.unique()
        #Adding info to static
        self.static = self.static.assign(had_head_scan = self.static.dw_pid.isin(had_head_scan))
        
        self.static.donor_type.fillna('Non_donor', inplace=True)
        self.static.donor_type = self.static.donor_type.replace({'Refered _Transfered_for_donation':'Refered_Transfered_for_donation',
          'Refered _Donated':'Refered_Donated'})
        
        return self.static

def static_lab_cleaning(df_lab, df_static, rarelabscleaningpercent = 0.9, timestep = 8):

    #Conjoint static and lab cleaning    
    
    #format data
    df_lab['resultdtm'] = pd.to_datetime(df_lab['resultdtm'], format = "%Y-%m-%d %H:%M:%S", errors='coerce')
    df_lab['dw_pid'] = df_lab['dw_pid'].astype(int)
    
    statcol = ['dw_pid','dhreadm','dhrefin']
    df_lab = df_lab.merge(df_static[statcol], on='dw_pid')
    
    #removing labs before and after hospital episode
    df_lab = df_lab[(df_lab.dhreadm <= df_lab.resultdtm)&
    (df_lab.resultdtm < df_lab.dhrefin)]
    
    #getting the timestep
    df_lab = df_lab.assign(delay_to_disch = df_lab['resultdtm']-df_lab['dhrefin'])
    df_lab['delay_to_disch'] = df_lab['delay_to_disch'].map(lambda x:x.days*24 + x.seconds/3600) #in hours
    df_lab['step_to_disch'] = (df_lab['delay_to_disch'] / timestep).map(np.floor)
    
    #Making the df more light
    df_lab = df_lab[df_lab.step_to_disch >= -30]
    df_lab.drop(columns = ['dhreadm','dhrefin','resultdtm'], inplace=True)
    df_lab.servacro = df_lab.servacro.astype(str)
    df_lab.step_to_disch = df_lab.step_to_disch.astype(int)
    
    ##Get last value per timestep
    df_lab = df_lab.loc[df_lab.groupby(['dw_pid','servacro', 'step_to_disch'])['delay_to_disch'].idxmax()]
    
    print('Before rare removing {} patients'.format(df_lab.dw_pid.nunique()))
    
    ## remove rare labs i.e. present in less than 90% of patients in last 6 days
    df_lab_6days = df_lab[df_lab.step_to_disch >= -18]
    
    #Removing labs that are present in less than 5% of patients in the last 6 days
    perc_miss = 1-(df_lab_6days.groupby('servacro')['dw_pid'].nunique() / df_lab_6days.dw_pid.nunique())
    df_lab = df_lab[df_lab.servacro.isin(perc_miss[perc_miss<rarelabscleaningpercent].index)]
    df_lab_6days = df_lab_6days[df_lab_6days.servacro.isin(perc_miss[perc_miss<rarelabscleaningpercent].index)]
    
    #Removing patients with less than 10% of the labs in the last 6 days
    perc_miss = 1-(df_lab_6days.groupby('dw_pid')['servacro'].nunique() / df_lab_6days.servacro.nunique())
    df_lab = df_lab[df_lab.dw_pid.isin(perc_miss[perc_miss<rarelabscleaningpercent].index)]
    df_lab_6days = df_lab_6days[df_lab_6days.dw_pid.isin(perc_miss[perc_miss<rarelabscleaningpercent].index)]
    
    print('After rare removing {} patients'.format(df_lab.dw_pid.nunique()))
    print('After rare removing {} lab'.format(df_lab.servacro.nunique()))
    
    #Remove patients that are not in both list
    pid_static = df_static.dw_pid.unique()
    pid_lab = df_lab.dw_pid.unique()
    dw_pid_both = list(set(pid_static)&set(pid_lab))
    df_lab = df_lab[df_lab.dw_pid.isin(dw_pid_both)]
    df_static = df_static[df_static.dw_pid.isin(dw_pid_both)]
    
    print('Final number of patients: {}'.format(df_lab.dw_pid.nunique()))
    print('Final number of labs: {}'.format(df_lab.servacro.nunique()))

    return df_lab[['dw_pid','servacro','lbres_ck', 'step_to_disch']], df_static

def get_ICD10(static, output_dir, raw_ICD10_file):
    #Getting ICD10 main diagnosis     
    ## Adding main ICD10
    ICD10 = pd.read_csv(output_dir+'/' + raw_ICD10_file,
        usecols= ['cddiagnostic','desctypediagnostic', 
        'hospital_mrg','dw_pid','noadm_mrg'])
    ICD10.rename(columns = {'noadm_mrg':'noadm','hospital_mrg':'hospital'}, inplace=True)
    
    ICD10 = ICD10.dropna(subset = ['desctypediagnostic']).drop(columns = 'desctypediagnostic')
    ICD10 = ICD10.drop_duplicates(subset = ['dw_pid','noadm','cddiagnostic'])
    #adding cols
    cols = ['dw_pid','pdonor','noadm','subset']
    ICD10 = ICD10.merge(static[cols], on = ['dw_pid','noadm'])
    
    return ICD10

def static_embed(static, traintag = 'train_set'):
    #Static values encoding with a mean target encoding approach

    dic_emb = {}
    train_static = static[static.subset == traintag]
    counts = train_static['servcode'].value_counts()
    steep = 20 #Low steepness
    av = train_static.pdonor.mean()
    inflex = counts.mean()
    
    for serv,count in counts.items():      
        av_n = train_static[train_static['servcode'] == serv].pdonor.mean()
        lam = 1/(1+np.exp(-(count-inflex)/steep))
        tag = lam*av_n + (1-lam)*av
        dic_emb.update({serv:tag})
        dic_emb.update({'<UNKN>':0})
        
        static['servcode_embed'] = static['servcode'].map(dic_emb)
        static.loc[static['servcode_embed'].isna(),'servcode_embed'] = 0

        static['had_head_scan'] = static['had_head_scan'].astype(int)

    return static