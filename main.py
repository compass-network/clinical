# %% Import
import sys, os
sys.path.append('/data8/projets/ChasseM_Compass_1013663/'+\
                'code/nsauthier/git_ChasseM_Compass_1013663/'+\
                'Clinical_Model/time_series_model/')
from importlib import reload
import pandas as pd, numpy as np
import time
import pickle

output_dir ='/data8/projets/ChasseM_Compass_1013663/output/clinical_data/'
file_name = "FINAL"
output_folder = output_dir+"/8_transplantqc_donors2020_ICUCARD/"

raw_static_file = '3_compass_episodes_tq_donors_ICUCARD_LastEpisd_2020-07-30'
raw_radiology_file = '4_all_radiology_exams_tq_donors_2020-06-15.csv.gz'
raw_lab_file = '5_lab_results_tq_donors_ICUCARD_LastEpisd_2020-07-30.csv.gz'
raw_ICD10_file = '6_medecho_ICD_diag_codes_tq_donors_2021-06-07.csv'

#Data cleaning
import DataCleaning
from DataCleaning import static_lab_cleaning, get_ICD10, static_embed
from get_ref_range import get_ref_range

#Data management and formatting
import TrainTestValidCreation
import ScalingAnd3DFormatting
from TemporalImputationAndFormat import TemporalImputationAndFormat
#
# %% First part: loading and cleaning temporal and static data
##############################################################################
t0 = time.time()
lab_cleaner = DataCleaning.lab_data(raw_lab_file,file_name,output_dir)
clean_lab = lab_cleaner.load().clean()
t1 = time.time()-t0
print('Total lab cleaning time: {0:4.0f} min and {1:2.0f} seconds'.format(t1//60, t1%60))

#Static cleaning
static_cleaner = DataCleaning.static_data(raw_static_file,file_name,output_dir, raw_radiology_file)
clean_static = static_cleaner.load().clean()
t2 = time.time()-t1+t0
print('Total static cleaning time: {0:4.0f} min and {1:2.0f} seconds'.format(t2//60, t2%60))

#Conjoint static and lab cleaning
clean_lab, clean_static = static_lab_cleaning(clean_lab, clean_static)
t3 = time.time()-t2
print('Total clean_static time: {0:4.0f} min and {1:2.0f} seconds'.format(t3//60, t3%60))

#Separating in train AE / train / test / valid
splitter = TrainTestValidCreation.TrainTestValidCreation(file_name, output_dir)
listing = splitter.traintestvalid(clean_static, load=True)
clean_static = clean_static.merge(listing[['dw_pid','subset']],on='dw_pid',how='left')

#Get ICD10 for statistics purpose
ICD10 = get_ICD10(clean_static, output_dir, raw_ICD10_file)

#Reference range
ref_range = get_ref_range(clean_lab)

#Exporting
clean_lab.to_csv(output_folder+file_name+'_clean_lab.csv.gz',compression='gzip', index=False)
clean_static.to_csv(output_folder+file_name+'_clean_static.csv', index=False)
ref_range.to_csv(output_folder+file_name+'_clean_ref_range.csv', index=False)

# %% Second part: Formatting and imputation of the temporal data
#############################################################################
temporal_final = TemporalImputationAndFormat(clean_lab,ref_range)

temporal_final.to_csv(output_folder+file_name+'_lab_imputated.csv.gz',compression='gzip', index=False)

static_final = clean_static

# %% Third part: Creating and training the auto-encoder
##############################################################################
import MachineLearningModels
import Article1_SensitivityAnalysis as compass_tsa

#Saving / exporting tabular results
result_folder = 'results_'+time.strftime("%Y%m%d_%H%M")
os.mkdir(output_folder+result_folder)
result_folder_path = output_folder+result_folder

#format the data in the 3D format instead of the tidy format
formatter = ScalingAnd3DFormatting.ScalingAnd3DFormatting()

AE_train = temporal_final[temporal_final.dw_pid.isin(static_final[static_final.subset == 'AE_train_set'].dw_pid)]

mean, std = formatter.scaler_train(AE_train)

# %% Fourth part: The auto-encoder
##############################################################################
print('Starting embedding...')
X_train_autoencoder = formatter.temporal_format(AE_train)

autoencoder = MachineLearningModels.autoencoder_model(result_folder_path,
                                                        embded_dimension=64,
                                                        N_timestep=X_train_autoencoder.shape[1],
                                                        N_features = X_train_autoencoder.shape[2],
                                                        conv_size=3,
                                                        mid_activation='sigmoid',
                                                        load=False )#To load already existing model

autoencoder.create_model()

autoencoder.fit(X_train_autoencoder,epoch = 500, batch_size=512, val_split=0.2,verbose=0.5, loss='mse')
autoencoder.save_embeding_model()

#To load
#autoencoder = compass_temporal_models.autoencoder_model(output_dir,load=True)

#Embedding the train / valid
train_set = ['train_set', 'valid_set']#['train_set']['valid_set','train_set']
test_set = 'test_set'#'test_set'#''valid_set 'valid_ucoro'

train = temporal_final[temporal_final.dw_pid.isin(static_final[static_final.subset.isin(train_set)].dw_pid)]
X_train_temporal = formatter.temporal_format(train)

valid = temporal_final[temporal_final.dw_pid.isin(static_final[static_final.subset == test_set].dw_pid)]
X_valid_temporal = formatter.temporal_format(valid)

X_train_embed = autoencoder.embed(X_train_temporal)
X_train_embed = X_train_embed.reshape((X_train_embed.shape[0],X_train_embed.shape[-1]))

X_valid_embed = autoencoder.embed(X_valid_temporal)
X_valid_embed = X_valid_embed.reshape((X_valid_embed.shape[0],X_valid_embed.shape[-1]))

# %% Fifth part: Creating and training the classifier
##############################################################################

#Train the static and concatenante static and dynamic
static_train = static_final[static_final.subset.isin(train_set)].sort_values(by='dw_pid')
X_train = np.hstack((X_train_embed, static_train[['servcode_embed', 'had_head_scan']].values))
X_train_lastval = formatter.last_val_format(train, static_train, ['servcode_embed','had_head_scan'])
Y_train = static_train.pdonor

static_valid = static_final[static_final.subset == test_set].sort_values(by='dw_pid')
X_valid = np.hstack((X_valid_embed, static_valid[['servcode_embed', 'had_head_scan']].values))
X_valid_lastval = formatter.last_val_format(valid, static_valid, ['servcode_embed','had_head_scan'])
Y_valid = static_valid.pdonor

bin_classif = MachineLearningModels.binary_classifier_model(result_folder_path)
_ = bin_classif.create_base_network(X_train.shape[1])
bin_classif.fit_model(128, 250, X_train, Y_train, verbose=0.5,
                      val_split = 0.2, lr = 0.001, loss= "binary_crossentropy")
bin_classif.save_classifier_model()

class_p = bin_classif.predict_class(X_valid, Y_valid)
class_p_LR = bin_classif.logistic_regression(X_train_lastval, Y_train,
                                X_valid_lastval, Y_valid)

#Get 90% sensitivity cutoff by KFold
cutoff_NN, cutoff_LR = bin_classif.get_cutoff(0.9)
#Get specific analysis of ROC curve by organ donor type
bin_classif.ROC_donor_type(static_valid.donor_type.values)

#Bootstrapping prediction for used model
Pred_boot_LR, Pred_boot_NN = bin_classif.bootstrap_prediction(2000, X_valid_lastval)
bin_classif.calibration_analysis(Pred_boot_LR,Pred_boot_NN)
bin_classif.AUC_Brier_analysis(Pred_boot_LR,Pred_boot_NN)

#Save Bootstrap
with open(result_folder_path+'/'+'pred_bootstrap_LR.pickle', 'wb') as f:
    pickle.dump(Pred_boot_LR, f)
with open(result_folder_path+'/'+'pred_bootstrap_NN.pickle', 'wb') as f:
    pickle.dump(Pred_boot_NN, f)

#Export prediction
static_valid['Predict_NN'] = class_p
static_valid['Predict_LR'] = class_p_LR
static_valid.to_excel(result_folder_path+'/'+ 'prediction.xls', index=False)

#Export subset for error checking
error_check = static_valid
error_check = error_check[(error_check.Predict_NN >= 0.75) | (error_check.Predict_LR >= 0.75)]
error_check = error_check[(error_check.pdonor == 0) & (error_check.death_in_icu == 1)]
error_check.to_excel(result_folder_path+'/'+'error_check.xls', index=False)

# %% Sixth part: Prospective simulation
##########################################################################
clean_lab = pd.read_csv(output_folder+file_name+'_clean_lab.csv.gz',compression='gzip')

prosp = compass_tsa.prospective_simulation(file_name, result_folder_path,
                                           autoencoder, bin_classif, formatter,
                                           static_final, clean_lab, ref_range)
pid = static_final[static_final['subset'] == test_set].dw_pid.values
pred = prosp.format_and_run(pid)

#Export tabular data
pred.to_excel(result_folder_path+'/prospective_prediction.xls', index=False)
prosp.ROC()

# %% Seventh part: Sensitivity Analysis
#############################################################################
#Labs removed by group of missingness
sensitivity = compass_tsa.sensitivity_analysis(file_name, result_folder_path)
sensitivity.remove_labs(2000,'grouped',
                        temporal_final, static_final, 
                        ref_range, result_folder_path)
sensitivity.remove_labs(2000,'single',
                        temporal_final, static_final, 
                        ref_range, result_folder)