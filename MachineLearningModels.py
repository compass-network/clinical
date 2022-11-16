
"""
Last updated Tue Apr 12 2022

@author: nsauthier@gmail.com

This file create and does most of the analysis on the AutoEncoder and the two classifiers.

"""

## for Model definition/training
from keras.models import Model, load_model
from keras.layers import Input, Dense, concatenate,  Dropout,Add 
from keras.layers import Activation, BatchNormalization, MaxPooling1D, Conv1D
from keras.layers import UpSampling1D, TimeDistributed
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import regularizers
from keras import backend as k
from keras.callbacks import EarlyStopping

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf
import scipy

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

## for visualizing and data manipulation
import matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LR

import progressbar
import pickle
import random

class binary_classifier_model:
    #This class handle all classifiers creation and analysis

    def __init__(self, save_path):
        self.path = save_path + "/"
        
        self.donor_type = {
                'All donors':['Refered_Transfered_for_donation',
                              'Not_refered',
                              'Refered_Not_eligible',
                              'Refered_Donated',
                              'Non_donor'],
                  'Transfered donors':['Refered_Transfered_for_donation','Non_donor'],
                  'Refered and transplanted':['Refered_Donated','Non_donor'],
                  'Refered but ineligible':['Refered_Not_eligible','Non_donor'],
                  'Not refered':['Not_refered','Non_donor']}
        
    def create_base_network(self,input_size):
        
        input_layer = Input(shape=(input_size,))
        
        classif = Dense(round(3*input_size/4),activation='relu', 
                        kernel_regularizer = regularizers.l2(0.01))(input_layer)
        classif = Dropout(0.5)(classif)
        classif = Dense(round(2*input_size/4),activation='relu',
                        kernel_regularizer = regularizers.l2(0.01))(classif)
        classif = Dropout(0.5)(classif)
        classif = Dense(round(input_size/4),activation='relu',
                        kernel_regularizer = regularizers.l2(0.01))(classif)
        classif = Dropout(0.5)(classif)
        classif = Dense(1, activation='sigmoid')(classif)
        
        self.classif_model = Model(input_layer, classif)
        
        return self.classif_model
    
    def fit_model(self, batch_size, epochs, x_train, label_train, 
                  verbose=0,val_split = 0.2,lr=0.001, loss = 'binary_crossentropy',
                  new_model=True):
        self.x_train = x_train
        self.label_train = label_train
        self.lr = lr
        self.batch_size =batch_size
        self.epochs =epochs
        self.val_split = val_split
        self.loss = loss
        
        early_stopping = EarlyStopping(patience=25) #arbitrary choice based on graph of history
        
        if new_model:
            self.create_base_network(self.x_train.shape[1])
        
        #Balanced class weights
        class_weights = [1,
                         round(len(self.label_train)/self.label_train.sum())]

        self.classif_model.compile(loss=self.loss,
                                   optimizer=Adam(lr=self.lr),
                                   metrics = ['accuracy']
                                   )

        self.history = self.classif_model.fit(
                        x=self.x_train,
                        y=self.label_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_split=self.val_split,
                        class_weight = dict(enumerate(class_weights)),
                        verbose=verbose,
                        callbacks = [early_stopping]
                        )
        
        loss_hist = pd.DataFrame(self.history.history)[['loss','val_loss']]
        loss_hist['epoch'] = np.linspace(1,len(loss_hist), len(loss_hist))
        loss_hist = loss_hist.melt(id_vars='epoch')
        plt.plot(loss_hist[loss_hist.variable == 'loss'].epoch, 
                 loss_hist[loss_hist.variable == 'loss'].value, 
                 'b', 
                 label = 'loss')
        plt.plot(loss_hist[loss_hist.variable == 'val_loss'].epoch, 
                 loss_hist[loss_hist.variable == 'val_loss'].value, 
                 'r', 
                 label = 'validation loss')
        plt.title('Loss history on classifier')
        plt.legend(loc ='upper right')
        plt.gcf().savefig(self.path+'classifier_loss_history.png', dpi=100)
        plt.show()
        
        pd.DataFrame(self.history.history).to_csv(self.path+'classifier_history.csv', index=False)
        
        
    def save_classifier_model(self):

      self.classif_model.save(self.path + 'classifier_model.h5')
        
    def predict_class(self, x_test, label_test): 
        self.x_test = x_test
        self.label_test = label_test
        self.predicts = self.classif_model.predict(self.x_test).ravel()
        
        return self.predicts
         
    def logistic_regression(self, X_train, y_train, X_valid, y_valid):
        #Model for the comparison logistical model

        predictors = {'Logistical Model':'LR',
                      'Neural Network':'NN'}

        data_NN = pd.DataFrame({
          'y_predict':self.predicts,
          'y_true':self.label_test,
          'predictor':'NN'})
        
        #Calcul of LR prediction
        self.clf = LR(max_iter=100000, solver = 'liblinear', class_weight = 'balanced')
        self.clf.fit(X_train, y_train)
        self.predicts_LR = self.clf.predict_proba(X_valid)[:,-1]
        
        data_LR = pd.DataFrame({
                'y_predict':self.predicts_LR,
                'y_true':y_valid,
                'predictor':'LR'})
        
        data = pd.concat([data_NN, data_LR])
        
        list_ROC = []
        for keys, value in predictors.items():
            
            fpr, tpr, ROC_Thresh = roc_curve(data[data.predictor == value].y_true,
                                             data[data.predictor == value].y_predict,
                                             pos_label=1)
            roc_auc = auc(fpr, tpr,)
            label = keys + ' AUC={0:1.3f}'.format(roc_auc)
            ROC = {'Sensitivity':tpr,
                   '1-Specificity':fpr,
                   'Threshold':ROC_Thresh,
                   'Predictor':[label for i in range(len(fpr))]}
            list_ROC.append(pd.DataFrame(ROC))
        ROC= pd.concat(list_ROC)
        
        sns.set(style="whitegrid")
        sns.lineplot(x='1-Specificity', 
                     y = 'Sensitivity',
                     hue='Predictor', 
                     data=ROC, 
                     ci=None)
        sns.lineplot(x=[0,1], y = [0,1])
        plt.title('ROC curve')
        plt.ylabel('Sensitivity')
        plt.xlabel('1-Specificity')
        plt.gcf().savefig(self.path+'logistic_comparison.png', dpi=100)
        plt.show()
        
        return self.predicts_LR
    
    def ROC_donor_type(self, donor):
        ## Donor ROC
        
        data_NN = pd.DataFrame({
                'y_predict':self.predicts,
                'y_true':self.label_test,
                'donor':donor})
        
        list_ROC = []
        
        for keys, value in self.donor_type.items():
            
            #Get ROC values only for donor type specified + non donor
            fpr, tpr, ROC_Thresh = roc_curve(data_NN[data_NN.donor.isin(value)].y_true,
                                             data_NN[data_NN.donor.isin(value)].y_predict,
                                             pos_label=1)

            roc_auc = auc(fpr, tpr,)
            label = keys + ' AUC={0:1.3f}'.format(roc_auc)
            ROC_donor = {'Sensitivity':tpr,
                   '1-Specificity':fpr,
                   'Threshold':ROC_Thresh,
                   'Donor Type':[label for i in range(len(fpr))]}
            list_ROC.append(pd.DataFrame(ROC_donor))
            
        ROC_donor = pd.concat(list_ROC)
        sns.set(style="whitegrid")
        
        sns.lineplot(x='1-Specificity', 
                     y = 'Sensitivity',
                     hue='Donor Type', 
                     data=ROC_donor, 
                     ci=None)
        sns.lineplot(x=[0,1], y = [0,1])
        
        plt.title('ROC for donor subtype vs rest (NN)')
        plt.ylabel('Sensitivity')
        plt.xlabel('1-Specificity')
        plt.gcf().savefig(self.path+'ROC_organ_donors_types_NN.png', dpi=300)
        plt.show()
        
        #Confusion matrix  
        f, axes = plt.subplots(3, 2,figsize=(6,9), sharex=True, sharey=True)
        
        for i, [donor_type, value] in enumerate(self.donor_type.items()):
            confusion = confusion_matrix(data_NN[data_NN.donor.isin(value)].y_true,
                                        (data_NN[data_NN.donor.isin(value)].y_predict > self.cutoff_NN).values)
            
            tn, fp, fn, tp = confusion.ravel()
            
            prec = tp/(tp+fp)
            spec = tn/(tn+fp)
            
            sns.heatmap(confusion, annot=True, fmt ='d', cbar=False, 
                        cmap = 'Blues', ax=axes.flat[i])
            
            axes.flat[i].set_title(donor_type)
        f.text(0.5, 0.04, 'Predicted organ donor by NN', ha='center')
        f.text(0.04, 0.5, 'True organ donor', ha='center', rotation='vertical')
        f.suptitle('Confusion plot - Neural Network prediction', fontsize=15)
        plt.gcf().savefig(self.path+'Confusion_sens_donor_type_NN.png', dpi=100)
        plt.show()
        
        ## Data logistic regression
        data_LR = pd.DataFrame({
                'y_predict':self.predicts_LR,
                'y_true':self.label_test,
                'donor':donor})
        
        list_ROC = []
        
        for keys, value in self.donor_type.items():
            
            #Get ROC values only for donor type specified + non donor
            fpr, tpr, ROC_Thresh = roc_curve(data_LR[data_LR.donor.isin(value)].y_true,
                                             data_LR[data_LR.donor.isin(value)].y_predict,
                                             pos_label=1)

            roc_auc = auc(fpr, tpr,)
            label = keys + ' AUC={0:1.3f}'.format(roc_auc)
            ROC_donor = {'Sensitivity':tpr,
                   '1-Specificity':fpr,
                   'Threshold':ROC_Thresh,
                   'Donor Type':[label for i in range(len(fpr))]}
            list_ROC.append(pd.DataFrame(ROC_donor))
            
        ROC_donor = pd.concat(list_ROC)
        sns.set(style="whitegrid")
        
        sns.lineplot(x='1-Specificity', 
                     y = 'Sensitivity',
                     hue='Donor Type', 
                     data=ROC_donor, 
                     ci=None)
        sns.lineplot(x=[0,1], y = [0,1])
        
        plt.title('ROC for donor subtype vs rest (LM)')
        plt.ylabel('Sensitivity')
        plt.xlabel('1-Specificity')
        plt.gcf().savefig(self.path+'ROC_organ_donors_types_LM.png', dpi=300)
        plt.show()
        
        #Confusion matrix  
        f, axes = plt.subplots(3, 2,figsize=(6,9), sharex=True, sharey=True)
        
        for i, [donor_type, value] in enumerate(self.donor_type.items()):
            confusion = confusion_matrix(data_LR[data_LR.donor.isin(value)].y_true,
                                        (data_LR[data_LR.donor.isin(value)].y_predict > self.cutoff_LR).values)
            
            tn, fp, fn, tp = confusion.ravel()
            
            prec = tp/(tp+fp)
            spec = tn/(tn+fp)
            
            sns.heatmap(confusion, annot=True, fmt ='d', cbar=False, cmap = 'Blues', ax=axes.flat[i])
            
            axes.flat[i].set_title(donor_type)
        f.text(0.5, 0.04, 'Predicted organ donor by LM', ha='center')
        f.text(0.04, 0.5, 'True organ donor', ha='center', rotation='vertical')
        f.suptitle('Confusion plot - logistic regression prediction', fontsize=15)
        plt.gcf().savefig(self.path+'Confusion_sens_donor_type_LM.png', dpi=300)
        plt.show()
        
    def get_cutoff(self, sensitivity_value, N_kfold = 3):
        #Obtain an optimal cutoff for high sensitivity detection
        #by cross validation on train_dataset

        kf = KFold(n_splits = N_kfold, shuffle=False)
        cutoff_LR = []
        cutoff_NN = []
        for train_index, test_index in kf.split(self.x_train):
            
            #NN
            model_KF = self.create_base_network(self.x_train.shape[1])
            
            #Balanced class weights
            class_weights = [1,round(len(self.label_train)/self.label_train.sum())]
    
            model_KF.compile(loss='binary_crossentropy',
                          optimizer=Adam(lr=self.lr))
            
            early_stopping = EarlyStopping(patience=25) #arbitrary choice based on graph of history

            _ = model_KF.fit(
                x=self.x_train[train_index,:],
                y=self.label_train.values[train_index],
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.val_split,
                class_weight = dict(enumerate(class_weights)),
                verbose=0,
                callbacks = [early_stopping])
            
            predict_KF = model_KF.predict(self.x_train[test_index,:]).ravel()
            
            #Get ROC values only for donor type specified + non donor
            fpr, tpr, ROC_Thresh = roc_curve(self.label_train.values[test_index],
                                             predict_KF,
                                             pos_label=1)
                
            cutoff_NN.append(ROC_Thresh[np.where(tpr>sensitivity_value)[0].min()])
            
            #LR
            clf = LR(max_iter=1000, class_weight = 'balanced')
            clf.fit(self.x_train[train_index,:],
                    self.label_train.values[train_index])
            predict_KF = clf.predict_proba(self.x_train[test_index,:])[:,-1]
            
            #Get ROC values only for donor type specified + non donor
            fpr, tpr, ROC_Thresh = roc_curve(self.label_train.values[test_index],
                                             predict_KF,
                                             pos_label=1)
                
            cutoff_LR.append(ROC_Thresh[np.where(tpr>sensitivity_value)[0].min()])
            
        self.cutoff_NN = np.mean(np.asarray(cutoff_NN))
        self.cutoff_LR = np.mean(np.asarray(cutoff_LR))
        
        return self.cutoff_NN, self.cutoff_LR
    
    def confusion_plot(self,sensitivity_value,donor, N_kfold = 3):
        #Obtain confusion matrix for a pre_specified cutoff.
        #Get cutoff by Kfold on train_data
        
        self.get_cutoff(sensitivity_value, N_kfold = 3)
        
        f, axes = plt.subplots(1, len(self.donor_type),figsize=(10,5))
        
        df_true = pd.DataFrame(self.label_test)
        df_predict = pd.DataFrame(self.predicts)
        df_label = pd.DataFrame(donor)
        
        for i, [donor_type, value] in enumerate(self.donor_type.items()):
            confusion = confusion_matrix(df_true[df_label.isin(value)[0]].values,
                                         (df_predict[df_label.isin(value)[0]] > self.cutoff.mean()).values)
            
            tn, fp, fn, tp = confusion.ravel()
            
            prec = tp/(tp+fp)
            spec = tn/(tn+fp)
            
            sns.heatmap(confusion, annot=True, fmt ='d', cbar=False, cmap = 'Blues', ax=axes[i])
            axes[i].set_ylim(2.0,0)
            axes[i].set_xlabel('Predicted organ donor')
            axes[i].set_ylabel('Actual organ donor')
            
            axes[i].set_title(donor_type+'''
            Prec:{0:1.2f} Spec:{1:1.2f}'''.format(prec,spec))
        plt.xlabel('Actual organ donor')
        plt.gcf().savefig(self.path+'Confusion_sens_donor_type.png', dpi=100)
        plt.show()
    
    def bootstrap_prediction(self,n_bootstrap,X_last_valid):
        
        NN_predictions =[]
        LR_predictions =[]
        
        
        for i in progressbar.progressbar(range(n_bootstrap)):
            while(1):
                #Bootstraping
                idx_valid = np.random.randint(0,
                                              len(self.x_test), 
                                              len(self.x_test))
                X_valid_boot = self.x_test[idx_valid,:]
                X_last_valid_boot = X_last_valid[idx_valid,:]
                Y_valid_boot = self.label_test.values[idx_valid].ravel()
                
                #In case the bootstrap output only non donors
                if Y_valid_boot.sum()>0:
                    break
            NN_predictions.append({'predict':self.classif_model.predict(X_valid_boot).ravel(),
                                   'true':Y_valid_boot})
            LR_predictions.append({'predict':self.clf.predict_proba(X_last_valid_boot)[:,-1],
                                   'true':Y_valid_boot})
                
        return LR_predictions,NN_predictions
    
    def AUC_Brier_analysis(self,Pred_boot_LR,Pred_boot_NN):
        def unpack_boot(data_dict, fct, out):
            output = []
            
            for data in data_dict:
                output.append(fct(data['true'],data['predict']))
            
            quant = np.quantile(np.array(output), [0.025, 0.975])
            
            if out == 'quant':
                return quant
            elif out == 'all':
                return np.array(output)
        
        def unpack_SNSP(data_dict, fct, cutoff, out):
            output = []
            
            for data in data_dict:
                output.append(fct(data['true'],data['predict']>cutoff))
            
            quant = np.quantile(np.array(output), [0.025, 0.975])
            
            if out == 'quant':
                return quant
            elif out == 'all':
                return np.array(output)
        
        def Brier(true, predict):
            brier_max = np.mean(true) * (1 - np.mean(predict))
            brier = np.mean((true-predict)**2)
            
            return 1- brier/brier_max
        
        def PRC(true, predict):
            precision, recall, _ = precision_recall_curve(true,predict)
            return auc(recall, precision,)
        
        def ROC(true, predict):
            fpr, tpr, _ = roc_curve(true,predict)
            return auc(fpr, tpr,)
        
        def sensitivity(true, predict):
            true = true.astype(bool)
            tp = np.sum(true&predict)
            fn = np.sum(true & ~predict)
            return tp/(fn+tp)
        
        def specificity(true, predict):
            true = true.astype(bool)
            tn = np.sum(~true & ~predict)
            fp = np.sum(~true & predict)
            return tn/(tn+fp)
        
        def z_test(boot_test1, boot_test2, test1_val, test2_val):
            z = (test1_val-test2_val)/np.std(boot_test1 - boot_test2)
            pval = 2*scipy.stats.norm.cdf(-abs(z))
            return pval

        qtsn = unpack_SNSP(Pred_boot_NN, sensitivity, self.cutoff_NN,'quant')
        qtsp = unpack_SNSP(Pred_boot_NN, specificity, self.cutoff_NN,'quant')
        qtROC = unpack_boot(Pred_boot_NN, ROC,'quant')
        qtPRC = unpack_boot(Pred_boot_NN, PRC,'quant')
        qtBrier = unpack_boot(Pred_boot_NN, Brier,'quant')
        valsn = sensitivity(self.label_test,self.predicts>self.cutoff_NN)
        valsp = specificity(self.label_test,self.predicts>self.cutoff_NN)
        valROC = ROC(self.label_test,self.predicts)
        valPRC = PRC(self.label_test,self.predicts)
        valBrier = Brier(self.label_test,self.predicts)
        
        f = open(self.path + + '/BootstrapOutcome.txt', 'w')
        f.write('Neural network, based on {0:3.0f} bootstraps\n'.format(len(Pred_boot_NN)))
        f.write('ROC AUC was {0:1.3f} with CI95 ({1:1.3f}-{2:1.3f})\n'.format(valROC,*qtROC))
        f.write('PRC AUC was {0:1.3f} with CI95 ({1:1.3f}-{2:1.3f})\n'.format(valPRC,*qtPRC))
        f.write('Brier was {0:1.3f} with CI95 ({1:1.3f}-{2:1.3f})\n'.format(valBrier,*qtBrier))
        f.write('Sensitivity was {0:1.3f} with CI95 ({1:1.3f}-{2:1.3f})\n'.format(valsn,*qtsn))
        f.write('Specificity was {0:1.3f} with CI95 ({1:1.3f}-{2:1.3f})\n'.format(valsp,*qtsp))
        f.write('with cutoff of {0:1.3f}\n\n'.format(self.cutoff_NN))
        
        qtsn = unpack_SNSP(Pred_boot_LR, sensitivity, self.cutoff_LR,'quant')
        qtsp = unpack_SNSP(Pred_boot_LR, specificity, self.cutoff_LR,'quant')
        qtROC = unpack_boot(Pred_boot_LR, ROC,'quant')
        qtPRC = unpack_boot(Pred_boot_LR, PRC,'quant')
        qtBrier = unpack_boot(Pred_boot_LR, Brier,'quant')
        valsn = sensitivity(self.label_test,self.predicts_LR>self.cutoff_LR)
        valsp = specificity(self.label_test,self.predicts_LR>self.cutoff_LR)
        valROC = ROC(self.label_test,self.predicts_LR)
        valPRC = PRC(self.label_test,self.predicts_LR)
        valBrier = Brier(self.label_test,self.predicts_LR)
        
        f.write('Logistic regression, based on {0:3.0f} bootstraps\n'.format(len(Pred_boot_LR)))
        f.write('ROC AUC was {0:1.3f} with CI95 ({1:1.3f}-{2:1.3f})\n'.format(valROC,*qtROC))
        f.write('PRC AUC was {0:1.3f} with CI95 ({1:1.3f}-{2:1.3f})\n'.format(valPRC,*qtPRC))
        f.write('Brier was {0:1.3f} with CI95 ({1:1.3f}-{2:1.3f})\n'.format(valBrier,*qtBrier))
        f.write('Sensitivity was {0:1.3f} with CI95 ({1:1.3f}-{2:1.3f})\n'.format(valsn,*qtsn))
        f.write('Specificity was {0:1.3f} with CI95 ({1:1.3f}-{2:1.3f})\n'.format(valsp,*qtsp))
        f.write('with cutoff of {0:1.3f}\n\n'.format(self.cutoff_LR))
        
        #Does statistical Z test for comparison of results
        p_sen_test = z_test(unpack_SNSP(Pred_boot_LR, sensitivity, self.cutoff_LR,'all'),
                       unpack_SNSP(Pred_boot_NN, sensitivity, self.cutoff_NN,'all'),
                       sensitivity(self.label_test,self.predicts_LR>self.cutoff_LR),
                       sensitivity(self.label_test,self.predicts>self.cutoff_NN))
        p_spe_test = z_test(unpack_SNSP(Pred_boot_LR, specificity, self.cutoff_LR,'all'),
                       unpack_SNSP(Pred_boot_NN, specificity, self.cutoff_NN,'all'),
                       specificity(self.label_test,self.predicts_LR>self.cutoff_LR),
                       specificity(self.label_test,self.predicts>self.cutoff_NN))
        p_ROC_test = z_test(unpack_boot(Pred_boot_LR, ROC,'all'),
                       unpack_boot(Pred_boot_NN, ROC,'all'),
                       ROC(self.label_test,self.predicts_LR),
                       ROC(self.label_test,self.predicts))
        p_PRC_test = z_test(unpack_boot(Pred_boot_LR, PRC,'all'),
                       unpack_boot(Pred_boot_NN, PRC,'all'),
                       PRC(self.label_test,self.predicts_LR),
                       PRC(self.label_test,self.predicts))
        p_Brier_test = z_test(unpack_boot(Pred_boot_LR, Brier,'all'),
                       unpack_boot(Pred_boot_NN, Brier,'all'),
                       Brier(self.label_test,self.predicts_LR),
                       Brier(self.label_test,self.predicts))
        
        f.write('pvalues for comparisons are:\n')
        f.write('ROC p value was {0:1.5f}\n'.format(p_ROC_test))
        f.write('sensitivity p value was {0:1.5f}\n'.format(p_sen_test))
        f.write('specificity p value was {0:1.5f}\n'.format(p_spe_test))
        f.write('PRC p value was {0:1.5f}\n'.format(p_PRC_test))
        f.write('Brier p value was {0:1.5f}\n'.format(p_Brier_test))
        f.write('with cutoff of {0:1.2f}\n\n'.format(cutoff))
        
        f.close()
        
        
    def calibration_analysis(self,Pred_boot_LR,Pred_boot_NN):
        def calibration(data_dict, n_groups):
            #Empty ndarray size nb_boot x nb_groups
            output_count = np.empty((len(data_dict),n_groups))
            output_mean_score = np.empty((len(data_dict),n_groups))
            output_proportion = np.empty((len(data_dict),n_groups))
            
            #Empty array to act as a mask
            compare =  np.empty_like(data_dict[0]['true'])
            compare[:] = np.NaN
            
            #Group limit
            grp = np.linspace(0,1,n_groups+1)
            
            for i, data in enumerate(data_dict):
                for j in range(n_groups):
                    output_count[i,j] =len(np.where(
                            (data['predict'] >= grp[j]) & \
                            (data['predict'] < grp[j+1])
                            )[0])
            
                    output_mean_score[i,j] = np.mean(data['predict'][np.where(
                                (data['predict'] >= grp[j]) & \
                                (data['predict'] < grp[j+1]))])
            
                    output_proportion[i,j] = np.mean(data['true'][np.where(
                                (data['predict'] >= grp[j]) & \
                                (data['predict'] < grp[j+1]))])
            
            
            count = np.nanquantile(output_count, [0.025, 0.5, 0.975], axis=0)
            mean_score = np.nanquantile(output_mean_score, [0.025, 0.5, 0.975], axis=0)
            true_proportion = np.nanquantile(output_proportion, [0.025, 0.5, 0.975], axis=0)
            
            return count, mean_score, true_proportion
        
        count_NN, mean_score_NN, true_proportion_NN = calibration(Pred_boot_NN, 10)
        count_LR, mean_score_LR, true_proportion_LR = calibration(Pred_boot_LR, 10)
        
        import matplotlib.pyplot as plt
        labels = ['0-0.1','0.1-0.2','0.2-0.3','0.3-0.4',
                  '0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8',
                  '0.8-0.9','0.9-1',]
        tick = [0.05, 0.15, 0.25, 0.35, 0.45,
                0.55, 0.65, 0.75, 0.85, 0.95]
        #Figure Calibration LR
        fig, ax = plt.subplots(2,1, sharex=True, figsize = (5,8),
                               gridspec_kw = {'height_ratios':[2,1]})
        ax[0].vlines(tick, true_proportion_LR[0,:], true_proportion_LR[2,:])
        ax[0].plot(mean_score_LR[1,:], true_proportion_LR[1,:], 'o')
        ax[0].plot([0,1], [0,1], 'k--', linewidth=1)
        ax[0].set_ylabel('Actual Proportion')
        
        ax[1].vlines(mean_score_LR[1,:], count_LR[0,:], count_LR[2,:])
        ax[1].bar(tick, count_LR[1,:], width=0.1)
        ax[1].set_xticks(tick)
        ax[1].set_xticklabels(labels, rotation=45)
        ax[1].set_xlabel('Predicted Proportion')
        ax[1].set_ylabel('Count')
        
        plt.suptitle('Calibration - Logistic Regression')
        plt.gcf().savefig(self.path+'Calibration_LR.png', dpi=100)
        plt.show()
        
        #Figure Calibration NN
        fig, ax = plt.subplots(2,1, sharex=True, figsize = (5,8),
                               gridspec_kw = {'height_ratios':[2,1]})
        ax[0].vlines(mean_score_NN[1,:], true_proportion_NN[0,:], true_proportion_NN[2,:])
        ax[0].plot(mean_score_NN[1,:], true_proportion_NN[1,:], 'o')
        ax[0].plot([0,1], [0,1], 'k--', linewidth=1)
        ax[0].set_ylabel('Actual Proportion')
        
        ax[1].vlines(tick, count_NN[0,:], count_NN[2,:])
        ax[1].bar(tick, count_NN[1,:], width=0.1)
        ax[1].set_xticks(tick)
        ax[1].set_xticklabels(labels, rotation=45)
        ax[1].set_xlabel('Predicted Proportion')
        ax[1].set_ylabel('Count')
        
        plt.suptitle('Calibration - Neural Network')
        plt.gcf().savefig(self.path+'Calibration_NN.png', dpi=100)
        plt.show()

class autoencoder_model:
    #Creation and analysis of the autoencoder

    def __init__(self,save_path, embded_dimension=512,N_timestep=8,
                 N_features = 1024,conv_size=3,mid_activation='sigmoid',
                 load=False, dropout_perc = 0.1):
        self.path=save_path +'/'
        self.embded_dimension = embded_dimension
        self.N_timestep = N_timestep
        self.N_features = N_features
        self.conv_size = conv_size
        self.mid_activation = mid_activation
        self.dropout_perc = dropout_perc
        
        if load:
            self.autoencoder_embed = load_model(self.path + 'auto_encoder_model.h5')
            
    
    def create_model(self):
        input_layer_cnn = Input(shape=(self.N_timestep,self.N_features,))
        
        input_encoder1 = input_layer_cnn
        cnn_encoder = Conv1D(self.N_features, self.conv_size, strides=1, padding='causal',activation=None)(input_encoder1)
        cnn_encoder = BatchNormalization(axis = 2)(cnn_encoder)
        out_encoder1 = Dropout(self.dropout_perc)(cnn_encoder)     
        
        input_encoder2 = Add()([out_encoder1,input_encoder1])
        input_encoder2 = Activation('relu')(input_encoder2)
        input_encoder2 = MaxPooling1D(pool_size=2)(input_encoder2)
    
        cnn_encoder = Conv1D(self.N_features, self.conv_size, strides=1, padding='causal',activation=None)(input_encoder2)
        cnn_encoder = BatchNormalization(axis = 2)(cnn_encoder)
        out_encoder2 = Dropout(self.dropout_perc)(cnn_encoder)     
        
        input_encoder3 = Add()([out_encoder2,input_encoder2])
        input_encoder3 = Activation('relu')(input_encoder3)
        input_encoder3 = MaxPooling1D(pool_size=2)(input_encoder3)
        
        cnn_encoder = Conv1D(self.N_features, self.conv_size, strides=1, padding='causal',activation=None)(input_encoder3)
        cnn_encoder = BatchNormalization(axis = 2)(cnn_encoder)
        out_encoder3 = Dropout(self.dropout_perc)(cnn_encoder)     
        
        input_encoder4 = Add()([out_encoder3,input_encoder3])
        input_encoder4 = Activation('relu')(input_encoder4)
        input_encoder4 = MaxPooling1D(pool_size=2)(input_encoder4)
        
        out_encoder = Conv1D(round(self.embded_dimension), self.conv_size, strides=1, padding='causal',activation=self.mid_activation)(input_encoder4)
    
        cnn_decoder = UpSampling1D(3)(out_encoder)
        
        cnn_decoder = Conv1D(round(self.embded_dimension), self.conv_size, strides=1, padding='causal',activation='relu')(cnn_decoder)
        cnn_decoder = BatchNormalization(axis = 2)(cnn_decoder)
        cnn_decoder = Dropout(self.dropout_perc)(cnn_decoder)
        cnn_decoder = UpSampling1D(3)(cnn_decoder)
        
        cnn_decoder = Conv1D(round(self.embded_dimension/2), self.conv_size, strides=1, padding='causal',activation='relu')(cnn_decoder)
        
        output_cnn_decoder = TimeDistributed(Dense(self.N_features, activation=None))(cnn_decoder)
        
        self.autoencoder_model = Model(input_layer_cnn, output_cnn_decoder)
        
        self.autoencoder_embed = Model(input_layer_cnn, out_encoder)
    
    def fit(self, X_train_autoencoder, epoch, batch_size, val_split,verbose=2, loss='mse'):
        early_stopping = EarlyStopping(patience=50) #arbitrary choice based on graph of history

        
        self.autoencoder_model.compile(loss=loss, optimizer=Adam())
        
        self.history = self.autoencoder_model.fit(X_train_autoencoder,
                              X_train_autoencoder,
                              epochs=epoch,
                              batch_size=batch_size,
                              validation_split=val_split,
                              verbose = verbose,
                              callbacks = [early_stopping])
                
        loss_hist = pd.DataFrame(self.history.history)[['loss','val_loss']]
        loss_hist['epoch'] = np.linspace(1,len(loss_hist), len(loss_hist))
        loss_hist = loss_hist.melt(id_vars='epoch')
        plt.plot(loss_hist[loss_hist.variable == 'loss'].epoch, 
                 loss_hist[loss_hist.variable == 'loss'].value, 
                 'b', 
                 label = 'loss')
        plt.plot(loss_hist[loss_hist.variable == 'val_loss'].epoch, 
                 loss_hist[loss_hist.variable == 'val_loss'].value, 
                 'r', 
                 label = 'validation loss')
        plt.legend(loc ='upper right')
        plt.title('Loss history on autoencoder')
        plt.gcf().savefig(self.path+'autoencoder_loss_history.png', dpi=100)
        plt.show()
        
        pd.DataFrame(self.history.history).to_csv(self.path+'autoencoder_history.csv', index=False)
        
    def save_embeding_model(self):
        self.autoencoder_embed.save(self.path + 'auto_encoder_model.h5')
        
    def embed(self, data):
        #Embed the data with the pretrain model
        self.embeded_data = self.autoencoder_embed.predict(data)
        
        return self.embeded_data