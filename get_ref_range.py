
"""
Created on Tue May 19  2020
Last updated Tue Apr 12 2022

@author: nsauthier@gmail.com

Create a dataframe with normal reference range for all values (along with statistics)

Reference range were either extractedfrom the oasis raw data
or from logic (i.e. min=0, max=0) or rarely manually decided
Some of them are missing or textually entered and must be manually written
Some are missing because the absence of specified labs is the norm
Some have multiples reference range
By default 0 is imputed in place of missing value and the largest range is
chosen when multiples values exist.

"""

import pandas as pd

reference_range = {
'white_blood_cell_count':{'low':4.00,'high':11.00, 'dtype':'float64'},
'vitamin_b12':{'low':130.00,'high':130.00, 'dtype':'float64'},#Normal value is >130 so no physiological max
'venous_po2':{'low':35.00,'high':95.00, 'dtype':'float64'},
'venous_ph':{'low':7.31,'high':7.43, 'dtype':'float64'},
'venous_pco2':{'low':38.00,'high':54.00, 'dtype':'float64'},
'venous_o2_sat':{'low':0.70,'high':1.00, 'dtype':'float64'},
'venous_lactic_acid':{'low':0.56,'high':2.40, 'dtype':'float64'},
'venous_bicarbonate':{'low':21.00,'high':29.00, 'dtype':'float64'},
'venous_base_excess':{'low':-2.00,'high':3.00, 'dtype':'float64'},
'vancomycin_pre_dose':{'low':0.00,'high':0.00, 'dtype':'float64'}, #normal is no vancomycin
'vancomycin_post_dose':{'low':0.00,'high':0.00, 'dtype':'float64'}, #normal is no vancomycin
'vancomycin':{'low':0.00,'high':0.00, 'dtype':'float64'},
'urinary_ac_ascorb':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_yeast':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_urothelials_cells_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_urobilinogen':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_urate':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_renal_cells_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_pus':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_protein':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_polychromia':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_ph':{'low':4.80,'high':8.00, 'dtype':'float64'},
'urinary_pavimentous_cells_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_osmolality':{'low':50.00,'high':1200.00, 'dtype':'float64'},
'urinary_nitrite':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_mucus_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_leucocytes':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_hyalogranulous_cyl_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_hyalin_cylinder_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_granulous_cyl_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_glucose':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_density':{'low':1.00,'high':1.03, 'dtype':'float64'},
'urinary_cetones':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_blood':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_bilirubin':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'urinary_bacteria':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'uric_acid':{'low':167.00,'high':441.00, 'dtype':'float64'},
'urea':{'low':2.80,'high':8.80, 'dtype':'float64'},
'triglycerides':{'low':0.43,'high':2.82, 'dtype':'float64'},
'transthyretin':{'low':200.00,'high':400.00, 'dtype':'float64'},
'transferrin':{'low':1.83,'high':3.51, 'dtype':'float64'},
'toxic_granulation_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'total_protein':{'low':63.00,'high':81.00, 'dtype':'float64'},
'total_calcium':{'low':2.17,'high':2.56, 'dtype':'float64'},
'total_bilirubin':{'low':7.00,'high':23.00, 'dtype':'float64'},
'thyroid_stimulating_hormone':{'low':0.35,'high':5.50, 'dtype':'float64'},
'thrombin_time':{'low':12.00,'high':18.00, 'dtype':'float64'},
'temperature':{'low':36.00,'high':38.00, 'dtype':'float64'},
'target_cells_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'tacrolimus':{'low':0.00,'high':0.00, 'dtype':'float64'}, #normal is no tacrolimus
'stomatocytes_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'spherocytes_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'sodium':{'low':135.00,'high':145.00, 'dtype':'float64'},
'serum_iron':{'low':5.60,'high':31.60, 'dtype':'float64'},
'schizocytes_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'rolls_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'reticulocyte_count':{'low':0.00,'high':110.00, 'dtype':'float64'},
'red_blood_cell_deviation_width':{'low':11.50,'high':20.00, 'dtype':'float64'},
'red_blood_cell_count':{'low':3.80,'high':6.20, 'dtype':'float64'},
'procalcitonin':{'low':0.00,'high':0.10, 'dtype':'float64'},
'potassium':{'low':3.50,'high':5.00, 'dtype':'float64'},
'poikilocytosis_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'plt_anisocytosis_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'platelet_count':{'low':140.00,'high':500.00, 'dtype':'float64'},
'platelet_clumping':{'low':0.00,'high':0.00, 'dtype':'float64'},
'phosphate':{'low':0.72,'high':1.64, 'dtype':'float64'},
'ph':{'low':7.37,'high':7.43, 'dtype':'float64'},
'partial_thromboplastin_time_11':{'low':22.00,'high':32.00, 'dtype':'float64'},
'partial_thromboplastin_time':{'low':22.00,'high':32.00, 'dtype':'float64'},
'osmolality':{'low':275.00,'high':300.00, 'dtype':'float64'},
'nucleated_red_blood_cells':{'low':0.00,'high':0.10, 'dtype':'float64'},
'nt_pro_bnp':{'low':5.00,'high':738.00, 'dtype':'float64'},
'neutrophil_vacuols_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'neutrophil_count':{'low':1.30,'high':7.70, 'dtype':'float64'},
'myelocyte_count':{'low':0.00,'high':0.00, 'dtype':'float64'},
'monocyte_count':{'low':0.00,'high':1.60, 'dtype':'float64'},
'microcytosis_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'methemoglobin':{'low':0.00,'high':3.00, 'dtype':'float64'},
'metamyelocyte_count':{'low':0.00,'high':0.00, 'dtype':'float64'},
'mean_platelet_volume':{'low':6.50,'high':13.50, 'dtype':'float64'},
'mean_corpuscular_volume':{'low':80.00,'high':101.00, 'dtype':'float64'},
'mean_corpuscular_hemoglobin_concentration':{'low':300.00,'high':365.00, 'dtype':'float64'},
'mean_corpuscular_hemoglobin':{'low':24.00,'high':33.50, 'dtype':'float64'},
'magnesium':{'low':0.70,'high':1.01, 'dtype':'float64'},
'macrocytosis_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'lymphocyte_count':{'low':1.00,'high':4.10, 'dtype':'float64'},
'lipemia_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'lipase':{'low':10.00,'high':102.00, 'dtype':'float64'},
'leucocytes_count':{'low':0.00,'high':2.00, 'dtype':'float64'},
'lactate_dehydrogenase':{'low':104.00,'high':205.00, 'dtype':'float64'},
'keratocyte_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'jolly_body_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'ionized_calcium_ph74':{'low':1.12,'high':1.32, 'dtype':'float64'},
'intact_pth':{'low':1.40,'high':6.80, 'dtype':'float64'},
'inr_11':{'low':0.80,'high':1.20, 'dtype':'float64'},
'inr':{'low':0.80,'high':1.20, 'dtype':'float64'},
'indirect_bilirubin':{'low':5.00,'high':20.00, 'dtype':'float64'},
'icterus_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'hypersegmented_neutrophils_presence':{'low':0.00,'high':0.00, 'dtype':'float64'},
'hs_troponin_t':{'low':0.00,'high':18.00, 'dtype':'float64'},
'hemolysis':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'hemoglobin':{'low':120.00,'high':196.00, 'dtype':'float64'},
'hematocrit':{'low':0.35,'high':0.60, 'dtype':'float64'},
'hdl_cholesterol':{'low':0.80,'high':2.38, 'dtype':'float64'},
'hcv_ag':{'low':0.00,'high':0.00, 'dtype':'float64'},
'hbv_sag':{'low':0.00,'high':0.00, 'dtype':'float64'},
'hbv_cag':{'low':0.00,'high':0.00, 'dtype':'float64'},
'hba1c':{'low':0.04,'high':0.06, 'dtype':'float64'},
'haptoglobin':{'low':0.35,'high':4.10, 'dtype':'float64'},
'glucose':{'low':4.00,'high':6.20, 'dtype':'float64'},
'globulins':{'low':21.00,'high':34.00, 'dtype':'float64'},
'giant_platelets_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'gamma_glutamyl_transferase':{'low':7.00,'high':47.00, 'dtype':'float64'},
'gamma_globulin':{'low':6.25,'high':13.20, 'dtype':'float64'},
'free_t4':{'low':8.00,'high':23.00, 'dtype':'float64'},
'folic_acid':{'low':7.70,'high':7.70, 'dtype':'float64'},
'fio2':{'low':0.21,'high':0.21, 'dtype':'float64'},
'fibrinogen':{'low':2.00,'high':4.50, 'dtype':'float64'},
'ferritin':{'low':10.00,'high':345.00, 'dtype':'float64'},
'erythrocytes':{'low':0.00,'high':2.00, 'dtype':'float64'},
'erythrocyte_sedimentation_rate':{'low':0.00,'high':46.00, 'dtype':'float64'},
'eosinophil_count':{'low':0.00,'high':0.80, 'dtype':'float64'},
'elliptocytes_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'echinocyts_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'doehle_body_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'direct_bilirubin':{'low':0.00,'high':3.60, 'dtype':'float64'},
'dacryocytes_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'd_dimer':{'low':0.00,'high':600.00, 'dtype':'float64'},
'creatinine':{'low':42.00,'high':112.00, 'dtype':'float64'},
'creatine_kinase':{'low':24.00,'high':213.00, 'dtype':'float64'},
'cortisol':{'low':60.00,'high':618.00, 'dtype':'float64'},
'corrected_total_calcium':{'low':2.20,'high':2.58, 'dtype':'float64'},
'cmv_units':{'low':0.00,'high':0.00, 'dtype':'float64'},
'cmv_igg':{'low':0.00,'high':0.00, 'dtype':'float64'},
'ck_mb':{'low':0.00,'high':19.00, 'dtype':'float64'},
'cholesterol':{'low':3.16,'high':7.30, 'dtype':'float64'},
'chloride':{'low':96.00,'high':106.00, 'dtype':'float64'},
'c_reactive_protein':{'low':0.00,'high':10.00, 'dtype':'float64'},
'basophile_punctuation_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'basophil_count':{'low':0.00,'high':0.30, 'dtype':'float64'},
'base_excess':{'low':-2.50,'high':2.50, 'dtype':'float64'},
'atypia_lympho_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'aspartate_aminotransferase':{'low':13.00,'high':39.00, 'dtype':'float64'},
'arterial_po2':{'low':70.00,'high':110.00, 'dtype':'float64'},
'arterial_ph':{'low':7.35,'high':7.45, 'dtype':'float64'},
'arterial_pco2':{'low':32.00,'high':45.00, 'dtype':'float64'},
'arterial_o2_sat':{'low':0.92,'high':1.00, 'dtype':'float64'},
'arterial_lactic_acid':{'low':0.60,'high':2.40, 'dtype':'float64'},
'lactic_acid':{'low':0.56,'high':2.40, 'dtype':'float64'},
'arterial_bicarbonate':{'low':19.00,'high':28.00, 'dtype':'float64'},
'anticoagulant':{'low':0.00,'high':0.00, 'dtype':'categorical'},
'anti_hbs':{'low':10.00,'high':10.00, 'dtype':'float64'},
'anisocytosis_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'anisochromia_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'anion_gap':{'low':4.00,'high':14.00, 'dtype':'float64'},
'amylase':{'low':20.00,'high':104.00, 'dtype':'float64'},
'alpha2_macroglobulin':{'low':1.10,'high':3.65, 'dtype':'float64'},
'alpha1_glycoprotein':{'low':0.15,'high':0.90, 'dtype':'float64'},
'alpha1_antitrypsin':{'low':0.90,'high':2.30, 'dtype':'float64'},
'alkaline_phosphatase':{'low':36.00,'high':110.00, 'dtype':'float64'},
'albumin':{'low':36.00,'high':52.00, 'dtype':'float64'},
'alanine_aminotransferase':{'low':8.00,'high':39.00, 'dtype':'float64'},
'acanthocytes_presence':{'low':0.00,'high':0.00, 'dtype':'ordered categorical'},
'25_oh_vitamin_d':{'low':75.00,'high':150.00, 'dtype':'float64'}
}

def get_ref_range(df_lab):

    #In case new labs are added without pre-specified reference range
    for lab in df_lab.servacro.unique():
        if lab not in reference_range.keys():
            print('This lab is not in pre-registered ref range: '+lab)

    #generate a df from pre registered labs
    ref_range = pd.DataFrame(reference_range).T.reset_index().rename(columns={'index': 'servacro'})

    #New columns for ulterior faster imputation
    ref_range = ref_range.assign(imputation_mean = lambda x:(x.high + x.low)/2)
    ref_range = ref_range.assign(imputation_std = lambda x:(x.high - x.low)/(1.96*2))

    #Percentage of missing lab pre-imputation for statistics purpose
    perc_lab_missing = 1-(df_lab.groupby('servacro')['dw_pid'].nunique()\
      / df_lab.dw_pid.nunique()).rename('missing_percent')
    ref_range = ref_range.merge(perc_lab_missing, left_on = 'servacro', right_index=True, how='left')

    #Adding values descriptive statistics
    stat = df_lab.groupby('servacro')['lbres_ck'].agg(['mean','std', lambda x:x.quantile(0.50),
        lambda x:x.quantile(0.75)-x.quantile(0.25)]).reset_index()
    stat.columns = ['servacro','value_mean','value_std','value_50e_p','IQR']

    ref_range = ref_range.merge(stat, on='servacro')

    return ref_range