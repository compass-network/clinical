
"""
Created Sept 2019
Last updated Tue Apr 12 2022

@author: nsauthier@gmail.com

Series of function that takes a uncleaned dataframe of labs from CITADEL
remove NaN results, remove some redundants labs, change somme litteral labs
to numeric (+-> 1, ++ -> 2, etc.).
Then merge some labs with different names (blood gas, etc.)

Return a dataframe of labs

"""

# Import section
import pandas as pd
import numpy as np
import re

## Servacro dict to replace, list to remove
remove_servacro = [
        #Dropping some analysis because of redundancy with other results:
        'CXLba:CA IONISE ACT.', #redudant with value corrected for pH
        'CXLba:LDL-C (calc)', #Simple calcul to obtain value
        'CXLba:NON-HDL CHOLESTEROL', #Simple calcul to obtain value
        'CXLba:T FILT GLOM CAL',
        'CXLha:STAB',  'CXLha:META',
        'CXLha:MYELO', 'CXLha:PROMYELO', 'CXLha:BLASTE',
        'CXLba:CO2 TOTAL','CXLba:CO2 TOTAL #','CXLba:CO2 TOTAL V #', #CO2 redundant with pCO2
        #Drop of specific test because of redundancy, litteral value content
        'CXLba:INT TROPT HS','CXLba:INT TROPT HS',
        'CXLba:COULEUR','CXLba:ASPECT (AU)', 
        'CXLha:TURBIDITE (LASC)',
        'CXLha:TURBIDITE (LPLEU)', 'CXLha:TURBIDITE (LHBHT)',
        'CXLha:TURBIDITE (LCR)', 'CXLha:TURBIDITE (LART)',
        'CXLha:TURBIDITE (LPERIT)','Lba:SYNTROID ?', 'CXLha:INT.(HD)',
        'CXLia:CVHCV CONT. CLINIQUE', 'Lxa:titrecontact', 
        'CXLza:COMM. ELEC. SERINGUE', 'CXLba:COMMENTAIRE',
        'Lba:COMMENTAIRE MICRO', 'Lha:COMMENT.LBA','Lxa:codeeve',
        'Lxa:datehreIA', 'Lxa:avis', 'Lxa:Analyse(s) visée(s)',
        'CXLha:PLT SUR LAME', 'Lba:DATE DEBUT COLL', 'Lba:DATE FIN COLL'
        ]

new_servacro = [
        'Lba:Sat O2 VEINEUX',
        'CXLba:PH ARTERIEL',
        'CXLba:PCO2 ARTERIEL',
        'CXLba:HCO3 VEINEUX',
        'Lba:PO2 VEINEUX',
        'Lba:PCO2 VEINEUX',
        'CXLba:HCO3 ARTERIEL',
        'CXLba:Sat O2 VEINEUX',
        'CXLba:ACIDE LACTIQUE VEINEUX',
        'Lba:PH VEINEUX',
        'CXLba:ACIDE LACTIQUE ARTERIEL',
        'CXLba:Sat O2 ARTERIEL',
        'CXLba:PO2 ARTERIEL',
        'Lha:META #',
        'Lha:MYELO #',
        'Lha:PROMYELO #',
        'Lha:STAB #',
        'Lha:BLASTE #']

rename_servacro = {
        #Double name because of change over time / different hospital
        "CXLha:ANTI-HISTONE":"CXLba:ANTI-HISTONE",
        "Lba:C4.":"CXLba:C4",
        "Lba:CA 125.":"CXLba:CA 125",
        "Lba:FERRITINE.":"CXLba:FERRITINE",
        "Lba:HAPTOGLOBINE.":"CXLba:HAPTOGLOBINE",
        "Lba:IgA":"CXLba:IgA",
        "Lba:IgM":"CXLba:IgM",
        "CXLza:TEMPERATURE (I)":"CXLba:TEMPERATURE",
        'CXLza:LACTATE ART.SER. (I)':'CXLba:AC LAC ART SER',
        "Lha:AC. FOLIQUE":"CXLha:ACIDE FOLIQUE",
        "Lba:VITAMINE B12.":"CXLha:VITAMINE B12",
        "CD:10485362":"CXLia:HBe Ag",
        "CXLha:GB CORRIGE":"CXLha:GB",

        #Blood gas
        #FIO2
        'CXLza:FIO2 (I)':'CXLba:FIO2',
        #Venous saturation
        'CXLba:SAT.O2 (VEINE)':'CXLba:Sat O2 VEINEUX',
        'CXLza:SATUR.O2 VEINEUX (I)':'CXLba:Sat O2 VEINEUX',
        "CXLba:SAT.O2 (VEINE)mesuree":'CXLba:Sat O2 VEINEUX',
        "Lba:Sat O2 VEINEUX":'CXLba:Sat O2 VEINEUX',
        "Lba:Sat O2 Vein":'CXLba:Sat O2 VEINEUX',
        #Arterial saturation
        'CXLba:SAT.O2':'CXLba:Sat O2 ARTERIEL',
        "CXLba:SAT.O2 (mesuree)":'CXLba:Sat O2 ARTERIEL',
        "CXLza:SATUR.O2 ARTER. (I)":'CXLba:Sat O2 ARTERIEL',
        "Lba:Sat O2 Art":'CXLba:Sat O2 ARTERIEL',
        #PH vein
        "CXLza:PH VEINEUX (I)":"CXLba:PH VEINEUX",
        'Lba:PH VEINEUX TC':"CXLba:PH VEINEUX",
        'Lba:PH VEINEUX':"CXLba:PH VEINEUX",
        #Ph Art
        "CXLza:PH ARTERIEL (I)":'CXLba:PH ARTERIEL',
        "Lba:PH TC":'CXLba:PH ARTERIEL',
        #Pco2 vein        
        "CXLza:PCO2 VEINEUX (I)":"CXLba:PCO2 VEINEUX",
        "Lba:PCO2 VEINEUX TC":"CXLba:PCO2 VEINEUX",
        "Lba:PCO2 VEINEUX":"CXLba:PCO2 VEINEUX",
        #Pco2 art
        "CXLza:PCO2 ARTERIEL (I)":'CXLba:PCO2 ARTERIEL',
        'CXLba:PCO2':'CXLba:PCO2 ARTERIEL',
        'Lba:PCO2 TC':'CXLba:PCO2 ARTERIEL',
        #BIC vein
        'CXLba:HCO3 ACT.(V)#':'CXLba:HCO3 VEINEUX',
        'CXLza:HCO3 ACT. VEINEUX (I)':'CXLba:HCO3 VEINEUX',
        #BIC art
        'CXLba:HCO3 ACTUEL':'CXLba:HCO3 ARTERIEL',
        'CXLza:HCO3 ACTUEL ART. (I)':'CXLba:HCO3 ARTERIEL',
        #Po2 vein        
        "CXLza:PO2 VEINEUX (I)":"CXLba:PO2 VEINEUX",
        "Lba:PO2 VEINEUX TC":"CXLba:PO2 VEINEUX",
        "Lba:PO2 VEINEUX":"CXLba:PO2 VEINEUX",
        #po2 art
        "CXLza:PO2 ARTERIEL (I)":'CXLba:PO2 ARTERIEL',
        "Lba:PO2 TC":'CXLba:PO2 ARTERIEL',
        'CXLba:PO2':'CXLba:PO2 ARTERIEL',
        #Lactic acide
        'CXLba:AC. LACT (TUBE VERT)':'CXLba:ACIDE LACTIQUE VEINEUX',
        'CXLba:ACIDE LACTIQUE':'CXLba:ACIDE LACTIQUE VEINEUX',
        'CXLza:LACTATE VEINE.SER.(I)':'CXLba:ACIDE LACTIQUE VEINEUX',
        'CXLba:AC LAC VEIN SER':'CXLba:ACIDE LACTIQUE VEINEUX',
        'CXLba:AC LAC ART SER':'CXLba:ACIDE LACTIQUE ARTERIEL',
        'CXLza:LACTATE ART.SER. (I)':'CXLba:ACIDE LACTIQUE ARTERIEL',
        #Electrolytes for resp therapist
        'CXLza:K SERINGUE (I)':'CXLba:POTASSIUM',
        'CXLza:CHLORE SERINGUE (I)':'CXLba:CHLORURE',
        'CXLza:GLUCOSE SERINGUE (I)':'CXLba:GLUCOSE',
        'CXLza:SODIUM SERINGUE (I)':'CXLba:SODIUM',
        'CXLba:K SERINGUE':'CXLba:POTASSIUM',
        'CXLba:CHLORURE SERINGUE':'CXLba:CHLORURE',
        'CXLba:GLUCOSE SERINGUE':'CXLba:GLUCOSE',
        'CXLba:SODIUM SERINGUE':'CXLba:SODIUM',

        'CD:44715080': 'Lha:ANTICOAGULANT ?',
        'Lba:CK.': 'CXLba:CK',
        'Lba:LIPASE.': 'CXLba:LIPASE',

        #Add manual measure to normal ones
        'CXLha:BASO # (MAN)':'CXLha:BASO #',
        'CXLha:EOSINO # (MAN)':'CXLha:EOSINO #',
        'CXLha:LYMPHO # (MAN)':'CXLha:LYMPHO #',
        'CXLha:MONO # (MAN)':'CXLha:MONO #',
        'CXLha:NEUTRO # (MAN)':'CXLha:NEUTRO #',
        'Lha:META # (MAN)':'Lha:META #',
        'Lha:MYELO # (MAN)':'Lha:MYELO #',
        'Lha:PROMYELO # (MAN)':'Lha:PROMYELO #',
        'Lha:STAB # (MAN)':'Lha:STAB #',
        'Lha:BLASTE # (MAN)':'Lha:BLASTE #',

        #Multiple analysis
        'CXLda:GLUCOSE (ADBD)':'CXLba:GLUCOSE',
        'Lba:GLUCOSE AC':'CXLba:GLUCOSE',
        'CXLba:GLUCOSE VEIN SER':'CXLba:GLUCOSE',
        'CXLba:GLUCOSE T 0':'CXLba:GLUCOSE',
        'CXLba:GLUCOSE CAPILLAIRE':'CXLba:GLUCOSE',
        'CXLba:GLUCOSE T 0 MIN':'CXLba:GLUCOSE',
        'CXLba:GLUCOSE ART SER':'CXLba:GLUCOSE',
        'Lba:Glu Sg tot':'CXLba:GLUCOSE',
        'CXLza:METHEMOGLOB.VEINE (I)':'CXLba:METHEMOGLOBINE',
        'CXLza:METHEMOGLOB. ART. (I)':'CXLba:METHEMOGLOBINE',
        'Lba:CKMB masse':'CXLba:CKMB',
        'CXLba:HAPTOGLOBINE, ELEC.':'CXLba:HAPTOGLOBINE',
        'CXLba:ALBUMINE, ELEC.':'CXLba:ALBUMINE',
        'CXLba:A1-ANTITRYPSINE, ELEC':'CXLba:A1 ANTITRYPSINE',
        'CXLba:A2-MACROGLOB., ELEC.':'CXLba:A2-MACROGLOB.',
        'CXLba:PHOS.ALC.(ELEC)':'CXLba:PHOSPHATASE ALC',
        'CXLba:TRANSFERRINE, ELEC.':'CXLba:TRANSFERRINE',
        'CXLba:OSMOLALITE CALC':'CXLba:OSMOLALITE SER.'
        }

new_longdesc = [
        'HCO3 Art', 'pCO2 Vein', 'pCO2 Art', 'pO2 Vein', 
        'Acide Lactique Vein', 'pH Vein', 'pH Art', 
        'Acide Lactique Art', 'HCO3 Vein', 'pO2 Art']

replace_longdesc = {
        'GB Corrigé':'GB',
        'Lactate Artér.seringue':'Acide Lac.Art. Ser.',
        'Sat. O2 (V.) mes.':'Sat O2 Vein',
        'Satur.O2 Veineux':'Sat O2 Vein',
        'Sat.O2(mesurée)':'Sat O2 Art',
        'Satur.O2 Artériel':'Sat O2 Art',
        'pH Veineux':'pH Vein',
        'pH Veineux TC':'pH Vein',
        'pH Artériel':'pH Art',
        'pH TC':'pH Art',
        'pCO2 Veineux':'pCO2 Vein',
        'pCO2':'pCO2 Art',
        'pCO2 TC':'pCO2 Art',
        'pCO2 Veineux TC':'pCO2 Vein',
        'pCO2 Artériel':'pCO2 Art',
        'HCO3 Actuel (V)':'HCO3 Vein',
        'HCO3 Act.(Veineux)':'HCO3 Vein',
        'HCO3 Actuel':'HCO3 Art',
        'HCO3 Actuel Art.':'HCO3 Art',
        'pO2 Veineux':'pO2 Vein',
        'pO2 Veineux TC':'pO2 Vein',
        'pO2 Artériel':'pO2 Art',
        'pO2 TC':'pO2 Art',
        'pO2':'pO2 Art',
        'Acide Lac.Veineux':'Acide Lactique Vein',
        'Acide Lactique':'Acide Lactique Vein',
        'Lactate Vein.seringue':'Acide Lactique Vein',
        'AC LACTIQUE VEINEUX SERINGUE':'Acide Lactique Vein',
        'Acide Lac.Art. Ser.':'Acide Lactique Art',
        'Lactate Artér.seringue':'Acide Lactique Art',
        'Glucose seringue':'Glucose ADBD',
        'K seringue':'K Seringue',
        'Chlore seringue':'Cl Seringue',
        'Sodium seringue':'Na Seringue',
        'CK (AU)': 'CK',
        'Lipase.': 'Lipase',
        'Anticoagulant': 'Anticoagulant ?',

        'Glucose AC':'Glucose',
        'Glucose ADBD':'Glucose', 
        'Glucose Art. Ser.':'Glucose',
        'Glucose Vein.Ser.':'Glucose',
        'Glucose T 0':'Glucose',
        'Glucose Capillaire':'Glucose',
        'Glucose Sg tot':'Glucose',
        'GLUCOSE T 0 MIN':'Glucose',

        'Baso # (Man)':'Baso #',
        'Lympho # (Man)':'Lympho #',
        'Neutro # (Man)':'Neutro #',
        'Mono # (Man)':'Mono #',
        'Éosino # (Man)':'Éosino #',

        }

lab_names= {
        'CXLha:ELLIPTOCYTES':'elliptocytes_presence',
        'CXLha:DACRYOCYTES':'dacryocytes_presence',
        'CXLha:DACRYOCYTES':'dacryocytes_presence',
        'CXLha:MACROCYTOSE':'macrocytosis_presence',
        'CXLha:PONC.BASOPHILES':'basophile_punctuation_presence',
        'CXLha:ECHINOCYTES':'echinocyts_presence',
        'CXLha:PLT ANISOCYTOSE':'plt_anisocytosis_presence',
        'CXLha:CELLULES CIBLES':'target_cells_presence',
        'CXLha:KERATOCYTES':'keratocyte_presence',
        'CXLha:ATYPIE LYMPHO.':'atypia_lympho_presence',
        'CXLha:POIKILOCYTOSE':'poikilocytosis_presence',
        'CXLha:SPHEROCYTES':'spherocytes_presence',
        'CXLha:STOMATOCYTES':'stomatocytes_presence',
        'CXLha:ANISOCHROMIE':'anisochromia_presence',
        'CXLha:ACANTHOCYTES':'acanthocytes_presence',
        'CXLha:ANISOCYTOSE':'anisocytosis_presence',
        'CXLha:NEUTRO.HYPERSEG':'hypersegmented_neutrophils_presence',
        'CXLha:PLT GEANTES':'giant_platelets_presence',
        'CXLha:SCHIZOCYTES':'schizocytes_presence',
        'CXLha:CORPS DE DOEHLE':'doehle_body_presence',
        'CXLha:CORPS DE JOLLY':'jolly_body_presence',
        'CXLha:MICROCYTOSE':'microcytosis_presence',
        'CXLha:NEUTRO.VACUOL':'neutrophil_vacuols_presence',
        'CXLha:GRANUL.TOXIQUES':'toxic_granulation_presence',
        'CXLha:ROULEAUX':'rolls_presence',
        'CXLba:CYL. HYALOGRANULEUX':'urinary_hyalogranulous_cyl_presence',
        'CXLba:CYL. GRANULEUX':'urinary_granulous_cyl_presence',
        'CXLia:CMV UNITES':'cmv_units',
        'CXLba:CELL. RENALES':'urinary_renal_cells_presence',
        'CXLba:C. UROTHELIALES':'urinary_urothelials_cells_presence',
        'CXLba:C.PAVIMENTEUSES':'urinary_pavimentous_cells_presence',
        'CXLba:CYL. HYALINS':'urinary_hyalin_cylinder_presence',
        'CXLba:UROBILINOGENE B':'urinary_urobilinogen',
        'CXLba:METHEMOGLOBINE':'methemoglobin',
        'CXLba:HBA1C':'hba1c',
        'CXLma:VANCO PRE DOSE':'vancomycin_pre_dose',
        'CXLba:LIPEMIE':'lipemia_presence',
        'CXLba:PUS':'urinary_pus',
        'CXLba:LEVURES':'urinary_yeast',
        'CXLba:A-1-GLYCOPROT., ELEC.': 'alpha1_glycoprotein',
        'Lha:ANTICOAGULANT ?':'anticoagulant',
        'CXLha:T.PROT.INR(1:1)':'inr_11',
        'CXLha:LYMPHO #':'lymphocyte_count',
        'CXLha:BASO #':'basophil_count',
        'CXLba:ACIDE LACTIQUE ARTERIEL':'arterial_lactic_acid',
        'CXLba:CKMB':'ck_mb',
        'CXLba:TEMPERATURE':'temperature',
        'CXLba:URATES AMORPHES':'urinary_urate',
        'CXLba:CA ION.(PH:7.4)':'ionized_calcium_ph74',
        'CXLha:HT':'hematocrit',
        'CXLba:PH VEINEUX':'venous_ph',
        'CXLha:EOSINO #':'eosinophil_count',
        'CXLba:PH ARTERIEL':'arterial_ph',
        'CXLha:T.THROMBINE':'thrombin_time',
        'CXLha:DVE':'red_blood_cell_deviation_width',
        'CXLba:A2-MACROGLOB.':'alpha2_macroglobulin',
        'CXLba:TACROLIMUS':'tacrolimus',
        'CXLba:MAGNESIUM':'magnesium',
        'CXLba:GAMMA GLOBULINE':'gamma_globulin',
        'CXLha:RETIC #':'reticulocyte_count',
        'CXLha:VS':'erythrocyte_sedimentation_rate',
        'CXLba:EXCES DE BASE':'base_excess',
        'CXLha:VPM':'mean_platelet_volume',
        'CXLma:VANCO POST DOSE':'vancomycin_post_dose',
        'CXLba:DENSITE (BAT.)':'urinary_density',
        'CXLba:TSH':'thyroid_stimulating_hormone',
        'CXLba:CALCIUM TOTAL':'total_calcium',
        'CXLba:TRIGLYCERIDES':'triglycerides',
        'CXLba:PO2 ARTERIEL':'arterial_po2',
        'CXLba:FIO2':'fio2',
        'CXLba:SODIUM':'sodium',
        'CXLha:ACIDE FOLIQUE':'folic_acid',
        'CXLha:TGMH':'mean_corpuscular_hemoglobin',
        'CXLha:D-DIMERE':'d_dimer',
        'CXLba:GGT':'gamma_glutamyl_transferase',
        'Lxa:AC.ASCORB.(BAT)':'urinary_ac_ascorb',
        'CXLma:VANCOMYCINE':'vancomycin',
        'CXLba:T4 LIBRE':'free_t4',
        'CXLba:LD':'lactate_dehydrogenase',
        'CXLba:TRANSTHYRETINE':'transthyretin',
        'CXLba:PROCALCITONINE':'procalcitonin',
        'CXLba:PROTEINES (BAT)':'urinary_protein',
        'CXLha:T.PROTHR. INR':'inr',
        'CXLba:GLUCOSE (BAT.)':'urinary_glucose',
        'CXLba:PH (BAT.)':'urinary_ph',
        'CXLba:AST':'aspartate_aminotransferase',
        'CXLba:HDL-CHOLESTEROL':'hdl_cholesterol',
        'CXLha:PLT':'platelet_count',
        'CXLha:MONO #':'monocyte_count',
        'CXLba:PROTEINE C REAC':'c_reactive_protein',
        'CXLha:PLT EN AMAS':'platelet_clumping',
        'CXLha:POLYCHROME':'urinary_polychromia',
        'CXLba:PH':'ph',
        'CXLba:Sat O2 VEINEUX':'venous_o2_sat',
        'CXLha:FIBRINOGENE':'fibrinogen',
        'CXLba:BILIRUBINE TOT.':'total_bilirubin',
        'CXLba:BACTERIES':'urinary_bacteria',
        'CXLba:PHOSPHORE':'phosphate',
        'CXLba:CREATININE':'creatinine',
        'CXLba:PCO2 VEINEUX':'venous_pco2',
        'CXLba:HCO3 VEINEUX':'venous_bicarbonate',
        'CXLia:Anti-HBc':'hbv_cag',
        'CXLba:PROTEINES TOT':'total_protein',
        'CXLba:ALBUMINE':'albumin',
        'CXLba:TROPONINE-T HS':'hs_troponin_t',
        'CXLia:ANTI-HCV':'hcv_ag',
        'CXLba:HEMOLYSE':'hemolysis',
        'CXLba:BILIRUBINE IND.':'indirect_bilirubin',
        'CXLba:POTASSIUM':'potassium',
        'CXLba:MUCUS':'urinary_mucus_presence',
        'CXLba:Sat O2 ARTERIEL':'arterial_o2_sat',
        'CXLba:GLOBULINES':'globulins',
        'CXLha:VGM':'mean_corpuscular_volume',
        'CXLba:CK':'creatine_kinase',
        'CXLha:T.CEPHALINE SEC':'partial_thromboplastin_time',
        'CXLia:CMV IgG':'cmv_igg',
        'CXLba:ALT':'alanine_aminotransferase',
        'CXLha:CGMH':'mean_corpuscular_hemoglobin_concentration',
        'CXLba:ERYTHROCYTES':'erythrocytes',
        'CXLha:VITAMINE B12':'vitamin_b12',
        'CXLba:AMYLASE':'amylase',
        'CXLba:HCO3 ARTERIEL':'arterial_bicarbonate',
        'CXLha:T.CEPH.(1:1)':'partial_thromboplastin_time_11',
        'CXLba:EXCES BASE(V) #':'venous_base_excess',
        'CXLba:PO2 VEINEUX':'venous_po2',
        'CXLba:A1 ANTITRYPSINE':'alpha1_antitrypsin',
        'CXLba:ACIDE LACTIQUE VEINEUX':'venous_lactic_acid',
        'CXLba:25(OH)-VIT. D':'25_oh_vitamin_d',
        'CXLba:CHLORURE':'chloride',
        'CXLba:CORTISOL (S)':'cortisol',
        'CXLba:UREE':'urea',
        'CXLba:OSMOLALITE URINAIRE':'urinary_osmolality',
        'CXLha:GR':'red_blood_cell_count',
        'CXLba:NITRITES (BAT.)':'urinary_nitrite',
        'CXLba:SANG (BAT.)':'urinary_blood',
        'CXLba:PCO2 ARTERIEL':'arterial_pco2',
        'CXLba:ICTERE':'icterus_presence',
        'CXLia:HBs Ag':'hbv_sag',
        'CXLha:NEUTRO #':'neutrophil_count',
        'CXLha:HYPO':'hypo',
        'CXLba:LIPASE':'lipase',
        'CXLba:PTH INTACTE':'intact_pth',
        'CXLha:HB':'hemoglobin',
        'CXLba:CETONES (BAT.)':'urinary_cetones',
        'CXLba:LEUCOCYTES':'leucocytes_count',
        'CXLia:VIH':'vih',
        'CXLha:GB':'white_blood_cell_count',
        'CXLba:BILIRUBINE(BAT)':'urinary_bilirubin',
        'CXLba:PHOSPHATASE ALC':'alkaline_phosphatase',
        'CXLha:NRBC#':'nucleated_red_blood_cells',
        'CXLba:CALCIUM TOTAL CORRIGE':'corrected_total_calcium',
        'CXLba:HAPTOGLOBINE':'haptoglobin',
        'CXLba:GAP ANION.':'anion_gap',
        'CXLba:GLUCOSE':'glucose',
        'CXLba:LEUCOCYTES(BAT)':'urinary_leucocytes',
        'CXLba:FER SERIQUE':'serum_iron',
        'CXLba:BILIRUBINE DIR.':'direct_bilirubin',
        'CXLia:Anti-HBs UNITES':'anti_hbs',
        'CXLba:TRANSFERRINE':'transferrin',
        'CXLba:FERRITINE':'ferritin',
        'CXLba:CHOLESTEROL':'cholesterol',
        'CXLba:OSMOLALITE SER.':'osmolality',
        'CXLba:ACIDE URIQUE':'uric_acid',
        'CXLba:NT-pro BNP':'nt_pro_bnp',
        'Lha:META #':'metamyelocyte_count', 
        'Lha:MYELO #':'myelocyte_count'}

def nonnum_to_num(non_num):
    #This fct transform frequent non numerical to numerical
    non_num=non_num.replace(regex = {
            #Replacement of categorical values
            r"0\s?[Aaà-]\s?2" : "1",
            r"3\s?[Aaà-]\s?5" : "4",
            r"6\s?[Aaà-]\s?10" : "8",
            r"11\s?[Aaà-]\s?100" : "55",
            #Replacement of case insentitives litteral values
            r"(?i)n[ée]g(?:atif)?.?":"0",
            r"(?i)normal.?":"0",
            r"(?i)non r[ée]actif.?":"0",
            r"(?i)absence.?":"0",
            r"(?i)[ée]quivoque.?":"0.5",
            r"(?i)pr[ée]sence.?":"1",
            r"(?i)positif.?":"1",
            r"(?i)r[ée]actif.?":"1",
            r"r(?i)ares?":"1",
            r"(?i)abondant?(?:ce)?":"2",
            r"^\+{4}$":"4",
            r"^\+{3}$":"3",
            r"^\+{2}$":"2",
            r"^\+{1}$":"1",
            r"^-{1}$":"0"})
            #Anticoagulant data
            
    non_num=non_num.replace({
            r'Aucun': '0',
            r'Inconnu(e)':'0',
            r'Héparine':'1',
            r'Lovenox(Enoxap)':'1',
            r'Argatroban':'1',
            r'Fragmin(Daltep)':'1',
            r'Coum.et Hépar.':'1',
            r'Coumadin':'1',
            r'Autre':'1',
            r'Orgaran(Danapa)':'1',
            r'Pradax(Dabigat)':'1',
            r'Xarelto(Rivaro)':'1',
            r'Arixtra(Fondap)':'1',
            r'HÉPARINE':'1',
            r'Apixaban':'1',
            r'Innohep(Tinzap)':'1'})
    
    #Replace virgule by dot for numbers
    non_num = non_num.str.replace(',', '.')
    
    #Remove values known to be adjacent to numerical values
    non_num = non_num.str.replace(' ', '') #Removing Spaces
    non_num = non_num.str.replace(r'[><*+=?]', '')
    non_num = non_num.str.replace(r'm?g\/d?L', '')
    non_num = non_num.str.replace(r'[muµ]mol\/d?L', '')
    non_num = non_num.str.replace('NONCONTROLEEGB','')
    non_num = non_num.str.replace('GB/uL', '')
    non_num = non_num.str.replace('/[µu]L', '')
    non_num = non_num.str.replace(r'Tr(a)?(ac)?(ace)?$', '',case=False)
    non_num = non_num.str.replace(',', '.')
    
    return non_num

def clean_percent(value):
    #Uniformisation of values with a percentages that are either aggregated of manually entered
    #mainly: FIO2, sat art sat veinous

    regexp = re.compile('^-?\d+\.*\d*$')
    if not regexp.search(str(value)):
        #not a potential number
        return value
    value = float(value)
    if value < 0:
        return '0'
    if (value >=0) & (value <=1):
        return str(value)
    if (value >1) & (value <=100):
        return str(value/100)
    if value >100:
        return "1"

def clean_servacro(serv):
    
    serv = serv.cat.remove_categories(remove_servacro)
    
    #removing aroung 1000 'lab analyses'. Either title, percentage etc
    reg = re.compile('%|/|^Lbt:')
    to_remove = list(serv.cat.categories[serv.cat.categories.str.contains(reg)])            
    #Case of pleural liquid / ascitis. AS it is now, both are coded the same. So normal value and impuration is impossible.
    #Moreover, all are between 90%-95% of missingness. Based on that, the analysis is removed
    reg = re.compile('LIQUIDE$|\(M\)$|MICT\.?$|\(L\)$',flags=re.IGNORECASE)
    to_remove += list(serv.cat.categories[serv.cat.categories.str.contains(reg)])
    
    serv = serv.cat.remove_categories(to_remove)
    
    #rename categories and merge
    serv = serv.cat.add_categories(new_servacro)
    cat_codes = dict(zip(serv.cat.categories,range(len(serv.cat.categories))))
    codes = {cat_codes.get(k):cat_codes.get(i) for (k,i) in rename_servacro.items()}
    serv = pd.Series(pd.Categorical.from_codes(serv.cat.codes.replace(codes), categories = serv.cat.categories))
    
    return serv

def clean_longdesc(desc):
    #Clean longdesc columns
    desc = desc.cat.add_categories(new_longdesc)
    cat_codes = dict(zip(desc.cat.categories,range(len(desc.cat.categories))))
    codes = {cat_codes.get(k):cat_codes.get(i) for (k,i) in replace_longdesc.items()}
    desc = pd.Categorical.from_codes(desc.cat.codes.replace(codes), categories = desc.cat.categories)
    
    return desc

def clean_values(values):
    #Clean lbres_ck
    values = nonnum_to_num(values)

    return values

def special_cases(df):
    #Some laboratory value needs multiples columsnt o be cleanes
    #method collection uniformisation with tag
    df.loc[((df.longdesc == 'pH')&
           (df.specimencollectionmethodcd == 'CYSang Artériel')), 'longdesc'] = 'pH Art'
    df.loc[((df.longdesc == 'pH')&
           (df.specimencollectionmethodcd == 'CYSang Artériel')), 'servacro'] = 'CXLba:PH ARTERIEL'
    df.loc[((df.longdesc == 'pH Art')&
           (df.servacro == 'CXLba:PH')), 'servacro'] = 'CXLba:PH ARTERIEL'
    df.loc[((df.longdesc == 'pH')&
           (df.specimencollectionmethodcd == 'CYSang Veineux')), 'longdesc'] = 'pH Vein'
    df.loc[((df.longdesc == 'pH')&
           (df.specimencollectionmethodcd == 'CYSang Veineux')), 'servacro'] = 'CXLba:PH VEINEUX'
    df.loc[((df.longdesc == 'pH Vein')&
           (df.servacro == 'CXLba:PH')), 'servacro'] = 'CXLba:PH VEINEUX'
    
    df = df.drop(columns = 'specimencollectionmethodcd')
    
    #Uniformization of percentage function (20 vs 0.2)
    percent_servacro = ['CXLba:Sat O2 ARTERIEL','CXLba:Sat O2 VEINEUX','CXLba:FIO2']
    df.loc[df.servacro.isin(percent_servacro), 'lbres_ck'] = df.loc[df.servacro.isin(percent_servacro), 'lbres_ck'].apply(clean_percent)

    ##In case of unknown anticoagulant values
    df.loc[(df['servacro'] == 'Lha:ANTICOAGULANT ?') & ~df['lbres_ck'].str.isnumeric(),'lbres_ck'] = '0'    
                                               
    return df

def rare_labs_patient_cleaning(df_rare_cleaning, percentage):
    #Remove rare laboratory analysis based on a percentag minima
    #Then remove patient with rare
    
    percent_to_drop = percentage
    pivot_nb_analyse_per_patient = pd.pivot_table(df_rare_cleaning[['servacro','dw_pid']],
                                                  index=['dw_pid'], 
                                                  columns='servacro', 
                                                  aggfunc=len)
    percent_missing = pivot_nb_analyse_per_patient.isna().sum() / pivot_nb_analyse_per_patient.shape[0]
    idx_rare_test_to_drop = (percent_missing[percent_missing > percent_to_drop]).index
    len_df_pre = len(df_rare_cleaning)
    n_labs_pre = df_rare_cleaning.servacro.nunique()
    n_patients_pre = df_rare_cleaning.dw_pid.nunique()
    df_rare_cleaning = df_rare_cleaning[~df_rare_cleaning.servacro.isin(idx_rare_test_to_drop)]
    
    #Droping patients that have less than a fixe percentage of the test
    percent_to_drop = percentage
    pivot_nb_analyse_per_patient = pd.pivot_table(df_rare_cleaning[['servacro','dw_pid']],
                                                  index=['servacro'], 
                                                  columns=['dw_pid'], 
                                                  aggfunc=len)
    percent_missing = pivot_nb_analyse_per_patient.isna().sum() / pivot_nb_analyse_per_patient.shape[0]
    idx_patient_missing_a_lot_to_drop = (percent_missing[percent_missing > percent_to_drop]).index
    df_rare_cleaning = df_rare_cleaning[~df_rare_cleaning.dw_pid.isin(idx_patient_missing_a_lot_to_drop)]
    len_df_post = len(df_rare_cleaning)
    n_labs_post = df_rare_cleaning.servacro.nunique()
    n_patients_post = df_rare_cleaning.dw_pid.nunique()
    
    print('With removing rare labs and rare patients'.format(n_labs_pre-n_labs_post))
    print('Loss of {0} unique labs and {1} unique patients representing {2:2.2f}% of lines of labs\n'.format(n_labs_pre-n_labs_post,n_patients_pre-n_patients_post,(len_df_pre-len_df_post)*100/len_df_pre))
    
    return(df_rare_cleaning)
    
def outliers_removal(outlier_df):
    #Values are etremely right skewed. Defintion of an 'outlier' is a complex topic
    #We want the output of this script to be as inclusive as possible
    #However, some specific labs have biological limits and manual or lab error
    #causes bug later in the algorithm
    #We remove these specifically chosen extremes.
    #Usual definition of outlier is 75percentile+1.5*IQR. 
    #We chose larger range as 75percentile+1.5*50*IQR
    #Also, some value cannot biologicaly be zero 
    
    outliers_labs_max = ['CXLba:PO2 ARTERIEL',
                     'CXLba:PO2 VEINEUX',
                     'CXLba:PH VEINEUX',
                     'CXLba:PH ARTERIEL',
                     'CXLba:PCO2 VEINEUX',
                     'CXLba:PCO2 ARTERIEL',
                     'CXLba:HCO3 ARTERIEL',
                     'CXLba:HCO3 VEINEUX',
                     'CXLba:TEMPERATURE',]
    
    max_value = outlier_df[outlier_df.servacro.isin(outliers_labs_max)].groupby('servacro')['lbres_ck'].apply(lambda x:(x.quantile(0.75)+((x.quantile(0.75)-x.quantile(0.25)))*1.5*50))
    
    for lab in outliers_labs_max:
        max_val = max_value[lab]
        index_max = outlier_df[(outlier_df.servacro == lab)&(outlier_df.lbres_ck > max_val)].index
        outlier_df = outlier_df.drop(index = index_max)
    
    outliers_labs_min = ['CXLba:CALCIUM TOTAL CORRIGE',
                         'CXLba:ACIDE LACTIQUE ARTERIEL',
                         'CXLba:ACIDE LACTIQUE VEINEUX',
                         'CXLba:ALBUMINE', 'CXLba:ALT',
                         'CXLba:BILIRUBINE IND.','CXLba:BILIRUBINE TOT.',
                         'CXLba:CREATININE', 'CXLba:FER SERIQUE',
                         'CXLba:FIO2', 'CXLba:HCO3 ARTERIEL', 'CXLba:LIPASE',
                         'CXLba:OSMOLALITE SER.', 'CXLba:OSMOLALITE URINAIRE',
                         'CXLba:PCO2 ARTERIEL', 'CXLba:PH ARTERIEL',
                         'CXLba:PHOSPHATASE ALC', 'CXLba:PO2 ARTERIEL', 'CXLba:PO2 VEINEUX',
                         'CXLba:Sat O2 ARTERIEL', 'CXLba:Sat O2 VEINEUX',
                         'CXLba:TEMPERATURE', 'CXLha:CGMH', 'CXLha:DVE', 'CXLha:GB',
                         'CXLha:GR', 'CXLha:LYMPHO #', 'CXLha:NEUTRO #', 'CXLha:TGMH',
                         'CXLha:VGM', 'CXLha:VPM']
    index_min = outlier_df[(outlier_df.servacro.isin(outliers_labs_min))&(outlier_df.lbres_ck <= 0)].index
    outlier_df = outlier_df.drop(index = index_min)
    
    return outlier_df