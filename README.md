# COMPASS pilot

## Authors informations
Corresponding author:
Dr Michael Chassé, MD, PhD
michael.chasse.med@ssss.gouv.qc.ca 

Main author:
Nicolas Sauthier, MD, MSc(c), BEng
nicolas.sauthier@umontreal.ca


## Project Description
This is a machine learning project. The goal is to classify patients in potential organ donors and non potential organ donors, based on temporal laboratory data and static data. This project start from the raw data all the way to the prectiction and sensitivity analysis and external validation.

Project made under supervision of Michael Chassé, MD, PhD.

Project financed by a grant from the CDTRP, done as part of a master's degree in biomedical sciences at Université de Montréal

Project start date: 2019

Project last update: 2022

## Files listing

#### Project files

|File name|File content|
|---|---|
|main.py|Main file. Call function and classes from all the other files.|
|DataCleaning.py|Function and classes used to clean raw temporal and static data. Use modules from DataCleaningFunctions|
|DataCleaningFunctions.py|Functions used in DataCleaning.py for raw data cleaning|
|TemporalImputationAndFormat.py|Functions used to reduce data to last three days, keeping one per 8h block and imputate missing data to have a complete dataset.|
|get_ref_range.py|Dictionnary, functions and classes used to attribute an normal reference range for laboratory data|
|ScalingAnd3DFormatting.py|Function and classes used to scale and transform a list of tidy laboratory to a 3D format (patient, lab, time)|
|TrainTestValidCreation.py|Classes to create a random separation in train, test, valid while keeping continuity between iteration of the dataset|
|MachineLearningModels.py|Classes to create, initiate, train and predict machine learning model i.e. autoencoder and classifier|


#### Files specific to article redaction
|File name|File content|
|---|---|
|Article_Table1.py|Function to generate the table 1 based on the static file|

## Example files
Two fake files are available in order to better understand the format of the data used. Theses fake files were randomly generated without any contact with the actual database.

## How to Use the Project

The main.py file walk through all the steps from training to sensitivity analysis. It's separated in 7 parts.

1. Loading, cleaning and exporting the raw data, based on the raw files extracted from EHR.
	In total four files are expected: 
	* raw_lab_file ('5_lab_results_tq_donors_ICUCARD_LastEpisd_2020-07-30.csv.gz')
	* raw_static_file ('3_compass_episodes_tq_donors_ICUCARD_LastEpisd_2020-07-30')
	* raw_ICD10_file ('6_medecho_ICD_diag_codes_tq_donors_2021-06-07.csv')
	* raw_radiology_file ('7_transplantqc_donors2020_allpatients_icu_urghosp_2012_2019/4_all_radiology_exams_tq_donors_2020-06-15.csv.gz')

2. Temporal data is reduced to last three days, one result per 8 hours.
3. Data are transformed from the tidy list into a 3D format including scaling and imputation.
4. The autoencodeur is initialized, trained and used to transform 3D temporal data to a vector of dim=64
5. Classifiers are initialized, trained and used to predict on the test set with metrics calculated
6. Sensitivity analysis i.e. removing labs and checking the accuracy *this takes a long time to compute*
7. Simulation of prospective, i.e. data are removed from the prediction time wise to blind model to the most recent lab results.



