This code was made by Natasja Bonde Andersen in the period 10th of January - 10th of July 2024. 
The codes cannot be used without permission from the author (08-07-2024). 
Codes for 5 datasets can be found in this repository (France, China patients, China controls, RBD patients and RBD controls). 
The china controls and RBD controls was handled separately from the china patients and RBD patient datasets.

One type of code exists for each of the dataset. This was due to different names in the patient IDs. Despite from the filenaming the codes are identical. 
The codes should be used in the following order for each dataset: 

1) The 'My_functions_scripts' covers several functions used per dataset in all other codes. 
These are functions developed for many purposes. This code is loaded in the top of all other codes.
1.a) My_functions_script_China
1.b) My_functions_script_China_controls
1.c) My_functions_script_France
1.d) My_functions_script_RBD
1.e) My_functions_script_RBD_controls

2) First the data were restructured using the code:
All datasets should have the same name of the electrodes used. All electrodes was given a proper name and was referenced it needed. New EDF files containing only the EEG signals were saved. 
2.a) Restructure_data_China
2.b) Restructure_data_China_controls
2.c) Restructure_data_France
2.d) Restructure_data_RBD
2.e) Restructure_data_RBD_controls

This part is the code for the sleep stage synchronization analysis: 
3) The hypnodensity features for each patient were calculated in each dataset. Epoch sizes of 1, 3, 5, 15 and 30 seconds were used.This was done for all electrodes.
A folder were created for each patient. An electrode name file called 'Name.txt' were generated containing the electrodes present. This file were kept in the patient folders. 
The code were build to detect possible missing hypnodensities if the code crashed after 12 hours due to the U-sleep token running out.
The code can pick up from were it stopped.
3.a) Usleep_mycode_Chinapatients
3.b) Usleep_mycode_Chinacontrols
3.c) Usleep_mycode_France
3.d) Usleep_mycode_RBD
3.e) Usleep_mycode_RBD_controls

This part is code for the outlier detector: 
4) The restructured data were loaded into the outlier detector code:
   This code detected outliers and saved them in a CSV file for each dataset containing the patientID and the outlier electrode/channel. 
4.a) Outlierdetection_China
4.b) Outlierdetection_China_controls
4.c) Outlierdetection_France
4.d) Outlierdetection_RBD
4.e) Outlierdetection_RBD_controls

The outliers were manually removed from the hypnodensity features since there were not a lot of them, and the outlier detector were designed after the implementation of U-sleep. 
When removing the outlier hypnodensities simply delete the hypnodensity with the correct name and remember to remove the electrode name from the 'Name.txt' file. 
If not done it will cause problems later.
The upcoming coherence codes removes outliers automatically. 


The coherence analysis code is shown in this part. The coherence codes covers multiple codes. 5 codes calculates the measures for the full night, 5 codes calculates the measures for the 1st sleep cycle, 
5 codes calculates the measures for the 2nd sleep cycle, 5 codes calculates the measures for the 3rd sleep cycle and 5 codes calculates the measures for the 4th sleep cycle. 
Due to collection of features and data in different lengths the features were calculated in different codes. They will be combined later in the project. 
These are presented in the following: 

5) Full night coherence: 
5.a) Coherence_script_fullnight_China_patients
5.b) Coherence_script_fullnight_China_controls
5.c) Coherence_script_fullnight_France
5.d) Coherence_script_fullnight_RBD
5.e) Coherence_script_fullnight_RBD_controls


6) 1st sleep cycle coherence: 
6.a) Coherence_script_1partnight_China_patients
6.b) Coherence_script_1partnight_China_controls
6.c) Coherence_script_1partnight_France
6.d) Coherence_script_1partnight_RBD
6.e) Coherence_script_1partnight_RBD_controls


7) 2nd sleep cycle coherence: 
7.a) Coherence_script_2partnight_China_patients
7.b) Coherence_script_2partnight_China_controls
7.c) Coherence_script_2partnight_France
7.d) Coherence_script_2partnight_RBD
7.e) Coherence_script_2partnight_RBD_controls

8) 3rd sleep cycle coherence: 
8.a) Coherence_script_3partnight_China_patients
8.b) Coherence_script_3partnight_China_controls
8.c) Coherence_script_3partnight_France
8.d) Coherence_script_3partnight_RBD
8.e) Coherence_script_3partnight_RBD_controls

9) 4th sleep cycle coherence: 
9.a) Coherence_script_4partnight_China_patients
9.b) Coherence_script_4partnight_China_controls
9.c) Coherence_script_4partnight_France
9.d) Coherence_script_4partnight_RBD
9.e) Coherence_script_4partnight_RBD_controls


Data combining: 
From the coherence feature analysis several CSV files were extracted containing features for different electrodes and measures. 
They are all combined to CSV files ready for machine learning analysis in the following codes. 
In the naming of these codes 'correlation' is used as a term covering both dynamic time warping and correlation, since dynamic time warping were implemented after the naming of these codes. 
However, the dynamic time warping measure is a part of the codes named something with 'correlation'. 

10) Data combining 
10.a) Coherence_Features_China_patientsandcontrols
10.b) Coherence_Features_France
10.c) Coherence_Features_RBD_controls
10.d) Correlation_Features_China
10.e) Correlation_Features_France
10.f) Correlation_Features_RBDandcontrols


11) Machine learning part 
Several models were made for NT1 (narolepsy type 1). Models were made for China, France, and a gathered dataset.
Furthermore, 3 models were made per dataset 1) overall model with all the data, 2) a coherence model with only the coherence features,
3) a sleep stage synchronization model called 'correlation' in this case. Dynamic time warping (dtw) is a part of the code.
The coherence and (correlation+dtw) features are all inside the machine learning documents.
In all of these codes the plots for the report can also be found in the bottom. 
   

11.a) Machinelearning_model_China_NT1
11.b) Machinelearning_model_France_NT1
11.c) Machinelearning_model_Fullmodel_NT1
11.d) Machinelearning_model_RBD

Baseline models were made for the same datasets and on the same terms as stated above regarding coherence, correlation and dtw.

11.e) Baseline_model_China_NT1
11.f) Baseline_model_France_NT1
11.g) Baseline_model_Fullmodel_NT1
11.h) Baseline_model_RBD

In the end statistics can be calculated for the the full dataset of NT1 (China and France combined) and RBD: 
The codes performed a t-test between the 10 most important features for each model. 

12.a) Statistics_NT1_fullmodel
12.b) Statistics_RBD


For questions feel free to contact me. 

Kind regards, 
Natasja Bonde Andersen




