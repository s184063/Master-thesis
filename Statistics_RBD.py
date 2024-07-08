# Load standard packages
import numpy as np
from numpy import shape, vstack
import pandas as pd
import copy
import matplotlib
from matplotlib import pyplot
# Load signal processing packages
import scipy # Signal processing 
from scipy.fft import fft, ifft, fftshift, ifftshift 
from scipy.signal import resample_poly, firwin, butter, bilinear, periodogram, welch, filtfilt, sosfiltfilt
# For processing EDF files 
import pyedflib 
from pyedflib import highlevel # Extra packages for EDF
import pyedflib as plib
#import mne
import sys
#import statsmodels
#from statsmodels import stats


sys.path.insert(0, 'C:/Users/natas/Documents/Master thesis code')
from My_functions_script_France import extract_numbers_from_filename, extract_letters_and_numbers, list_files_in_folder, split_string_by_length, Usleep_2channels, correlation_multiple_electrodes

#import mne
import itertools 
from itertools import combinations
# form correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

### Loading data ###########################
#Loading data frame 

# Coherence
df_fullnight=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/RBD controls/Coherence_features_combined_fullnight_RBD_and_controls.csv')
df_1part=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/RBD controls/Coherence_features_combined_1partnight_RBDandcontrols.csv')
df_2part=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/RBD controls/Coherence_features_combined_2partnight_RBDandcontrols.csv')
df_3part=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/RBD controls/Coherence_features_combined_3partnight_RBDandcontrols.csv')
df_4part=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/RBD controls/Coherence_features_combined_4partnight_RBDandcontrols.csv')



# Combining all coherence features 

# Dropping the first patienID folder for most of the files in order to make a concatenation of all files 
df_1part=df_1part.drop(['PatientID','Dianosis','Sex_F1_M2','Age'],axis=1)
df_2part=df_2part.drop(['PatientID','Dianosis','Sex_F1_M2','Age'],axis=1)
df_3part=df_3part.drop(['PatientID','Dianosis','Sex_F1_M2','Age'],axis=1)
df_4part=df_4part.drop(['PatientID','Dianosis','Sex_F1_M2','Age'],axis=1)


df_1part=df_1part.add_suffix('_1part')
df_2part=df_2part.add_suffix('_2part')
df_3part=df_3part.add_suffix('_3part')
df_4part=df_4part.add_suffix('_4part')


# cropping dataframes
df_fullnight_edited = df_fullnight.iloc[:,1:557]
df_part1_edited=df_1part.iloc[:,1:557]
df_part2_edited=df_2part.iloc[:,1:557]
df_part3_edited=df_3part.iloc[:,1:557]
df_part4_edited=df_4part.iloc[:,1:557]


# Combining the dataframes. Only one column contains patientID
df_coherence=pd.concat([df_fullnight,df_part1_edited,df_part2_edited,df_part3_edited,df_part4_edited],axis=1)
df_coherence.to_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/RBD controls/Coherence_All_combined.csv')

df_coherence_model=df_coherence # defining the dataset for a separate coherence model 
print(df_coherence)




# Correlation 
df_correlation=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Correlation Features/RBD/All_correlation_features_combined_RBDandcontrols.csv')
print(df_correlation)
df_correlation_model=df_correlation # defining the dataset for a separate correlatiom model 


### All combined coherence and correlation ####

df_coherence=df_coherence.drop(['PatientID','Dianosis','Sex_F1_M2','Age'],axis=1)
length=df_coherence.shape[1]
df_coherence_cropped=df_coherence.iloc[:,1:length]
print(df_coherence_cropped)

all_features=pd.concat([df_correlation,df_coherence_cropped],axis=1)
all_features.to_csv('C:/Users/natas/Documents/Master thesis code/All features/RBD dataset/All_features_RBDandcontrols_coherence_and_correlation.csv')

print(all_features)




############### Enable the model you want to run ################################
#df_combined=df_correlation_model
#df_combined=df_coherence_model
df_combined=all_features
#################################################################################


df_iRBD=df_combined[df_combined['Dianosis'] == 'I']
#df_PD=df_combined[df_combined['Dianosis'] == 'P']
#df_PD_D=df_combined[df_combined['Dianosis'] == 'D']

df_patients=pd.concat([df_iRBD])#,df_PD,df_PD_D])

print('df patients')
print(df_patients)

df_controls=df_combined[df_combined['Dianosis'] == 'Control']

print('df controls')
print(df_controls['N3_dwt_F3M2O1M2_epocssize_15'])




Important_features_10=['REMcoh_gammaF3M2O2M1','REMcoh_gammaC4M1O1M2','N1_dwt_F3M2F4M1_epocssize_1_','REMcoh_gammaC3M2O1M2','N3_dwt_diff_F3M2F4M1_epocssize_30','REMcoh_gammaF3M2F4M1','REM_dwt_F3M2F4M1_epocssize_1_','N1_dwt_diff_F3M2F4M1_epocssize_1_','N1_dwt_F4M1C4M1_epocssize_1_','REMcoh_gammaO1M2O2M1']#,'N1_dwt_diff_F3M2F4M1_epocssize_1_','N3_dwt_F3M2O1M2_epocssize_15','REMcoh_gammaF4M1C3M2','N3coh_gammaF4M1O2M1','N1_dwt_F3M2F4M1_epocssize_1_','REMcoh_gammaC4M1O1M2','Wake_dwt_F3M2F4M1_epocssize_30']

df_controls_important=df_controls[Important_features_10]
df_patients_important=df_patients[Important_features_10]

print(df_controls_important)
print(df_patients_important)

############################################


#### T-test between features (RBD vs. controls) ####


temp_pvalue=[]


#df_controls=df_controls.iloc[:,5:125]
#df_NT1= df_NT1.iloc[:,5:125]

#print('Cropped version')
#print(df_controls)
#print(df_NT1)

temp_actual_pvalue=[]

for column_name in df_controls_important.columns:

    column_df_controls = df_controls_important[column_name]
    column_df_controls = column_df_controls.dropna()
    #print(type(column_df_controls))
    #print(column_df_controls)


    column_df_NT1 = df_patients_important[column_name]
    column_df_NT1 = column_df_NT1.dropna()
    #print(type(column_df_NT1))
    #print(column_df_NT1)
    
    ttest=scipy.stats.ttest_ind(column_df_controls, column_df_NT1,alternative='two-sided')

   
    
    if ttest.pvalue < 0.05:
        temp_pvalue.append(column_name)
        temp_actual_pvalue.append(ttest.pvalue)


    del column_df_controls, column_df_NT1
    
    print(column_name)
    print(ttest)


print('Temp variable - pvalues significant')
print(temp_pvalue)
print('P-values significant')
print(temp_actual_pvalue)

temp_NT1_mean=[]
temp_controls_mean=[]

temp_NT1_std=[]
temp_controls_std=[]

for j in temp_pvalue:
    
    Controls=df_controls_important[j]
    RBD=df_patients_important[j]

    Mean_Controls=np.mean(Controls)
    Std_controls=np.std(Controls)

    Mean_NT1=np.mean(RBD)
    std_NT1=np.std(RBD)

    temp_NT1_mean.append(Mean_NT1)
    temp_controls_mean.append(Mean_Controls)

    temp_NT1_std.append(std_NT1)
    temp_controls_std.append(Std_controls)


print(temp_pvalue)
print('Mean RBD')
print(temp_NT1_mean)
print('Std RBD')
print(temp_NT1_std)

print('Mean Controls')
print(temp_controls_mean)
print('Std Controls')
print(temp_controls_std)
