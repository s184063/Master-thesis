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

### Loading data China ###########################
df_fullnight_China=pd.read_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/All_Coherence_fullnight_features_China_patientsandcontrols.csv') # full night
df_1part_China=pd.read_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/All_Coherence_1part_features_China_patientsandcontrols.csv') # part 1 
df_2part_China=pd.read_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/All_Coherence_2part_features_China_patientsandcontrols.csv') # part 2 
df_3part_China=pd.read_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/All_Coherence_3part_features_China_patientsandcontrols.csv') # part 3
df_4part_China=pd.read_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/All_Coherence_4part_features_China_patientsandcontrols.csv') # part 4

### Loading data France ###########################
df_fullnight_France=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/France/NT1_and_controls_France.csv') # full night
df_1part_France=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/France/NT1_and_controls_1partnight_France.csv') # part 1 
df_2part_France=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/France/NT1_and_controls_2partnight_France.csv') # part 2 
df_3part_France=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/France/NT1_and_controls_3partnight_France.csv') # part 3
df_4part_France=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/France/NT1_and_controls_4partnight_France.csv') # part 4





# Combining all coherence features 

# Dropping the first patienID folder for most of the files in order to make a concatenation of all files 
df_1part_China=df_1part_China.drop(['PatientID','Dx','Sex','Cohort'],axis=1)
df_2part_China=df_2part_China.drop(['PatientID','Dx','Sex','Cohort'],axis=1)
df_3part_China=df_3part_China.drop(['PatientID','Dx','Sex','Cohort'],axis=1)
df_4part_China=df_4part_China.drop(['PatientID','Dx','Sex','Cohort'],axis=1)

df_1part_China=df_1part_China.add_suffix('_1part')
df_2part_China=df_2part_China.add_suffix('_2part')
df_3part_China=df_3part_China.add_suffix('_3part')
df_4part_China=df_4part_China.add_suffix('_4part')

print(df_1part_China)


# cropping dataframes
#print(df_1part)
df_fullnight_edited_China = df_fullnight_China.iloc[:,1:556]
df_part1_edited_China=df_1part_China.iloc[:,1:556]
df_part2_edited_China=df_2part_China.iloc[:,1:556]
df_part3_edited_China=df_3part_China.iloc[:,1:556]
df_part4_edited_China=df_4part_China.iloc[:,1:556]





##### France #############
# Combining all coherence features 

# Dropping the first patienID folder for most of the files in order to make a concatenation of all files 
df_1part_France=df_1part_France.drop(['PatientID','Dx','Sex','Cohort'],axis=1)
df_2part_France=df_2part_France.drop(['PatientID','Dx','Sex','Cohort'],axis=1)
df_3part_France=df_3part_France.drop(['PatientID','Dx','Sex','Cohort'],axis=1)
df_4part_France=df_4part_France.drop(['PatientID','Dx','Sex','Cohort'],axis=1)

print(df_1part_France)
df_1part_France=df_1part_France.add_suffix('_1part')
df_2part_France=df_2part_France.add_suffix('_2part')
df_3part_France=df_3part_France.add_suffix('_3part')
df_4part_France=df_4part_France.add_suffix('_4part')



# cropping dataframes
#print(df_1part)
df_fullnight_edited_France = df_fullnight_France.iloc[:,1:122]
df_part1_edited_France=df_1part_France.iloc[:,1:122]
df_part2_edited_France=df_2part_France.iloc[:,1:122]
df_part3_edited_France=df_3part_France.iloc[:,1:122]
df_part4_edited_France=df_4part_France.iloc[:,1:122]


print(df_part1_edited_France)
print(df_part1_edited_China)

# Combining the dataframes. Only one column contains patientID
df_coherence_China=pd.concat([df_fullnight_China,df_part1_edited_China,df_part2_edited_China,df_part3_edited_China,df_part4_edited_China],axis=1)
df_coherence_France=pd.concat([df_fullnight_France,df_part1_edited_France,df_part2_edited_France,df_part3_edited_France,df_part4_edited_France],axis=1)
df_coherence=pd.concat([df_coherence_China,df_coherence_France])
df_coherence.to_csv('C:/Users/natas/Documents/Master thesis code/All features/NT1 China and France combined/Coherence_All_combined_ChinaandFrance.csv')

df_coherence_model=df_coherence # defining the dataset for a separate coherence model 
print(df_coherence)




# Correlation 
df_correlation_China=pd.read_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/All_Correlation_features_China_patientsandcontrols.csv')
print(df_correlation_China)

df_correlation_France=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Correlation Features/France/All_Correlation_features_France.csv')
print(df_correlation_France)


df_correlation_model=pd.concat([df_correlation_China, df_correlation_France]) # defining the dataset for a separate correlatiom model 
df_correlation=df_correlation_model

### All combined coherence and correlation ####

df_coherence_=df_coherence.drop(['PatientID','Dx','Sex','Cohort'],axis=1)
length=df_coherence_.shape[1]
df_coherence_cropped=df_coherence_.iloc[:,1:length]
print(df_coherence_cropped)


all_features=pd.concat([df_correlation_model,df_coherence_cropped],axis=1)
all_features.to_csv('C:/Users/natas/Documents/Master thesis code/All features/NT1 China and France combined/Final_All_features_China_and France_coherence_and_correlation.csv')

print(all_features)



############### Enable the model you want to run ################################
#df_combined=df_correlation_model
#df_combined=df_coherence_model
df_combined=all_features
#################################################################################

df_NT1=df_combined[df_combined['Dx'] == 'NT1']


df_patients=pd.concat([df_NT1])

print('df patients')
print(df_patients)

df_controls=df_combined[df_combined['Dx'] == 'Control']

print('df controls')
print(df_controls['P1_alpha_C4M1_3part'])


Important_features_10=['REM_dwt_diff_C3M2C4M1_epocssize_30','Wake_dwt_C3M2C4M1_epocssize_15', 'REM_dwt_diff_C3M2C4M1_epocssize_15','REM_dwt_diff_C3M2C4M1_epocssize_5','REM_dwt_C3M2C4M1_epocssize_3_','Wake_corrdiff_C3M2C4M1_epocssize_15','Wake_corrdiff_C3M2C4M1_epocssize_5','REM_dwt_C3M2C4M1_epocssize_1_','REM_C3M2C4M1_epocssize_5','P1_alpha_C4M1_3part']#,'REM_dwt_C3M2C4M1_epocssize_1_','Wake_dwt_C3M2C4M1_epocssize_30','REM_dwt_diff_C3M2C4M1_epocssize_15','Wake_corrdiff_C3M2C4M1_epocssize_5','N1_corrdiff_C3M2C4M1_epocssize_15','REM_dwt_diff_C3M2C4M1_epocssize_5','Wake_corrdiff_C3M2C4M1_epocssize_15','REM_dwt_C3M2O2M1_epocssize_15']

df_controls_important=df_controls[Important_features_10]
df_patients_important=df_patients[Important_features_10]

print(df_controls_important)
print(df_patients_important)

############################################


#### T-test between features (NT1 vs. controls) ####


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

   
    
    #if ttest.pvalue < 0.05:
    #    temp_pvalue.append(column_name)
    #    temp_actual_pvalue.append(ttest.pvalue)

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
    NT1=df_patients_important[j]

    Mean_Controls=np.mean(Controls)
    Std_controls=np.std(Controls)

    Mean_NT1=np.mean(NT1)
    std_NT1=np.std(NT1)

    temp_NT1_mean.append(Mean_NT1)
    temp_controls_mean.append(Mean_Controls)

    temp_NT1_std.append(std_NT1)
    temp_controls_std.append(Std_controls)


print(temp_pvalue)
print('Mean NT1')
print(temp_NT1_mean)
print('Std NT1')
print(temp_NT1_std)

print('Mean Controls')
print(temp_controls_mean)
print('Std Controls')
print(temp_controls_std)


'''
#### France ##############
Electrodes=['C3M2','C4M1','O2M1']

iterable=Electrodes
r=2 # Length of subsequence 
E_combinations=list(itertools.combinations(iterable,r))

print('Combinations of electrode names in file')
print(E_combinations)
print(type(E_combinations))
print(len(E_combinations))


##### Significant factors in recording     
boxplot_features=temp_pvalue
plt.suptitle('Coherence SIGNIFICANT features of full night - NT1')
sns.boxplot(data=df_NT1[boxplot_features])
plt.xticks(rotation=45, ha='right')
plt.title(' NT1 ')
plt.tight_layout()
plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_NT1_SIGNIFICANT_coherence_fullnight.png')
#plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_NT1_SIGNIFICANT_coherence_1partnight.png')
#plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_NT1_SIGNIFICANT_coherence_2partnight.png')
#plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_NT1_SIGNIFICANT_coherence_3partnight.png')
#plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_NT1_SIGNIFICANT_coherence_4partnight.png')
plt.clf()
del boxplot_features

boxplot_features=temp_pvalue
sns.boxplot(data=df_controls[boxplot_features])
plt.suptitle('Coherence SIGNIFICANT features of full night - controls')
plt.xticks(rotation=45, ha='right')
plt.title('Controls ')
plt.tight_layout()

print('Figure was saved')
    
plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_controls_SIGNIFICANT_coherence_fullnight.png')
#plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_controls_SIGNIFICANT_coherence_1partnight.png')   
#plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_controls_SIGNIFICANT_coherence_2partnight.png')
#plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_controls_SIGNIFICANT_coherence_3partnight.png')
#plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_controls_SIGNIFICANT_coherence_4partnight.png')
plt.clf()
del boxplot_features




### Coherence full recording ##############
for d in range(len(E_combinations)):

    print('Finding new electrode combination in same main folder')
    print('full E_combinations')
    print(len(E_combinations))
    print('D')
    print(d)
            

    # Defining electrode combination for naming the CSV files in the end 
    Electrode_combination = [E_combinations[d][0], E_combinations[d][1]]
    Electrode_combination = extract_letters_and_numbers(Electrode_combination)
    print('Electrode combination')
    print(Electrode_combination)

    
    boxplot_features=['Delta_coh_'+str(Electrode_combination), 'Theta_coh_'+str(Electrode_combination),'Alpha_coh_'+str(Electrode_combination),'Beta_coh_'+str(Electrode_combination),'Gamma_coh_'+str(Electrode_combination)]
    plt.suptitle('Coherence of full night')
    plt.subplot(1, 2, 1) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_NT1[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title(' NT1 '+str(Electrode_combination))
    plt.tight_layout()


    plt.subplot(1, 2, 2) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_controls[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title('Controls '+str(Electrode_combination))
    plt.tight_layout()

    print('Figure was saved')
    plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_coherence_fullnight_'+str(Electrode_combination)+'.png')
    
    plt.clf()
    del boxplot_features

    



### Coherence 30sec intervals ##############
for d in range(len(E_combinations)):

    print('Finding new electrode combination in same main folder')
    print('full E_combinations')
    print(len(E_combinations))
    print('D')
    print(d)
            

    # Defining electrode combination for naming the CSV files in the end 
    Electrode_combination = [E_combinations[d][0], E_combinations[d][1]]
    Electrode_combination = extract_letters_and_numbers(Electrode_combination)
    print('Electrode combination')
    print(Electrode_combination)

    
    boxplot_features=['Deltacoh_av30s'+str(Electrode_combination), 'Thetacoh_av30s'+str(Electrode_combination),'Alphacoh_av30s'+str(Electrode_combination),'Betacoh_av30s'+str(Electrode_combination),'Gammacoh_av30s'+str(Electrode_combination)]
    plt.suptitle('Coherence 30 seconds intervals')
    plt.subplot(1, 2, 1) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_NT1[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title(' NT1 '+str(Electrode_combination))
    plt.tight_layout()


    plt.subplot(1, 2, 2) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_controls[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title('Controls '+str(Electrode_combination))
    plt.tight_layout()

    print('Figure was saved')
    plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_coherence_30scoh_'+str(Electrode_combination)+'.png')
    
    plt.clf()
    del boxplot_features





### Wake coherence 30sec intervals ##############
for d in range(len(E_combinations)):

    print('Finding new electrode combination in same main folder')
    print('full E_combinations')
    print(len(E_combinations))
    print('D')
    print(d)
            

    # Defining electrode combination for naming the CSV files in the end 
    Electrode_combination = [E_combinations[d][0], E_combinations[d][1]]
    Electrode_combination = extract_letters_and_numbers(Electrode_combination)
    print('Electrode combination')
    print(Electrode_combination)

    
    boxplot_features=['Wakecoh_delta'+str(Electrode_combination), 'Wakecoh_theta'+str(Electrode_combination),'Wakecoh_alpha'+str(Electrode_combination),'Wakecoh_beta'+str(Electrode_combination),'Wakecoh_gamma'+str(Electrode_combination)]
    plt.suptitle('Wake coherence 30 seconds intervals')
    plt.subplot(1, 2, 1) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_NT1[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title(' NT1 '+str(Electrode_combination))
    plt.tight_layout()


    plt.subplot(1, 2, 2) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_controls[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title('Controls '+str(Electrode_combination))
    plt.tight_layout()

    print('Figure was saved')
    plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_wakecoherence_'+str(Electrode_combination)+'.png')
    
    plt.clf()
    del boxplot_features





### N1 coherence 30sec intervals ##############
for d in range(len(E_combinations)):

    print('Finding new electrode combination in same main folder')
    print('full E_combinations')
    print(len(E_combinations))
    print('D')
    print(d)
            

    # Defining electrode combination for naming the CSV files in the end 
    Electrode_combination = [E_combinations[d][0], E_combinations[d][1]]
    Electrode_combination = extract_letters_and_numbers(Electrode_combination)
    print('Electrode combination')
    print(Electrode_combination)

    
    boxplot_features=['N1coh_delta'+str(Electrode_combination), 'N1coh_theta'+str(Electrode_combination),'N1coh_alpha'+str(Electrode_combination),'N1coh_beta'+str(Electrode_combination),'N1coh_gamma'+str(Electrode_combination)]
    plt.suptitle('N1 coherence 30 seconds intervals')
    plt.subplot(1, 2, 1) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_NT1[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title(' NT1 '+str(Electrode_combination))
    plt.tight_layout()


    plt.subplot(1, 2, 2) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_controls[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title('Controls '+str(Electrode_combination))
    plt.tight_layout()

    print('Figure was saved')
    plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_N1coherence_'+str(Electrode_combination)+'.png')
    
    plt.clf()
    del boxplot_features




### N2 coherence 30sec intervals ##############
for d in range(len(E_combinations)):

    print('Finding new electrode combination in same main folder')
    print('full E_combinations')
    print(len(E_combinations))
    print('D')
    print(d)
            

    # Defining electrode combination for naming the CSV files in the end 
    Electrode_combination = [E_combinations[d][0], E_combinations[d][1]]
    Electrode_combination = extract_letters_and_numbers(Electrode_combination)
    print('Electrode combination')
    print(Electrode_combination)

    
    boxplot_features=['N2coh_delta'+str(Electrode_combination), 'N2coh_theta'+str(Electrode_combination),'N2coh_alpha'+str(Electrode_combination),'N2coh_beta'+str(Electrode_combination),'N2coh_gamma'+str(Electrode_combination)]
    plt.suptitle('N2 coherence 30 seconds intervals')
    plt.subplot(1, 2, 1) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_NT1[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title(' NT1 '+str(Electrode_combination))
    plt.tight_layout()


    plt.subplot(1, 2, 2) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_controls[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title('Controls '+str(Electrode_combination))
    plt.tight_layout()

    print('Figure was saved')
    plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_N2coherence_'+str(Electrode_combination)+'.png')
    
    plt.clf()
    del boxplot_features




### N3 coherence 30sec intervals ##############
for d in range(len(E_combinations)):

    print('Finding new electrode combination in same main folder')
    print('full E_combinations')
    print(len(E_combinations))
    print('D')
    print(d)
            

    # Defining electrode combination for naming the CSV files in the end 
    Electrode_combination = [E_combinations[d][0], E_combinations[d][1]]
    Electrode_combination = extract_letters_and_numbers(Electrode_combination)
    print('Electrode combination')
    print(Electrode_combination)

    
    boxplot_features=['N3coh_delta'+str(Electrode_combination), 'N3coh_theta'+str(Electrode_combination),'N3coh_alpha'+str(Electrode_combination),'N3coh_beta'+str(Electrode_combination),'N3coh_gamma'+str(Electrode_combination)]
    plt.suptitle('N3 coherence 30 seconds intervals')
    plt.subplot(1, 2, 1) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_NT1[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title(' NT1 '+str(Electrode_combination))
    plt.tight_layout()


    plt.subplot(1, 2, 2) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_controls[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title('Controls '+str(Electrode_combination))
    plt.tight_layout()

    print('Figure was saved')
    plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_N3coherence_'+str(Electrode_combination)+'.png')
    
    plt.clf()
    del boxplot_features




### REM coherence 30sec intervals ##############
for d in range(len(E_combinations)):

    print('Finding new electrode combination in same main folder')
    print('full E_combinations')
    print(len(E_combinations))
    print('D')
    print(d)
            

    # Defining electrode combination for naming the CSV files in the end 
    Electrode_combination = [E_combinations[d][0], E_combinations[d][1]]
    Electrode_combination = extract_letters_and_numbers(Electrode_combination)
    print('Electrode combination')
    print(Electrode_combination)

    
    boxplot_features=['REMcoh_delta'+str(Electrode_combination), 'REMcoh_theta'+str(Electrode_combination),'REMcoh_alpha'+str(Electrode_combination),'REMcoh_beta'+str(Electrode_combination),'REMcoh_gamma'+str(Electrode_combination)]
    plt.suptitle('REM coherence 30 seconds intervals')
    plt.subplot(1, 2, 1) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_NT1[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title(' NT1 '+str(Electrode_combination))
    plt.tight_layout()


    plt.subplot(1, 2, 2) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_controls[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title('Controls '+str(Electrode_combination))
    plt.tight_layout()

    print('Figure was saved')
    plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_REMcoherence_'+str(Electrode_combination)+'.png')
    
    plt.clf()
    del boxplot_features



del Electrode_combination

### Bandpower  ##############
for d in range(len(Electrodes)):

    print('Finding new electrode combination in same main folder')
    print('full E_combinations')
    print(len(Electrodes))
    print('D')
    print(d)
            

    # Defining electrode combination for naming the CSV files in the end 
    #Electrode_combination = [E_combinations[d][0], E_combinations[d][1]]
    Electrode_combination = Electrodes[d]
    print('Electrode combination')
    print(Electrode_combination)

    
    boxplot_features=['P1_delta_'+str(Electrode_combination), 'P1_theta_'+str(Electrode_combination),'P1_alpha_'+str(Electrode_combination),'P1_beta_'+str(Electrode_combination),'P1_gamma_'+str(Electrode_combination)]
    plt.suptitle('Bandpower electrode')
    plt.subplot(1, 2, 1) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_NT1[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title(' NT1 '+str(Electrode_combination))
    plt.tight_layout()


    plt.subplot(1, 2, 2) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_controls[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title('Controls '+str(Electrode_combination))
    plt.tight_layout()

    print('Figure was saved')
    plt.savefig('C:/Users/natas/Documents/Master thesis code/Coherence features/France/Boxplot_France_Bandpower_'+str(Electrode_combination)+'.png')
    
    plt.clf()
    del boxplot_features

'''