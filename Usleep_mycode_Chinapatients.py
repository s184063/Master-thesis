############################################################ 
# Usleep my code
# Made by Natasja Bonde Andersen 05.03.2024

# Part 1: 
# This code generates the hypnodensity features for each patient and saved as np files. 
# These features are then loaded in as np.arrays and correlation between two chosen
# electrodes and their sleep stage probabilities are performed. 

# Part 2
# A correlation value is saved for each sleep stage (N1, N2, N3, Wake and REM).
# These values are saved in a CSV file taking the number of electrodes into account and the epoch size. 
# The epoch sizes are [1s 3s 5s 15s 30s]

# Part 3
# The CSV files are combined such that all epochs for a given set of electrodes are gathered in one file. 
# This means the end result will be a CSV file containing all 2-electrode or 3-electrode recordings 
# and their correlation values for each epoch. 

# Part 4 
# Simple multiple linear regression model of correlation features 

#############################################################################################

#################### Loading packages ###############################
import os
import numpy as np
import usleep_api
from usleep_api import USleepAPI
import pandas as pd
import matplotlib
from matplotlib import pyplot
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_example")
import re
import copy
import sys
import itertools 
from itertools import combinations
import statsmodels.api as sm
#import seaborn as sns
# Using sys function to import 'My_functions_script'
sys.path.insert(0, '/home/users/s184063')
# Import My_functions_script
from My_functions_script_China import extract_numbers_from_filename, extract_letters_and_numbers, list_files_in_folder, split_string_by_length, Usleep_2channels, correlation_multiple_electrodes

################################################################################################


####### Setting paths ##################################
# File paths 
input_path =r"/scratch/users/s184063/China restructured EDF patients correct/"
#input_file=f'A0001_4 165907.EDF' # Easier later for namechange according to the patient ID 
#input_file_path = os.path.join(input_path,input_file)
output_path = r'/scratch/users/s184063/hypnograms_China_patients/' 
#######################################################

############## Part 1 #####################################
# Looping over all EDF files and generating all hypnodensity features. 
# They are saved in folders according to the patientID. 
# Each file gets a name related to patientID, visitnumber,electrode combination and epoch size. 


# Choose epoch size in seconds 
epoch_size_in_seconds =[1, 3, 5, 15, 30]   

temp_visit=[]
error_dict=[]

print(input_path)
print(type(input_path))
print(output_path)
print(type(output_path))


# Looping over all EDF files in folder 
edf_files_list_unsorted = list_files_in_folder(input_path)

print('Unsorted edf files')
print(edf_files_list_unsorted)

edf_files_list=sorted(edf_files_list_unsorted)

print('Sorted edf list')
print(edf_files_list)

print("EDF files in the folder:")
for edf_file in edf_files_list:
    print(edf_file)

    numbers_found = os.path.basename(edf_file)
    print("Numbers found in the filename:", numbers_found)

    filename, file_extension = os.path.splitext(numbers_found)

    print('filename')
    print(filename)

    numbers_found=filename

    patientID=filename
    # Recording the visit number for later 
    
    
    #temp_visit.append(numbers_found[0])
    #print(numbers_found[0])
    #visitnumber=np.stack(temp_visit,axis=0)
    
    

    # create output folder 
    make_folder_path = os.path.join(output_path, str(numbers_found))
    print('Make folder path')
    print(make_folder_path)
    
    # Check if the folder already exists before creating it
    if not os.path.exists(make_folder_path):
        os.makedirs(make_folder_path)
        print(f"Folder created for Patient ID {str(numbers_found)}")


        # Generate hypnodensity in the clean folder
        for loop_factor in epoch_size_in_seconds:
            # Running function to generate hypnodensity features 

            try:
                print('Generating folder and looping through epoch sizes and generating hypnograms')
                Name_electrodes =Usleep_2channels(edf_file,make_folder_path,loop_factor,numbers_found)
                ### Saving the electrode names as a tuple ####
                # Converting a tuple and saving it 
                with open(os.path.join(make_folder_path, f'Name_electrodes.txt'),'w') as file:
                    for item in Name_electrodes:
                        file.write(str(item))
                ##############################################
            except Exception as e:
                print(f"An error occurred: {e}. Moving on to the next iteration...")
                
                error_dict.append({'error':e,'filename':edf_file})
                continue


                


    else:
        print(f"Folder for Patient ID {str(numbers_found)} already exists")

        # Checking if the textfile with the electrode names are present - this means hypnograms has been generated before 
        text_file_path = os.path.join(make_folder_path, f'Name_electrodes.txt') 
        print('Text file path')
        print(text_file_path)


        if os.path.exists(text_file_path):
            print('Loading Names_electrodes.txt')
            text_file_path = os.path.join(make_folder_path, f'Name_electrodes.txt') 
            
            # Initialize an empty list to store the elements of the tuple
            my_list = []

            # Open the file for reading
            with open(text_file_path, 'r') as file:
                # Read each line from the file
                for line in file:
                    # Strip any trailing newline characters and append the line to the list
                    my_list.append(line.strip())

            # Convert the list to a tuple
            Name_electrodes2 = tuple(my_list)

            # Split this part into the set names (consistent parts)
            Name_electrodes=extract_letters_and_numbers(str(Name_electrodes2))
            print(Name_electrodes)
            print(type(Name_electrodes))
            print(len(Name_electrodes))

            Electrodes=split_string_by_length(Name_electrodes,4) # 4 is the length of the electrode name e.g. F3M2
            print('Split string')
            print(Electrodes)
            print(str(Electrodes[0]))
            print(str(Electrodes[1]))
            #print(numbers_found[0])
            #print(numbers_found[1])


            if len(Electrodes)==2:
                print('Two electrodes where found ')
                for loop_factor in epoch_size_in_seconds:
                    print(str(loop_factor))
                    print(make_folder_path+f"/hypnogram_ID_{numbers_found}_electrode_{Electrodes[0]}_epocssize_{loop_factor}.npy")
                    print(make_folder_path+f"/hypnogram_ID_{numbers_found}_electrode_{Electrodes[1]}_epocssize_{loop_factor}.npy")
                    Electrode0_exists=os.path.exists(make_folder_path+f"/hypnogram_ID_{numbers_found}_electrode_{Electrodes[0]}_epocssize_{loop_factor}.npy") # Later add patient ID in the name)
                    Electrode1_exists=os.path.exists(make_folder_path+f"/hypnogram_ID_{numbers_found}_electrode_{Electrodes[1]}_epocssize_{loop_factor}.npy") # Later add patient ID in the name)
                    print(Electrode0_exists)
                    print(Electrode1_exists)
                    
                    if Electrode0_exists and Electrode1_exists:
                        print('Both electrode files excist in all epoch sizes ')
                        continue
                    else:
                        # Running function to generate hypnodensity features 
                            print('Not all epoch sizes where present - they are being generated now')

                            try: 
                                Name_electrodes =Usleep_2channels(edf_file,make_folder_path,loop_factor,numbers_found)

                                ### Saving the electrode names as a tuple ####
                                # Converting a tuple and saving it 
                                with open(os.path.join(make_folder_path, f'Name_electrodes.txt'),'w') as file:
                                    for item in Name_electrodes:
                                        file.write(str(item))
                                ##############################################
                            except Exception as e:
                                print(f"An error occurred: {e}. Moving on to the next iteration...")
                
                                error_dict.append({'error':e,'filename':edf_file})
                                continue
            
            
            if len(Electrodes)==3:
                print('Three electrodes where found')
                for loop_factor in epoch_size_in_seconds:
                    Electrode0_exists=os.path.exists(make_folder_path+f"/hypnogram_ID_{numbers_found}_electrode_{Electrodes[0]}_epocssize_{loop_factor}.npy") # Later add patient ID in the name)
                    Electrode1_exists=os.path.exists(make_folder_path+f"/hypnogram_ID_{numbers_found}_electrode_{Electrodes[1]}_epocssize_{loop_factor}.npy") # Later add patient ID in the name)
                    Electrode2_exists=os.path.exists(make_folder_path+f"/hypnogram_ID_{numbers_found}_electrode_{Electrodes[2]}_epocssize_{loop_factor}.npy") # Later add patient ID in the name)
                    
                    print(Electrode0_exists)
                    print(Electrode1_exists)
                    print(Electrode2_exists)

                    if Electrode0_exists and Electrode1_exists and Electrode2_exists:
                        print('All epoch sizes for the three electrodes excist')
                        continue
                    else:
                        # Running function to generate hypnodensity features 
                            try: 
                                print('Not all epoch sizes for the three electrodes excisted - they are generated now')
                                Name_electrodes =Usleep_2channels(edf_file,make_folder_path,loop_factor,numbers_found)

                                ### Saving the electrode names as a tuple ####
                                # Converting a tuple and saving it 
                                with open(os.path.join(make_folder_path, f'Name_electrodes.txt'),'w') as file:
                                    for item in Name_electrodes:
                                        file.write(str(item))
                                ##############################################
                            except Exception as e:
                                print(f"An error occurred: {e}. Moving on to the next iteration...")
                
                                error_dict.append({'error':e,'filename':edf_file})
                                continue
                                    

            if len(Electrodes)==6:
                print('Six electrodes where found')
                
                for loop_factor in epoch_size_in_seconds:
                    Electrode0_exists=os.path.exists(make_folder_path+f"/hypnogram_ID_{numbers_found}_electrode_{Electrodes[0]}_epocssize_{loop_factor}.npy") # Later add patient ID in the name)
                    Electrode1_exists=os.path.exists(make_folder_path+f"/hypnogram_ID_{numbers_found}_electrode_{Electrodes[1]}_epocssize_{loop_factor}.npy") # Later add patient ID in the name)
                    Electrode2_exists=os.path.exists(make_folder_path+f"/hypnogram_ID_{numbers_found}_electrode_{Electrodes[2]}_epocssize_{loop_factor}.npy") # Later add patient ID in the name)
                    Electrode3_exists=os.path.exists(make_folder_path+f"/hypnogram_ID_{numbers_found}_electrode_{Electrodes[3]}_epocssize_{loop_factor}.npy") # Later add patient ID in the name)
                    Electrode4_exists=os.path.exists(make_folder_path+f"/hypnogram_ID_{numbers_found}_electrode_{Electrodes[4]}_epocssize_{loop_factor}.npy") # Later add patient ID in the name)
                    Electrode5_exists=os.path.exists(make_folder_path+f"/hypnogram_ID_{numbers_found}_electrode_{Electrodes[5]}_epocssize_{loop_factor}.npy") # Later add patient ID in the name)
                    
                    print(Electrode0_exists)
                    print(Electrode1_exists)
                    print(Electrode2_exists)
                    print(Electrode3_exists)
                    print(Electrode4_exists)
                    print(Electrode5_exists)

                    if Electrode0_exists and Electrode1_exists and Electrode2_exists and Electrode3_exists and Electrode4_exists and Electrode5_exists:
                        print('All epoch sizes for the six electrodes excist')
                        continue
                    else:
                        # Running function to generate hypnodensity features 
                            try: 
                                print('Not all epoch sizes for the three electrodes excisted - they are generated now')
                                Name_electrodes =Usleep_2channels(edf_file,make_folder_path,loop_factor,numbers_found)

                                ### Saving the electrode names as a tuple ####
                                # Converting a tuple and saving it 
                                with open(os.path.join(make_folder_path, f'Name_electrodes.txt'),'w') as file:
                                    for item in Name_electrodes:
                                        file.write(str(item))
                                ##############################################
                            except Exception as e:
                                print(f"An error occurred: {e}. Moving on to the next iteration...")
                
                                error_dict.append({'error':e,'filename':edf_file})
                                continue
                                 

        else:
            print('The Name_electrodes.txt does not exist')
        
            for loop_factor in epoch_size_in_seconds:
                # Running function to generate hypnodensity features 
                try: 
                    print('The text file did not excist and the hypnogram in all epochs are being generated now')
                    Name_electrodes =Usleep_2channels(edf_file,make_folder_path,loop_factor,numbers_found)

                    ### Saving the electrode names as a tuple ####
                    # Converting a tuple and saving it 
                    with open(os.path.join(make_folder_path, f'Name_electrodes.txt'),'w') as file:
                        for item in Name_electrodes:
                            file.write(str(item))
                    ##############################################
                except Exception as e:
                    print(f"An error occurred: {e}. Moving on to the next iteration...")
                
                    error_dict.append({'error':e,'filename':edf_file})
                    continue


    

Errors=pd.DataFrame(error_dict)   
Errors.to_csv('Errors_happened.csv') 
#visitnumber_df=pd.DataFrame({'wsc_vst':visitnumber})
#visitnumber_df.to_csv('Visit_number.csv')

'''

##### Part 2 ###########################
#### Load the hypnograms and do correlation between combination of pairs of electrodes ####

epoch_size_in_seconds = 30 #[1s 3s 5s 15s 30s]

# The path loaded in should be the one, where the hypnograms are stored 

correlation_multiple_electrodes(output_path,epoch_size_in_seconds)

# The Pearson (product-moment) correlation coefficient is a measure of the linear relationship between two features.
# Pearson correlation coefficient can take on any real value in the range −1 ≤ r ≤ 1.
#########################################



########### Part 3 #################
# Combining CSV files 

# Loading CSV files for 2-electrode data for all epocs
E2_epoc1=pd.read_csv('C:/Users/natas/Documents/Master thesis code/U-Sleep-API-Python-Bindings/Correlation_2E_epocssize_1_allpatients_data.csv')
E2_epoc3=pd.read_csv('C:/Users/natas/Documents/Master thesis code/U-Sleep-API-Python-Bindings/Correlation_2E_epocssize_3_allpatients_data.csv')
E2_epoc5=pd.read_csv('C:/Users/natas/Documents/Master thesis code/U-Sleep-API-Python-Bindings/Correlation_2E_epocssize_5_allpatients_data.csv')
E2_epoc15=pd.read_csv('C:/Users/natas/Documents/Master thesis code/U-Sleep-API-Python-Bindings/Correlation_2E_epocssize_15_allpatients_data.csv')
E2_epoc30=pd.read_csv('C:/Users/natas/Documents/Master thesis code/U-Sleep-API-Python-Bindings/Correlation_2E_epocssize_30_allpatients_data.csv')


# Dropping the first patienID folder for most of the files in order to make a concatenation of all files 
E2_epoc3=E2_epoc3.drop('PatientID_2E',axis=1)
E2_epoc5=E2_epoc5.drop('PatientID_2E',axis=1)
E2_epoc15=E2_epoc15.drop('PatientID_2E',axis=1)
E2_epoc30=E2_epoc30.drop('PatientID_2E',axis=1)

#print(E2_epoc5)

# Combining the dataframes. Only one column contains patientID
df_2E=pd.concat([E2_epoc1,E2_epoc3,E2_epoc5,E2_epoc15,E2_epoc30],axis=1)
df_2E.to_csv('Correlation_2E_all_epocs.csv')
print(df_2E)



# Loading CSV files for 3-electrode data for all epocs
E3_epoc1=pd.read_csv('C:/Users/natas/Documents/Master thesis code/U-Sleep-API-Python-Bindings/Correlation_3E_epocssize_1_allpatients_data.csv')
E3_epoc3=pd.read_csv('C:/Users/natas/Documents/Master thesis code/U-Sleep-API-Python-Bindings/Correlation_3E_epocssize_3_allpatients_data.csv')
E3_epoc5=pd.read_csv('C:/Users/natas/Documents/Master thesis code/U-Sleep-API-Python-Bindings/Correlation_3E_epocssize_5_allpatients_data.csv')
E3_epoc15=pd.read_csv('C:/Users/natas/Documents/Master thesis code/U-Sleep-API-Python-Bindings/Correlation_3E_epocssize_15_allpatients_data.csv')
E3_epoc30=pd.read_csv('C:/Users/natas/Documents/Master thesis code/U-Sleep-API-Python-Bindings/Correlation_3E_epocssize_30_allpatients_data.csv')


# Dropping the first patienID folder for most of the files in order to make a concatenation of all files 
E3_epoc3=E3_epoc3.drop('PatientID_3E',axis=1)
E3_epoc5=E3_epoc5.drop('PatientID_3E',axis=1)
E3_epoc15=E3_epoc15.drop('PatientID_3E',axis=1)
E3_epoc30=E3_epoc30.drop('PatientID_3E',axis=1)

print(E3_epoc5)

# Combining dataframes to one large dataframe with only one column containing patientID
df_3E=pd.concat([E3_epoc1,E3_epoc3,E3_epoc5,E3_epoc15,E3_epoc30],axis=1)
df_3E.to_csv('Correlation_3E_all_epocs.csv')
print(df_3E)


########### part 4 ###############################################################################
# Multiple linear regression 

# Predict: age 
# Input: correlation features for each epoch length 

################## Extract age from the Wisconsin data set ######################
demographics=pd.read_csv('C:/Users/natas/Documents/Master thesis code/U-Sleep-API-Python-Bindings/wsc-dataset-0.6.0.csv')
print(demographics.shape)
ID_age_vst=demographics[['wsc_id','age','wsc_vst']] #wisconsinID =wsc_id, age=at visit, wsc_vst=visit number
#print(ID_age_vst)

# patientID from 2-electrode data
patientID_2E=df_2E['PatientID_2E']

## Delete later !!!!!!!
visitnumber_fake = pd.Series(1, index=patientID_2E.index) # making it the same length as patientID
# load the csv file instead 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# creating dataframe 
ID_vst_2E=pd.DataFrame({'wsc_id':patientID_2E,'wsc_vst':visitnumber_fake}) # using same names in order to merge later 
print('ID_vst_2E')
print(ID_vst_2E)
# Finding the indexes where the patientID match 
result_2E=pd.merge(ID_age_vst,ID_vst_2E,how='inner',on=['wsc_id','wsc_vst'])
print('Results - finding indexes for age')
print(result_2E)

# Extracting age and adding the age to the full dataframe for 2E
age_vst=result_2E['age']
print('Age_vst')
print(result_2E['age'])
print(type(result_2E['age']))
df_2E['age_vst']=age_vst
print(df_2E)


# patientID from 3-electrode data
patientID_3E=df_3E['PatientID_3E']

# Delete later !!!!!!!!!!!!!!!!!!!!
visitnumber_fake = pd.Series(1, index=patientID_3E.index)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# creating dataframe 
ID_vst_3E=pd.DataFrame({'wsc_id':patientID_3E,'wsc_vst':visitnumber_fake})
#print(ID_vst_3E)
# Finding the indexes where the patientID match 
result_3E=pd.merge(ID_age_vst,ID_vst_3E,how='inner',on=['wsc_id','wsc_vst'])
print(result_3E)


# adding the age to the full dataframe for 3E
age_vst=result_3E['age']
df_3E['age_vst']=age_vst
print(df_3E)



################### Combining all electrode C3_M2 and O1_M2 in one CSV ######################

# Extract all columns with the electrode name and combine to another dataframe 
# Epoch 1 - extracting from df3E
Wake_epoch1=df_3E['Wake_3E_C3M2O1M2_epocssize_1']
N1_epoch1=df_3E['N1_3E_C3M2O1M2_epocssize_1']
N2_epoch1=df_3E['N2_3E_C3M2O1M2_epocssize_1']
N3_epoch1=df_3E['N3_3E_C3M2O1M2_epocssize_1']
REM_epoch1=df_3E['REM_3E_C3M2O1M2_epocssize_1']

# Epoch 3 - extracting from df3E
Wake_epoch3=df_3E['Wake_3E_C3M2O1M2_epocssize_3']
N1_epoch3=df_3E['N1_3E_C3M2O1M2_epocssize_3']
N2_epoch3=df_3E['N2_3E_C3M2O1M2_epocssize_3']
N3_epoch3=df_3E['N3_3E_C3M2O1M2_epocssize_3']
REM_epoch3=df_3E['REM_3E_C3M2O1M2_epocssize_3']

# Epoch 5 - extracting from df3E
Wake_epoch5=df_3E['Wake_3E_C3M2O1M2_epocssize_5']
N1_epoch5=df_3E['N1_3E_C3M2O1M2_epocssize_5']
N2_epoch5=df_3E['N2_3E_C3M2O1M2_epocssize_5']
N3_epoch5=df_3E['N3_3E_C3M2O1M2_epocssize_5']
REM_epoch5=df_3E['REM_3E_C3M2O1M2_epocssize_5']

# Epoch 15 - extracting from df3E
Wake_epoch15=df_3E['Wake_3E_C3M2O1M2_epocssize_15']
N1_epoch15=df_3E['N1_3E_C3M2O1M2_epocssize_15']
N2_epoch15=df_3E['N2_3E_C3M2O1M2_epocssize_15']
N3_epoch15=df_3E['N3_3E_C3M2O1M2_epocssize_15']
REM_epoch15=df_3E['REM_3E_C3M2O1M2_epocssize_15']

# Epoch 30 - extracting from df3E
Wake_epoch30=df_3E['Wake_3E_C3M2O1M2_epocssize_30']
N1_epoch30=df_3E['N1_3E_C3M2O1M2_epocssize_30']
N2_epoch30=df_3E['N2_3E_C3M2O1M2_epocssize_30']
N3_epoch30=df_3E['N3_3E_C3M2O1M2_epocssize_30']
REM_epoch30=df_3E['REM_3E_C3M2O1M2_epocssize_30']

# Combining all extracted values into one dataframe matching the structure for correlation_2E_all_epochs

# Epoch 1 - changning name and adding C3_M2 vs. O1_M2 to the df_2E dataframe from df_3E
df_C3M2O1M2=pd.DataFrame({'PatientID_2E':patientID_3E}) # changing name to merge to df
df_C3M2O1M2['Wake_2E_C3M2O1M2_epocssize_1']=Wake_epoch1
df_C3M2O1M2['N1_2E_C3M2O1M2_epocssize_1']=N1_epoch1
df_C3M2O1M2['N2_2E_C3M2O1M2_epocssize_1']=N2_epoch1
df_C3M2O1M2['N3_2E_C3M2O1M2_epocssize_1']=N3_epoch1
df_C3M2O1M2['REM_2E_C3M2O1M2_epocssize_1']=REM_epoch1

# Epoch 3 - changning name and adding C3_M2 vs. O1_M2 to the df_2E dataframe from df_3E
df_C3M2O1M2['Wake_2E_C3M2O1M2_epocssize_3']=Wake_epoch3
df_C3M2O1M2['N1_2E_C3M2O1M2_epocssize_3']=N1_epoch3
df_C3M2O1M2['N2_2E_C3M2O1M2_epocssize_3']=N2_epoch3
df_C3M2O1M2['N3_2E_C3M2O1M2_epocssize_3']=N3_epoch3
df_C3M2O1M2['REM_2E_C3M2O1M2_epocssize_3']=REM_epoch3

# Epoch 5 - changning name and adding C3_M2 vs. O1_M2 to the df_2E dataframe from df_3E
df_C3M2O1M2['Wake_2E_C3M2O1M2_epocssize_5']=Wake_epoch5
df_C3M2O1M2['N1_2E_C3M2O1M2_epocssize_5']=N1_epoch5
df_C3M2O1M2['N2_2E_C3M2O1M2_epocssize_5']=N2_epoch5
df_C3M2O1M2['N3_2E_C3M2O1M2_epocssize_5']=N3_epoch5
df_C3M2O1M2['REM_2E_C3M2O1M2_epocssize_5']=REM_epoch5

# Epoch 15 - changning name and adding C3_M2 vs. O1_M2 to the df_2E dataframe from df_3E
df_C3M2O1M2['Wake_2E_C3M2O1M2_epocssize_15']=Wake_epoch15
df_C3M2O1M2['N1_2E_C3M2O1M2_epocssize_15']=N1_epoch15
df_C3M2O1M2['N2_2E_C3M2O1M2_epocssize_15']=N2_epoch15
df_C3M2O1M2['N3_2E_C3M2O1M2_epocssize_15']=N3_epoch15
df_C3M2O1M2['REM_2E_C3M2O1M2_epocssize_15']=REM_epoch15

# EPoch 30 - changning name and adding C3_M2 vs. O1_M2 to the df_2E dataframe from df_3E
df_C3M2O1M2['Wake_2E_C3M2O1M2_epocssize_30']=Wake_epoch30
df_C3M2O1M2['N1_2E_C3M2O1M2_epocssize_30']=N1_epoch30
df_C3M2O1M2['N2_2E_C3M2O1M2_epocssize_30']=N2_epoch30
df_C3M2O1M2['N3_2E_C3M2O1M2_epocssize_30']=N3_epoch30
df_C3M2O1M2['REM_2E_C3M2O1M2_epocssize_30']=REM_epoch30

df_C3M2O1M2['age_vst']=age_vst

print(df_C3M2O1M2)




# Afterwards concatenate like this: 
df_combined_C3M2O1M2=pd.concat([df_2E, df_C3M2O1M2])
print(df_combined_C3M2O1M2)

df_combined_C3M2O1M2.to_csv('Combined_C3M2O1M2.csv')




################## Statistical investigation of features ###################


# Loading data frame 
df_combined_C3M2O1M2=pd.read_csv('Combined_C3M2O1M2.csv')
df_combined_C3M2O1M2=df_combined_C3M2O1M2.drop(['PatientID_2E'],axis=1)
print(df_combined_C3M2O1M2)

print('Interested in this one')
print(df_combined_C3M2O1M2.iloc[:,1:25])

# Selecting the interesting columns
df_combined_C3M2O1M2=df_combined_C3M2O1M2.iloc[:,1:27]





##### Correlation matrix between features #########

# form correlation matrix
import matplotlib.pyplot as plt

# Calculate correlation matrix
matrix = df_combined_C3M2O1M2.corr()

# Print correlation matrix and save to a CSV
print("Correlation matrix is : ")
print(matrix)
matrix.to_csv('Matrix_correlation.csv')

# Initialize a new figure and set its size
plt.figure(figsize=(8, 6))

# Plotting correlation matrix 
plt.imshow(matrix, cmap='Blues')

# Adding a color bar
plt.colorbar()

# Extracting variable names 
variables = matrix.columns.tolist()

# Adding labels to the matrix
plt.xticks(range(len(matrix)), variables, rotation=45, ha='right')
plt.yticks(range(len(matrix)), variables)

# Adding the title
plt.title('Correlation matrix of correlated hypnodensity features')

# Adjust the padding
plt.tight_layout()

# Display the plot
plt.show()
'''
'''

#####Multiple linear regression ##########

# Define the dependent variable and the independent variables.
Y = df_combined_C3M2O1M2['age_vst'] # replace 'Y' with your actual column name for the dependent variable
X = df_combined_C3M2O1M2.iloc[:,24] # differ this variable to select single features to compare with age 



# Add a constant to the independent value
X = sm.add_constant(X)

# Conduct the linear regression
model = sm.OLS(Y, X)
results = model.fit()

# Print the summary statistics of the regression model.
print(results.summary())


#### Standard statistics #####
mean_list=[]

for j in range(25):
    chosen_signal=df_combined_C3M2O1M2.iloc[:,j]
    print('Chosen signal')
    print(chosen_signal)

    #mean
    mean= np.mean(chosen_signal)
    print('Mean')
    print(mean)

    mean_list.append(mean)
    mean_stack=np.stack(mean_list)


mean=pd.DataFrame({'mean':mean_stack})
mean.to_csv('Mean values.csv')




#std

std_list=[]

for j in range(25):
    chosen_signal=df_combined_C3M2O1M2.iloc[:,j]
    print('Chosen signal')
    print(chosen_signal)

    #mean
    std= np.std(chosen_signal)
    

    std_list.append(std)
    std_stack=np.stack(std_list)


std=pd.DataFrame({'std':std_stack})
std.to_csv('Std values.csv')



#min value 
min_list=[]

for j in range(25):
    chosen_signal=df_combined_C3M2O1M2.iloc[:,j]
    print('Chosen signal')
    print(chosen_signal)

    #mean
    min= np.min(chosen_signal)
    

    min_list.append(min)
    min_stack=np.stack(min_list)


min=pd.DataFrame({'minimum':min_stack})
min.to_csv('min values.csv')


#max value
max_list=[]

for j in range(25):
    chosen_signal=df_combined_C3M2O1M2.iloc[:,j]
    print('Chosen signal')
    print(chosen_signal)

    #mean
    max= np.max(chosen_signal)
    

    max_list.append(max)
    max_stack=np.stack(max_list)


max=pd.DataFrame({'max':max_stack})
max.to_csv('Max values.csv')
'''


#### Scatter plots ######
'''
# Convert into np for all features 
age=np.array(df_combined_C3M2O1M2['age_vst'])
print(age)

feature1=np.array(df_combined_C3M2O1M2.iloc[:,0])
print(feature1)
print(df_combined_C3M2O1M2.iloc[:,0])





# All single features vs. age
wake=[0, 4, 9, 15, 20]
matplotlib.pyplot.subplot(2, 3, 1) # (dimension1, dimension2, number of plots)
for i in wake:
    # Wake   
    
    matplotlib.pyplot.scatter(age,np.array(df_combined_C3M2O1M2.iloc[:,i]))
    matplotlib.pyplot.xlabel('Age')
    matplotlib.pyplot.ylabel('Wake all epochs')
    matplotlib.pyplot.legend(['1s','3s','5s','15s','30s'])

print(df_combined_C3M2O1M2.iloc[:,20])

# N1   
matplotlib.pyplot.subplot(2, 3, 2) # (dimension1, dimension2, number of plots)
N1=[1, 5, 11, 16, 21]
for j in N1:
    matplotlib.pyplot.scatter(age,np.array(df_combined_C3M2O1M2.iloc[:,j]))
    matplotlib.pyplot.xlabel('Age')
    matplotlib.pyplot.ylabel('N1 all epochs')
    matplotlib.pyplot.legend(['1s','3s','5s','15s','30s'])
    matplotlib.pyplot.suptitle("Scatterplot of features vs. age- Features epochsize ")

# N2
matplotlib.pyplot.subplot(2, 3, 3) # (dimension1, dimension2, number of plots)

N2 =[2, 6, 12, 17, 22]
for k in N2:
    matplotlib.pyplot.scatter(age,np.array(df_combined_C3M2O1M2.iloc[:,k]))
    matplotlib.pyplot.xlabel('Age')
    matplotlib.pyplot.ylabel('N2 all epochs')
    matplotlib.pyplot.legend(['1s','3s','5s','15s','30s'])
    matplotlib.pyplot.suptitle("Scatterplot of features vs. age- Features epoch ")

# N3   
N3 = [3, 7, 13, 18, 23]
for h in N3:
    matplotlib.pyplot.subplot(2, 3, 4) # (dimension1, dimension2, number of plots)
    matplotlib.pyplot.scatter(age,np.array(df_combined_C3M2O1M2.iloc[:,h]))
    matplotlib.pyplot.xlabel('Age')
    matplotlib.pyplot.ylabel('N3 all epochs')
    matplotlib.pyplot.legend(['1s','3s','5s','15s','30s'])
    matplotlib.pyplot.suptitle("Scatterplot of features vs. age- Features epoch ")


# REM
REM = [4, 8, 14, 19, 24]
for dd in REM:
    matplotlib.pyplot.subplot(2, 3, 5) # (dimension1, dimension2, number of plots)
    matplotlib.pyplot.scatter(age,np.array(df_combined_C3M2O1M2.iloc[:,dd]))
    matplotlib.pyplot.xlabel('Age')
    matplotlib.pyplot.ylabel('REM all epochs')
    matplotlib.pyplot.legend(['1s','3s','5s','15s','30s'])
    matplotlib.pyplot.suptitle("Scatterplot of features vs. age- Features epoch ")


# All
matplotlib.pyplot.subplot(2, 3, 6) # (dimension1, dimension2, number of plots)
for l in range(24):
    matplotlib.pyplot.scatter(age,np.array(df_combined_C3M2O1M2.iloc[:,l]))
    matplotlib.pyplot.xlabel('Age')
    matplotlib.pyplot.ylabel('All epochs mixed')
    matplotlib.pyplot.legend(['Wake','N1','N2','N3','REM'])
    matplotlib.pyplot.suptitle("Scatterplot of sleep stages vs. age ")

matplotlib.pyplot.show()
'''


########## box plot ########
'''
# for all features 
matplotlib.pyplot.figure(figsize=(8,6)) # Adjust these numbers as per your requirement.
sns.boxplot(data=df_combined_C3M2O1M2.iloc[:,0:25])
matplotlib.pyplot.xticks(rotation=45, ha='right')
matplotlib.pyplot.title('Correlation features in box plot [Wake, N1, N2, N3, REM]')
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.show()
'''
######histograms #######

# for all features 
#matplotlib.pyplot.figure(figsize=(8,6)) # Adjust these numbers as per your requirement.
'''
matplotlib.pyplot.subplot(2,3,1)
sns.histplot(data=df_combined_C3M2O1M2.iloc[:,20])
#matplotlib.pyplot.xticks(rotation=45, ha='right')
matplotlib.pyplot.suptitle('Histogram - Epoch size 30s for all plots')
matplotlib.pyplot.tight_layout()

matplotlib.pyplot.subplot(2,3,2)
sns.histplot(data=df_combined_C3M2O1M2.iloc[:,21])
#matplotlib.pyplot.xticks(rotation=45, ha='right')
matplotlib.pyplot.suptitle('Histogram - Epoch size 30s for all plots')
matplotlib.pyplot.tight_layout()

matplotlib.pyplot.subplot(2,3,3)
sns.histplot(data=df_combined_C3M2O1M2.iloc[:,22])
#matplotlib.pyplot.xticks(rotation=45, ha='right')
matplotlib.pyplot.suptitle('Histogram - Epoch size 30s for all plots')
matplotlib.pyplot.tight_layout()

matplotlib.pyplot.subplot(2,3,4)
sns.histplot(data=df_combined_C3M2O1M2.iloc[:,23])
#matplotlib.pyplot.xticks(rotation=45, ha='right')
matplotlib.pyplot.suptitle('Histogram - Epoch size 30s for all plots')
matplotlib.pyplot.tight_layout()


matplotlib.pyplot.subplot(2,3,5)
sns.histplot(data=df_combined_C3M2O1M2.iloc[:,24])
#matplotlib.pyplot.xticks(rotation=45, ha='right')
matplotlib.pyplot.suptitle('Histogram - Epoch size 30s for all plots')
matplotlib.pyplot.tight_layout()

matplotlib.pyplot.show()

'''




print('Done')