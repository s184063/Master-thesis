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
import seaborn as sns
# Using sys function to import 'My_functions_script'
sys.path.insert(0, '/home/users/s184063')
# Import My_functions_script
from My_functions_script_RBD import extract_numbers_from_filename, extract_letters_and_numbers, list_files_in_folder, split_string_by_length, Usleep_2channels, correlation_multiple_electrodes

################################################################################################


####### Setting paths ##################################
# File paths 
input_path =r'/scratch/users/s184063/RBD_Restructure_firsttry/'
#input_file=f'A0001_4 165907.EDF' # Easier later for namechange according to the patient ID 
#input_file_path = os.path.join(input_path,input_file)
output_path = r'/scratch/users/s184063/hypnograms_RBD_firsttry/' 
#######################################################


'''
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

    # For naming in the RBD dataset include both names and numbers
    #numbers_found = extract_numbers_from_filename(edf_file)
    #numbers_found = extract_letters_and_numbers(edf_file)
    numbers_found = os.path.basename(edf_file)
    print("Numbers found in the filename:", numbers_found)

    filename, file_extension = os.path.splitext(numbers_found)

    print('filename')
    print(filename)

    numbers_found=filename

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

epoch_size_in_seconds = [1, 3, 5, 15, 30]

for epoch in epoch_size_in_seconds:
    # The path loaded in should be the one, where the hypnograms are stored 

    correlation_multiple_electrodes(output_path,epoch)


# The Pearson (product-moment) correlation coefficient is a measure of the linear relationship between two features.
# Pearson correlation coefficient can take on any real value in the range −1 ≤ r ≤ 1.
#########################################


########### Part 3 #################
# In this part of the code all CSV files with correlation features will be merged to one large CSV file and dataframe
# Age, sex, BMI, label for disease and cohort will be extracted from a major CSV file from a previous study. 
# In the end a major file for the RBD cohort will be gathered and ready for modelling. 

# Combining CSV files 

# Loading CSV files for 2-electrode data for all epocs
E6_epoc1=pd.read_csv('/scratch/users/s184063/RBD_Features/Correlation_6E_epocssize_1_RBD.csv')
E6_epoc3=pd.read_csv('/scratch/users/s184063/RBD_Features/Correlation_6E_epocssize_3_RBD.csv')
E6_epoc5=pd.read_csv('/scratch/users/s184063/RBD_Features/Correlation_6E_epocssize_5_RBD.csv')
E6_epoc15=pd.read_csv('/scratch/users/s184063/RBD_Features/Correlation_6E_epocssize_15_RBD.csv')
E6_epoc30=pd.read_csv('/scratch/users/s184063/RBD_Features/Correlation_6E_epocssize_30_RBD.csv')


# Dropping the first patienID folder for most of the files in order to make a concatenation of all files 
E6_epoc3=E6_epoc3.drop('PatientID',axis=1)
E6_epoc5=E6_epoc5.drop('PatientID',axis=1)
E6_epoc15=E6_epoc15.drop('PatientID',axis=1)
E6_epoc30=E6_epoc30.drop('PatientID',axis=1)


# Combining the dataframes. Only one column contains patientID
df_6E=pd.concat([E6_epoc1,E6_epoc3,E6_epoc5,E6_epoc15,E6_epoc30],axis=1)
df_6E.to_csv('/scratch/users/s184063/RBD_Features/Correlation_6E_RBD_all_epocs.csv')




# Extract age, sex, and diagnosis label (Dx)
demographics=pd.read_csv('/scratch/users/s184063/during_v2.csv')
print('demographic features for use')
Demographic_features_for_use=demographics[['DTU ID','Age in 2021','Gender','Diagnosis']] # Dx is the label (RBD or control)
print(Demographic_features_for_use)



# Extracting patient ID's from the combined CSV file 
# They are used to match the demographic_features_for_use
patientID=df_6E['PatientID'] # Extracting patientID's 
print('Length of patientID list')
print(len(patientID))


# Create a temporary dataframe with patientID and cohort, to extract the demographic information 
# using same names in order to merge later 
temporary_df=pd.DataFrame({'DTU ID':patientID})
print('Temporary df')
print(temporary_df)

# Merging and extratcing the features with overlapping patientIDs and cohorts 
result_df=pd.merge(Demographic_features_for_use,temporary_df,how='inner',on=['DTU ID'])
print('Results df - first time')
print(result_df)

# Changing name back to 'patientID'
result_df=result_df.rename(columns={'DTU ID':'PatientID'})
print('Results df - After renaming')
print(result_df)


# Final document containing all features of interest including the correlation features for RBD FHC
result_final_df=pd.merge(result_df,df_6E,how='inner',on=['PatientID'])
result_final_df.to_csv('/scratch/users/s184063/RBD_Features/All_features_combined_RBD.csv')

print('Results df - final')
print(result_final_df)




################## Statistical investigation of features ###################


#Loading data frame 
df_combined=pd.read_csv('/scratch/users/s184063/RBD_Features/All_features_combined_RBD.csv')



print('All patients - patientID included')
print(df_combined)
df_combined=df_combined.drop(['PatientID'],axis=1)
df_combined=df_combined.iloc[:,1:381]

print('Interested in this one')
print(df_combined)




##### Correlation matrix between features #########

# form correlation matrix
import matplotlib.pyplot as plt

# Calculate correlation matrix
matrix = df_combined.corr()

# Print correlation matrix and save to a CSV
matrix.to_csv('/scratch/users/s184063/RBD_Features/Matrix_correlation_RBD.csv')

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
plt.title('RBD - Correlation matrix of correlated hypnodensity features')

# Adjust the padding
plt.tight_layout()

# Display the plot
plt.savefig('/scratch/users/s184063/RBD_Features/Correlation_matrix_RBD.png')

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

'''
'''
#### Standard statistics #####
mean_list=[]
std_list=[]
min_list=[]
max_list=[]



print(df_combined.shape[1])
print(df_combined.shape[0])
print(df_combined.shape)

temp=df_combined.drop('Gender',axis=1)
temp=temp.drop('Diagnosis',axis=1)
print(temp)
for j in range(temp.shape[1]): # running over number of columns 
    chosen_signal=temp.iloc[:,j]

    print(j)
   

    #mean of each column 
    mean= np.mean(chosen_signal)
    mean_list.append(mean)
    mean_stack=np.stack(mean_list)

    # std of each column 
    std= np.std(chosen_signal)
    std_list.append(std)
    std_stack=np.stack(std_list)

    # min of each column 
    min_val= np.min(chosen_signal)
    min_list.append(min_val)
    min_stack=np.stack(min_list)

    # max of each column 
    max_val= np.max(chosen_signal)
    max_list.append(max_val)
    max_stack=np.stack(max_list)


# Collecting values calculated 
mean=pd.DataFrame({'mean':mean_stack})
mean.to_csv('/scratch/users/s184063/RBD_Features/Mean values.csv')
std=pd.DataFrame({'std':std_stack})
std.to_csv('/scratch/users/s184063/RBD_Features/Std values.csv')
min_val=pd.DataFrame({'minimum':min_stack})
min_val.to_csv('/scratch/users/s184063/RBD_Features/Min values.csv')
max_val=pd.DataFrame({'max':max_stack})
max_val.to_csv('/scratch/users/s184063/RBD_Features/Max values.csv')


print(temp)
'''

########## Scatter plots ##############
######## RBD #################

Electrodes=['F3M2','F4M1','C3M2','C4M1','O1M2','O2M1']

iterable=Electrodes
r=2 # Length of subsequence 
E_combinations=list(itertools.combinations(iterable,r))

print('Combinations of electrode names in file')
print(E_combinations)
print(type(E_combinations))
print(len(E_combinations))


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



    # All single features vs. age
    wake=['Wake_'+str(Electrode_combination)+'_epocssize_1', 'Wake_'+str(Electrode_combination)+'_epocssize_3', 'Wake_'+str(Electrode_combination)+'_epocssize_5', 'Wake_'+str(Electrode_combination)+'_epocssize_15', 'Wake_'+str(Electrode_combination)+'_epocssize_30']
    plt.subplot(2, 3, 1) # (dimension1, dimension2, number of plots)
    for i in wake:
        # Wake   
        plt.scatter(np.array(df_combined['Age in 2021']),np.array(df_combined[i]))
    plt.xlabel('Age in 2021')
    plt.ylabel('Wake all epochs')
    plt.legend(['1s','3s','5s','15s','30s'])
    plt.tight_layout()



    # N1   
    plt.subplot(2, 3, 2) # (dimension1, dimension2, number of plots)
    N1=['N1_'+str(Electrode_combination)+'_epocssize_1', 'N1_'+str(Electrode_combination)+'_epocssize_3', 'N1_'+str(Electrode_combination)+'_epocssize_5', 'N1_'+str(Electrode_combination)+'_epocssize_15', 'N1_'+str(Electrode_combination)+'_epocssize_30']
    for j in N1:
        plt.scatter(np.array(df_combined['Age in 2021']),np.array(df_combined[j]))
        #print(df_combined[j])
    plt.xlabel('Age in 2021')
    plt.ylabel('N1 all epochs')
    plt.legend(['1s','3s','5s','15s','30s'])
    plt.tight_layout()

    # N2
    plt.subplot(2, 3, 3) # (dimension1, dimension2, number of plots)

    N2 =['N2_'+str(Electrode_combination)+'_epocssize_1', 'N2_'+str(Electrode_combination)+'_epocssize_3', 'N2_'+str(Electrode_combination)+'_epocssize_5', 'N2_'+str(Electrode_combination)+'_epocssize_15', 'N2_'+str(Electrode_combination)+'_epocssize_30']
    for k in N2:
        plt.scatter(np.array(df_combined['Age in 2021']),np.array(df_combined[k]))
    plt.xlabel('Age in 2021')
    plt.ylabel('N2 all epochs')
    plt.legend(['1s','3s','5s','15s','30s'])
    plt.tight_layout()

    # N3   
    N3 = ['N3_'+str(Electrode_combination)+'_epocssize_1', 'N3_'+str(Electrode_combination)+'_epocssize_3', 'N3_'+str(Electrode_combination)+'_epocssize_5', 'N3_'+str(Electrode_combination)+'_epocssize_15', 'N3_'+str(Electrode_combination)+'_epocssize_30']
    for h in N3:
        plt.subplot(2, 3, 4) # (dimension1, dimension2, number of plots)
        plt.scatter(np.array(df_combined['Age in 2021']),np.array(df_combined[h]))
    plt.xlabel('Age in 2021')
    plt.ylabel('N3 all epochs')
    plt.legend(['1s','3s','5s','15s','30s'])
    plt.tight_layout()
    plt.tight_layout()

    # REM
    REM = ['REM_'+str(Electrode_combination)+'_epocssize_1', 'REM_'+str(Electrode_combination)+'_epocssize_3', 'REM_'+str(Electrode_combination)+'_epocssize_5', 'REM_'+str(Electrode_combination)+'_epocssize_15', 'REM_'+str(Electrode_combination)+'_epocssize_30']
    for dd in REM:
        plt.subplot(2, 3, 5) # (dimension1, dimension2, number of plots)
        plt.scatter(np.array(df_combined['Age in 2021']),np.array(df_combined[dd]))
    plt.xlabel('Age in 2021')
    plt.ylabel('REM all epochs')
    plt.legend(['1s','3s','5s','15s','30s'])
    plt.tight_layout()
    plt.suptitle(" RBD - Scatterplot of sleep stages vs. age - "+str(Electrode_combination))
    plt.tight_layout()
    
    print('Figure was saved')
    plt.savefig('/scratch/users/s184063/RBD_Features/Scatter_RBD_'+str(Electrode_combination)+'.png')

    plt.clf()
    del wake, N1, N2, N3, REM, Electrode_combination


########## box plot ########

#### RBD ##############
Electrodes=['F3M2','F4M1','C3M2','C4M1','O1M2','O2M1']

iterable=Electrodes
r=2 # Length of subsequence 
E_combinations=list(itertools.combinations(iterable,r))

print('Combinations of electrode names in file')
print(E_combinations)
print(type(E_combinations))
print(len(E_combinations))


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

    
    boxplot_features=['Wake_'+str(Electrode_combination)+'_epocssize_1', 'N1_'+str(Electrode_combination)+'_epocssize_1','N2_'+str(Electrode_combination)+'_epocssize_1','N3_'+str(Electrode_combination)+'_epocssize_1','REM_'+str(Electrode_combination)+'_epocssize_1','Wake_'+str(Electrode_combination)+'_epocssize_3', 'N1_'+str(Electrode_combination)+'_epocssize_3','N2_'+str(Electrode_combination)+'_epocssize_3','N3_'+str(Electrode_combination)+'_epocssize_3','REM_'+str(Electrode_combination)+'_epocssize_3','Wake_'+str(Electrode_combination)+'_epocssize_5', 'N1_'+str(Electrode_combination)+'_epocssize_5','N2_'+str(Electrode_combination)+'_epocssize_5','N3_'+str(Electrode_combination)+'_epocssize_5','REM_'+str(Electrode_combination)+'_epocssize_5','Wake_'+str(Electrode_combination)+'_epocssize_15', 'N1_'+str(Electrode_combination)+'_epocssize_15','N2_'+str(Electrode_combination)+'_epocssize_15','N3_'+str(Electrode_combination)+'_epocssize_15','REM_'+str(Electrode_combination)+'_epocssize_15','Wake_'+str(Electrode_combination)+'_epocssize_30', 'N1_'+str(Electrode_combination)+'_epocssize_30','N2_'+str(Electrode_combination)+'_epocssize_30','N3_'+str(Electrode_combination)+'_epocssize_30','REM_'+str(Electrode_combination)+'_epocssize_30']
    plt.figure(figsize=(8,6)) # Adjust these numbers as per your requirement.
    sns.boxplot(data=df_combined[boxplot_features])
    plt.xticks(rotation=45, ha='right')
    plt.title('Correlation features - RBD [Wake, N1, N2, N3, REM] for all epochs '+str(Electrode_combination))
    plt.tight_layout()

    print('Figure was saved')
    plt.savefig('/scratch/users/s184063/RBD_Features/Boxplot_RBD_'+str(Electrode_combination)+'.png')
    
    plt.clf()
    del boxplot_features


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