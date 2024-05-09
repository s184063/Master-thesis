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
sys.path.insert(0, '/home/users/s184063')
#import mne

# Import My_functions_script
from My_functions_script_RBD import preprocessing, extract_numbers_from_filename, extract_letters_and_numbers, list_files_in_folder, split_string_by_length, Usleep_2channels, correlation_multiple_electrodes

# Made by Natasja Bonde Andersen 25.04.2024 

##### Algorithm description ##########
# 1) Calculate median and std and (and median + std for spectrum) for full signal and the full dataset (the settings and amplitudes may differ significantly between datasets)

# 2) Loop over 30 sec epochs and check if the signal deviates significantly from the overall median and std for time signal and spectrum

# 3) make an array to mark of the signal was within limit or not (True, False)

# 4) Evaluate if the channel should be excluded
# Use method: 
# Between electrode comparison using this formula: 
# bad channel threshold = 5% + factor*min(percent for all electrodes for one patient)

# Plot the bad signals and the period to check 

##########################################

# Input path 
path_edf=r'/scratch/users/s184063/RBD_Restructure_firsttry/'

# Looping over all EDF files in folder 
edf_files_list = list_files_in_folder(path_edf)

edf_files_list=sorted(edf_files_list)

# Defining temporary variables 
temp_median=[]
temp_std=[]
temp_spectrum_median=[]
temp_spectrum_std=[]

print("EDF files in the folder:")
for edf_file in edf_files_list:
    print(edf_file)

    # PatientID
    file_temp=str(edf_file) 
    filename=file_temp[48:] # /scratch/users/s184063/RBD_Restructure_firsttry/restructuredfile_RBD_82001_(1).EDF
    print('Filename')
    print(filename)
  
    signals, signal_headers, header = plib.highlevel.read_edf(edf_file)


    # Step 1 
    # Calculate average and std for full signal and the full dataset (the settings and amplitudes may differ significantly between datasets)

    for k in range(len(signal_headers)):

        signals_for_calculations=signals[k]
        
        signals_median=np.median(signals_for_calculations)
        signals_std=np.std(signals_for_calculations)
        spectrum_signals=scipy.fft.fft(signals_for_calculations)
        spectrum_median=np.median(spectrum_signals)
        spectrum_std=np.std(spectrum_signals)

        # Saving the values for later calulations
        temp_median.append(signals_median)
        median_structure= np.stack(temp_median ,axis=0)
        
        temp_std.append(signals_std)
        std_structure=np.stack(temp_std,axis=0)

        temp_spectrum_median.append(spectrum_median)
        spectrum_structure_median=np.stack(temp_spectrum_median,axis=0)

        temp_spectrum_std.append(spectrum_std)
        spectrum_structure_std=np.stack(temp_spectrum_std,axis=0)

# All channels are mixed to calculate this measure (which is ok)
print('Median structure')
print(median_structure.shape)
print(median_structure)
overall_median=np.median(median_structure)
print('Overall median')
print(overall_median)

print('Std structure')
print(std_structure.shape)
print(std_structure)
overall_std=np.std(std_structure)
print('Overall Std ')
print(overall_std)

print('Spectrum structure median and std')
print(spectrum_structure_median.shape)
print(spectrum_structure_median)
print(spectrum_structure_std.shape)
print(spectrum_structure_std)


overall_spectrum_median=np.median(spectrum_structure_median)
overall_spectrum_std=np.std(spectrum_structure_std)
print('Overall spectrum median')
print(overall_spectrum_median)
print('Overall spectrum std')
print(overall_spectrum_std)


temp_bad_signal=[]


'''

# Step 2 
# Loop over 30 sec epochs and check if the signals deviates with a 
# chosen factor from the overall median and std of the time signal and spectrum
'''
# Input path 
#path_edf=r'/scratch/users/s184063/RBD_onepatient/'

# Looping over all EDF files in folder 
#edf_files_list = list_files_in_folder(path_edf)

#edf_files_list=sorted(edf_files_list)
'''
print("EDF files in the folder:")
for edf_file in edf_files_list:
    print(edf_file)

    # PatientID
    file_temp=str(edf_file) 
    filename=file_temp[48:] # /scratch/users/s184063/RBD_Restructure_firsttry/restructuredfile_RBD_82001_(1).EDF
    print('Filename')
    print(filename)

  
    signals, signal_headers, header = plib.highlevel.read_edf(edf_file)

    temp_percentage=[]
    temp_electrodes=[]
    
    # Looping over the EDF signals within a patient 
    for k in range(len(signal_headers)):
        
        
        time_signal=signals[k]
     
       
        signal_header_of_interest=signal_headers[k]
        Electrode=signal_header_of_interest['label']
        Electrode=extract_letters_and_numbers(Electrode)
        Electrode=f'{Electrode}'
        print('Electrode under investigation')
        print(Electrode)
        print(type(Electrode))
        temp_electrodes.append(Electrode)
        Electrodes_in_data=np.stack(temp_electrodes,axis=0)

        
        # calculate the amount of samples that would be 30 seconds 
        fs=signal_header_of_interest['sample_frequency']#Hz Extracting true sampling frequency 
        fs=int(fs)
        print('Sampling frequency')
        print(type(fs))
        print(fs)
        time_30s_calculate=fs*30 # fs is the sampling frequency (number of samples per second) and we would like the amount of samples for 30 seconds
        idx_30=time_30s_calculate
        print('Idx_30')
        print(idx_30)

        temp_out_of_range=[]      
                
        # Indexing for 30 seconds intervals using the amount of samples for 30 seconds 'idx_30'
        for i in range(0,len(time_signal),time_30s_calculate):
                    
            # indexing in the signal for 30 second intervals 
            intervals_30_signal1=time_signal[i:i+idx_30] # channel 1
            
                    
            # Step 2) 
            # Calculating median for 30 second intervals 
            median_30sec = np.median(intervals_30_signal1)
            std_30sec = np.std(intervals_30_signal1)

            # Spectrum of 30 sec signals
            spectrum_30sec=scipy.fft.fft(intervals_30_signal1)
            spectrum_median_30sec=np.median(spectrum_30sec)
            spectrum_std_30sec=np.std(spectrum_30sec)

            # Checking if the signal are out of range 
            if 50*overall_median <= median_30sec and 4*overall_std <= std_30sec:
                # The factors multiplied on the median and std are found by trial and error 
                # This part checks for large positive amplitudes in the signal 
                out_of_range='True'
                temp_out_of_range.append(out_of_range)
                Outlier=np.stack(temp_out_of_range,axis=0)
                print('Normal median and std was used to detect an outlier')

            elif abs(spectrum_median_30sec) < 0.01*overall_spectrum_median and abs(spectrum_std_30sec) < 0.01*overall_spectrum_std:
                #Checking for flatlines 
                # The factors multiplied on the median and std are found by trial and error 
                out_of_range='True'
                temp_out_of_range.append(out_of_range)
                Outlier=np.stack(temp_out_of_range,axis=0)
                print('Spectrum was used to detect an outlier')
            else: 
                out_of_range='False'
                temp_out_of_range.append(out_of_range)
                Outlier=np.stack(temp_out_of_range,axis=0)

        # This is printed for each signal in the EDF file 
        print('Outlier variable')
        print(Outlier.shape)
        print(Outlier)

        print('Electrodes in data')
        print(Electrodes_in_data)
        print(type(Electrodes_in_data))

        #### Check how much of the signal are outliers ######
        # Find the indexes where the value is 'True'
        indices = np.where(Outlier == 'True')
    
        amount_of_outliers=len(indices[0])
        print('The amount of outlier 30sec signals')
        print(amount_of_outliers)

        print('Dimensions of outliers - full length')
        print(Outlier.shape[0])
        
        # Calculating the percent of the signal, which consists of outliers 
        Full_signal_length=Outlier.shape[0]
        
        percentage_of_signal=(amount_of_outliers/Full_signal_length)*100
        print('Percent of the signal that is outliers')
        print(percentage_of_signal)

        # Collecting the percentage for each signal within a patient
        temp_percentage.append(percentage_of_signal)
        percent_stacked=np.stack(temp_percentage,axis=0)
        

        # Plotting the pieces of signal out of range 
        timefs= np.arange(0, time_30s_calculate/fs, 1/fs)
        print('Length time axis for plotting')
        print(len(timefs))
        
        signal_index=indices[0]*30
        print(signal_index)

        
        #for loop_factor in signal_index:
        #    fig = matplotlib.pyplot.figure()
        #    fig.suptitle(f'Outlier detection {loop_factor}')
        #    matplotlib.pyplot.plot(timefs, time_signal[loop_factor:loop_factor+len(timefs)], 'k')
        #    #matplotlib.pyplot.title("After resampling")
        #    matplotlib.pyplot.xlabel('Time [s]') 
        #    matplotlib.pyplot.ylabel('Amplitude [muV]') 
        #    matplotlib.pyplot.savefig(f'/scratch/users/s184063/RBD subsample/Outlier_number_{loop_factor}.png')
        #    # Clearing plot
        #    matplotlib.pyplot.clf()
        
    
        print('New signal and the outlier variable is deleted')


        del Outlier, temp_out_of_range, 
        
    # Clearing the variables, so they only collect data for one patient at a time 
    del temp_electrodes, temp_percentage

    # Looking into the variables collected for one patient 
    print('Percentage for patient - full structure and minimum value')
    print(percent_stacked)
    print(np.min(percent_stacked))

    print('Electrodes')
    print(Electrodes_in_data)
    

    #### Detemining if the channel is too bad and should be excluded ###############
    # Between electrode comparison using this formula: 
    # bad channel threshold = 5% + factor*min(percent for all electrodes for one patient)
    factor=3
    bad_channel_threshold=(5+factor*np.min(percent_stacked)) # This is in percent 

    print('Bad channel threshold')
    print(bad_channel_threshold)

    # Comparing the percent per channel with the 'bad_channel_threshold'
    # If the channels are above the threshold they are excluded from the study. 
    # The excluded electrodes are noted down to a list for removal later. 
    for l in range(len(percent_stacked)):
        if percent_stacked[l] > bad_channel_threshold:
            print('Bad channel - exclude it from the study')
            print('Electrode - bad')
            print(Electrodes_in_data[l])
            # Noting the bad signal and the electrode 

            label=Electrodes_in_data[l]
            print(label)
            bad_signals=f'file:{filename}_Electrode:{label}'
            print('Generating bad signals names')
            print(bad_signals)

            temp_bad_signal.append(bad_signals)
            bad_signals_structure=np.stack(temp_bad_signal,axis=0)  
            bad_channels_for_csv={'Bad signals':bad_signals_structure}


        else: 
            print('No bad channels for this patient')
            print(filename)
    

if 'bad_channels_for_csv' in locals():
    full_dataframe=pd.DataFrame(bad_channels_for_csv)
    full_dataframe.to_csv('/scratch/users/s184063/RBD_Features/Outliers_RBD.csv',index=False)
else: 
    print('No bad channels where found in this dataset')

'''
print('Done')













