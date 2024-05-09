# Load standard packages
import numpy as np
import os
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
# Load wavelet package 
import pywt 
from pywt import wavedec
import sys
import itertools 
from itertools import combinations
# Using sys function to import 'My_functions_script'
sys.path.insert(0, '/home/users/s184063')

# Import My_functions_script
from My_functions_script import list_files_in_folder, extract_numbers_from_filename, extract_letters_and_numbers, split_string_by_length


# Get indices for the EEG files - the names of electrodes has to be known 
def get_indices(data, labels):
    indices = {label: None for label in labels}
    for i, d in enumerate(data):
        if d['label'] in labels:
            indices[d['label']] = i
            
    return indices


error_dict = []

# File paths 
input_path =r"/oak/stanford/groups/mignot/psg/FHC_France/all/" # select correct path and folder


# Make output path to scratch !!!!!



# Looping over all EDF files in folder 
edf_files_list_unsorted = list_files_in_folder(input_path)

print('Unsorted edf files')
print(edf_files_list_unsorted)

edf_files_list=sorted(edf_files_list_unsorted)

del edf_files_list_unsorted

print('Sorted edf list')
print(edf_files_list)


print("EDF files in the folder:")
for edf_file in edf_files_list:
    print(edf_file)

    try: 
    

        # Loading EDF file and extracting signals, signal headers and headers for the file
        signals, signal_headers, header = plib.highlevel.read_edf(edf_file)


        print('Information about original edf file')
        print('Signal header')
        print(signal_headers)
        invest=signal_headers[0]

        print('invest signalheader physical_min')
        print(invest['physical_min'])

        print('indexed type')
        print(type(signal_headers[0]))
        print('shape')
        print(len(signal_headers))

        print('Signals')
        print(type(signals))
        print('indexed type')
        print(type(signals[0]))
        print('shape')
        print(signals.shape)



        # Extracting patientID and visiting number
        numbers_found = os.path.basename(edf_file)
        print("Numbers found in the filename:", numbers_found)

        filename, file_extension = os.path.splitext(numbers_found)

        print('filename')
        print(filename)

        numbers_found=filename

        patientID=filename


        # Save visit number and patient ID in dict here!!!!
        
        output_path='/scratch/users/s184063/France restructure EDF all correct final/' 
        filename_output=f'restructuredfile_France_all_{patientID}.EDF'
        output_file_path=os.path.join(output_path,filename_output)
        print(output_file_path)
        print(type(output_file_path))



        # Extracting the indices for the EEG signals in the data
        # This 'labels_to_find' variable should be corrected for each dataset 
        labels_to_find = ['EEG A1','EEG A2', 'EEG C3','EEG C4','EEG O1','EEG O2','EEG Fp1','EEG Fp2']
        indices = get_indices(signal_headers, labels_to_find)
        #print(indices)
        
        
        
        for label in labels_to_find:
            print(f"The label '{label}' is at index {indices[label]}")
                    
                
            # Extracting the electrode indexes 
            if label=='EEG A1':
                M1_index_trial=indices[label]
                print('M1 label trial')
                print(M1_index_trial)
                print(type(M1_index_trial))
                
                
                if type(M1_index_trial)==int:
                    M1_index = M1_index_trial
                    print('M1_index true')

                    print("M1 variable exists.")
                    M1=signals[M1_index]
                    print('M1 defined')
                    print(type(M1))
                    print(len(M1))

                    
                    
                else:
                    print("The variable M1 does not exist.")
                    

            # Extracting the electrode indexes 
            if label=='EEG A2':
                M2_index_trial=indices[label]
                print('M2 label trial')
                print(M2_index_trial)
                print(type(M2_index_trial))
                
                
                if type(M2_index_trial)==int:
                    M2_index = M2_index_trial
                    print('M2_index true')

                    print("M2 variable exists.")
                    M2=signals[M2_index]
                    print('M2 defined')
                    print(type(M2))
                    print(len(M2))
                    
                else:
                    print("The variable M2 does not exist.")
                    
            # Extracting the electrode indexes 
            if label=='EEG C3':
                C3_index_trial=indices[label]
                print('C3 label trial')
                print(C3_index_trial)
                print(type(C3_index_trial))
                
                
                if type(C3_index_trial)==int:
                    C3_index = C3_index_trial
                    print('C3_index true')

                    print("C3 variable exists.")
                    C3=signals[C3_index]
                    print('C3 defined')
                    print(type(C3))
                    print(len(C3))
                    
                else:
                    print("The variable C3 does not exist.")
            

            # Extracting the electrode indexes 
            if label=='EEG C4':
                C4_index_trial=indices[label]
                print('C4 label trial')
                print(C4_index_trial)
                print(type(C4_index_trial))
                
                
                if type(C4_index_trial)==int:
                    C4_index = C4_index_trial
                    print('C4_index true')

                    print("C4 variable exists.")
                    C4=signals[C4_index]
                    print('C4 defined')
                    print(type(C4))
                    print(len(C4))
                    
                else:
                    print("The variable C4 does not exist.")
                    
            
            # Extracting the electrode indexes 
            if label=='EEG O2':
                O2_index_trial=indices[label]
                print('O2 label trial')
                print(O2_index_trial)
                print(type(O2_index_trial))
                
                
                if type(O2_index_trial)==int:
                    O2_index = O2_index_trial
                    print('O2_index true')

                    print("O2 variable exists.")
                    O2=signals[O2_index]
                    print('O2 defined')
                    print(type(O2))
                    print(len(O2))
                    
                else:
                    print("The variable O2 does not exist.")
                    

            # Extracting the electrode indexes 
            if label=='EEG O1':
                O1_index_trial=indices[label]
                print('O1 label trial')
                print(O1_index_trial)
                print(type(O1_index_trial))
                
                
                if type(O1_index_trial)==int:
                    O1_index = O1_index_trial
                    print('O1_index true')

                    print("O1 variable exists.")
                    O1=signals[O1_index]
                    print('O1 defined')
                    print(type(O1))
                    print(len(O1))


                    
                else:
                    print("The variable O1 does not exist.")


            
            # Extracting the electrode indexes 
            if label=='EEG Fp1':
                Fp1_index_trial=indices[label]
                print('Fp1 label trial')
                print(Fp1_index_trial)
                print(type(Fp1_index_trial))
                
                
                if type(Fp1_index_trial)==int:
                    Fp1_index = Fp1_index_trial
                    print('Fp1_index true')

                    print("Fp1 variable exists.")
                    Fp1=signals[Fp1_index]
                    print('Fp1 defined')
                    print(type(Fp1))
                    print(len(Fp1))

                    
                else:
                    print("The variable Fp1 does not exist.")
                    

                    
            # Extracting the electrode indexes 
            if label=='EEG Fp2':
                Fp2_index_trial=indices[label]
                print('Fp2 label trial')
                print(Fp2_index_trial)
                print(type(Fp2_index_trial))
                
                
                if type(Fp2_index_trial)==int:
                    Fp2_index = Fp2_index_trial
                    print('Fp2_index true')

                    print("Fp2 variable exists.")
                    Fp2=signals[Fp2_index]
                    print('Fp2 defined')
                    print(type(Fp2))
                    print(len(Fp2))

        
                    
                else:
                    print("The variable Fp2 does not exist.")
                    

        # Referencing the electrodes to M1
        if 'M1' in locals() and 'C4' in locals(): 
            print('M1 and C4 exists - now the referenced C4M1 is calculated')      
            
            C4M1=np.subtract(C4,M1)


            # Extract values from old signal header to pack it in the new
            invest_C4_signalheader=signal_headers[C4_index]
            dimension_C4=invest_C4_signalheader['dimension']
            sample_rate_C4=invest_C4_signalheader['sample_rate']
            sample_frequency_C4=invest_C4_signalheader['sample_frequency']
            #physical_min_C4=invest_C4_signalheader['physical_min']
            #physical_max_C4=invest_C4_signalheader['physical_max']
            physical_min_C4=-12000
            physical_max_C4=12000
            digital_min_C4=invest_C4_signalheader['digital_min']
            digital_max_C4=invest_C4_signalheader['digital_max']
            transducer_C4=invest_C4_signalheader['transducer']
            prefiler_C4=invest_C4_signalheader['prefilter']


            print('Physical min')
            print(physical_min_C4)
            print(type(physical_min_C4))

            print('Physical max')
            print(physical_max_C4)
            print(type(physical_max_C4))

            

            

            print('Using function to generate signal header for C4M1')
            new_signal_header_C4M1=pyedflib.highlevel.make_signal_header(label='C4M1', dimension=dimension_C4, sample_rate=sample_rate_C4, sample_frequency=sample_frequency_C4, physical_min=physical_min_C4, physical_max=physical_max_C4, digital_min=digital_min_C4, digital_max=digital_max_C4, transducer=transducer_C4, prefiler=prefiler_C4)
            print(new_signal_header_C4M1)


            del C4


        if 'M1' in locals() and 'O2' in locals(): 
            print('M1 and O2 exists - now the referenced O2M1 is calculated')      

            O2M1=np.subtract(O2,M1)

            
            # Extract values from old signal header to pack it in the new
            invest_O2_signalheader=signal_headers[O2_index]
            dimension_O2=invest_O2_signalheader['dimension']
            sample_rate_O2=invest_O2_signalheader['sample_rate']
            sample_frequency_O2=invest_O2_signalheader['sample_frequency']
            physical_min_O2=-12000
            physical_max_O2=12000
            #physical_min_O2=invest_O2_signalheader['physical_min']
            #physical_max_O2=invest_O2_signalheader['physical_max']
            digital_min_O2=invest_O2_signalheader['digital_min']
            digital_max_O2=invest_O2_signalheader['digital_max']
            transducer_O2=invest_O2_signalheader['transducer']
            prefiler_O2=invest_O2_signalheader['prefilter']


            print('Physical min')
            print(physical_min_O2)
            print(type(physical_min_O2))

            print('Physical max')
            print(physical_max_O2)
            print(type(physical_max_O2))


            print('Using function to generate signal header for O2M1')
            new_signal_header_O2M1=pyedflib.highlevel.make_signal_header(label='O2M1', dimension=dimension_O2, sample_rate=sample_rate_O2, sample_frequency=sample_frequency_O2, physical_min=physical_min_O2, physical_max=physical_max_O2, digital_min=digital_min_O2, digital_max=digital_max_O2, transducer=transducer_O2, prefiler=prefiler_O2)
            print(new_signal_header_O2M1)
            

            del O2


        if 'M1' in locals() and 'Fp2' in locals(): 
            print('M1 and Fp2 exists - now the referenced Fp2M1 is calculated')      

            Fp2M1=np.subtract(Fp2,M1)

            
            # Extract values from old signal header to pack it in the new
            invest_Fp2_signalheader=signal_headers[Fp2_index]
            dimension_Fp2=invest_Fp2_signalheader['dimension']
            sample_rate_Fp2=invest_Fp2_signalheader['sample_rate']
            sample_frequency_Fp2=invest_Fp2_signalheader['sample_frequency']
            #physical_min_Fp2=invest_Fp2_signalheader['physical_min']
            #physical_max_Fp2=invest_Fp2_signalheader['physical_max']
            physical_min_Fp2=-12000
            physical_max_Fp2=12000
            digital_min_Fp2=invest_Fp2_signalheader['digital_min']
            digital_max_Fp2=invest_Fp2_signalheader['digital_max']
            transducer_Fp2=invest_Fp2_signalheader['transducer']
            prefiler_Fp2=invest_Fp2_signalheader['prefilter']


            print('Physical min')
            print(physical_min_Fp2)
            print(type(physical_min_Fp2))

            print('Physical max')
            print(physical_max_Fp2)
            print(type(physical_max_Fp2))

    
            print('Using function to generate signal header for F2M1 --> change to F4M1')
            # Changing name to F2M1 and not Fp2M1
            new_signal_header_Fp2M1=pyedflib.highlevel.make_signal_header(label='F4M1', dimension=dimension_Fp2, sample_rate=sample_rate_Fp2, sample_frequency=sample_frequency_Fp2, physical_min=physical_min_Fp2, physical_max=physical_max_Fp2, digital_min=digital_min_Fp2, digital_max=digital_max_Fp2, transducer=transducer_Fp2, prefiler=prefiler_Fp2)
            print(new_signal_header_Fp2M1)

            del Fp2


        # Referencing electrodes to M2
        
        if 'M2' in locals() and 'O1' in locals(): 
            print('M2 and O1 exists - now the referenced O1M2 is calculated')      

            O1M2=np.subtract(O1,M2)

            
            #Extract values from old signal header to pack it in the new
            invest_O1_signalheader=signal_headers[O1_index]
            dimension_O1=invest_O1_signalheader['dimension']
            sample_rate_O1=invest_O1_signalheader['sample_rate']
            sample_frequency_O1=invest_O1_signalheader['sample_frequency']
            #physical_min_O1=invest_O1_signalheader['physical_min']
            #physical_max_O1=invest_O1_signalheader['physical_max']
            physical_min_O1=-12000
            physical_max_O1=12000
            digital_min_O1=invest_O1_signalheader['digital_min']
            digital_max_O1=invest_O1_signalheader['digital_max']
            transducer_O1=invest_O1_signalheader['transducer']
            prefiler_O1=invest_O1_signalheader['prefilter']


            print('Physical min')
            print(physical_min_O1)
            print(type(physical_min_O1))

            print('Physical max')
            print(physical_max_O1)
            print(type(physical_max_O1))            

            print('Using function to generate signal header for O1M2')
            new_signal_header_O1M2=pyedflib.highlevel.make_signal_header(label='O1M2', dimension=dimension_O1, sample_rate=sample_rate_O1, sample_frequency=sample_frequency_O1, physical_min=physical_min_O1, physical_max=physical_max_O1, digital_min=digital_min_O1, digital_max=digital_max_O1, transducer=transducer_O1, prefiler=prefiler_O1)
            print(new_signal_header_O1M2)

            del O1


        if 'M2' in locals() and 'C3' in locals(): 
            print('M2 and C3 exists - now the referenced C3M2 is calculated')      

            C3M2=np.subtract(C3,M2)

            
            #Extract values from old signal header to pack it in the new
            invest_C3_signalheader=signal_headers[C3_index]
            dimension_C3=invest_C3_signalheader['dimension']
            sample_rate_C3=invest_C3_signalheader['sample_rate']
            sample_frequency_C3=invest_C3_signalheader['sample_frequency']
            #physical_min_C3=invest_C3_signalheader['physical_min']
            #physical_max_C3=invest_C3_signalheader['physical_max']
            physical_min_C3=-12000
            physical_max_C3=12000
            digital_min_C3=invest_C3_signalheader['digital_min']
            digital_max_C3=invest_C3_signalheader['digital_max']
            transducer_C3=invest_C3_signalheader['transducer']
            prefiler_C3=invest_C3_signalheader['prefilter']


            print('Physical min')
            print(physical_min_C3)
            print(type(physical_min_C3))

            print('Physical max')
            print(physical_max_C3)
            print(type(physical_max_C3))


            print('Using function to generate signal header for C3M2')
            new_signal_header_C3M2=pyedflib.highlevel.make_signal_header(label='C3M2', dimension=dimension_C3, sample_rate=sample_rate_C3, sample_frequency=sample_frequency_C3, physical_min=physical_min_C3, physical_max=physical_max_C3, digital_min=digital_min_C3, digital_max=digital_max_C3, transducer=transducer_C3, prefiler=prefiler_C3)
            print(new_signal_header_C3M2)

            del C3


        if 'M2' in locals() and 'Fp1' in locals(): 
            print('M2 and Fp1 exists - now the referenced Fp1M2 is calculated')   

               

            Fp1M2=np.subtract(Fp1,M2)

            
            #Extract values from old signal header to pack it in the new
            invest_Fp1_signalheader=signal_headers[Fp1_index]
            dimension_Fp1=invest_Fp1_signalheader['dimension']
            sample_rate_Fp1=invest_Fp1_signalheader['sample_rate']
            sample_frequency_Fp1=invest_Fp1_signalheader['sample_frequency']
            #physical_min_Fp1=invest_Fp1_signalheader['physical_min']
            #physical_max_Fp1=invest_Fp1_signalheader['physical_max']
            physical_min_Fp1=-12000
            physical_max_Fp1=12000
            digital_min_Fp1=invest_Fp1_signalheader['digital_min']
            digital_max_Fp1=invest_Fp1_signalheader['digital_max']
            transducer_Fp1=invest_Fp1_signalheader['transducer']
            prefiler_Fp1=invest_Fp1_signalheader['prefilter']


            print('Physical min')
            print(physical_min_Fp1)
            print(type(physical_min_Fp1))

            print('Physical max')
            print(physical_max_Fp1)
            print(type(physical_max_Fp1))

            print('Using function to generate signal header for Fp1M2 --> changed to F3M2')
            new_signal_header_Fp1M2=pyedflib.highlevel.make_signal_header(label='F3M2', dimension=dimension_Fp1, sample_rate=sample_rate_Fp1, sample_frequency=sample_frequency_Fp1, physical_min=physical_min_Fp1, physical_max=physical_max_Fp1, digital_min=digital_min_Fp1, digital_max=digital_max_Fp1, transducer=transducer_Fp1, prefiler=prefiler_Fp1)
            print(new_signal_header_Fp1M2)

            del Fp1
                    



        # Packing the EDF files 
        if 'Fp1M2' in locals() and 'Fp2M1' in locals() and 'C3M2' in locals() and 'C4M1' in locals()and 'O1M2' in locals() and 'O2M1' in locals(): 
            
            print('The combination: Fp1M2, Fp2M1, C3M2, C4M1, O1M2, O2M1 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            signals_matrix_6E=np.vstack((Fp1M2,Fp2M1,C3M2,C4M1,O1M2,O2M1))
            print('Signal matrix - 6 electrodes referenced')
            print(type(signals_matrix_6E))
            print(signals_matrix_6E.shape)
            
            print('Signal header type')
            new_signal_header_6E=[new_signal_header_Fp1M2,new_signal_header_Fp2M1,new_signal_header_C3M2,new_signal_header_C4M1, new_signal_header_O1M2,new_signal_header_O2M1]
            print(new_signal_header_6E)
            print(len(new_signal_header_6E))
            
            pyedflib.highlevel.write_edf(output_file_path,signals_matrix_6E, new_signal_header_6E) 

            del Fp1M2, Fp2M1, C3M2, C4M1, O1M2, O2M1, M1, M2


        
        if 'C3M2' in locals() and 'C4M1' in locals() and 'O1M2' in locals() and 'O2M1' in locals(): 
            
            print('The combination: C3M2, C4M1, O1M2, O2M1 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            signals_matrix_4E=np.vstack((C3M2,C4M1,O1M2,O2M1))
            print('Signal matrix - 4 electrodes referenced')
            print(type(signals_matrix_4E))
            print(signals_matrix_4E.shape)
            
            print('Signal header type')
            new_signal_header_4E=[new_signal_header_C3M2,new_signal_header_C4M1, new_signal_header_O1M2,new_signal_header_O2M1]
            print(new_signal_header_4E)
            print(len(new_signal_header_4E))
            
            pyedflib.highlevel.write_edf(output_file_path,signals_matrix_4E, new_signal_header_4E) 

            del C3M2, C4M1, O1M2, O2M1, M1, M2

        
        if 'C3M2' in locals() and 'C4M1' in locals() and 'O2M1' in locals(): 
            
            print('The combination: C3M2, C4M1, O2M1 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            signals_matrix_3E=np.vstack((C3M2,C4M1,O2M1))
            print('Signal matrix - 3 electrodes referenced')
            print(type(signals_matrix_3E))
            print(signals_matrix_3E.shape)
            
            print('Signal header type')
            new_signal_header_3E=[new_signal_header_C3M2,new_signal_header_C4M1,new_signal_header_O2M1]
            print(new_signal_header_3E)
            print(len(new_signal_header_3E))
            
            pyedflib.highlevel.write_edf(output_file_path,signals_matrix_3E, new_signal_header_3E) 

            del C3M2, C4M1, O2M1, M1, M2

        
        if 'C3M2' in locals() and 'C4M1' in locals() and 'O1M2' in locals(): 
            
            print('The combination: C3M2, C4M1, O1M2 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            signals_matrix_3E=np.vstack((C3M2,C4M1,O1M2))
            print('Signal matrix - 3 electrodes referenced')
            print(type(signals_matrix_3E))
            print(signals_matrix_3E.shape)
            
            print('Signal header type')
            new_signal_header_3E=[new_signal_header_C3M2,new_signal_header_C4M1, new_signal_header_O1M2]
            print(new_signal_header_3E)
            print(len(new_signal_header_3E))
            
            pyedflib.highlevel.write_edf(output_file_path,signals_matrix_3E, new_signal_header_3E) 

            del C3M2, C4M1, O1M2, M1, M2


        if 'C3M2' in locals() and 'C4M1' in locals(): 
            
            print('The combination: C3M2, C4M1 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            signals_matrix_2E=np.vstack((C3M2,C4M1))
            print('Signal matrix - 2 electrodes referenced')
            print(type(signals_matrix_2E))
            print(signals_matrix_2E.shape)
            
            print('Signal header type')
            new_signal_header_2E=[new_signal_header_C3M2,new_signal_header_C4M1]
            print(new_signal_header_2E)
            print(len(new_signal_header_2E))
            
            pyedflib.highlevel.write_edf(output_file_path,signals_matrix_2E, new_signal_header_2E) 

            del C3M2, C4M1, M1, M2

        
        # Delete variables in locals - to prevent errors

        if 'signals' in locals(): 
            del signals
        
        if 'signalheaders' in locals(): 
            del signalheaders

        if 'headers' in locals():
            del headers

        if 'C3' in locals(): 
            del C3
        
        if 'C4' in locals(): 
            del C4

        if 'O1' in locals():
            del O1

        if 'O2' in locals():
            del O2

        if 'M1' in locals():
            del M1

        if 'M2' in locals():
            del M2
        
        if 'Fp1' in locals ():
            del Fp1

        if 'Fp2' in locals():
            del Fp2

        if 'Fp1M2' in locals(): 
            del Fp1M2

        if 'Fp2M1' in locals():
            del Fp2M1
        
        if 'C3M2' in locals():
            del C3M2
            
            
        if 'C4M1' in locals():
            
            del C4M1
            
        if 'O1M2' in locals():
            del O1M2
        
        if 'O2M1' in locals():
            del O2M1
        
    
    except Exception as e:
        print(f"An error occurred: {e}. Moving on to the next iteration...")
        error_dict.append({'error':e,'filename':edf_file})
        # Delete variables in locals - to prevent errors

        if 'signals' in locals(): 
            del signals
        
        if 'signalheaders' in locals(): 
            del signalheaders

        if 'headers' in locals():
            del headers

        if 'C3' in locals(): 
            del C3
        
        if 'C4' in locals(): 
            del C4

        if 'O1' in locals():
            del O1

        if 'O2' in locals():
            del O2

        if 'M1' in locals():
            del M1

        if 'M2' in locals():
            del M2
        
        if 'Fp1' in locals ():
            del Fp1

        if 'Fp2' in locals():
            del Fp2

        if 'Fp1M2' in locals(): 
            del Fp1M2

        if 'Fp2M1' in locals():
            del Fp2M1
        
        if 'C3M2' in locals():
            del C3M2
            
            
        if 'C4M1' in locals():
            del C4M1
            
        if 'O1M2' in locals():
            del O1M2
        
        if 'O2M1' in locals():
            del O2M1

        

        continue




    

    

    


    



    
    
    
    #pyedflib.highlevel.write_edf(output_path,signals_matrix, new_signal_header) 




print('Done')
    

                
        






## Functions that might work ####

# pyedflib.highlevel.rename_channels(edf_file, mapping, new_file=None, verbose=False) 

# pyedflib.highlevel.drop_channels(edf_source, edf_target=None, to_keep=None, to_drop=None, verbose=False)
# pyedflib.highlevel.read_edf_header(edf_file, read_annotations=True)

# pyedflib.highlevel.write_edf(edf_file, signals, signal_headers, header=None, digital=False, file_type=-1, block_size=1) 

# pyedflib.highlevel.make_signal_header(label, dimension='uV', sample_rate=256, sample_frequency=None, physical_min=-200, physical_max=200, digital_min=-32768, digital_max=32767, transducer='', prefiler='')


# Check for referenced electrodes 


