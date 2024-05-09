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


# File paths 
input_path =r"/oak/stanford/groups/mignot/projects/irbd_during_v2/" # select correct path and folder


# Make output path to scratch !!!!!
error_dict=[]

# Looping over all EDF files in folder 
edf_files_list = list_files_in_folder(input_path)

edf_files_list=sorted(edf_files_list)

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
        print(type(signal_headers[1]))
        print(signal_headers[1])
        print('shape')
        print(len(signal_headers))

        print('Signals')
        #print(type(signals))
        print(signals)
        print('indexed type')
        print(type(signals[1]))
        print(signals[1])
        print('shape')
        print(len(signals))

        print('header')
        print(type(header))
        print('indexed type')
        print(header)
        print('shape')
        print(len(header))



        # Extracting patientID and visiting number
        numbers_found = os.path.basename(edf_file)
        print("Numbers found in the filename:", numbers_found)

        filename, file_extension = os.path.splitext(numbers_found)

        print('filename')
        print(filename)

        numbers_found=filename

        patientID=filename



        
        

        # Save visit number and patient ID in dict here!!!!

        output_path=r'/scratch/users/s184063/RBD_Restructure_firsttry/' 
        filename=f'restructuredfile_RBD_{patientID}.EDF'
        output_file_path=os.path.join(output_path,filename)
        print(output_file_path)
        print(type(output_file_path))

        # Extracting the indices for the EEG signals in the data
        # This 'labels_to_find' variable should be corrected for each dataset 
        labels_to_find = ['F3:M2','F4:M1','C3:M2','C4:M1', 'O1:M2','O2:M1','M2:F3','M1:F4','M2:C3','M1:C4', 'M2:O1','M1:O2']

        indices = get_indices(signal_headers, labels_to_find)
        #print(indices)

        #Looping over the possible labels 
        for label in labels_to_find:
            print(f"The label '{label}' is at index {indices[label]}")
                        
                    
            # Extracting the electrode indexes 
            if label=='F3:M2' or label=='M2:F3':
                F3M2_index_trial=indices[label]
                print('F3M2 label trial')
                print(F3M2_index_trial)
                print(type(F3M2_index_trial))
                    
                    
                if type(F3M2_index_trial)==int:
                    F3M2_index = F3M2_index_trial
                    print('F3M2_index true')

                    print("F3M2 variable exists.")
                    F3M2=signals[F3M2_index]
                    print('F3M2 defined')
                    print(type(F3M2))
                    print(len(F3M2))

                    
                    # Extract values from old signal header to pack it in the new
                    invest_F3M2_signalheader=signal_headers[F3M2_index]
                    dimension_F3M2=invest_F3M2_signalheader['dimension']
                    sample_rate_F3M2=invest_F3M2_signalheader['sample_rate']
                    sample_frequency_F3M2=invest_F3M2_signalheader['sample_frequency']
                    physical_min_F3M2=invest_F3M2_signalheader['physical_min']
                    physical_max_F3M2=invest_F3M2_signalheader['physical_max']
                    #physical_min_C4=-12000
                    #physical_max_C4=12000
                    digital_min_F3M2=invest_F3M2_signalheader['digital_min']
                    digital_max_F3M2=invest_F3M2_signalheader['digital_max']
                    transducer_F3M2=invest_F3M2_signalheader['transducer']
                    prefiler_F3M2=invest_F3M2_signalheader['prefilter']


                    print('Physical min')
                    print(physical_min_F3M2)
                    print(type(physical_min_F3M2))

                    print('Physical max')
                    print(physical_max_F3M2)
                    print(type(physical_max_F3M2))


                    print('Using function to generate signal header for F3M2')
                    new_signal_header_F3M2=pyedflib.highlevel.make_signal_header(label='F3M2', dimension=dimension_F3M2, sample_rate=sample_rate_F3M2, sample_frequency=sample_frequency_F3M2, physical_min=physical_min_F3M2, physical_max=physical_max_F3M2, digital_min=digital_min_F3M2, digital_max=digital_max_F3M2, transducer=transducer_F3M2, prefiler=prefiler_F3M2)
                    print(new_signal_header_F3M2)

                        
                else:
                    print("The variable F3M2 does not exist.")
                        


            # Extracting the electrode indexes 
            if label=='F4:M1' or label=='M1:F4':
                F4M1_index_trial=indices[label]
                print('F4M1 label trial')
                print(F4M1_index_trial)
                print(type(F4M1_index_trial))
                    
                    
                if type(F4M1_index_trial)==int:
                    F4M1_index = F4M1_index_trial
                    print('F4M1_index true')

                    print("F4M1 variable exists.")
                    F4M1=signals[F4M1_index]
                    print('F4M1 defined')
                    print(type(F4M1))
                    print(len(F4M1))

                    
                    # Extract values from old signal header to pack it in the new
                    invest_F4M1_signalheader=signal_headers[F4M1_index]
                    dimension_F4M1=invest_F4M1_signalheader['dimension']
                    sample_rate_F4M1=invest_F4M1_signalheader['sample_rate']
                    sample_frequency_F4M1=invest_F4M1_signalheader['sample_frequency']
                    physical_min_F4M1=invest_F4M1_signalheader['physical_min']
                    physical_max_F4M1=invest_F4M1_signalheader['physical_max']
                    #physical_min_C4=-12000
                    #physical_max_C4=12000
                    digital_min_F4M1=invest_F4M1_signalheader['digital_min']
                    digital_max_F4M1=invest_F4M1_signalheader['digital_max']
                    transducer_F4M1=invest_F4M1_signalheader['transducer']
                    prefiler_F4M1=invest_F4M1_signalheader['prefilter']


                    print('Physical min')
                    print(physical_min_F4M1)
                    print(type(physical_min_F4M1))

                    print('Physical max')
                    print(physical_max_F4M1)
                    print(type(physical_max_F4M1))


                    print('Using function to generate signal header for F4M1')
                    new_signal_header_F4M1=pyedflib.highlevel.make_signal_header(label='F4M1', dimension=dimension_F4M1, sample_rate=sample_rate_F4M1, sample_frequency=sample_frequency_F4M1, physical_min=physical_min_F4M1, physical_max=physical_max_F4M1, digital_min=digital_min_F4M1, digital_max=digital_max_F4M1, transducer=transducer_F4M1, prefiler=prefiler_F4M1)
                    print(new_signal_header_F4M1)
                        
                else:
                    print("The variable F4M1 does not exist.")
                        

            # Extracting the electrode indexes 
            if label=='C3:M2' or label=='M2:C3':
                C3M2_index_trial=indices[label]
                print('C3M2 label trial')
                print(C3M2_index_trial)
                print(type(C3M2_index_trial))
                    
                    
                if type(C3M2_index_trial)==int:
                    C3M2_index = C3M2_index_trial
                    print('C3M2_index true')
                    print(type(C3M2_index))


                    print("C3M2 variable exists.")
                    C3M2=signals[C3M2_index]
                    print('C3M2 defined')
                    print(type(C3M2))
                    print(len(C3M2))

                    
                    # Extract values from old signal header to pack it in the new
                    invest_C3M2_signalheader=signal_headers[C3M2_index]
                    dimension_C3M2=invest_C3M2_signalheader['dimension']
                    sample_rate_C3M2=invest_C3M2_signalheader['sample_rate']
                    sample_frequency_C3M2=invest_C3M2_signalheader['sample_frequency']
                    physical_min_C3M2=invest_C3M2_signalheader['physical_min']
                    physical_max_C3M2=invest_C3M2_signalheader['physical_max']
                    #physical_min_C4=-12000
                    #physical_max_C4=12000
                    digital_min_C3M2=invest_C3M2_signalheader['digital_min']
                    digital_max_C3M2=invest_C3M2_signalheader['digital_max']
                    transducer_C3M2=invest_C3M2_signalheader['transducer']
                    prefiler_C3M2=invest_C3M2_signalheader['prefilter']


                    print('Physical min')
                    print(physical_min_C3M2)
                    print(type(physical_min_C3M2))

                    print('Physical max')
                    print(physical_max_C3M2)
                    print(type(physical_max_C3M2))

                
                    print('Using function to generate signal header for C3M2')
                    new_signal_header_C3M2=pyedflib.highlevel.make_signal_header(label='C3M2', dimension=dimension_C3M2, sample_rate=sample_rate_C3M2, sample_frequency=sample_frequency_C3M2, physical_min=physical_min_C3M2, physical_max=physical_max_C3M2, digital_min=digital_min_C3M2, digital_max=digital_max_C3M2, transducer=transducer_C3M2, prefiler=prefiler_C3M2)
                    print(new_signal_header_C3M2)
                        
                else:
                    print("The variable C3M2 does not exist.")
                

            # Extracting the electrode indexes 
            if label=='C4:M1' or label=='M1:C4':
                C4M1_index_trial=indices[label]
                print('C4M1 label trial')
                print(C4M1_index_trial)
                print(type(C4M1_index_trial))
                    
                    
                if type(C4M1_index_trial)==int:
                    C4M1_index = C4M1_index_trial
                    print('C4M1_index true')

                    print("C4M1 variable exists.")
                    C4M1=signals[C4M1_index]
                    print('C4M1 defined')
                    print(type(C4M1))
                    print(len(C4M1))

                    # Extract values from old signal header to pack it in the new
                    invest_C4M1_signalheader=signal_headers[C4M1_index]
                    dimension_C4M1=invest_C4M1_signalheader['dimension']
                    sample_rate_C4M1=invest_C4M1_signalheader['sample_rate']
                    sample_frequency_C4M1=invest_C4M1_signalheader['sample_frequency']
                    physical_min_C4M1=invest_C4M1_signalheader['physical_min']
                    physical_max_C4M1=invest_C4M1_signalheader['physical_max']
                    #physical_min_C4=-12000
                    #physical_max_C4=12000
                    digital_min_C4M1=invest_C4M1_signalheader['digital_min']
                    digital_max_C4M1=invest_C4M1_signalheader['digital_max']
                    transducer_C4M1=invest_C4M1_signalheader['transducer']
                    prefiler_C4M1=invest_C4M1_signalheader['prefilter']


                    print('Physical min')
                    print(physical_min_C4M1)
                    print(type(physical_min_C4M1))

                    print('Physical max')
                    print(physical_max_C4M1)
                    print(type(physical_max_C4M1))
                    

                    print('Using function to generate signal header for C4M1')
                    new_signal_header_C4M1=pyedflib.highlevel.make_signal_header(label='C4M1', dimension=dimension_C4M1, sample_rate=sample_rate_C4M1, sample_frequency=sample_frequency_C4M1, physical_min=physical_min_C4M1, physical_max=physical_max_C4M1, digital_min=digital_min_C4M1, digital_max=digital_max_C4M1, transducer=transducer_C4M1, prefiler=prefiler_C4M1)
                    print(new_signal_header_C4M1)

                        
                else:
                    print("The variable C4M1 does not exist.")
                        
                
            # Extracting the electrode indexes 
            if label=='O2:M1' or label=='M1:O2':
                O2M1_index_trial=indices[label]
                print('O2M1 label trial')
                print(O2M1_index_trial)
                print(type(O2M1_index_trial))
                    
                    
                if type(O2M1_index_trial)==int:
                    O2M1_index = O2M1_index_trial
                    print('O2M1_index true')

                    print("O2M1 variable exists.")
                    O2M1=signals[O2M1_index]
                    print('O2M1 defined')
                    print(type(O2M1))
                    print(len(O2M1))

                    
                    # Extract values from old signal header to pack it in the new
                    invest_O2M1_signalheader=signal_headers[O2M1_index]
                    dimension_O2M1=invest_O2M1_signalheader['dimension']
                    sample_rate_O2M1=invest_O2M1_signalheader['sample_rate']
                    sample_frequency_O2M1=invest_O2M1_signalheader['sample_frequency']
                    physical_min_O2M1=invest_O2M1_signalheader['physical_min']
                    physical_max_O2M1=invest_O2M1_signalheader['physical_max']
                    #physical_min_C4=-12000
                    #physical_max_C4=12000
                    digital_min_O2M1=invest_O2M1_signalheader['digital_min']
                    digital_max_O2M1=invest_O2M1_signalheader['digital_max']
                    transducer_O2M1=invest_O2M1_signalheader['transducer']
                    prefiler_O2M1=invest_O2M1_signalheader['prefilter']


                    print('Physical min')
                    print(physical_min_O2M1)
                    print(type(physical_min_O2M1))

                    print('Physical max')
                    print(physical_max_O2M1)
                    print(type(physical_max_O2M1))


                    print('Using function to generate signal header for O2M1')
                    new_signal_header_O2M1=pyedflib.highlevel.make_signal_header(label='O2M1', dimension=dimension_O2M1, sample_rate=sample_rate_O2M1, sample_frequency=sample_frequency_O2M1, physical_min=physical_min_O2M1, physical_max=physical_max_O2M1, digital_min=digital_min_O2M1, digital_max=digital_max_O2M1, transducer=transducer_O2M1, prefiler=prefiler_O2M1)
                    print(new_signal_header_O2M1)
                        
                else:
                    print("The variable O2M1 does not exist.")
                        

            # Extracting the electrode indexes 
            if label=='O1:M2' or label=='M2:O1':
                O1M2_index_trial=indices[label]
                print('O1M2 label trial')
                print(O1M2_index_trial)
                print(type(O1M2_index_trial))
                    
                    
                if type(O1M2_index_trial)==int:
                    O1M2_index = O1M2_index_trial
                    print('O1M2_index true')

                    print("O1M2 variable exists.")
                    O1M2=signals[O1M2_index]
                    print('O1M2 defined')
                    print(type(O1M2))
                    print(len(O1M2))

                    
                    # Extract values from old signal header to pack it in the new
                    invest_O1M2_signalheader=signal_headers[O1M2_index]
                    dimension_O1M2=invest_O1M2_signalheader['dimension']
                    sample_rate_O1M2=invest_O1M2_signalheader['sample_rate']
                    sample_frequency_O1M2=invest_O1M2_signalheader['sample_frequency']
                    physical_min_O1M2=invest_O1M2_signalheader['physical_min']
                    physical_max_O1M2=invest_O1M2_signalheader['physical_max']
                    #physical_min_C4=-12000
                    #physical_max_C4=12000
                    digital_min_O1M2=invest_O1M2_signalheader['digital_min']
                    digital_max_O1M2=invest_O1M2_signalheader['digital_max']
                    transducer_O1M2=invest_O1M2_signalheader['transducer']
                    prefiler_O1M2=invest_O1M2_signalheader['prefilter']


                    print('Physical min')
                    print(physical_min_O1M2)
                    print(type(physical_min_O1M2))

                    print('Physical max')
                    print(physical_max_O1M2)
                    print(type(physical_max_O1M2))


                    print('Using function to generate signal header for O1M2')
                    new_signal_header_O1M2=pyedflib.highlevel.make_signal_header(label='O1M2', dimension=dimension_O1M2, sample_rate=sample_rate_O1M2, sample_frequency=sample_frequency_O1M2, physical_min=physical_min_O1M2, physical_max=physical_max_O1M2, digital_min=digital_min_O1M2, digital_max=digital_max_O1M2, transducer=transducer_O1M2, prefiler=prefiler_O1M2)
                    print(new_signal_header_O1M2)


                        
                else:
                    print("The variable O1M2 does not exist.")


            
            
    
        # Packing the EDF files 
        if 'F3M2' in locals() and 'F4M1' in locals() and 'C3M2' in locals() and 'C4M1' in locals() and 'O1M2' in locals() and 'O2M1' in locals(): 
                
            print('The combination: F3M2, F4M1, C3M2, C4M1, O1M2, O2M1 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            signals_matrix_6E=np.vstack((F3M2,F4M1,C3M2,C4M1,O1M2,O2M1))
            print('Signal matrix - 6 electrodes referenced')
            print(type(signals_matrix_6E))
            print(signals_matrix_6E.shape)
                
            print('Signal header type')
            new_signal_header_6E=[new_signal_header_F3M2,new_signal_header_F4M1,new_signal_header_C3M2,new_signal_header_C4M1, new_signal_header_O1M2,new_signal_header_O2M1]
            print(new_signal_header_6E)
            print(len(new_signal_header_6E))
                
            pyedflib.highlevel.write_edf(output_file_path,signals_matrix_6E, new_signal_header_6E) 
            print('Have been packed: F3M2, F4M1, C3M2, C4M1, O1M2, O2M1 ')
            del F3M2, F4M1, C3M2, C4M1, O1M2, O2M1


            
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
            print('Have been packed: C3M2, C4M1, O1M2, O2M1')
            del C3M2, C4M1, O1M2, O2M1

            
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
            print('Have been packed: C3M2, C4M1, O2M1')
            del C3M2, C4M1, O2M1

            
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
            print('Have been packed: C3M2, C4M1, O1M2')
            del C3M2, C4M1, O1M2


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
            print('Have been packed: C3M2, C4M1')
            del C3M2, C4M1

            
        # Delete variables in locals - to prevent errors

        if 'signals' in locals(): 
            del signals
            
        if 'signalheaders' in locals(): 
            del signalheaders

        if 'headers' in locals():
            del headers

        if 'labels_to_find' in locals():
            del labels_to_find
    
        if 'F3M2' in locals(): 
            del F3M2

        if 'F4M1' in locals():
            del F4M1
            
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
        
        if 'labels_to_find' in locals():
                del labels_to_find


        if 'F3M2' in locals(): 
            del F3M2

        if 'F4M1' in locals():
            del F4M1
        
        if 'C3M2' in locals():
            del C3M2
            
            
        if 'C4M1' in locals():
            del C4M1
            
        if 'O1M2' in locals():
            del O1M2
        
        if 'O2M1' in locals():
            del O2M1

        

        continue



print('Done')
    

                
        






## Functions that might work ####

# pyedflib.highlevel.rename_channels(edf_file, mapping, new_file=None, verbose=False) 

# pyedflib.highlevel.drop_channels(edf_source, edf_target=None, to_keep=None, to_drop=None, verbose=False)
# pyedflib.highlevel.read_edf_header(edf_file, read_annotations=True)

# pyedflib.highlevel.write_edf(edf_file, signals, signal_headers, header=None, digital=False, file_type=-1, block_size=1) 

# pyedflib.highlevel.make_signal_header(label, dimension='uV', sample_rate=256, sample_frequency=None, physical_min=-200, physical_max=200, digital_min=-32768, digital_max=32767, transducer='', prefiler='')


# Check for referenced electrodes 


