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


textfile_path=r'/scratch/users/s184063/STAGES_controls_matched_to_irbd.txt'

temp_names=[]

with open(textfile_path, 'r') as file:
    file_content = ''
    line = file.readline()
     
    while line:
        file_content += line
        line = file.readline()
    
        # Extracting patientID and visiting number
        numbers_found = os.path.basename(line)
        print("Numbers found in the filename:", numbers_found)
        filename, file_extension = os.path.splitext(numbers_found)
        filename=filename[30:] #G:\stanford_irbd\edf_control\STNF00379.edf
        filename=filename+'.edf'
        #print('filename')
        #print(filename)
        #print(type)

        path_oak=r'/oak/stanford/groups/mignot/psg/STAGES/all/'

        path_edf=os.path.join(path_oak,filename)
        #print('Path edfs')
        #print(path_edf)

        temp_names.append(path_edf)
        

edf_files_list = temp_names[:-1]
print('temp_names')
print(temp_names)



# Make output path to scratch !!!!!
error_dict=[]

# Looping over all EDF files in folder 
#edf_files_list = list_files_in_folder(input_path)

#edf_files_list=sorted(edf_files_list)
print('Length of edf_files_list')
print(len(edf_files_list))

print("EDF files in the folder:")
for edf_file in edf_files_list:
    print(edf_file)

    print('Starting a new loop for patient')
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

        output_path=r'/scratch/users/s184063/RBD_controls_Restructure_firsttry/' 
        filename=f'restructuredfile_RBD_controls_{patientID}.EDF'
        output_file_path=os.path.join(output_path,filename)
        print(output_file_path)
        print(type(output_file_path))

        # Extracting the indices for the EEG signals in the data
        # This 'labels_to_find' variable should be corrected for each dataset 
        labels_to_find = ['F3','F4','C3','C4', 'O1','O2','M2','M1']

        indices = get_indices(signal_headers, labels_to_find)
        #print(indices)

        #Looping over the possible labels 
        for label in labels_to_find:
            print(f"The label '{label}' is at index {indices[label]}")
                        
                    
            # Extracting the electrode indexes 
            if label=='F3':
                F3_index_trial=indices[label]
                print('F3 label trial')
                print(F3_index_trial)
                print(type(F3_index_trial))
                    
                    
                if type(F3_index_trial)==int:
                    F3_index = F3_index_trial
                    print('F3_index true')

                    print("F3 variable exists.")
                    F3_signal=signals[F3_index]
                    print('F3 defined')
                    print(type(F3_signal))
                    print(len(F3_signal))

                        
                else:
                    print("The variable F3 does not exist.")
                        


            # Extracting the electrode indexes 
            if label=='F4':
                F4_index_trial=indices[label]
                print('F4 label trial')
                print(F4_index_trial)
                print(type(F4_index_trial))
                    
                    
                if type(F4_index_trial)==int:
                    F4_index = F4_index_trial
                    print('F4_index true')

                    print("F4 variable exists.")
                    F4_signal=signals[F4_index]
                    print('F4 defined')
                    print(type(F4_signal))
                    print(len(F4_signal))

                else:
                    print("The variable F4 does not exist.")
                        

            # Extracting the electrode indexes 
            if label=='C3':
                C3_index_trial=indices[label]
                print('C3 label trial')
                print(C3_index_trial)
                print(type(C3_index_trial))
                    
                    
                if type(C3_index_trial)==int:
                    C3_index = C3_index_trial
                    print('C3_index true')
                    print(type(C3_index))


                    print("C3 variable exists.")
                    C3_signal=signals[C3_index]
                    print('C3 defined')
                    print(type(C3_signal))
                    print(len(C3_signal))

                    
                                        
                else:
                    print("The variable C3 does not exist.")
                

            # Extracting the electrode indexes 
            if label=='C4':
                C4_index_trial=indices[label]
                print('C4 label trial')
                print(C4_index_trial)
                print(type(C4_index_trial))
                    
                    
                if type(C4_index_trial)==int:
                    C4_index = C4_index_trial
                    print('C4_index true')

                    print("C4 variable exists.")
                    C4_signal=signals[C4_index]
                    print('C4 defined')
                    print(type(C4_signal))
                    print(len(C4_signal))

                        
                else:
                    print("The variable C4 does not exist.")
                        
                
            # Extracting the electrode indexes 
            if label=='O2':
                O2_index_trial=indices[label]
                print('O2 label trial')
                print(O2_index_trial)
                print(type(O2_index_trial))
                    
                    
                if type(O2_index_trial)==int:
                    O2_index = O2_index_trial
                    print('O2_index true')

                    print("O2 variable exists.")
                    O2_signal=signals[O2_index]
                    print('O2 defined')
                    print(type(O2_signal))
                    print(len(O2_signal))

                        
                else:
                    print("The variable O2 does not exist.")
                        

            # Extracting the electrode indexes 
            if label=='O1':
                O1_index_trial=indices[label]
                print('O1 label trial')
                print(O1_index_trial)
                print(type(O1_index_trial))
                    
                    
                if type(O1_index_trial)==int:
                    O1_index = O1_index_trial
                    print('O1_index true')

                    print("O1 variable exists.")
                    O1_signal=signals[O1_index]
                    print('O1 defined')
                    print(type(O1_signal))
                    print(len(O1_signal))

                    

                        
                else:
                    print("The variable O1 does not exist.")


        
            # Extracting the electrode indexes 
            if label=='M2':
                M2_index_trial=indices[label]
                print('M2 label trial')
                print(M2_index_trial)
                print(type(M2_index_trial))
                    
                    
                if type(M2_index_trial)==int:
                    M2_index = M2_index_trial
                    print('M2_index true')

                    print("M2 variable exists.")
                    M2_signal=signals[M2_index]
                    print('M2 defined')
                    print(type(M2_signal))
                    print(len(M2_signal))

                        
                else:
                    print("The variable M2 does not exist.")
                        

            # Extracting the electrode indexes 
            if label=='M1':
                M1_index_trial=indices[label]
                print('M1 label trial')
                print(M1_index_trial)
                print(type(M1_index_trial))
                    
                    
                if type(M1_index_trial)==int:
                    M1_index = M1_index_trial
                    print('M1_index true')

                    print("M1 variable exists.")
                    M1_signal=signals[M1_index]
                    print('M1 defined')
                    print(type(M1_signal))
                    print(len(M1_signal))

                    

                        
                else:
                    print("The variable M1 does not exist.")





        # Referencing the electrodes to M1
        if 'M1_signal' in locals() and 'C4_signal' in locals(): 
            print('M1 and C4 exists - now the referenced C4M1 is calculated')      
            
            C4M1=np.subtract(C4_signal,M1_signal)


            # Extract values from old signal header to pack it in the new
            invest_C4_signalheader=signal_headers[C4_index]
            dimension_C4=invest_C4_signalheader['dimension']
            sample_rate_C4=invest_C4_signalheader['sample_rate']
            sample_frequency_C4=invest_C4_signalheader['sample_frequency']
            #physical_min_C4=invest_C4_signalheader['physical_min']
            #physical_max_C4=invest_C4_signalheader['physical_max']
            physical_min_C4=-2000
            physical_max_C4=2000
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


            del C4_signal


        if 'M1_signal' in locals() and 'O2_signal' in locals(): 
            print('M1 and O2 exists - now the referenced O2M1 is calculated')      

            O2M1=np.subtract(O2_signal,M1_signal)

            
            # Extract values from old signal header to pack it in the new
            invest_O2_signalheader=signal_headers[O2_index]
            dimension_O2=invest_O2_signalheader['dimension']
            sample_rate_O2=invest_O2_signalheader['sample_rate']
            sample_frequency_O2=invest_O2_signalheader['sample_frequency']
            physical_min_O2=-2000
            physical_max_O2=2000
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
            

            del O2_signal


        if 'M1_signal' in locals() and 'F4_signal' in locals(): 
            print('M1 and F4 exists - now the referenced Fp2M1 is calculated')      

            F4M1=np.subtract(F4_signal,M1_signal)

            
            # Extract values from old signal header to pack it in the new
            invest_F4_signalheader=signal_headers[F4_index]
            dimension_F4=invest_F4_signalheader['dimension']
            sample_rate_F4=invest_F4_signalheader['sample_rate']
            sample_frequency_F4=invest_F4_signalheader['sample_frequency']
            #physical_min_F4=invest_F4_signalheader['physical_min']
            #physical_max_F4=invest_F4_signalheader['physical_max']
            physical_min_F4=-2000
            physical_max_F4=2000
            digital_min_F4=invest_F4_signalheader['digital_min']
            digital_max_F4=invest_F4_signalheader['digital_max']
            transducer_F4=invest_F4_signalheader['transducer']
            prefiler_F4=invest_F4_signalheader['prefilter']


            print('Physical min')
            print(physical_min_F4)
            print(type(physical_min_F4))

            print('Physical max')
            print(physical_max_F4)
            print(type(physical_max_F4))

    
            print('Using function to generate signal header for F2M1 --> change to F4M1')
            # Changing name to F2M1 and not Fp2M1
            new_signal_header_F4M1=pyedflib.highlevel.make_signal_header(label='F4M1', dimension=dimension_F4, sample_rate=sample_rate_F4, sample_frequency=sample_frequency_F4, physical_min=physical_min_F4, physical_max=physical_max_F4, digital_min=digital_min_F4, digital_max=digital_max_F4, transducer=transducer_F4, prefiler=prefiler_F4)
            print(new_signal_header_F4M1)

            del F4_signal


        # Referencing electrodes to M2
        
        if 'M2_signal' in locals() and 'O1_signal' in locals(): 
            print('M2 and O1 exists - now the referenced O1M2 is calculated')      

            O1M2=np.subtract(O1_signal,M2_signal)

            
            #Extract values from old signal header to pack it in the new
            invest_O1_signalheader=signal_headers[O1_index]
            dimension_O1=invest_O1_signalheader['dimension']
            sample_rate_O1=invest_O1_signalheader['sample_rate']
            sample_frequency_O1=invest_O1_signalheader['sample_frequency']
            #physical_min_O1=invest_O1_signalheader['physical_min']
            #physical_max_O1=invest_O1_signalheader['physical_max']
            physical_min_O1=-2000
            physical_max_O1=2000
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

            del O1_signal


        if 'M2_signal' in locals() and 'C3_signal' in locals(): 
            print('M2 and C3 exists - now the referenced C3M2 is calculated')      

            C3M2=np.subtract(C3_signal,M2_signal)

            
            #Extract values from old signal header to pack it in the new
            invest_C3_signalheader=signal_headers[C3_index]
            dimension_C3=invest_C3_signalheader['dimension']
            sample_rate_C3=invest_C3_signalheader['sample_rate']
            sample_frequency_C3=invest_C3_signalheader['sample_frequency']
            #physical_min_C3=invest_C3_signalheader['physical_min']
            #physical_max_C3=invest_C3_signalheader['physical_max']
            physical_min_C3=-2000
            physical_max_C3=2000
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

            del C3_signal


        if 'M2_signal' in locals() and 'F3_signal' in locals(): 
            print('M2 and F3 exists - now the referenced Fp1M2 is calculated')   

               

            F3M2=np.subtract(F3_signal,M2_signal)

            
            #Extract values from old signal header to pack it in the new
            invest_F3_signalheader=signal_headers[F3_index]
            dimension_F3=invest_F3_signalheader['dimension']
            sample_rate_F3=invest_F3_signalheader['sample_rate']
            sample_frequency_F3=invest_F3_signalheader['sample_frequency']
            #physical_min_F3=invest_F3_signalheader['physical_min']
            #physical_max_F3=invest_F3_signalheader['physical_max']
            physical_min_F3=-2000
            physical_max_F3=2000
            digital_min_F3=invest_F3_signalheader['digital_min']
            digital_max_F3=invest_F3_signalheader['digital_max']
            transducer_F3=invest_F3_signalheader['transducer']
            prefiler_F3=invest_F3_signalheader['prefilter']


            print('Physical min')
            print(physical_min_F3)
            print(type(physical_min_F3))

            print('Physical max')
            print(physical_max_F3)
            print(type(physical_max_F3))

            print('Using function to generate signal header for F3M2')
            new_signal_header_F3M2=pyedflib.highlevel.make_signal_header(label='F3M2', dimension=dimension_F3, sample_rate=sample_rate_F3, sample_frequency=sample_frequency_F3, physical_min=physical_min_F3, physical_max=physical_max_F3, digital_min=digital_min_F3, digital_max=digital_max_F3, transducer=transducer_F3, prefiler=prefiler_F3)
            print(new_signal_header_F3M2)

            del F3_signal
                    



            
    
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



        
        # Packing the EDF files 
        if 'F3M2' in locals() and 'F4M1' in locals() and 'C3M2' in locals() and 'C4M1' in locals() and 'O1M2' in locals(): 
                
            print('The combination: F3M2, F4M1, C3M2, C4M1, O1M2 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            signals_matrix_5E=np.vstack((F3M2,F4M1,C3M2,C4M1,O1M2))
            print('Signal matrix - 5 electrodes referenced')
            print(type(signals_matrix_5E))
            print(signals_matrix_5E.shape)
                
            print('Signal header type')
            new_signal_header_5E=[new_signal_header_F3M2,new_signal_header_F4M1,new_signal_header_C3M2,new_signal_header_C4M1, new_signal_header_O1M2]
            print(new_signal_header_5E)
            print(len(new_signal_header_5E))
                
            pyedflib.highlevel.write_edf(output_file_path,signals_matrix_5E, new_signal_header_5E) 
            print('Have been packed: F3M2, F4M1, C3M2, C4M1, O1M2 ')
            del F3M2, F4M1, C3M2, C4M1, O1M2

        

        
        # Packing the EDF files 
        if 'F3M2' in locals() and 'F4M1' in locals() and 'C3M2' in locals() and 'C4M1' in locals() and 'O2M1' in locals(): 
                
            print('The combination: F3M2, F4M1, C3M2, C4M1, O2M1 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            signals_matrix_5E=np.vstack((F3M2,F4M1,C3M2,C4M1,O2M1))
            print('Signal matrix - 5 electrodes referenced')
            print(type(signals_matrix_5E))
            print(signals_matrix_5E.shape)
                
            print('Signal header type')
            new_signal_header_5E=[new_signal_header_F3M2,new_signal_header_F4M1,new_signal_header_C3M2,new_signal_header_C4M1, new_signal_header_O2M1]
            print(new_signal_header_5E)
            print(len(new_signal_header_5E))
                
            pyedflib.highlevel.write_edf(output_file_path,signals_matrix_5E, new_signal_header_5E) 
            print('Have been packed: F3M2, F4M1, C3M2, C4M1, O2M1 ')
            del F3M2, F4M1, C3M2, C4M1, O2M1


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

        if 'F3' in locals():
            del F3
        
        if 'F3_signal' in locals():
            del F3_signal

        if 'F4' in locals():
            del F4
        
        if 'F4_signal' in locals():
            del F4_signal

        if 'C3' in locals():
            del F3
        
        if 'C3_signal' in locals():
            del C3_signal

        if 'C4' in locals():
            del C4
        
        if 'C4_signal' in locals():
            del C4_signal

        if 'O2' in locals():
            del O2
        
        if 'O2_signal' in locals():
            del O2_signal

        if 'O1' in locals():
            del O1
        
        if 'O1_signal' in locals():
            del O1_signal

        if 'M2' in locals():
            del M2
        
        if 'M2_signal' in locals():
            del M2_signal

        if 'M1' in locals():
            del M1
        
        if 'M1_signal' in locals():
            del M1_signal

        
        if 'F3_index' in locals():
            del F3_index
        
        if 'F4_index' in locals():
            del F4_index

        if 'C3_index' in locals():
            del C3_index
        
        if 'C4_index' in locals():
            del C4_index

        if 'O1_index' in locals():
            del O1_index
        
        if 'O2_index' in locals():
            del O2_index

        if 'M1_index' in locals():
            del M1_index
        
        if 'M2_index' in locals():
            del M2_index
        

        
    except Exception as e:
        print(f"An error occurred: {e}. Moving on to the next iteration...")
        error_dict.append({'error':e,'filename':edf_file})
        # Delete variables in locals - to prevent errors

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

        if 'F3' in locals():
            del F3
        
        if 'F3_signal' in locals():
            del F3_signal

        if 'F4' in locals():
            del F4
        
        if 'F4_signal' in locals():
            del F4_signal

        if 'C3' in locals():
            del F3
        
        if 'C3_signal' in locals():
            del C3_signal

        if 'C4' in locals():
            del C4
        
        if 'C4_signal' in locals():
            del F4_signal

        if 'O2' in locals():
            del O2
        
        if 'O2_signal' in locals():
            del O2_signal

        if 'O1' in locals():
            del O1
        
        if 'O1_signal' in locals():
            del O1_signal

        if 'M2' in locals():
            del M2
        
        if 'M2_signal' in locals():
            del M2_signal

        if 'M1' in locals():
            del M1
        
        if 'M1_signal' in locals():
            del M1_signal

        
        if 'F3_index' in locals():
            del F3_index
        
        if 'F4_index' in locals():
            del F4_index

        if 'C3_index' in locals():
            del C3_index
        
        if 'C4_index' in locals():
            del C4_index

        if 'O1_index' in locals():
            del O1_index
        
        if 'O2_index' in locals():
            del O2_index

        if 'M1_index' in locals():
            del M1_index
        
        if 'M2_index' in locals():
            del M2_index
        
        

        continue



print('Done')
    

                
        






## Functions that might work ####

# pyedflib.highlevel.rename_channels(edf_file, mapping, new_file=None, verbose=False) 

# pyedflib.highlevel.drop_channels(edf_source, edf_target=None, to_keep=None, to_drop=None, verbose=False)
# pyedflib.highlevel.read_edf_header(edf_file, read_annotations=True)

# pyedflib.highlevel.write_edf(edf_file, signals, signal_headers, header=None, digital=False, file_type=-1, block_size=1) 

# pyedflib.highlevel.make_signal_header(label, dimension='uV', sample_rate=256, sample_frequency=None, physical_min=-200, physical_max=200, digital_min=-32768, digital_max=32767, transducer='', prefiler='')


# Check for referenced electrodes 


