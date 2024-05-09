
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
# Load wavelet package 
import pywt 
from pywt import wavedec
import os
import re
import usleep_api
from usleep_api import USleepAPI
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_example")
import itertools 
from itertools import combinations
import fractions
from fractions import Fraction



def preprocessing(input_signal,input_signal_header): #,input_signal_header
    # Made by Natasja Bonde Andersen 28-02-2024
    # This function pre-processes the data by resampling and filtering 
    # the data using a notch filter and highpass filter. 
    

    
    ########### Resampling ##################
    # final sampling rate = (up/down)*original sampling rate
    # The function upsamples by adding zeros, then it low pass FIR filters and then downsamples again
    signal_header_selected = copy.deepcopy(input_signal_header)
    signal_selected = copy.deepcopy(input_signal)

    #print('Information about selected signal')
    #print(type(signal_selected))
    #print(len(signal_selected))

    #sample_rate=signal_header_selected['sample_rate']
    sample_frequency=signal_header_selected['sample_frequency'] # 256 Hz for RBD dataset 
    fs_old=int(sample_frequency)
    print('Sampling frequency')
    print(type(fs_old))
    print(fs_old)

    
    fs_new=128

    resample_frac = Fraction(fs_new/fs_old).limit_denominator(100)
    signal_new = scipy.signal.resample_poly(signal_selected, resample_frac.numerator, resample_frac.denominator)

    signal_old=signal_selected

    
   
    '''
    # Time axis for 128 Hz and 100 Hz signals 
    t_128 = np.arange(0, len(signal_new)/fs_new, 1/fs_new)
    t_100 = np.arange(0,len(signal_old)/fs_old,1/fs_old)
    
    print('Length of signals after resampling')
    print(signal_new.shape)
    '''
    '''
    # Visualising the resampling
    fig, axs = matplotlib.pyplot.subplots(2) # Creating subplot 
    fig.suptitle('After resampling')
    axs[0].plot(t_128, signal_new, 'k')
    #matplotlib.pyplot.title("After resampling")
    matplotlib.pyplot.xlabel('Time [s]') 
    matplotlib.pyplot.ylabel('Amplitude [muV]') 
    axs[1].plot(t_100, signal_old, 'purple')  
    matplotlib.pyplot.title("Before resampling")
    matplotlib.pyplot.xlabel('Time [s]') 
    matplotlib.pyplot.ylabel('Amplitude [muV]') 
    #matplotlib.pyplot.legend(['Resampled polyphase filter','Raw data'], loc='best')
    matplotlib.pyplot.show()
    '''


    ###### Looking into spectra before and after resampling #######
    '''
    # Fast fourier transform 
    y_old=fft(signal_old)
    y_new=fft(signal_new)

    y_new=scipy.fft.fftshift(y_new)
    y_old=scipy.fft.fftshift(y_old)

    #Creating frequency axis 
    f_old=np.linspace(-fs_old/2,fs_old/2,len(signal_old))
    f_new=np.linspace(-fs_new/2,fs_new/2,len(signal_new))
    '''
    '''
    # Plot magnitude spectra dB 
    fig, axs = matplotlib.pyplot.subplots(2)
    fig.suptitle("Magnitude spectra - resampled signal")
    axs[0].magnitude_spectrum(signal_new,fs_new,scale='dB') # input a time signal and fs
    matplotlib.pyplot.title("Magnitude spectra - resampled signal in time domain")
    matplotlib.pyplot.xlabel('Frequency [Hz]') 
    matplotlib.pyplot.ylabel('Magnitude [dB]') 
    axs[1].magnitude_spectrum(signal_old,fs_old,scale='dB') # input a time signal and 
    matplotlib.pyplot.title("Magnitude spectra - raw signal ")
    matplotlib.pyplot.xlabel('Frequency [Hz]') 
    matplotlib.pyplot.ylabel('Magnitude [dB]') 
    matplotlib.pyplot.show()
    '''
   

    #### Creating and applying highpass filter ######

    # IIR filter (highpass, 0.5 Hz )
    sos = scipy.signal.iirfilter(25, [0.5], rs=60, btype='highpass',analog=False, ftype='cheby2', fs=128, output='sos')
    
    '''
    w, h = scipy.signal.sosfreqz(sos, fs=128)
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5))) # w = frequencies corresponding to the h value = frequency response in complex values 
    ax.set_title('Chebyshev Type II highpass frequency response')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    #ax.axis((10, 1000, -100, 10))
    ax.grid(which='both', axis='both')
    matplotlib.pyplot.show()
    '''

    # Applying highpass filter to signal using filtfilt in order to make zero phase filtering 
    time_filtered_HP=scipy.signal.sosfiltfilt(sos, signal_new, axis=-1, padtype='odd', padlen=None)

    
    # Making notch IIR filter 50 Hz and 60 Hz 
    # fs = Sample frequency (Hz)
    # f0 =  Frequency to be removed from signal (Hz)
    # Q = Quality factor
   
    b_50, a_50 = scipy.signal.iirnotch(50, 30, fs_new)
    b_60, a_60 = scipy.signal.iirnotch(60, 30, fs_new)



    # Frequency response for notch 50 Hz 
    #freq, h = scipy.signal.freqz(b, a, fs=fs_new)
    # Plot
    #fig, ax = matplotlib.pyplot.subplots(2, 1, figsize=(8, 6))
    #ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
    #ax[0].set_title("Frequency Response")
    #ax[0].set_ylabel("Amplitude (dB)", color='blue')
    #ax[0].set_xlim([0, 100])
    #ax[0].set_ylim([-25, 10])
    #ax[0].grid(True)
    #ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    #ax[1].set_ylabel("Angle (degrees)", color='green')
    #ax[1].set_xlabel("Frequency (Hz)")
    #ax[1].set_xlim([0, 100])
    #ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    #ax[1].set_ylim([-90, 90])
    #ax[1].grid(True)
    #matplotlib.pyplot.show()
    
    
    # Apply notch filters 
    outputSignal_50 = scipy.signal.filtfilt(b_50, a_50, time_filtered_HP)

    outputSignal_60 = scipy.signal.filtfilt(b_60, a_60, outputSignal_50)

    time_filtered_HP = outputSignal_60

    '''
    # Magnitude spectrum dB of highpass filtered time signal 
    fig, axs = matplotlib.pyplot.subplots(2)
    fig.suptitle("Magnitude spectra highpass and notch filtered signal - frequency domain")
    axs[0].magnitude_spectrum(time_filtered_HP,fs_new,scale='dB')
    #matplotlib.pyplot.title('Magnitude spectra highpass and notch filtered signal - frequency domain')
    matplotlib.pyplot.xlabel('Frequency [Hz]') 
    matplotlib.pyplot.ylabel('Magnitude [dB]') 
    axs[1].magnitude_spectrum(signal_new,fs_new,scale='dB')
    matplotlib.pyplot.title('Magnitude spectra raw signal - frequency domain')
    matplotlib.pyplot.xlabel('Frequency [Hz]') 
    matplotlib.pyplot.ylabel('Magnitude [dB]') 
    matplotlib.pyplot.show()

    
    # Time signal before and after filtering
    fig, axs = matplotlib.pyplot.subplots(2)
    fig.suptitle('Raw signal - time domain')
    axs[0].plot(t_128, signal_new, 'k')
    matplotlib.pyplot.title('Raw signal - time domain')
    matplotlib.pyplot.xlabel('Time [s]') 
    matplotlib.pyplot.ylabel('Amplitude [muV]') 
    axs[1].plot(t_128,time_filtered_HP, 'purple') 
    matplotlib.pyplot.title('Highpass and notch filtered signal - time domain') 
    matplotlib.pyplot.xlabel('Time [s]') 
    matplotlib.pyplot.ylabel('Amplitude [muV]') 
    matplotlib.pyplot.show()
    '''
    

    print('Success')
    return signal_new, fs_new, time_filtered_HP




def inverse_wavelet_5_levels(coeffs_5_levels_input,idx_input):
    # Made by Natasja Bonde Andersen 28-02-2024
    # This function makes inverse wavelet transform on a signal 
    # that was wavelet transformed using 'db4' and 5 levels of decomposition 
    
    ##### Inputs #####
    # idx = index for coefficient to express (0,1,2,3,4,5)= cA5,cD5,cD4,cD3,cD2,cD1
    # coeffs = coefficients from the wavelet transform using 'db4' and 5 level decomposition 
    idx=idx_input
    coeffs_5_levels=copy.deepcopy(coeffs_5_levels_input)


    # Creating empty index np.array for coeffs_idx
    coeffs_idx=[]

    # Saving index values from coeffs
    coeffs_idx=coeffs_5_levels[idx]

    # Setting coeffs as a new structure for changes 
    coeffs_zero=coeffs_5_levels
    #Settting all values to zero in the new structure 
    coeffs_zero[-6]=np.zeros_like(coeffs_zero[-6])
    coeffs_zero[-5]=np.zeros_like(coeffs_zero[-5])
    coeffs_zero[-4]=np.zeros_like(coeffs_zero[-4])
    coeffs_zero[-3]=np.zeros_like(coeffs_zero[-3])
    coeffs_zero[-2]=np.zeros_like(coeffs_zero[-2])
    coeffs_zero[-1]=np.zeros_like(coeffs_zero[-1])


    # Inputting values in index into the zero-structure 
    coeffs_zero[idx]=coeffs_idx

    # Reconstructing the signal based on the coefficients expressed
    signal_reconstructed=pywt.waverec(coeffs_zero, 'db4', mode='symmetric', axis=-1)

    return signal_reconstructed




def relative_power_for_frequencyband(signal_reconstructed_x_input,signal_reconstructed_full_input):
    # Made by Natasja Bonde Andersen 28-02-2024

    # This function calculates the relative energy for a specific frequency band 
    # The data loaded in should be bandpass filtered or filtered using wavelet transform 
    
    
    # Input signals are copied 
    signal_reconstructed_x=copy.deepcopy(signal_reconstructed_x_input)
    signal_reconstructed_full=copy.deepcopy(signal_reconstructed_full_input)

    signal_reconstructed_x=np.array(signal_reconstructed_x)
    signal_reconstructed_full=np.array(signal_reconstructed_full)

    #print(type(signal_reconstructed_x))
    #print(type(signal_reconstructed_full))
    '''
    # Squaring the signals
    signal_squared = [n**2 for n in signal_reconstructed_x]
    signal_squared_full = [n**2 for n in signal_reconstructed_full]
    '''

    # Squaring the signals
    signal_squared = np.power(signal_reconstructed_x, 2)
    signal_squared_full = np.power(signal_reconstructed_full, 2)

    '''
    # Calculating absolute value of signal
    res = [abs(ele) for ele in signal_squared]
    res2 = [abs(ele) for ele in signal_squared_full]
    


    #Calculating the energy of the signal in one frequency band and the total energy 
    E_j=sum(res)
    E_total=sum(res2)

    # Calculating the normalised power for a frequency band 
    p_j=E_j/E_total

    print('Relative power for chosen frequency band')
    print(p_j)
    '''

    # Calculating absolute value of signal
    res = np.abs(signal_squared)
    res2 = np.abs(signal_squared_full)

    #Calculating the energy of the signal in one frequency band and the total energy 
    E_j = np.sum(res)
    E_total = np.sum(res2)

    # Calculating the normalised power for a frequency band 
    p_j = E_j / E_total

    print('Relative power for chosen frequency band')
    print(p_j)

    return p_j



def coherence_features(Coherence_xy_input,f_input):
    # Made by Natasja Bonde Andersen 28-02-2024

    # This function extracts the average coherence values for each EEG frequency band 

    # Inputs (coherence signal and frequency from coherence function )
    Cxy=copy.deepcopy(Coherence_xy_input)
    f=copy.deepcopy(f_input)
    
    # EEG frequency bands 
    # delta (1-3 Hz)
    # theta (4-7 Hz)
    # alpha (8-12 Hz)
    # beta band (13-30 Hz)
    # gamma band (31-100 Hz)
    
    # Finding the indexes 
    #Finding the last value in an np.array [-1]
    idx_1=np.where(f<0.5)
    idx_1=idx_1[-1][-1]
    #print(idx_1)
    #print(f[idx_1])

    idx_3=np.where(f<4)
    idx_3=idx_3[-1][-1]
    #print(idx_3)
    #print(f[idx_3])


    #idx_4=np.where(f==4)
    #idx_4=idx_4[0][0]

    idx_7=np.where(f<8)
    idx_7=idx_7[-1][-1]
    #print(idx_7)
    #print(f[idx_7])


    #idx_8=np.where(f==8)
    #idx_8=idx_8[0][0]

    idx_12=np.where(f<13)
    idx_12=idx_12[-1][-1]
    #print(idx_12)
    #print(f[idx_12])


    #idx_13=np.where(f==13)
    #idx_13=idx_13[0][0]


    idx_30=np.where(f<31)
    idx_30=idx_30[-1][-1]
    #print(idx_30)
    #print(f[idx_30])


    #idx_31=np.where(f==31)
    #idx_31=idx_31[0][0]

    idx_100=np.where(f<100)
    idx_100=idx_100[-1][-1]
    #print(idx_100)
    #print(f[idx_100])

    

    
    # Converting into integers 
    idx_1=int(idx_1)
    idx_3=int(idx_3)
    #idx_4=int(idx_4)
    idx_7=int(idx_7)
    #idx_8=int(idx_8)
    idx_12=int(idx_12)
    #idx_13=int(idx_13)
    idx_30=int(idx_30)
    #idx_31=int(idx_31)
    idx_100=int(idx_100)


    

   
    # Indexing and averaging the frequency bands 
    #Delta
    delta=Cxy[idx_1:idx_3]
    deltaband_coh_2E=np.average(delta)


    #Theta 
    theta=Cxy[idx_3:idx_7]
    thetaband_coh_2E=np.average(theta)


    #Alpha 
    alpha=Cxy[idx_7:idx_12]
    alphaband_coh_2E=np.average(alpha)


    #Beta
    beta=Cxy[idx_12:idx_30]
    betaband_coh_2E=np.average(beta)


    #gamma 
    gamma=Cxy[idx_30:idx_100]
    gammaband_coh_2E=np.average(gamma)

    return deltaband_coh_2E, thetaband_coh_2E, alphaband_coh_2E,betaband_coh_2E,gammaband_coh_2E


def bandpass_frequency_band():
            
    # This function creates IIR bandpass filters for the EEG frequency bands 
    # It outputs the filters ready for application using sosfiltfilt 


    # EEG frequency bands 
    # delta (1-3 Hz)
    # theta (4-7 Hz)
    # alpha (8-12 Hz)
    # beta band (13-30 Hz)
    # gamma band (31-100 Hz)

        

    # IIR filter bandpass delta (1-3 Hz)
    sos_delta = scipy.signal.iirfilter(15, [1, 3], rs=60, btype='bandpass',analog=False, ftype='cheby2', fs=128, output='sos')
    # rs = minimum damping in dB of the signal 
    # 15 = filter size
    # 1-3 Hz = cutoff 

    w_delta, h_delta = scipy.signal.sosfreqz(sos_delta, fs=128)
    '''
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(w_delta, 20 * np.log10(np.maximum(abs(h_delta), 1e-5))) # w = frequencies corresponding to the h value = frequency response in complex values 
    ax.set_title('Chebyshev Type II bandpass (1-3 Hz) frequency response')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    #ax.axis((10, 1000, -100, 10))
    ax.grid(which='both', axis='both')
    matplotlib.pyplot.show()
    '''

    #### theta (4-7 Hz) #####
    # IIR filter bandpass (4-7 Hz)
    sos_theta = scipy.signal.iirfilter(15, [4, 7], rs=60, btype='bandpass',analog=False, ftype='cheby2', fs=128, output='sos')
    # rs = minimum damping in dB of the signal 
    # 15 = filter size
    # 4-7 Hz = cutoff 

    w_theta, h_theta = scipy.signal.sosfreqz(sos_theta, fs=128)
    '''
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(w_theta, 20 * np.log10(np.maximum(abs(h_theta), 1e-5))) # w = frequencies corresponding to the h value = frequency response in complex values 
    ax.set_title('Chebyshev Type II bandpass (4-7 Hz) frequency response')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    #ax.axis((10, 1000, -100, 10))
    ax.grid(which='both', axis='both')
    matplotlib.pyplot.show()
    '''

    ###### alpha (8-12 Hz)############
    # IIR filter bandpass (8-12 Hz)
    sos_alpha = scipy.signal.iirfilter(15, [8, 12], rs=60, btype='bandpass',analog=False, ftype='cheby2', fs=128, output='sos')
    # rs = minimum damping in dB of the signal 
    # 15 = filter size
    # 8-12 Hz = cutoff 

    w_alpha, h_alpha = scipy.signal.sosfreqz(sos_alpha, fs=128)
    '''
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(w_alpha, 20 * np.log10(np.maximum(abs(h_alpha), 1e-5))) # w = frequencies corresponding to the h value = frequency response in complex values 
    ax.set_title('Chebyshev Type II bandpass (8-12 Hz) frequency response')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    #ax.axis((10, 1000, -100, 10))
    ax.grid(which='both', axis='both')
    matplotlib.pyplot.show()
    '''

    # beta band (13-30 Hz)
    # IIR filter bandpass (13-30 Hz)
    sos_beta = scipy.signal.iirfilter(15, [13, 30], rs=60, btype='bandpass',analog=False, ftype='cheby2', fs=128, output='sos')
    # rs = minimum damping in dB of the signal 
    # 15 = filter size
    # 13-30 Hz = cutoff 

    w_beta, h_beta = scipy.signal.sosfreqz(sos_beta, fs=128)
    '''
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(w_beta, 20 * np.log10(np.maximum(abs(h_beta), 1e-5))) # w = frequencies corresponding to the h value = frequency response in complex values 
    ax.set_title('Chebyshev Type II bandpass (13-30 Hz) frequency response')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    #ax.axis((10, 1000, -100, 10))
    ax.grid(which='both', axis='both')
    matplotlib.pyplot.show()
    '''

    # gamma band (31-100 Hz)
    # IIR filter bandpass (31-64 Hz)
    sos_gamma = scipy.signal.iirfilter(15, [31, 63], rs=60, btype='bandpass',analog=False, ftype='cheby2', fs=128, output='sos')
    # rs = minimum damping in dB of the signal 
    # 15 = filter size
    # 31-100 Hz = cutoff 

    w_gamma, h_gamma = scipy.signal.sosfreqz(sos_gamma, fs=128)
    '''
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(w_gamma, 20 * np.log10(np.maximum(abs(h_gamma), 1e-5))) # w = frequencies corresponding to the h value = frequency response in complex values 
    ax.set_title('Chebyshev Type II bandpass (31-63 Hz) frequency response')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    #ax.axis((10, 1000, -100, 10))
    ax.grid(which='both', axis='both')
    matplotlib.pyplot.show()
    '''

    '''
    # Gathered plot of filters designed 
    fig, axs = matplotlib.pyplot.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns

    # Assuming your data is for the first plot at position (0, 0)
    axs[0, 0].semilogx(w_delta, 20 * np.log10(np.maximum(abs(h_delta), 1e-5))) # w = frequencies corresponding to the h value = frequency response in complex values 
    axs[0, 0].set_title('Chebyshev Type II bandpass (1-3 Hz) frequency response')
    #axs[0, 0].set_xlabel('frequency [Hz]')
    axs[0, 0].set_ylabel('Amplitude [dB]')

    # Add other plots here accordingly, 
    # the index axs[i, j] will change according to the position you want for each plot.
    # axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]
    axs[0, 1].semilogx(w_theta, 20 * np.log10(np.maximum(abs(h_theta), 1e-5))) # w = frequencies corresponding to the h value = frequency response in complex values 
    axs[0, 1].set_title('Chebyshev Type II bandpass (4-7 Hz) frequency response')
    #axs[0, 1].set_xlabel('frequency [Hz]')
    axs[0, 1].set_ylabel('Amplitude [dB]')

    axs[0, 2].semilogx(w_alpha, 20 * np.log10(np.maximum(abs(h_alpha), 1e-5))) # w = frequencies corresponding to the h value = frequency response in complex values 
    axs[0, 2].set_title('Chebyshev Type II bandpass (8-12 Hz) frequency response')
    #axs[0, 2].set_xlabel('frequency [Hz]')
    axs[0, 2].set_ylabel('Amplitude [dB]')

    axs[1, 0].semilogx(w_beta, 20 * np.log10(np.maximum(abs(h_beta), 1e-5))) # w = frequencies corresponding to the h value = frequency response in complex values 
    axs[1, 0].set_title('Chebyshev Type II bandpass (13-30 Hz) frequency response')
    axs[1, 0].set_xlabel('frequency [Hz]')
    axs[1, 0].set_ylabel('Amplitude [dB]')

    axs[1, 1].semilogx(w_gamma, 20 * np.log10(np.maximum(abs(h_gamma), 1e-5))) # w = frequencies corresponding to the h value = frequency response in complex values 
    axs[1, 1].set_title('Chebyshev Type II bandpass (30-63 Hz) frequency response')
    axs[1, 1].set_xlabel('frequency [Hz]')
    axs[1, 1].set_ylabel('Amplitude [dB]')

            

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()
    '''
    return sos_delta, sos_theta, sos_alpha, sos_beta, sos_gamma


# Defining function to extract numbers for the name of the hypnodensity files 
def extract_numbers_from_filename(filename):
    # Define a regular expression pattern to match numbers
    pattern = r'\d+'

    # Use re.findall to find all matches in the filename
    matches = re.findall(pattern, filename)

    # Convert the matched strings to integers
    numbers = [int(match) for match in matches]

    return numbers


# Remove special signs - extracting numbers and letters 
def extract_letters_and_numbers(input_string):
    # Input= string
    # Output = string 
    return ''.join(char for char in input_string if char.isalnum())


# Function to extract only the EDF files in a folder and loop over dem 
def list_files_in_folder(folder_path, file_extension=".EDF"): # or EDF (depending on the file name)
    edf_files = []

    # Loop through files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            # If the file has the specified extension, add it to the list
            edf_files.append(os.path.join(folder_path, filename))

    return edf_files



# Function for splitting a string by length 
def split_string_by_length(s, length):
    return [s[i:i+length] for i in range(0, len(s), length)]





# Defining function to generate hypnograms for electrodes with correct names and epoch sizes 

def Usleep_2channels(edf_file_uploaded,make_folder_path_uploaded,epoch_size_in_seconds_uploaded,numbers_found_uploaded ):

    #### Describtion of function ##########
    # This function generates hypnodensity features for each EDF file and the chosen epoch size. 


    ##### Packages neeeded to run this function: ####
    # import usleep_api
    # from usleep_api import USleepAPI
    # import os
    # import numpy as np
    # import copy
    # import sys
    # Using sys function to import 'My_functions_script'
    # sys.path.insert(0, 'C:/Users/natas/Documents/Master thesis code')
    # Import My_functions_script
    # from My_functions_script import extract_numbers_from_filename, extract_letters_and_numbers, list_files_in_folder


    # Paste this piece of code in before running this function: 

    #numbers_found = extract_numbers_from_filename(edf_file)
    #print("Numbers found in the filename:", numbers_found)
    #create output folder 
    #make_folder_path = os.path.join(output_path, str(numbers_found[1]))
    # Check if the folder already exists before creating it
    #if not os.path.exists(make_folder_path):
    #   os.makedirs(make_folder_path)
    #   print(f"Folder created for Patient ID {str(numbers_found[1])}")
    #else:
    #   print(f"Folder for Patient ID {str(numbers_found[1])} already exists")
    
        
    # Copying the files in the function 
    edf_file=copy.deepcopy(edf_file_uploaded)
    make_folder_path=copy.deepcopy(make_folder_path_uploaded)
    epoch_size_in_seconds = copy.deepcopy(epoch_size_in_seconds_uploaded)
    numbers_found = copy.deepcopy(numbers_found_uploaded)

    print('Numbers found')
    print(numbers_found)

    # Saving token - this should be changed every 12 hours 
    os.environ ['USLEEP_API_TOKEN']='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3MTQ5NzM3ODksImlhdCI6MTcxNDkzMDU4OSwibmJmIjoxNzE0OTMwNTg5LCJpZGVudGl0eSI6ImZmYzUxYTJiYzc2MiJ9.tqKheV1gn14v3X-oQmXkXtESAzb_GLiCEf0xXBmmyyI'

    # Create an API object with API token stored in environment variable 
    api = usleep_api.USleepAPI(api_token=os.environ['USLEEP_API_TOKEN'])


    # Create an API object and (optionally) a new session.
    session = api.new_session(session_name="my_session")

    # See a list of valid models and set which model to use
    #logger.info(f"Available models: {session.get_model_names()}")

    # Setting the model 
    session.set_model("U-Sleep-EEG v2.0") # Uses only one EEG channel 


    # Upload a local file (usually .edf format)
    session.upload_file(edf_file, anonymize_before_upload=False)
    print('done uploading one file')


    # Extracting a list with channels to loop over 
    groups_found=session._infer_channel_groups()
    groups=copy.deepcopy(groups_found)
    print('The final list with groups are printed')
    print(groups)
    print('length of groups list')
    n=len(groups)

    # Looping over the EEG channels, such that a hypnodensity matrix is generated for a single EEG channel 
    for i in range(n):
        # Start the prediction on a single channel from channel groups
            
        print('Printing i')
        print(i)

        # Setting the model for each loop 
        session.set_model("U-Sleep-EEG v2.0")
            
        ########## Predicting #################
        # Choosing a epoch_size_in_seconds and the 128 is the sampling frequency 
        session.predict(data_per_prediction=128*epoch_size_in_seconds,channel_groups=[groups[i]]) #[["C3_M2"],["O1_M2"]]
        #######################################

        # Checking the channel the loop is looking at 
        print('Group channel - check')
        print(groups[i])

        #session.stream_prediction_log()

        success = session.wait_for_completion()


        if success:
            # Fetch hypnogram
            hyp = session.get_hypnogram()
            logger.info(hyp["hypnogram"])


            # Download hypnogram file with correct name 
            # Extract only letters function for naming the hypnodensity feature 
            Electrode=extract_letters_and_numbers(str(groups[i]))
            print('Extracting letters and numbers')
            print(Electrode)
            print(type(Electrode))
                
            # Downloading the hypnodensity features 
            output_filename =f"hypnogram_ID_"+str(numbers_found)+"_electrode_"+str(Electrode)+"_epocssize_"+str(epoch_size_in_seconds)+".tsv" # Later add patient ID in the name
            output_file_path=os.path.join(make_folder_path,output_filename)
            print(output_file_path)
                
            # To generate hypnodensity features this must be set = with_confidence_scores=True 
            # To generate a hypnogram this must be set = with_confidence_scores=False 
            session.download_hypnogram(output_file_path, file_type="npy",with_confidence_scores=True)

        else:
            logger.error("Prediction failed.")
        
    # Delete session (i.e., uploaded file, prediction and logs)
    # Can only handle 5 sessions at a time --> This is why the session is deleted after each run.        

    print('Function done')  
    session.delete_session()          
    return groups



def correlation_multiple_electrodes (input_path_uploaded,epoch_size_in_seconds_uploaded):
    
    # The input path uploaded should be where the hypnograms are saved 

    # This function outputs CSV files 

    # Copying input variables 
    output_path = copy.deepcopy(input_path_uploaded)
    epoch_size_in_seconds = copy.deepcopy(epoch_size_in_seconds_uploaded)
    epoch_size_in_seconds = f"epocssize_"+str(epoch_size_in_seconds) # This part makes sure that the function selects the exact name and number


    # Defining temporary variables: 
    # Defining variables to collect information for dataframe
    # Combination of electrodes E1E2 (electrode 1 and 2 for two electrodes (_2))
    temp_patient_id_E1E2_2=[]
    temp_correlation_Wake_E1E2_2=[]
    temp_correlation_N1_E1E2_2=[]
    temp_correlation_N2_E1E2_2=[]
    temp_correlation_N3_E1E2_2=[]
    temp_correlation_REM_E1E2_2=[]
    # Combination of electrodes E1E2 (electrode 1 and 2 for three electrodes (_3))
    temp_patient_id_E1E2_3=[]
    temp_correlation_Wake_E1E2_3=[]
    temp_correlation_N1_E1E2_3=[]
    temp_correlation_N2_E1E2_3=[]
    temp_correlation_N3_E1E2_3=[]
    temp_correlation_REM_E1E2_3=[]
    # Combinaton of electrodes E1E3
    temp_patient_id_E1E3_3=[]
    temp_correlation_Wake_E1E3_3=[]
    temp_correlation_N1_E1E3_3=[]
    temp_correlation_N2_E1E3_3=[]
    temp_correlation_N3_E1E3_3=[]
    temp_correlation_REM_E1E3_3=[]
    # Combination of electrodes E2E3
    temp_patient_id_E2E3_3=[]
    temp_correlation_Wake_E2E3_3=[]
    temp_correlation_N1_E2E3_3=[]
    temp_correlation_N2_E2E3_3=[]
    temp_correlation_N3_E2E3_3=[]
    temp_correlation_REM_E2E3_3=[]

    # Combination of 4 electrods --> 6 combinations 
    # first combination
    temp_patient_id_E1E2_4=[]
    temp_correlation_Wake_E1E2_4=[]
    temp_correlation_N1_E1E2_4=[]
    temp_correlation_N2_E1E2_4=[]
    temp_correlation_N3_E1E2_4=[]
    temp_correlation_REM_E1E2_4=[]
    # second combination 
    temp_patient_id_E1E3_4=[]
    temp_correlation_Wake_E1E3_4=[]
    temp_correlation_N1_E1E3_4=[]
    temp_correlation_N2_E1E3_4=[]
    temp_correlation_N3_E1E3_4=[]
    temp_correlation_REM_E1E3_4=[]
    # third combination 
    temp_patient_id_E2E3_4=[]
    temp_correlation_Wake_E2E3_4=[]
    temp_correlation_N1_E2E3_4=[]
    temp_correlation_N2_E2E3_4=[]
    temp_correlation_N3_E2E3_4=[]
    temp_correlation_REM_E2E3_4=[]
    # fourth combination 
    temp_patient_id_E1E4_4=[]
    temp_correlation_Wake_E1E4_4=[]
    temp_correlation_N1_E1E4_4=[]
    temp_correlation_N2_E1E4_4=[]
    temp_correlation_N3_E1E4_4=[]
    temp_correlation_REM_E1E4_4=[]
    # fifth combination  
    temp_patient_id_E2E4_4=[]
    temp_correlation_Wake_E2E4_4=[]
    temp_correlation_N1_E2E4_4=[]
    temp_correlation_N2_E2E4_4=[]
    temp_correlation_N3_E2E4_4=[]
    temp_correlation_REM_E2E4_4=[]
    # sixth combination 
    temp_patient_id_E3E4_4=[]
    temp_correlation_Wake_E3E4_4=[]
    temp_correlation_N1_E3E4_4=[]
    temp_correlation_N2_E3E4_4=[]
    temp_correlation_N3_E3E4_4=[]
    temp_correlation_REM_E3E4_4=[]

     # Combination of 5 electrods --> 10 combinations 
    # first combination of 10
    temp_patient_id_E1E2_5=[]
    temp_correlation_Wake_E1E2_5=[]
    temp_correlation_N1_E1E2_5=[]
    temp_correlation_N2_E1E2_5=[]
    temp_correlation_N3_E1E2_5=[]
    temp_correlation_REM_E1E2_5=[]
    # second combination of 10
    temp_patient_id_E1E3_5=[]
    temp_correlation_Wake_E1E3_5=[]
    temp_correlation_N1_E1E3_5=[]
    temp_correlation_N2_E1E3_5=[]
    temp_correlation_N3_E1E3_5=[]
    temp_correlation_REM_E1E3_5=[]
    # third combination of 10
    temp_patient_id_E2E3_5=[]
    temp_correlation_Wake_E2E3_5=[]
    temp_correlation_N1_E2E3_5=[]
    temp_correlation_N2_E2E3_5=[]
    temp_correlation_N3_E2E3_5=[]
    temp_correlation_REM_E2E3_5=[]
    # fourth combination of 10
    temp_patient_id_E1E4_5=[]
    temp_correlation_Wake_E1E4_5=[]
    temp_correlation_N1_E1E4_5=[]
    temp_correlation_N2_E1E4_5=[]
    temp_correlation_N3_E1E4_5=[]
    temp_correlation_REM_E1E4_5=[]
    # fifth combination of 10
    temp_patient_id_E2E4_5=[]
    temp_correlation_Wake_E2E4_5=[]
    temp_correlation_N1_E2E4_5=[]
    temp_correlation_N2_E2E4_5=[]
    temp_correlation_N3_E2E4_5=[]
    temp_correlation_REM_E2E4_5=[]
    # sixth combination of 10
    temp_patient_id_E3E4_5=[]
    temp_correlation_Wake_E3E4_5=[]
    temp_correlation_N1_E3E4_5=[]
    temp_correlation_N2_E3E4_5=[]
    temp_correlation_N3_E3E4_5=[]
    temp_correlation_REM_E3E4_5=[]
    # seventh combination of 10
    temp_patient_id_E1E5_5=[]
    temp_correlation_Wake_E1E5_5=[]
    temp_correlation_N1_E1E5_5=[]
    temp_correlation_N2_E1E5_5=[]
    temp_correlation_N3_E1E5_5=[]
    temp_correlation_REM_E1E5_5=[]
    # 8th combination of 10
    temp_patient_id_E2E5_5=[]
    temp_correlation_Wake_E2E5_5=[]
    temp_correlation_N1_E2E5_5=[]
    temp_correlation_N2_E2E5_5=[]
    temp_correlation_N3_E2E5_5=[]
    temp_correlation_REM_E2E5_5=[]
    # 9th combination of 10
    temp_patient_id_E3E5_5=[]
    temp_correlation_Wake_E3E5_5=[]
    temp_correlation_N1_E3E5_5=[]
    temp_correlation_N2_E3E5_5=[]
    temp_correlation_N3_E3E5_5=[]
    temp_correlation_REM_E3E5_5=[]
    # 10th combination of 10
    temp_patient_id_E4E5_5=[]
    temp_correlation_Wake_E4E5_5=[]
    temp_correlation_N1_E4E5_5=[]
    temp_correlation_N2_E4E5_5=[]
    temp_correlation_N3_E4E5_5=[]
    temp_correlation_REM_E4E5_5=[]

    
     # Combination of 6 electrods --> 15 combinations 
    # first combination of 15
    temp_patient_id_E1E2_6=[]
    temp_correlation_Wake_E1E2_6=[]
    temp_correlation_N1_E1E2_6=[]
    temp_correlation_N2_E1E2_6=[]
    temp_correlation_N3_E1E2_6=[]
    temp_correlation_REM_E1E2_6=[]
    # second combination of 15
    temp_patient_id_E1E3_6=[]
    temp_correlation_Wake_E1E3_6=[]
    temp_correlation_N1_E1E3_6=[]
    temp_correlation_N2_E1E3_6=[]
    temp_correlation_N3_E1E3_6=[]
    temp_correlation_REM_E1E3_6=[]
    # third combination of 15
    temp_patient_id_E2E3_6=[]
    temp_correlation_Wake_E2E3_6=[]
    temp_correlation_N1_E2E3_6=[]
    temp_correlation_N2_E2E3_6=[]
    temp_correlation_N3_E2E3_6=[]
    temp_correlation_REM_E2E3_6=[]
    # fourth combination of 15
    temp_patient_id_E1E4_6=[]
    temp_correlation_Wake_E1E4_6=[]
    temp_correlation_N1_E1E4_6=[]
    temp_correlation_N2_E1E4_6=[]
    temp_correlation_N3_E1E4_6=[]
    temp_correlation_REM_E1E4_6=[]
    # fifth combination of 15
    temp_patient_id_E2E4_6=[]
    temp_correlation_Wake_E2E4_6=[]
    temp_correlation_N1_E2E4_6=[]
    temp_correlation_N2_E2E4_6=[]
    temp_correlation_N3_E2E4_6=[]
    temp_correlation_REM_E2E4_6=[]
    # sixth combination of 15
    temp_patient_id_E3E4_6=[]
    temp_correlation_Wake_E3E4_6=[]
    temp_correlation_N1_E3E4_6=[]
    temp_correlation_N2_E3E4_6=[]
    temp_correlation_N3_E3E4_6=[]
    temp_correlation_REM_E3E4_6=[]
    # seventh combination of 15
    temp_patient_id_E1E5_6=[]
    temp_correlation_Wake_E1E5_6=[]
    temp_correlation_N1_E1E5_6=[]
    temp_correlation_N2_E1E5_6=[]
    temp_correlation_N3_E1E5_6=[]
    temp_correlation_REM_E1E5_6=[]
    # 8th combination of 15
    temp_patient_id_E2E5_6=[]
    temp_correlation_Wake_E2E5_6=[]
    temp_correlation_N1_E2E5_6=[]
    temp_correlation_N2_E2E5_6=[]
    temp_correlation_N3_E2E5_6=[]
    temp_correlation_REM_E2E5_6=[]
    # 9th combination of 15
    temp_patient_id_E3E5_6=[]
    temp_correlation_Wake_E3E5_6=[]
    temp_correlation_N1_E3E5_6=[]
    temp_correlation_N2_E3E5_6=[]
    temp_correlation_N3_E3E5_6=[]
    temp_correlation_REM_E3E5_6=[]
    # 10th combination of 15
    temp_patient_id_E4E5_6=[]
    temp_correlation_Wake_E4E5_6=[]
    temp_correlation_N1_E4E5_6=[]
    temp_correlation_N2_E4E5_6=[]
    temp_correlation_N3_E4E5_6=[]
    temp_correlation_REM_E4E5_6=[]
    # 11th combination of 15
    temp_patient_id_E1E6_6=[]
    temp_correlation_Wake_E1E6_6=[]
    temp_correlation_N1_E1E6_6=[]
    temp_correlation_N2_E1E6_6=[]
    temp_correlation_N3_E1E6_6=[]
    temp_correlation_REM_E1E6_6=[]
    # 12th combination of 15
    temp_patient_id_E2E6_6=[]
    temp_correlation_Wake_E2E6_6=[]
    temp_correlation_N1_E2E6_6=[]
    temp_correlation_N2_E2E6_6=[]
    temp_correlation_N3_E2E6_6=[]
    temp_correlation_REM_E2E6_6=[]  
    # 13th combination of 15
    temp_patient_id_E3E6_6=[]
    temp_correlation_Wake_E3E6_6=[]
    temp_correlation_N1_E3E6_6=[]
    temp_correlation_N2_E3E6_6=[]
    temp_correlation_N3_E3E6_6=[]
    temp_correlation_REM_E3E6_6=[]
    # 14th combination of 15
    temp_patient_id_E4E6_6=[]
    temp_correlation_Wake_E4E6_6=[]
    temp_correlation_N1_E4E6_6=[]
    temp_correlation_N2_E4E6_6=[]
    temp_correlation_N3_E4E6_6=[]
    temp_correlation_REM_E4E6_6=[]
    # 15th combination of 15
    temp_patient_id_E5E6_6=[]
    temp_correlation_Wake_E5E6_6=[]
    temp_correlation_N1_E5E6_6=[]
    temp_correlation_N2_E5E6_6=[]
    temp_correlation_N3_E5E6_6=[]
    temp_correlation_REM_E5E6_6=[]

    templist_visit=[]

    # List of folder paths
    main_folders = os.listdir(output_path)#["folder1", "folder2", "folder3"]  # Replace with your actual folder paths
    print('List of main folders unsorted')
    print(main_folders)  

    main_folders=sorted(main_folders)

    print('List of main folders sorted')
    print(main_folders)  

    # Loop through main folders
    for main_folder in main_folders:
        folder_path = os.path.join(output_path, main_folder)  # Update with your actual parent folder path
            
        # Extracting patient ID
        #patientID=str(main_folder) 
        file_name=str(main_folder) 
        print('Folder') # Check which folder we are inside 
        print(file_name )

        print('PatientID')
        # Skipping the first part of the filename to extract the real patientID 
        patientID=file_name[30:] #restructuredfile_RBD_controls_STNF00006
        print(patientID)
        
        ############## Loading electrode names ###################
        # Loading text file for each patient ID containing the name of the electrodes present for this patient
        # This file was a tuple before and will be converted to a tuple
        text_file_path = os.path.join(folder_path, f'Name_electrodes.txt') 
        
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
        ############### Done loading electrode names in correct variable type ##########################################

        
        
        ########### Combining electrodes for input to the correlation ######################
        # Make combination of electrode names to loop over. Used to select the electrode-pair used in correlation. 
        iterable=Electrodes
        r=2 # Length of subsequence 
        E_combinations=list(itertools.combinations(iterable,r))

        print('Combinations of electrode names in file')
        print(E_combinations)
        print(E_combinations[0][0])
        print(E_combinations[0][1])
        print(type(E_combinations))
        print(len(E_combinations))

        ############# Done combining the electrodes ###############################


        ######## Looping over the electrode combinations #############
        # The loop will select a combination of two electrodes for the correlation 

        for d in range(len(E_combinations)):
            print('Finding new electrode combination in same main folder')
            print('full E_combinations')
            print(len(E_combinations))
            #print(E_combinations[0])
            #print(E_combinations[1])
            print('D')
            print(d)
            
            # Selecting the consistent part of the filename
            consistent_part_1 = E_combinations[d][0] # e.g "C3M2"  # Replace with the consistent part of the first array's name
            consistent_part_2 = E_combinations[d][1] # e.g "O1M2"  # Replace with the consistent part of the second array's name

            # Defining electrode combination for naming the CSV files in the end 
            Electrode_combination_naming = [consistent_part_1, consistent_part_2]
            Electrode_combination_naming = extract_letters_and_numbers(Electrode_combination_naming)
            print('Electrode combination')
            print(Electrode_combination_naming)

            # Checking the consistent part 
            print('Consistent part 1')
            print(consistent_part_1)
            print('Consistent part 2')
            print(consistent_part_2)


            # Loop through files in the folder 
            for filename in os.listdir(folder_path):
                
            

                # Finding the consistent parts of the filename to select the correct files 
                if consistent_part_1 in filename and f'{epoch_size_in_seconds}.npy' in filename:
                    # Load array for the first consistent part
                    print('Loading file 1')
                    Electrode1 = np.load(os.path.join(folder_path, filename))
                    #Electrode1 = copy.deepcopy(Electrode1)
                    print(filename)
                    print(epoch_size_in_seconds)
                        
                elif consistent_part_2 in filename and f'{epoch_size_in_seconds}.npy' in filename:
                    # Load array for the second consistent part
                    print('Loading file 2')
                    Electrode2 = np.load(os.path.join(folder_path, filename))
                    #Electrode2 = copy.deepcopy(Electrode2)
                    print(filename)
                    print(epoch_size_in_seconds)

                
            
            
            # Preparing for correlation - creating empty lists 
            print('Length Electrode[:,j]')
            print(Electrode1.shape)
            k=Electrode2.shape[1] # value = 5
            print(k)
            correlation_structure =[]
            temporary_list =[]


            ##### Correlation loop ################

            for j in range(k):
                # Correlation between column1 and column1 in each hypnodensity matrix 
                # and keep going like this for all columns. 
                # This will end up with a result of correlation within sleep stages: 
                # Wake vs. Wake, N1 vs. N1, N2 vs. N2, N3 vs. N3, REM vs. REM
                # for the chosen electrodes and their hypnodensity matrices 

                # This code will generate 5 (2,2) matrices and select the relevant value from the matrix 
                # This is done using Pearson correlation coefficient 
                print('Lengt of electrode1 and electrode2 in correlation loop')
                print(len(Electrode1[:,j]))
                print(len(Electrode2[:,j]))
                # Using Pearson correlation coefficient 
                correlation_matrix=np.corrcoef(Electrode1[:,j],Electrode2[:,j])
                print('Correlation matrix inside loop')
                print(correlation_matrix[0,1]) # Saving the relevant value from the (2,2) matrix
                # The Pearson (product-moment) correlation coefficient is a measure of the linear relationship between two features.
                # Pearson correlation coefficient can take on any real value in the range −1 ≤ r ≤ 1.
                
                # Saving the value in a temporary list 
                temporary_list.append(correlation_matrix[0,1])
                
                # Creating a 3D structure with correlation matrices 
                correlation_structure= np.stack(temporary_list,axis=0)

            # Generating the correlation structure 
            print('Correlation structure!!!')
            print(correlation_structure)
            print(correlation_structure.shape)

            # Making sure not to save wrong electrodes 
            del Electrode1, Electrode2
            



            ######### Storing the correlation structures according to number of electrodes in the data ############
            
            # Using if-statements to determine how many electrodes present in the data for the single patient
            # Storing the correlation values in temporary lists and dictonaries  

            # Explanation of names: 
            
            # Two electrodes = one combination of E1E2_2 
            #(E1 and E2 representing the electrode1 and electrode2 and 
            # the _2 to indicate only two electrodes are present)

            # Three electrodes = three combinations of E1E2_3, E1E3_3 and E2E3_3 
            # Applying same name rules as for the two electrodes 

                
            
            # Only two electrodes with one combination of electrodes 
            if len(E_combinations) ==1:
                
                print('Temporary E-combinations length 1')

                # Filling out temporary variables: 
                temp_patient_id_E1E2_2.append(patientID)
                temp_correlation_Wake_E1E2_2.append(correlation_structure[0])
                temp_correlation_N1_E1E2_2.append(correlation_structure[1])
                temp_correlation_N2_E1E2_2.append(correlation_structure[2])
                temp_correlation_N3_E1E2_2.append(correlation_structure[3])
                temp_correlation_REM_E1E2_2.append(correlation_structure[4])

                
                # Stacking the variables 
                patient_id_stacked_E1E2_2=np.stack(temp_patient_id_E1E2_2)
                N1_E1E2_2=np.stack(temp_correlation_N1_E1E2_2)
                N2_E1E2_2=np.stack(temp_correlation_N2_E1E2_2)
                N3_E1E2_2=np.stack(temp_correlation_N3_E1E2_2)
                Wake_E1E2_2=np.stack(temp_correlation_Wake_E1E2_2)
                REM_E1E2_2=np.stack(temp_correlation_REM_E1E2_2)

                ##### Saving values in a dictonary #####
                patient_ids_E1E2_2=patient_id_stacked_E1E2_2

                print('Electrode combination')
                print(Electrode_combination_naming)

                # Create a dictionary to store patient ID and corresponding information
                patient_data_dict_E1E2_2 = {
                    'PatientID': patient_ids_E1E2_2,
                    'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E2_2.tolist(),
                    'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E2_2.tolist(),
                    'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E2_2.tolist(),
                    'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E2_2.tolist(),
                    'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E2_2.tolist(),
                }

                print('Patient dictionary E1E2_2 - two electrodes, one combination')
                print(patient_data_dict_E1E2_2)


            # Three electrodes are present giving three combinations 
            elif len(E_combinations) ==3:
                
                
                print('Temporary E-combinations length 3')

                # Storing the first combination E1E2_3
                if E_combinations[0]==E_combinations[d]:

                    print('E_combinations[0] chosen for three electrodes first combination')
                    print(E_combinations[0])
                    
                    # Filling out temporary variables: 
                    temp_patient_id_E1E2_3.append(patientID)
                    temp_correlation_Wake_E1E2_3.append(correlation_structure[0])
                    temp_correlation_N1_E1E2_3.append(correlation_structure[1])
                    temp_correlation_N2_E1E2_3.append(correlation_structure[2])
                    temp_correlation_N3_E1E2_3.append(correlation_structure[3])
                    temp_correlation_REM_E1E2_3.append(correlation_structure[4])

                    
                    # Stacking the variables 
                    patient_id_stacked_E1E2_3=np.stack(temp_patient_id_E1E2_3)
                    N1_E1E2_3=np.stack(temp_correlation_N1_E1E2_3)
                    N2_E1E2_3=np.stack(temp_correlation_N2_E1E2_3)
                    N3_E1E2_3=np.stack(temp_correlation_N3_E1E2_3)
                    Wake_E1E2_3=np.stack(temp_correlation_Wake_E1E2_3)
                    REM_E1E2_3=np.stack(temp_correlation_REM_E1E2_3)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E2_3=patient_id_stacked_E1E2_3

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E2_3 = {
                        'PatientID': patient_ids_E1E2_3,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E2_3.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E2_3.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E2_3.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E2_3.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E2_3.tolist(),
                    }

                    print('Patient dictionary E1E2_3 - three electrodes, first combination')
                    print(patient_data_dict_E1E2_3)
                
                # Storing the second combination E1E3_3
                elif E_combinations[1]==E_combinations[d]:

                    print('E_combinations[1] chosen for three electrodes second combination')
                    print(E_combinations[1])

                    # Filling out temporary variables: 
                    temp_patient_id_E1E3_3.append(patientID)
                    temp_correlation_Wake_E1E3_3.append(correlation_structure[0])
                    temp_correlation_N1_E1E3_3.append(correlation_structure[1])
                    temp_correlation_N2_E1E3_3.append(correlation_structure[2])
                    temp_correlation_N3_E1E3_3.append(correlation_structure[3])
                    temp_correlation_REM_E1E3_3.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E1E3_3=np.stack(temp_patient_id_E1E3_3)
                    N1_E1E3_3=np.stack(temp_correlation_N1_E1E3_3)
                    N2_E1E3_3=np.stack(temp_correlation_N2_E1E3_3)
                    N3_E1E3_3=np.stack(temp_correlation_N3_E1E3_3)
                    Wake_E1E3_3=np.stack(temp_correlation_Wake_E1E3_3)
                    REM_E1E3_3=np.stack(temp_correlation_REM_E1E3_3)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E3_3=patient_id_stacked_E1E3_3

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E3_3 = {
                        'PatientID': patient_ids_E1E3_3,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E3_3.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E3_3.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E3_3.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E3_3.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E3_3.tolist(),
                    }
                    
                    print('Patient dictionary E1E3_3 - three electrodes, second combination')
                    print(patient_data_dict_E1E3_3)

                # Storing the third combination E2E3_3
                elif E_combinations[2]==E_combinations[d]:

                    print('E_combinations[2] chosen for three electrodes third combination')
                    print(E_combinations[2])

                    # Filling out temporary variables: 
                    temp_patient_id_E2E3_3.append(patientID)
                    temp_correlation_Wake_E2E3_3.append(correlation_structure[0])
                    temp_correlation_N1_E2E3_3.append(correlation_structure[1])
                    temp_correlation_N2_E2E3_3.append(correlation_structure[2])
                    temp_correlation_N3_E2E3_3.append(correlation_structure[3])
                    temp_correlation_REM_E2E3_3.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E2E3_3=np.stack(temp_patient_id_E2E3_3)
                    N1_E2E3_3=np.stack(temp_correlation_N1_E2E3_3)
                    N2_E2E3_3=np.stack(temp_correlation_N2_E2E3_3)
                    N3_E2E3_3=np.stack(temp_correlation_N3_E2E3_3)
                    Wake_E2E3_3=np.stack(temp_correlation_Wake_E2E3_3)
                    REM_E2E3_3=np.stack(temp_correlation_REM_E2E3_3)

                    ##### Saving values in a dataframe #####
                    patient_ids_E2E3_3=patient_id_stacked_E2E3_3

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E2E3_3 = {
                        'PatientID': patient_ids_E2E3_3,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E2E3_3.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E2E3_3.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E2E3_3.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E2E3_3.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E2E3_3.tolist(),
                    }

                    print('Patient dictionary E2E3_3 - three electrodes, third combination')
                    print(patient_data_dict_E2E3_3)

            # 4 electrodes will give 6 combinations 

            elif len(E_combinations) ==6:
            
                
                print('Temporary E-combinations length 6')

                # Storing the first combination E1E2_4
                if E_combinations[0]==E_combinations[d]:

                    print('E_combinations[0] chosen for four electrodes first combination')
                    print(E_combinations[0])
                    
                    # Filling out temporary variables: 
                    temp_patient_id_E1E2_4.append(patientID)
                    temp_correlation_Wake_E1E2_4.append(correlation_structure[0])
                    temp_correlation_N1_E1E2_4.append(correlation_structure[1])
                    temp_correlation_N2_E1E2_4.append(correlation_structure[2])
                    temp_correlation_N3_E1E2_4.append(correlation_structure[3])
                    temp_correlation_REM_E1E2_4.append(correlation_structure[4])

                    
                    # Stacking the variables 
                    patient_id_stacked_E1E2_4=np.stack(temp_patient_id_E1E2_4)
                    N1_E1E2_4=np.stack(temp_correlation_N1_E1E2_4)
                    N2_E1E2_4=np.stack(temp_correlation_N2_E1E2_4)
                    N3_E1E2_4=np.stack(temp_correlation_N3_E1E2_4)
                    Wake_E1E2_4=np.stack(temp_correlation_Wake_E1E2_4)
                    REM_E1E2_4=np.stack(temp_correlation_REM_E1E2_4)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E2_4=patient_id_stacked_E1E2_4

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E2_4 = {
                        'PatientID': patient_ids_E1E2_4,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E2_4.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E2_4.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E2_4.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E2_4.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E2_4.tolist(),
                    }

                    print('Patient dictionary E1E2_4 - three electrodes, first combination')
                    print(patient_data_dict_E1E2_4)
                
                # Storing the second combination E1E3_4
                elif E_combinations[1]==E_combinations[d]:

                    print('E_combinations[1] chosen for four electrodes second combination')
                    print(E_combinations[1])

                    # Filling out temporary variables: 
                    temp_patient_id_E1E3_4.append(patientID)
                    temp_correlation_Wake_E1E3_4.append(correlation_structure[0])
                    temp_correlation_N1_E1E3_4.append(correlation_structure[1])
                    temp_correlation_N2_E1E3_4.append(correlation_structure[2])
                    temp_correlation_N3_E1E3_4.append(correlation_structure[3])
                    temp_correlation_REM_E1E3_4.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E1E3_4=np.stack(temp_patient_id_E1E3_4)
                    N1_E1E3_4=np.stack(temp_correlation_N1_E1E3_4)
                    N2_E1E3_4=np.stack(temp_correlation_N2_E1E3_4)
                    N3_E1E3_4=np.stack(temp_correlation_N3_E1E3_4)
                    Wake_E1E3_4=np.stack(temp_correlation_Wake_E1E3_4)
                    REM_E1E3_4=np.stack(temp_correlation_REM_E1E3_4)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E3_4=patient_id_stacked_E1E3_4

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E3_4 = {
                        'PatientID': patient_ids_E1E3_4,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E3_4.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E3_4.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E3_4.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E3_4.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E3_4.tolist(),
                    }
                    
                    print('Patient dictionary E1E3_4 - four electrodes, second combination')
                    print(patient_data_dict_E1E3_4)

                # Storing the third combination E2E3_4
                elif E_combinations[2]==E_combinations[d]:

                    print('E_combinations[2] chosen for four electrodes third combination')
                    print(E_combinations[2])

                    # Filling out temporary variables: 
                    temp_patient_id_E2E3_4.append(patientID)
                    temp_correlation_Wake_E2E3_4.append(correlation_structure[0])
                    temp_correlation_N1_E2E3_4.append(correlation_structure[1])
                    temp_correlation_N2_E2E3_4.append(correlation_structure[2])
                    temp_correlation_N3_E2E3_4.append(correlation_structure[3])
                    temp_correlation_REM_E2E3_4.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E2E3_4=np.stack(temp_patient_id_E2E3_4)
                    N1_E2E3_4=np.stack(temp_correlation_N1_E2E3_4)
                    N2_E2E3_4=np.stack(temp_correlation_N2_E2E3_4)
                    N3_E2E3_4=np.stack(temp_correlation_N3_E2E3_4)
                    Wake_E2E3_4=np.stack(temp_correlation_Wake_E2E3_4)
                    REM_E2E3_4=np.stack(temp_correlation_REM_E2E3_4)

                    ##### Saving values in a dataframe #####
                    patient_ids_E2E3_4=patient_id_stacked_E2E3_4

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E2E3_4 = {
                        'PatientID': patient_ids_E2E3_4,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E2E3_4.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E2E3_4.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E2E3_4.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E2E3_4.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E2E3_4.tolist(),
                    }

                    print('Patient dictionary E2E3_4 - four electrodes, third combination')
                    print(patient_data_dict_E2E3_4)

                    
                # Storing the fourth combination E1E4_4
                elif E_combinations[3]==E_combinations[d]:

                    print('E_combinations[3] chosen for four electrodes fourth combination')
                    print(E_combinations[3])

                    # Filling out temporary variables: 
                    temp_patient_id_E1E4_4.append(patientID)
                    temp_correlation_Wake_E1E4_4.append(correlation_structure[0])
                    temp_correlation_N1_E1E4_4.append(correlation_structure[1])
                    temp_correlation_N2_E1E4_4.append(correlation_structure[2])
                    temp_correlation_N3_E1E4_4.append(correlation_structure[3])
                    temp_correlation_REM_E1E4_4.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E1E4_4=np.stack(temp_patient_id_E1E4_4)
                    N1_E1E4_4=np.stack(temp_correlation_N1_E1E4_4)
                    N2_E1E4_4=np.stack(temp_correlation_N2_E1E4_4)
                    N3_E1E4_4=np.stack(temp_correlation_N3_E1E4_4)
                    Wake_E1E4_4=np.stack(temp_correlation_Wake_E1E4_4)
                    REM_E1E4_4=np.stack(temp_correlation_REM_E1E4_4)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E4_4=patient_id_stacked_E1E4_4

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E4_4 = {
                        'PatientID': patient_ids_E1E4_4,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E4_4.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E4_4.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E4_4.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E4_4.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E4_4.tolist(),
                    }

                    print('Patient dictionary E1E4_4 - four electrodes, fourth combination')
                    print(patient_data_dict_E1E4_4)
                
                # Storing the fourth combination E2E4_4
                elif E_combinations[4]==E_combinations[d]:

                    print('E_combinations[4] chosen for four electrodes fifth combination')
                    print(E_combinations[4])

                    # Filling out temporary variables: 
                    temp_patient_id_E2E4_4.append(patientID)
                    temp_correlation_Wake_E2E4_4.append(correlation_structure[0])
                    temp_correlation_N1_E2E4_4.append(correlation_structure[1])
                    temp_correlation_N2_E2E4_4.append(correlation_structure[2])
                    temp_correlation_N3_E2E4_4.append(correlation_structure[3])
                    temp_correlation_REM_E2E4_4.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E2E4_4=np.stack(temp_patient_id_E2E4_4)
                    N1_E2E4_4=np.stack(temp_correlation_N1_E2E4_4)
                    N2_E2E4_4=np.stack(temp_correlation_N2_E2E4_4)
                    N3_E2E4_4=np.stack(temp_correlation_N3_E2E4_4)
                    Wake_E2E4_4=np.stack(temp_correlation_Wake_E2E4_4)
                    REM_E2E4_4=np.stack(temp_correlation_REM_E2E4_4)

                    ##### Saving values in a dataframe #####
                    patient_ids_E2E4_4=patient_id_stacked_E2E4_4

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E2E4_4 = {
                        'PatientID': patient_ids_E2E4_4,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E2E4_4.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E2E4_4.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E2E4_4.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E2E4_4.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E2E4_4.tolist(),
                    }

                    print('Patient dictionary E2E4_4 - four electrodes, fifth combination')
                    print(patient_data_dict_E2E4_4)

                
                # Storing the fourth combination E3E4_4
                elif E_combinations[5]==E_combinations[d]:

                    print('E_combinations[5] chosen for four electrodes sixth combination')
                    print(E_combinations[5])

                    # Filling out temporary variables: 
                    temp_patient_id_E3E4_4.append(patientID)
                    temp_correlation_Wake_E3E4_4.append(correlation_structure[0])
                    temp_correlation_N1_E3E4_4.append(correlation_structure[1])
                    temp_correlation_N2_E3E4_4.append(correlation_structure[2])
                    temp_correlation_N3_E3E4_4.append(correlation_structure[3])
                    temp_correlation_REM_E3E4_4.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E3E4_4=np.stack(temp_patient_id_E3E4_4)
                    N1_E3E4_4=np.stack(temp_correlation_N1_E3E4_4)
                    N2_E3E4_4=np.stack(temp_correlation_N2_E3E4_4)
                    N3_E3E4_4=np.stack(temp_correlation_N3_E3E4_4)
                    Wake_E3E4_4=np.stack(temp_correlation_Wake_E3E4_4)
                    REM_E3E4_4=np.stack(temp_correlation_REM_E3E4_4)

                    ##### Saving values in a dataframe #####
                    patient_ids_E3E4_4=patient_id_stacked_E3E4_4

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E3E4_4 = {
                        'PatientID': patient_ids_E3E4_4,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E3E4_4.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E3E4_4.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E3E4_4.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E3E4_4.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E3E4_4.tolist(),
                    }

                    print('Patient dictionary E3E4_4 - four electrodes, sixth combination')
                    print(patient_data_dict_E3E4_4)

                
            # 5 electrodes will give 10 combinations 

            elif len(E_combinations) ==10:
            
                
                print('Temporary E-combinations length 10')

                # Storing the first combination E1E2_5
                if E_combinations[0]==E_combinations[d]:

                    print('E_combinations[0] chosen for five electrodes first combination')
                    print(E_combinations[0])
                    
                    # Filling out temporary variables: 
                    temp_patient_id_E1E2_5.append(patientID)
                    temp_correlation_Wake_E1E2_5.append(correlation_structure[0])
                    temp_correlation_N1_E1E2_5.append(correlation_structure[1])
                    temp_correlation_N2_E1E2_5.append(correlation_structure[2])
                    temp_correlation_N3_E1E2_5.append(correlation_structure[3])
                    temp_correlation_REM_E1E2_5.append(correlation_structure[4])

                    
                    # Stacking the variables 
                    patient_id_stacked_E1E2_5=np.stack(temp_patient_id_E1E2_5)
                    N1_E1E2_5=np.stack(temp_correlation_N1_E1E2_5)
                    N2_E1E2_5=np.stack(temp_correlation_N2_E1E2_5)
                    N3_E1E2_5=np.stack(temp_correlation_N3_E1E2_5)
                    Wake_E1E2_5=np.stack(temp_correlation_Wake_E1E2_5)
                    REM_E1E2_5=np.stack(temp_correlation_REM_E1E2_5)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E2_5=patient_id_stacked_E1E2_5

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E2_5 = {
                        'PatientID': patient_ids_E1E2_5,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E2_5.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E2_5.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E2_5.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E2_5.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E2_5.tolist(),
                    }

                    print('Patient dictionary E1E2_5 - five electrodes, first combination')
                    print(patient_data_dict_E1E2_5)
                
                # Storing the second combination E1E3_5
                elif E_combinations[1]==E_combinations[d]:

                    print('E_combinations[1] chosen for five electrodes second combination')
                    print(E_combinations[1])

                    # Filling out temporary variables: 
                    temp_patient_id_E1E3_5.append(patientID)
                    temp_correlation_Wake_E1E3_5.append(correlation_structure[0])
                    temp_correlation_N1_E1E3_5.append(correlation_structure[1])
                    temp_correlation_N2_E1E3_5.append(correlation_structure[2])
                    temp_correlation_N3_E1E3_5.append(correlation_structure[3])
                    temp_correlation_REM_E1E3_5.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E1E3_5=np.stack(temp_patient_id_E1E3_5)
                    N1_E1E3_5=np.stack(temp_correlation_N1_E1E3_5)
                    N2_E1E3_5=np.stack(temp_correlation_N2_E1E3_5)
                    N3_E1E3_5=np.stack(temp_correlation_N3_E1E3_5)
                    Wake_E1E3_5=np.stack(temp_correlation_Wake_E1E3_5)
                    REM_E1E3_5=np.stack(temp_correlation_REM_E1E3_5)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E3_5=patient_id_stacked_E1E3_5

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E3_5 = {
                        'PatientID': patient_ids_E1E3_5,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E3_5.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E3_5.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E3_5.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E3_5.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E3_5.tolist(),
                    }
                    
                    print('Patient dictionary E1E3_5 - five electrodes, second combination')
                    print(patient_data_dict_E1E3_5)

                # Storing the third combination E2E3_5
                elif E_combinations[2]==E_combinations[d]:

                    print('E_combinations[2] chosen for five electrodes third combination')
                    print(E_combinations[2])

                    # Filling out temporary variables: 
                    temp_patient_id_E2E3_5.append(patientID)
                    temp_correlation_Wake_E2E3_5.append(correlation_structure[0])
                    temp_correlation_N1_E2E3_5.append(correlation_structure[1])
                    temp_correlation_N2_E2E3_5.append(correlation_structure[2])
                    temp_correlation_N3_E2E3_5.append(correlation_structure[3])
                    temp_correlation_REM_E2E3_5.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E2E3_5=np.stack(temp_patient_id_E2E3_5)
                    N1_E2E3_5=np.stack(temp_correlation_N1_E2E3_5)
                    N2_E2E3_5=np.stack(temp_correlation_N2_E2E3_5)
                    N3_E2E3_5=np.stack(temp_correlation_N3_E2E3_5)
                    Wake_E2E3_5=np.stack(temp_correlation_Wake_E2E3_5)
                    REM_E2E3_5=np.stack(temp_correlation_REM_E2E3_5)

                    ##### Saving values in a dataframe #####
                    patient_ids_E2E3_5=patient_id_stacked_E2E3_5

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E2E3_5 = {
                        'PatientID': patient_ids_E2E3_5,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E2E3_5.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E2E3_5.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E2E3_5.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E2E3_5.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E2E3_5.tolist(),
                    }

                    print('Patient dictionary E2E3_5 - five electrodes, third combination')
                    print(patient_data_dict_E2E3_5)

                    
                # Storing the fourth combination E1E4_5
                elif E_combinations[3]==E_combinations[d]:

                    print('E_combinations[3] chosen for five electrodes fourth combination')
                    print(E_combinations[3])

                    # Filling out temporary variables: 
                    temp_patient_id_E1E4_5.append(patientID)
                    temp_correlation_Wake_E1E4_5.append(correlation_structure[0])
                    temp_correlation_N1_E1E4_5.append(correlation_structure[1])
                    temp_correlation_N2_E1E4_5.append(correlation_structure[2])
                    temp_correlation_N3_E1E4_5.append(correlation_structure[3])
                    temp_correlation_REM_E1E4_5.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E1E4_5=np.stack(temp_patient_id_E1E4_5)
                    N1_E1E4_5=np.stack(temp_correlation_N1_E1E4_5)
                    N2_E1E4_5=np.stack(temp_correlation_N2_E1E4_5)
                    N3_E1E4_5=np.stack(temp_correlation_N3_E1E4_5)
                    Wake_E1E4_5=np.stack(temp_correlation_Wake_E1E4_5)
                    REM_E1E4_5=np.stack(temp_correlation_REM_E1E4_5)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E4_5=patient_id_stacked_E1E4_5

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E4_5 = {
                        'PatientID': patient_ids_E1E4_5,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E4_5.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E4_5.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E4_5.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E4_5.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E4_5.tolist(),
                    }

                    print('Patient dictionary E1E4_5 - five electrodes, fourth combination')
                    print(patient_data_dict_E1E4_5)
                
                # Storing the fourth combination E2E4_5
                elif E_combinations[4]==E_combinations[d]:

                    print('E_combinations[4] chosen for five electrodes fifth combination')
                    print(E_combinations[4])

                    # Filling out temporary variables: 
                    temp_patient_id_E2E4_5.append(patientID)
                    temp_correlation_Wake_E2E4_5.append(correlation_structure[0])
                    temp_correlation_N1_E2E4_5.append(correlation_structure[1])
                    temp_correlation_N2_E2E4_5.append(correlation_structure[2])
                    temp_correlation_N3_E2E4_5.append(correlation_structure[3])
                    temp_correlation_REM_E2E4_5.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E2E4_5=np.stack(temp_patient_id_E2E4_5)
                    N1_E2E4_5=np.stack(temp_correlation_N1_E2E4_5)
                    N2_E2E4_5=np.stack(temp_correlation_N2_E2E4_5)
                    N3_E2E4_5=np.stack(temp_correlation_N3_E2E4_5)
                    Wake_E2E4_5=np.stack(temp_correlation_Wake_E2E4_5)
                    REM_E2E4_5=np.stack(temp_correlation_REM_E2E4_5)

                    ##### Saving values in a dataframe #####
                    patient_ids_E2E4_5=patient_id_stacked_E2E4_5

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E2E4_5 = {
                        'PatientID': patient_ids_E2E4_5,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E2E4_5.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E2E4_5.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E2E4_5.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E2E4_5.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E2E4_5.tolist(),
                    }

                    print('Patient dictionary E2E4_5 - five electrodes, fifth combination')
                    print(patient_data_dict_E2E4_5)

                
                # Storing the sixth combination E3E4_5
                elif E_combinations[5]==E_combinations[d]:

                    print('E_combinations[5] chosen for five electrodes sixth combination')
                    print(E_combinations[5])

                    # Filling out temporary variables: 
                    temp_patient_id_E3E4_5.append(patientID)
                    temp_correlation_Wake_E3E4_5.append(correlation_structure[0])
                    temp_correlation_N1_E3E4_5.append(correlation_structure[1])
                    temp_correlation_N2_E3E4_5.append(correlation_structure[2])
                    temp_correlation_N3_E3E4_5.append(correlation_structure[3])
                    temp_correlation_REM_E3E4_5.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E3E4_5=np.stack(temp_patient_id_E3E4_5)
                    N1_E3E4_5=np.stack(temp_correlation_N1_E3E4_5)
                    N2_E3E4_5=np.stack(temp_correlation_N2_E3E4_5)
                    N3_E3E4_5=np.stack(temp_correlation_N3_E3E4_5)
                    Wake_E3E4_5=np.stack(temp_correlation_Wake_E3E4_5)
                    REM_E3E4_5=np.stack(temp_correlation_REM_E3E4_5)

                    ##### Saving values in a dataframe #####
                    patient_ids_E3E4_5=patient_id_stacked_E3E4_5

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E3E4_5 = {
                        'PatientID': patient_ids_E3E4_5,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E3E4_5.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E3E4_5.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E3E4_5.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E3E4_5.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E3E4_5.tolist(),
                    }

                    print('Patient dictionary E3E4_5 - five electrodes, sixth combination')
                    print(patient_data_dict_E3E4_5)
                
                # Storing the seventh combination E1E5_5
                elif E_combinations[6]==E_combinations[d]:

                    print('E_combinations[6] chosen for five electrodes seventh combination')
                    print(E_combinations[6])

                    # Filling out temporary variables: 
                    temp_patient_id_E1E5_5.append(patientID)
                    temp_correlation_Wake_E1E5_5.append(correlation_structure[0])
                    temp_correlation_N1_E1E5_5.append(correlation_structure[1])
                    temp_correlation_N2_E1E5_5.append(correlation_structure[2])
                    temp_correlation_N3_E1E5_5.append(correlation_structure[3])
                    temp_correlation_REM_E1E5_5.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E1E5_5=np.stack(temp_patient_id_E1E5_5)
                    N1_E1E5_5=np.stack(temp_correlation_N1_E1E5_5)
                    N2_E1E5_5=np.stack(temp_correlation_N2_E1E5_5)
                    N3_E1E5_5=np.stack(temp_correlation_N3_E1E5_5)
                    Wake_E1E5_5=np.stack(temp_correlation_Wake_E1E5_5)
                    REM_E1E5_5=np.stack(temp_correlation_REM_E1E5_5)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E5_5=patient_id_stacked_E1E5_5

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E5_5 = {
                        'PatientID': patient_ids_E1E5_5,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E5_5.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E5_5.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E5_5.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E5_5.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E5_5.tolist(),
                    }

                    print('Patient dictionary E1E5_5 - five electrodes, seventh combination')
                    print(patient_data_dict_E1E5_5)


                # Storing the combination E2E5_5
                elif E_combinations[7]==E_combinations[d]:

                    print('E_combinations[7] chosen for five electrodes eight combination')
                    print(E_combinations[7])

                    # Filling out temporary variables: 
                    temp_patient_id_E2E5_5.append(patientID)
                    temp_correlation_Wake_E2E5_5.append(correlation_structure[0])
                    temp_correlation_N1_E2E5_5.append(correlation_structure[1])
                    temp_correlation_N2_E2E5_5.append(correlation_structure[2])
                    temp_correlation_N3_E2E5_5.append(correlation_structure[3])
                    temp_correlation_REM_E2E5_5.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E2E5_5=np.stack(temp_patient_id_E2E5_5)
                    N1_E2E5_5=np.stack(temp_correlation_N1_E2E5_5)
                    N2_E2E5_5=np.stack(temp_correlation_N2_E2E5_5)
                    N3_E2E5_5=np.stack(temp_correlation_N3_E2E5_5)
                    Wake_E2E5_5=np.stack(temp_correlation_Wake_E2E5_5)
                    REM_E2E5_5=np.stack(temp_correlation_REM_E2E5_5)

                    ##### Saving values in a dataframe #####
                    patient_ids_E2E5_5=patient_id_stacked_E2E5_5

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E2E5_5 = {
                        'PatientID': patient_ids_E2E5_5,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E2E5_5.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E2E5_5.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E2E5_5.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E2E5_5.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E2E5_5.tolist(),
                    }

                    print('Patient dictionary E2E5_5 - five electrodes, 8th combination')
                    print(patient_data_dict_E2E5_5)
            
                # Storing the fourth combination E3E5_5
                elif E_combinations[8]==E_combinations[d]:

                    print('E_combinations[8] chosen for five electrodes 9th combination')
                    print(E_combinations[8])

                    # Filling out temporary variables: 
                    temp_patient_id_E3E5_5.append(patientID)
                    temp_correlation_Wake_E3E5_5.append(correlation_structure[0])
                    temp_correlation_N1_E3E5_5.append(correlation_structure[1])
                    temp_correlation_N2_E3E5_5.append(correlation_structure[2])
                    temp_correlation_N3_E3E5_5.append(correlation_structure[3])
                    temp_correlation_REM_E3E5_5.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E3E5_5=np.stack(temp_patient_id_E3E5_5)
                    N1_E3E5_5=np.stack(temp_correlation_N1_E3E5_5)
                    N2_E3E5_5=np.stack(temp_correlation_N2_E3E5_5)
                    N3_E3E5_5=np.stack(temp_correlation_N3_E3E5_5)
                    Wake_E3E5_5=np.stack(temp_correlation_Wake_E3E5_5)
                    REM_E3E5_5=np.stack(temp_correlation_REM_E3E5_5)

                    ##### Saving values in a dataframe #####
                    patient_ids_E3E5_5=patient_id_stacked_E3E5_5

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E3E5_5 = {
                        'PatientID': patient_ids_E3E5_5,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E3E5_5.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E3E5_5.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E3E5_5.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E3E5_5.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E3E5_5.tolist(),
                    }

                    print('Patient dictionary E3E5_5 - five electrodes, 9th combination')
                    print(patient_data_dict_E3E5_5)

                
                # Storing the fourth combination E4E5_5
                elif E_combinations[9]==E_combinations[d]:

                    print('E_combinations[9] chosen for five electrodes 10th combination')
                    print(E_combinations[9])

                    # Filling out temporary variables: 
                    temp_patient_id_E4E5_5.append(patientID)
                    temp_correlation_Wake_E4E5_5.append(correlation_structure[0])
                    temp_correlation_N1_E4E5_5.append(correlation_structure[1])
                    temp_correlation_N2_E4E5_5.append(correlation_structure[2])
                    temp_correlation_N3_E4E5_5.append(correlation_structure[3])
                    temp_correlation_REM_E4E5_5.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E4E5_5=np.stack(temp_patient_id_E4E5_5)
                    N1_E4E5_5=np.stack(temp_correlation_N1_E4E5_5)
                    N2_E4E5_5=np.stack(temp_correlation_N2_E4E5_5)
                    N3_E4E5_5=np.stack(temp_correlation_N3_E4E5_5)
                    Wake_E4E5_5=np.stack(temp_correlation_Wake_E4E5_5)
                    REM_E4E5_5=np.stack(temp_correlation_REM_E4E5_5)

                    ##### Saving values in a dataframe #####
                    patient_ids_E4E5_5=patient_id_stacked_E4E5_5

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E4E5_5 = {
                        'PatientID': patient_ids_E4E5_5,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E4E5_5.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E4E5_5.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E4E5_5.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E4E5_5.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E4E5_5.tolist(),
                    }

                    print('Patient dictionary E4E5_5 - five electrodes, 10th combination')
                    print(patient_data_dict_E4E5_5)
                

            elif len(E_combinations) ==15:
            
                
                print('Temporary E-combinations length 15')

                # Storing the first combination E1E2_6
                if E_combinations[0]==E_combinations[d]:

                    print('E_combinations[0] chosen for six electrodes first combination')
                    print(E_combinations[0])
                    
                    # Filling out temporary variables: 
                    temp_patient_id_E1E2_6.append(patientID)
                    temp_correlation_Wake_E1E2_6.append(correlation_structure[0])
                    temp_correlation_N1_E1E2_6.append(correlation_structure[1])
                    temp_correlation_N2_E1E2_6.append(correlation_structure[2])
                    temp_correlation_N3_E1E2_6.append(correlation_structure[3])
                    temp_correlation_REM_E1E2_6.append(correlation_structure[4])

                    
                    # Stacking the variables 
                    patient_id_stacked_E1E2_6=np.stack(temp_patient_id_E1E2_6)
                    N1_E1E2_6=np.stack(temp_correlation_N1_E1E2_6)
                    N2_E1E2_6=np.stack(temp_correlation_N2_E1E2_6)
                    N3_E1E2_6=np.stack(temp_correlation_N3_E1E2_6)
                    Wake_E1E2_6=np.stack(temp_correlation_Wake_E1E2_6)
                    REM_E1E2_6=np.stack(temp_correlation_REM_E1E2_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E2_6=patient_id_stacked_E1E2_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E2_6 = {
                        'PatientID': patient_ids_E1E2_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E2_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E2_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E2_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E2_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E2_6.tolist(),
                    }

                    print('Patient dictionary E1E2_6 - six electrodes, first combination')
                    print(patient_data_dict_E1E2_6)
                
                # Storing the second combination E1E3_6
                elif E_combinations[1]==E_combinations[d]:

                    print('E_combinations[1] chosen for six electrodes second combination')
                    print(E_combinations[1])

                    # Filling out temporary variables: 
                    temp_patient_id_E1E3_6.append(patientID)
                    temp_correlation_Wake_E1E3_6.append(correlation_structure[0])
                    temp_correlation_N1_E1E3_6.append(correlation_structure[1])
                    temp_correlation_N2_E1E3_6.append(correlation_structure[2])
                    temp_correlation_N3_E1E3_6.append(correlation_structure[3])
                    temp_correlation_REM_E1E3_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E1E3_6=np.stack(temp_patient_id_E1E3_6)
                    N1_E1E3_6=np.stack(temp_correlation_N1_E1E3_6)
                    N2_E1E3_6=np.stack(temp_correlation_N2_E1E3_6)
                    N3_E1E3_6=np.stack(temp_correlation_N3_E1E3_6)
                    Wake_E1E3_6=np.stack(temp_correlation_Wake_E1E3_6)
                    REM_E1E3_6=np.stack(temp_correlation_REM_E1E3_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E3_6=patient_id_stacked_E1E3_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E3_6 = {
                        'PatientID': patient_ids_E1E3_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E3_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E3_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E3_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E3_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E3_6.tolist(),
                    }
                    
                    print('Patient dictionary E1E3_6 - six electrodes, second combination')
                    print(patient_data_dict_E1E3_6)

                # Storing the third combination E2E3_6
                elif E_combinations[2]==E_combinations[d]:

                    print('E_combinations[2] chosen for six electrodes third combination')
                    print(E_combinations[2])

                    # Filling out temporary variables: 
                    temp_patient_id_E2E3_6.append(patientID)
                    temp_correlation_Wake_E2E3_6.append(correlation_structure[0])
                    temp_correlation_N1_E2E3_6.append(correlation_structure[1])
                    temp_correlation_N2_E2E3_6.append(correlation_structure[2])
                    temp_correlation_N3_E2E3_6.append(correlation_structure[3])
                    temp_correlation_REM_E2E3_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E2E3_6=np.stack(temp_patient_id_E2E3_6)
                    N1_E2E3_6=np.stack(temp_correlation_N1_E2E3_6)
                    N2_E2E3_6=np.stack(temp_correlation_N2_E2E3_6)
                    N3_E2E3_6=np.stack(temp_correlation_N3_E2E3_6)
                    Wake_E2E3_6=np.stack(temp_correlation_Wake_E2E3_6)
                    REM_E2E3_6=np.stack(temp_correlation_REM_E2E3_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E2E3_6=patient_id_stacked_E2E3_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E2E3_6 = {
                        'PatientID': patient_ids_E2E3_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E2E3_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E2E3_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E2E3_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E2E3_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E2E3_6.tolist(),
                    }

                    print('Patient dictionary E2E3_6 - six electrodes, third combination')
                    print(patient_data_dict_E2E3_6)

                    
                # Storing the fourth combination E1E4_6
                elif E_combinations[3]==E_combinations[d]:

                    print('E_combinations[3] chosen for six electrodes fourth combination')
                    print(E_combinations[3])

                    # Filling out temporary variables: 
                    temp_patient_id_E1E4_6.append(patientID)
                    temp_correlation_Wake_E1E4_6.append(correlation_structure[0])
                    temp_correlation_N1_E1E4_6.append(correlation_structure[1])
                    temp_correlation_N2_E1E4_6.append(correlation_structure[2])
                    temp_correlation_N3_E1E4_6.append(correlation_structure[3])
                    temp_correlation_REM_E1E4_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E1E4_6=np.stack(temp_patient_id_E1E4_6)
                    N1_E1E4_6=np.stack(temp_correlation_N1_E1E4_6)
                    N2_E1E4_6=np.stack(temp_correlation_N2_E1E4_6)
                    N3_E1E4_6=np.stack(temp_correlation_N3_E1E4_6)
                    Wake_E1E4_6=np.stack(temp_correlation_Wake_E1E4_6)
                    REM_E1E4_6=np.stack(temp_correlation_REM_E1E4_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E4_6=patient_id_stacked_E1E4_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E4_6 = {
                        'PatientID': patient_ids_E1E4_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E4_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E4_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E4_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E4_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E4_6.tolist(),
                    }

                    print('Patient dictionary E1E4_6 - six electrodes, fourth combination')
                    print(patient_data_dict_E1E4_6)
                
                # Storing the fourth combination E2E4_6
                elif E_combinations[4]==E_combinations[d]:

                    print('E_combinations[4] chosen for six electrodes fifth combination')
                    print(E_combinations[4])

                    # Filling out temporary variables: 
                    temp_patient_id_E2E4_6.append(patientID)
                    temp_correlation_Wake_E2E4_6.append(correlation_structure[0])
                    temp_correlation_N1_E2E4_6.append(correlation_structure[1])
                    temp_correlation_N2_E2E4_6.append(correlation_structure[2])
                    temp_correlation_N3_E2E4_6.append(correlation_structure[3])
                    temp_correlation_REM_E2E4_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E2E4_6=np.stack(temp_patient_id_E2E4_6)
                    N1_E2E4_6=np.stack(temp_correlation_N1_E2E4_6)
                    N2_E2E4_6=np.stack(temp_correlation_N2_E2E4_6)
                    N3_E2E4_6=np.stack(temp_correlation_N3_E2E4_6)
                    Wake_E2E4_6=np.stack(temp_correlation_Wake_E2E4_6)
                    REM_E2E4_6=np.stack(temp_correlation_REM_E2E4_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E2E4_6=patient_id_stacked_E2E4_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E2E4_6 = {
                        'PatientID': patient_ids_E2E4_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E2E4_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E2E4_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E2E4_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E2E4_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E2E4_6.tolist(),
                    }

                    print('Patient dictionary E2E4_6 - six electrodes, fifth combination')
                    print(patient_data_dict_E2E4_6)

                
                # Storing the sixth combination E3E4_6
                elif E_combinations[5]==E_combinations[d]:

                    print('E_combinations[5] chosen for six electrodes sixth combination')
                    print(E_combinations[5])

                    # Filling out temporary variables: 
                    temp_patient_id_E3E4_6.append(patientID)
                    temp_correlation_Wake_E3E4_6.append(correlation_structure[0])
                    temp_correlation_N1_E3E4_6.append(correlation_structure[1])
                    temp_correlation_N2_E3E4_6.append(correlation_structure[2])
                    temp_correlation_N3_E3E4_6.append(correlation_structure[3])
                    temp_correlation_REM_E3E4_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E3E4_6=np.stack(temp_patient_id_E3E4_6)
                    N1_E3E4_6=np.stack(temp_correlation_N1_E3E4_6)
                    N2_E3E4_6=np.stack(temp_correlation_N2_E3E4_6)
                    N3_E3E4_6=np.stack(temp_correlation_N3_E3E4_6)
                    Wake_E3E4_6=np.stack(temp_correlation_Wake_E3E4_6)
                    REM_E3E4_6=np.stack(temp_correlation_REM_E3E4_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E3E4_6=patient_id_stacked_E3E4_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E3E4_6 = {
                        'PatientID': patient_ids_E3E4_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E3E4_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E3E4_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E3E4_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E3E4_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E3E4_6.tolist(),
                    }

                    print('Patient dictionary E3E4_6 - six electrodes, sixth combination')
                    print(patient_data_dict_E3E4_6)
                
                # Storing the seventh combination E1E5_6
                elif E_combinations[6]==E_combinations[d]:

                    print('E_combinations[6] chosen for six electrodes seventh combination')
                    print(E_combinations[6])

                    # Filling out temporary variables: 
                    temp_patient_id_E1E5_6.append(patientID)
                    temp_correlation_Wake_E1E5_6.append(correlation_structure[0])
                    temp_correlation_N1_E1E5_6.append(correlation_structure[1])
                    temp_correlation_N2_E1E5_6.append(correlation_structure[2])
                    temp_correlation_N3_E1E5_6.append(correlation_structure[3])
                    temp_correlation_REM_E1E5_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E1E5_6=np.stack(temp_patient_id_E1E5_6)
                    N1_E1E5_6=np.stack(temp_correlation_N1_E1E5_6)
                    N2_E1E5_6=np.stack(temp_correlation_N2_E1E5_6)
                    N3_E1E5_6=np.stack(temp_correlation_N3_E1E5_6)
                    Wake_E1E5_6=np.stack(temp_correlation_Wake_E1E5_6)
                    REM_E1E5_6=np.stack(temp_correlation_REM_E1E5_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E5_6=patient_id_stacked_E1E5_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E5_6 = {
                        'PatientID': patient_ids_E1E5_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E5_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E5_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E5_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E5_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E5_6.tolist(),
                    }

                    print('Patient dictionary E1E5_6 - six electrodes, seventh combination')
                    print(patient_data_dict_E1E5_6)


                # Storing the combination E2E5_6
                elif E_combinations[7]==E_combinations[d]:

                    print('E_combinations[7] chosen for six electrodes eight combination')
                    print(E_combinations[7])

                    # Filling out temporary variables: 
                    temp_patient_id_E2E5_6.append(patientID)
                    temp_correlation_Wake_E2E5_6.append(correlation_structure[0])
                    temp_correlation_N1_E2E5_6.append(correlation_structure[1])
                    temp_correlation_N2_E2E5_6.append(correlation_structure[2])
                    temp_correlation_N3_E2E5_6.append(correlation_structure[3])
                    temp_correlation_REM_E2E5_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E2E5_6=np.stack(temp_patient_id_E2E5_6)
                    N1_E2E5_6=np.stack(temp_correlation_N1_E2E5_6)
                    N2_E2E5_6=np.stack(temp_correlation_N2_E2E5_6)
                    N3_E2E5_6=np.stack(temp_correlation_N3_E2E5_6)
                    Wake_E2E5_6=np.stack(temp_correlation_Wake_E2E5_6)
                    REM_E2E5_6=np.stack(temp_correlation_REM_E2E5_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E2E5_6=patient_id_stacked_E2E5_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E2E5_6 = {
                        'PatientID': patient_ids_E2E5_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E2E5_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E2E5_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E2E5_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E2E5_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E2E5_6.tolist(),
                    }

                    print('Patient dictionary E2E5_6 - six electrodes, 8th combination')
                    print(patient_data_dict_E2E5_6)
            
                # Storing the fourth combination E3E5_6
                elif E_combinations[8]==E_combinations[d]:

                    print('E_combinations[8] chosen for six electrodes 9th combination')
                    print(E_combinations[8])

                    # Filling out temporary variables: 
                    temp_patient_id_E3E5_6.append(patientID)
                    temp_correlation_Wake_E3E5_6.append(correlation_structure[0])
                    temp_correlation_N1_E3E5_6.append(correlation_structure[1])
                    temp_correlation_N2_E3E5_6.append(correlation_structure[2])
                    temp_correlation_N3_E3E5_6.append(correlation_structure[3])
                    temp_correlation_REM_E3E5_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E3E5_6=np.stack(temp_patient_id_E3E5_6)
                    N1_E3E5_6=np.stack(temp_correlation_N1_E3E5_6)
                    N2_E3E5_6=np.stack(temp_correlation_N2_E3E5_6)
                    N3_E3E5_6=np.stack(temp_correlation_N3_E3E5_6)
                    Wake_E3E5_6=np.stack(temp_correlation_Wake_E3E5_6)
                    REM_E3E5_6=np.stack(temp_correlation_REM_E3E5_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E3E5_6=patient_id_stacked_E3E5_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E3E5_6 = {
                        'PatientID': patient_ids_E3E5_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E3E5_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E3E5_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E3E5_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E3E5_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E3E5_6.tolist(),
                    }

                    print('Patient dictionary E3E5_6 - six electrodes, 9th combination')
                    print(patient_data_dict_E3E5_6)

                
                # Storing the fourth combination E4E5_6
                elif E_combinations[9]==E_combinations[d]:

                    print('E_combinations[9] chosen for six electrodes 10th combination')
                    print(E_combinations[9])

                    # Filling out temporary variables: 
                    temp_patient_id_E4E5_6.append(patientID)
                    temp_correlation_Wake_E4E5_6.append(correlation_structure[0])
                    temp_correlation_N1_E4E5_6.append(correlation_structure[1])
                    temp_correlation_N2_E4E5_6.append(correlation_structure[2])
                    temp_correlation_N3_E4E5_6.append(correlation_structure[3])
                    temp_correlation_REM_E4E5_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E4E5_6=np.stack(temp_patient_id_E4E5_6)
                    N1_E4E5_6=np.stack(temp_correlation_N1_E4E5_6)
                    N2_E4E5_6=np.stack(temp_correlation_N2_E4E5_6)
                    N3_E4E5_6=np.stack(temp_correlation_N3_E4E5_6)
                    Wake_E4E5_6=np.stack(temp_correlation_Wake_E4E5_6)
                    REM_E4E5_6=np.stack(temp_correlation_REM_E4E5_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E4E5_6=patient_id_stacked_E4E5_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E4E5_6 = {
                        'PatientID': patient_ids_E4E5_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E4E5_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E4E5_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E4E5_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E4E5_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E4E5_6.tolist(),
                    }

                    print('Patient dictionary E4E5_6 - six electrodes, 10th combination')
                    print(patient_data_dict_E4E5_6)

                
                # Storing the fourth combination E1E6_6
                elif E_combinations[10]==E_combinations[d]:

                    print('E_combinations[10] chosen for six electrodes 11th combination')
                    print(E_combinations[10])

                    # Filling out temporary variables: 
                    temp_patient_id_E1E6_6.append(patientID)
                    temp_correlation_Wake_E1E6_6.append(correlation_structure[0])
                    temp_correlation_N1_E1E6_6.append(correlation_structure[1])
                    temp_correlation_N2_E1E6_6.append(correlation_structure[2])
                    temp_correlation_N3_E1E6_6.append(correlation_structure[3])
                    temp_correlation_REM_E1E6_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E1E6_6=np.stack(temp_patient_id_E1E6_6)
                    N1_E1E6_6=np.stack(temp_correlation_N1_E1E6_6)
                    N2_E1E6_6=np.stack(temp_correlation_N2_E1E6_6)
                    N3_E1E6_6=np.stack(temp_correlation_N3_E1E6_6)
                    Wake_E1E6_6=np.stack(temp_correlation_Wake_E1E6_6)
                    REM_E1E6_6=np.stack(temp_correlation_REM_E1E6_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E1E6_6=patient_id_stacked_E1E6_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E1E6_6 = {
                        'PatientID': patient_ids_E1E6_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E1E6_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E1E6_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E1E6_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E1E6_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E1E6_6.tolist(),
                    }

                    print('Patient dictionary E1E6_6 - six electrodes, 11th combination')
                    print(patient_data_dict_E1E6_6)

                
                # Storing the 12th combination E2E6_6
                elif E_combinations[11]==E_combinations[d]:

                    print('E_combinations[11] chosen for six electrodes 12th combination')
                    print(E_combinations[11])

                    # Filling out temporary variables: 
                    temp_patient_id_E2E6_6.append(patientID)
                    temp_correlation_Wake_E2E6_6.append(correlation_structure[0])
                    temp_correlation_N1_E2E6_6.append(correlation_structure[1])
                    temp_correlation_N2_E2E6_6.append(correlation_structure[2])
                    temp_correlation_N3_E2E6_6.append(correlation_structure[3])
                    temp_correlation_REM_E2E6_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E2E6_6=np.stack(temp_patient_id_E2E6_6)
                    N1_E2E6_6=np.stack(temp_correlation_N1_E2E6_6)
                    N2_E2E6_6=np.stack(temp_correlation_N2_E2E6_6)
                    N3_E2E6_6=np.stack(temp_correlation_N3_E2E6_6)
                    Wake_E2E6_6=np.stack(temp_correlation_Wake_E2E6_6)
                    REM_E2E6_6=np.stack(temp_correlation_REM_E2E6_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E2E6_6=patient_id_stacked_E2E6_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E2E6_6 = {
                        'PatientID': patient_ids_E2E6_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E2E6_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E2E6_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E2E6_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E2E6_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E2E6_6.tolist(),
                    }

                    print('Patient dictionary E2E6_6 - six electrodes, 12th combination')
                    print(patient_data_dict_E2E6_6)
                
                
                # Storing the fourth combination E3E6_6
                elif E_combinations[12]==E_combinations[d]:

                    print('E_combinations[12] chosen for six electrodes 13th combination')
                    print(E_combinations[12])

                    # Filling out temporary variables: 
                    temp_patient_id_E3E6_6.append(patientID)
                    temp_correlation_Wake_E3E6_6.append(correlation_structure[0])
                    temp_correlation_N1_E3E6_6.append(correlation_structure[1])
                    temp_correlation_N2_E3E6_6.append(correlation_structure[2])
                    temp_correlation_N3_E3E6_6.append(correlation_structure[3])
                    temp_correlation_REM_E3E6_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E3E6_6=np.stack(temp_patient_id_E3E6_6)
                    N1_E3E6_6=np.stack(temp_correlation_N1_E3E6_6)
                    N2_E3E6_6=np.stack(temp_correlation_N2_E3E6_6)
                    N3_E3E6_6=np.stack(temp_correlation_N3_E3E6_6)
                    Wake_E3E6_6=np.stack(temp_correlation_Wake_E3E6_6)
                    REM_E3E6_6=np.stack(temp_correlation_REM_E3E6_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E3E6_6=patient_id_stacked_E3E6_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E3E6_6 = {
                        'PatientID': patient_ids_E3E6_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E3E6_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E3E6_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E3E6_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E3E6_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E3E6_6.tolist(),
                    }

                    print('Patient dictionary E3E6_6 - six electrodes, 13th combination')
                    print(patient_data_dict_E3E6_6)



                # Storing the fourth combination E4E6_6
                elif E_combinations[13]==E_combinations[d]:

                    print('E_combinations[13] chosen for six electrodes 14th combination')
                    print(E_combinations[13])

                    # Filling out temporary variables: 
                    temp_patient_id_E4E6_6.append(patientID)
                    temp_correlation_Wake_E4E6_6.append(correlation_structure[0])
                    temp_correlation_N1_E4E6_6.append(correlation_structure[1])
                    temp_correlation_N2_E4E6_6.append(correlation_structure[2])
                    temp_correlation_N3_E4E6_6.append(correlation_structure[3])
                    temp_correlation_REM_E4E6_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E4E6_6=np.stack(temp_patient_id_E4E6_6)
                    N1_E4E6_6=np.stack(temp_correlation_N1_E4E6_6)
                    N2_E4E6_6=np.stack(temp_correlation_N2_E4E6_6)
                    N3_E4E6_6=np.stack(temp_correlation_N3_E4E6_6)
                    Wake_E4E6_6=np.stack(temp_correlation_Wake_E4E6_6)
                    REM_E4E6_6=np.stack(temp_correlation_REM_E4E6_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E4E6_6=patient_id_stacked_E4E6_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E4E6_6 = {
                        'PatientID': patient_ids_E4E6_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E4E6_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E4E6_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E4E6_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E4E6_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E4E6_6.tolist(),
                    }

                    print('Patient dictionary E4E6_6 - six electrodes, 14th combination')
                    print(patient_data_dict_E4E6_6)
                
                
                # Storing the fourth combination E5E6_6
                elif E_combinations[14]==E_combinations[d]:

                    print('E_combinations[14] chosen for six electrodes 15th combination')
                    print(E_combinations[14])

                    # Filling out temporary variables: 
                    temp_patient_id_E5E6_6.append(patientID)
                    temp_correlation_Wake_E5E6_6.append(correlation_structure[0])
                    temp_correlation_N1_E5E6_6.append(correlation_structure[1])
                    temp_correlation_N2_E5E6_6.append(correlation_structure[2])
                    temp_correlation_N3_E5E6_6.append(correlation_structure[3])
                    temp_correlation_REM_E5E6_6.append(correlation_structure[4])

                
                    # Stacking the variables 
                    patient_id_stacked_E5E6_6=np.stack(temp_patient_id_E5E6_6)
                    N1_E5E6_6=np.stack(temp_correlation_N1_E5E6_6)
                    N2_E5E6_6=np.stack(temp_correlation_N2_E5E6_6)
                    N3_E5E6_6=np.stack(temp_correlation_N3_E5E6_6)
                    Wake_E5E6_6=np.stack(temp_correlation_Wake_E5E6_6)
                    REM_E5E6_6=np.stack(temp_correlation_REM_E5E6_6)

                    ##### Saving values in a dataframe #####
                    patient_ids_E5E6_6=patient_id_stacked_E5E6_6

                    print('Electrode combination')
                    print(Electrode_combination_naming)

                    # Create a dictionary to store patient ID and corresponding information
                    patient_data_dict_E5E6_6 = {
                        'PatientID': patient_ids_E5E6_6,
                        'Wake_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): Wake_E5E6_6.tolist(),
                        'N1_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N1_E5E6_6.tolist(),
                        'N2_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N2_E5E6_6.tolist(),
                        'N3_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): N3_E5E6_6.tolist(),
                        'REM_'+str(Electrode_combination_naming)+'_'+str(epoch_size_in_seconds): REM_E5E6_6.tolist(),
                    }

                    print('Patient dictionary E5E6_6 - six electrodes, 15th combination')
                    print(patient_data_dict_E5E6_6)

    ################ Generating dataframes and CSV files #####################

    # Merging the dataframes containing the three combinations of electrodes 
    def Merge3(dict1, dict2, dict3):
        res = {**dict1, **dict2, **dict3}
        return res

    def Merge6(dict1, dict2, dict3, dict4, dict5, dict6):
        res = {**dict1, **dict2, **dict3, **dict4, **dict5, **dict6}
        return res

    def Merge10(dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9, dict10):
        res = {**dict1, **dict2, **dict3, **dict4, **dict5, **dict6, **dict7, **dict8, **dict9, **dict10}
        return res


    def Merge15(dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9, dict10, dict11, dict12, dict13, dict14, dict15):
        res = {**dict1, **dict2, **dict3, **dict4, **dict5, **dict6, **dict7, **dict8, **dict9, **dict10, **dict11,**dict12,**dict13,**dict14,**dict15}
        return res
    

    
    

    ############ 6 electrodes ##################################################
    # Merge15 is used because 15 combinations of electrodes are present, when there are 6 electrodes 
    if 'patient_data_dict_E1E2_6' in locals():
        print(patient_data_dict_E1E2_6)
        full_dict_6E = Merge15(patient_data_dict_E1E2_6,patient_data_dict_E1E3_6,patient_data_dict_E2E3_6,patient_data_dict_E1E4_6,patient_data_dict_E2E4_6,patient_data_dict_E3E4_6,patient_data_dict_E1E5_6,patient_data_dict_E2E5_6,patient_data_dict_E3E5_6,patient_data_dict_E4E5_6,patient_data_dict_E1E6_6,patient_data_dict_E2E6_6,patient_data_dict_E3E6_6,patient_data_dict_E4E6_6,patient_data_dict_E5E6_6)
        
        # Generating full data frame for the 3 electrode data 
        full_dataframe_6E=pd.DataFrame(full_dict_6E)

        filename_6E=f"Correlation_6E_"+str(epoch_size_in_seconds)+"_RBD_controls.csv"
        RBD_output_path_6E=os.path.join('/scratch/users/s184063/RBD_controls_Features/', filename_6E)

        full_dataframe_6E.to_csv(RBD_output_path_6E, index=False) # change filename using os
        print(RBD_output_path_6E)
        #full_dataframe_6E.to_csv(f"Correlation_6E_"+str(epoch_size_in_seconds)+"_RBD.csv", index=False) # change filename using os
        del patient_data_dict_E1E2_6,patient_data_dict_E1E3_6,patient_data_dict_E2E3_6,patient_data_dict_E1E4_6,patient_data_dict_E2E4_6,patient_data_dict_E3E4_6,patient_data_dict_E1E5_6,patient_data_dict_E2E5_6,patient_data_dict_E3E5_6,patient_data_dict_E4E5_6,patient_data_dict_E1E6_6,patient_data_dict_E2E6_6,patient_data_dict_E3E6_6,patient_data_dict_E4E6_6,patient_data_dict_E5E6_6
    
        print('6 electrode documents where packed')

    ############ 5 electrodes ##################################################
    # Merge10 is used because 10 combinations of electrodes are present, when there are 5 electrodes 
    if 'patient_data_dict_E1E2_5' in locals():
        print(patient_data_dict_E1E2_5)
        full_dict_5E = Merge10(patient_data_dict_E1E2_5,patient_data_dict_E1E3_5,patient_data_dict_E2E3_5,patient_data_dict_E1E4_5,patient_data_dict_E2E4_5,patient_data_dict_E3E4_5,patient_data_dict_E1E5_5,patient_data_dict_E2E5_5,patient_data_dict_E3E5_5,patient_data_dict_E4E5_5)
        
        # Generating full data frame for the 3 electrode data 
        full_dataframe_5E=pd.DataFrame(full_dict_5E)
        
        filename_5E=f"Correlation_5E_"+str(epoch_size_in_seconds)+"_RBD_controls.csv"
        RBD_output_path_5E=os.path.join('/scratch/users/s184063/RBD_controls_Features/', filename_5E)
        print(RBD_output_path_5E)
        full_dataframe_5E.to_csv(RBD_output_path_5E, index=False) # change filename using os
        
        #full_dataframe_5E.to_csv(f"Correlation_5E_"+str(epoch_size_in_seconds)+"_RBD.csv", index=False) # change filename using os
        del patient_data_dict_E1E2_5,patient_data_dict_E1E3_5,patient_data_dict_E2E3_5,patient_data_dict_E1E4_5,patient_data_dict_E2E4_5,patient_data_dict_E3E4_5,patient_data_dict_E1E5_5,patient_data_dict_E2E5_5,patient_data_dict_E3E5_5,patient_data_dict_E4E5_5
        print('5 electrode documents where packed')
    
    ############ 4 electrodes ##################################################
    # Merge6 is used because 6 combinations of electrodes are present, when there are 4 electrodes 
    if 'patient_data_dict_E1E2_4' in locals():
        print(patient_data_dict_E1E2_4)
        full_dict_4E = Merge6(patient_data_dict_E1E2_4,patient_data_dict_E1E3_4,patient_data_dict_E2E3_4,patient_data_dict_E1E4_4,patient_data_dict_E2E4_4,patient_data_dict_E3E4_4)
        
        # Generating full data frame for the 3 electrode data 
        full_dataframe_4E=pd.DataFrame(full_dict_4E)
        
        filename_4E=f"Correlation_4E_"+str(epoch_size_in_seconds)+"_RBD_controls.csv"
        RBD_output_path_4E=os.path.join('/scratch/users/s184063/RBD_controls_Features/', filename_4E)

        print(RBD_output_path_4E)
        full_dataframe_4E.to_csv(RBD_output_path_4E, index=False) # change filename using os
    
        
        #full_dataframe_4E.to_csv(f"Correlation_4E_"+str(epoch_size_in_seconds)+"_RBD.csv", index=False) # change filename using os
        
        del patient_data_dict_E1E2_4,patient_data_dict_E1E3_4,patient_data_dict_E2E3_4,patient_data_dict_E1E4_4,patient_data_dict_E2E4_4,patient_data_dict_E3E4_4
        print('4 electrode documents where packed')

    ############ 3 electrodes ###################################################
    # Generating dictonary for the merged dictornary 
    if 'patient_data_dict_E1E2_3' in locals():

        print(patient_data_dict_E1E2_3)
        full_dict_3E = Merge3(patient_data_dict_E1E2_3,patient_data_dict_E1E3_3,patient_data_dict_E2E3_3)
        print('Full dictornary 3 electrodes')
        # One row per patient 

        # Generating full data frame for the 3 electrode data 
        full_dataframe_3E=pd.DataFrame(full_dict_3E)
        
        filename_3E=f"Correlation_3E_"+str(epoch_size_in_seconds)+"_RBD_controls.csv"
        RBD_output_path_3E=os.path.join('/scratch/users/s184063/RBD_controls_Features/', filename_3E)

        print(RBD_output_path_3E)
        full_dataframe_3E.to_csv(RBD_output_path_3E, index=False) # change filename using os
        
        
        #full_dataframe_3E.to_csv(f"Correlation_3E_"+str(epoch_size_in_seconds)+"_RBD.csv", index=False) # change filename using os
        del patient_data_dict_E1E2_3,patient_data_dict_E1E3_3,patient_data_dict_E2E3_3
        print('3 electrode documents where packed')
    ############################################################################
    

    
    ##################### 2 electrodes #########################################
    # Generating full data frame for the 2 electrode data 
    if 'patient_data_dict_E1E2_2' in locals():
        print('Generating dataframe and CSV file for 2-electrode data')
        print(patient_data_dict_E1E2_2)
        full_dataframe_2E=pd.DataFrame(patient_data_dict_E1E2_2)
        
        filename_2E=f"Correlation_2E_"+str(epoch_size_in_seconds)+"_RBD_controls.csv"
        RBD_output_path_2E=os.path.join('/scratch/users/s184063/RBD_controls_Features/', filename_2E)

        print(RBD_output_path_2E)
        full_dataframe_2E.to_csv(RBD_output_path_2E, index=False) # change filename using os
        
        
        #full_dataframe_2E.to_csv(f"Correlation_2E_"+str(epoch_size_in_seconds)+"_RBD.csv", index=False) # change filename using os
        ############################################################################
        del patient_data_dict_E1E2_2
        print('2 electrode documents where packed')

    return print('Correlation_multiple_electrodes() are done') 
