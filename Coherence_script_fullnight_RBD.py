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
import mne
# Using sys function to import 'My_functions_script'
sys.path.insert(0, '/home/users/s184063/')

# Import My_functions_script
from My_functions_script_RBD import list_files_in_folder, preprocessing, bandpass_frequency_band, relative_power_for_frequencyband, coherence_features, extract_numbers_from_filename, extract_letters_and_numbers


# Get indices for the EEG files - the names of electrodes has to be known 
def get_indices(data, labels):
    indices = {label: None for label in labels}
    for i, d in enumerate(data):
        if d['label'] in labels:
            indices[d['label']] = i
            
    return indices


# File paths 
input_path =r'/scratch/users/s184063/RBD_restructured_1_10/' # select correct path and folder


temp_patientID_2E_E1E2=[]
temp_visit=[]
temp_deltaband_coh_2E_E1E2=[]
temp_thetaband_coh_2E_E1E2=[]
temp_alphaband_coh_2E_E1E2=[]
temp_betaband_coh_2E_E1E2=[]
temp_gammaband_coh_2E_E1E2=[]

# three electrodes - first combination 
temp_patientID_3E_E1E2=[]
temp_deltaband_coh_3E_E1E2=[]
temp_thetaband_coh_3E_E1E2=[]
temp_alphaband_coh_3E_E1E2=[]
temp_betaband_coh_3E_E1E2=[]
temp_gammaband_coh_3E_E1E2=[]

# three electrodes - second combination 
temp_patientID_3E_E1E3=[]
temp_deltaband_coh_3E_E1E3=[]
temp_thetaband_coh_3E_E1E3=[]
temp_alphaband_coh_3E_E1E3=[]
temp_betaband_coh_3E_E1E3=[]
temp_gammaband_coh_3E_E1E3=[]

# three electrodes - third combination 
temp_patientID_3E_E2E3=[]
temp_deltaband_coh_3E_E2E3=[]
temp_thetaband_coh_3E_E2E3=[]
temp_alphaband_coh_3E_E2E3=[]
temp_betaband_coh_3E_E2E3=[]
temp_gammaband_coh_3E_E2E3=[]

# Four electrodes - first combination 
temp_patientID_4E_E1E2=[]
temp_deltaband_coh_4E_E1E2=[]
temp_thetaband_coh_4E_E1E2=[]
temp_alphaband_coh_4E_E1E2=[]
temp_betaband_coh_4E_E1E2=[]
temp_gammaband_coh_4E_E1E2=[]
temp_deltacoh_av30_4E_E1E2=[]
temp_thetacoh_av30_4E_E1E2=[]
temp_alphacoh_av30_4E_E1E2=[]
temp_betacoh_av30_4E_E1E2=[]
temp_gammacoh_av30_4E_E1E2=[]

# Four electrodes - second combination 
temp_patientID_4E_E1E3=[]
temp_deltaband_coh_4E_E1E3=[]
temp_thetaband_coh_4E_E1E3=[]
temp_alphaband_coh_4E_E1E3=[]
temp_betaband_coh_4E_E1E3=[]
temp_gammaband_coh_4E_E1E3=[]
temp_deltacoh_av30_4E_E1E3=[]
temp_thetacoh_av30_4E_E1E3=[]
temp_alphacoh_av30_4E_E1E3=[]
temp_betacoh_av30_4E_E1E3=[]
temp_gammacoh_av30_4E_E1E3=[]

# Four electrodes - third combination 
temp_patientID_4E_E2E3=[]
temp_deltaband_coh_4E_E2E3=[]
temp_thetaband_coh_4E_E2E3=[]
temp_alphaband_coh_4E_E2E3=[]
temp_betaband_coh_4E_E2E3=[]
temp_gammaband_coh_4E_E2E3=[]
temp_deltacoh_av30_4E_E2E3=[]
temp_thetacoh_av30_4E_E2E3=[]
temp_alphacoh_av30_4E_E2E3=[]
temp_betacoh_av30_4E_E2E3=[]
temp_gammacoh_av30_4E_E2E3=[]


# Four electrodes - fourth combination 
temp_patientID_4E_E1E4=[]
temp_deltaband_coh_4E_E1E4=[]
temp_thetaband_coh_4E_E1E4=[]
temp_alphaband_coh_4E_E1E4=[]
temp_betaband_coh_4E_E1E4=[]
temp_gammaband_coh_4E_E1E4=[]
temp_deltacoh_av30_4E_E1E4=[]
temp_thetacoh_av30_4E_E1E4=[]
temp_alphacoh_av30_4E_E1E4=[]
temp_betacoh_av30_4E_E1E4=[]
temp_gammacoh_av30_4E_E1E4=[]


# Four electrodes - fifth combination 
temp_patientID_4E_E2E4=[]
temp_deltaband_coh_4E_E2E4=[]
temp_thetaband_coh_4E_E2E4=[]
temp_alphaband_coh_4E_E2E4=[]
temp_betaband_coh_4E_E2E4=[]
temp_gammaband_coh_4E_E2E4=[]
temp_deltacoh_av30_4E_E2E4=[]
temp_thetacoh_av30_4E_E2E4=[]
temp_alphacoh_av30_4E_E2E4=[]
temp_betacoh_av30_4E_E2E4=[]
temp_gammacoh_av30_4E_E2E4=[]

# Four electrodes - sixth combination 
temp_patientID_4E_E3E4=[]
temp_deltaband_coh_4E_E3E4=[]
temp_thetaband_coh_4E_E3E4=[]
temp_alphaband_coh_4E_E3E4=[]
temp_betaband_coh_4E_E3E4=[]
temp_gammaband_coh_4E_E3E4=[]
temp_deltacoh_av30_4E_E3E4=[]
temp_thetacoh_av30_4E_E3E4=[]
temp_alphacoh_av30_4E_E3E4=[]
temp_betacoh_av30_4E_E3E4=[]
temp_gammacoh_av30_4E_E3E4=[]




# Five electrodes - first combination 
temp_patientID_5E_E1E2=[]
temp_deltaband_coh_5E_E1E2=[]
temp_thetaband_coh_5E_E1E2=[]
temp_alphaband_coh_5E_E1E2=[]
temp_betaband_coh_5E_E1E2=[]
temp_gammaband_coh_5E_E1E2=[]
temp_deltacoh_av30_5E_E1E2=[]
temp_thetacoh_av30_5E_E1E2=[]
temp_alphacoh_av30_5E_E1E2=[]
temp_betacoh_av30_5E_E1E2=[]
temp_gammacoh_av30_5E_E1E2=[]

# Five electrodes - second combination 
temp_patientID_5E_E1E3=[]
temp_deltaband_coh_5E_E1E3=[]
temp_thetaband_coh_5E_E1E3=[]
temp_alphaband_coh_5E_E1E3=[]
temp_betaband_coh_5E_E1E3=[]
temp_gammaband_coh_5E_E1E3=[]
temp_deltacoh_av30_5E_E1E3=[]
temp_thetacoh_av30_5E_E1E3=[]
temp_alphacoh_av30_5E_E1E3=[]
temp_betacoh_av30_5E_E1E3=[]
temp_gammacoh_av30_5E_E1E3=[]

# Five electrodes - third combination 
temp_patientID_5E_E2E3=[]
temp_deltaband_coh_5E_E2E3=[]
temp_thetaband_coh_5E_E2E3=[]
temp_alphaband_coh_5E_E2E3=[]
temp_betaband_coh_5E_E2E3=[]
temp_gammaband_coh_5E_E2E3=[]
temp_deltacoh_av30_5E_E2E3=[]
temp_thetacoh_av30_5E_E2E3=[]
temp_alphacoh_av30_5E_E2E3=[]
temp_betacoh_av30_5E_E2E3=[]
temp_gammacoh_av30_5E_E2E3=[]


# Five electrodes - fourth combination 
temp_patientID_5E_E1E4=[]
temp_deltaband_coh_5E_E1E4=[]
temp_thetaband_coh_5E_E1E4=[]
temp_alphaband_coh_5E_E1E4=[]
temp_betaband_coh_5E_E1E4=[]
temp_gammaband_coh_5E_E1E4=[]
temp_deltacoh_av30_5E_E1E4=[]
temp_thetacoh_av30_5E_E1E4=[]
temp_alphacoh_av30_5E_E1E4=[]
temp_betacoh_av30_5E_E1E4=[]
temp_gammacoh_av30_5E_E1E4=[]


# Five electrodes - fifth combination 
temp_patientID_5E_E2E4=[]
temp_deltaband_coh_5E_E2E4=[]
temp_thetaband_coh_5E_E2E4=[]
temp_alphaband_coh_5E_E2E4=[]
temp_betaband_coh_5E_E2E4=[]
temp_gammaband_coh_5E_E2E4=[]
temp_deltacoh_av30_5E_E2E4=[]
temp_thetacoh_av30_5E_E2E4=[]
temp_alphacoh_av30_5E_E2E4=[]
temp_betacoh_av30_5E_E2E4=[]
temp_gammacoh_av30_5E_E2E4=[]

# Five electrodes - sixth combination 
temp_patientID_5E_E3E4=[]
temp_deltaband_coh_5E_E3E4=[]
temp_thetaband_coh_5E_E3E4=[]
temp_alphaband_coh_5E_E3E4=[]
temp_betaband_coh_5E_E3E4=[]
temp_gammaband_coh_5E_E3E4=[]
temp_deltacoh_av30_5E_E3E4=[]
temp_thetacoh_av30_5E_E3E4=[]
temp_alphacoh_av30_5E_E3E4=[]
temp_betacoh_av30_5E_E3E4=[]
temp_gammacoh_av30_5E_E3E4=[]

# Five electrodes - seventh combination 
temp_patientID_5E_E1E5=[]
temp_deltaband_coh_5E_E1E5=[]
temp_thetaband_coh_5E_E1E5=[]
temp_alphaband_coh_5E_E1E5=[]
temp_betaband_coh_5E_E1E5=[]
temp_gammaband_coh_5E_E1E5=[]
temp_deltacoh_av30_5E_E1E5=[]
temp_thetacoh_av30_5E_E1E5=[]
temp_alphacoh_av30_5E_E1E5=[]
temp_betacoh_av30_5E_E1E5=[]
temp_gammacoh_av30_5E_E1E5=[]

# Five electrodes - 8th combination 
temp_patientID_5E_E2E5=[]
temp_deltaband_coh_5E_E2E5=[]
temp_thetaband_coh_5E_E2E5=[]
temp_alphaband_coh_5E_E2E5=[]
temp_betaband_coh_5E_E2E5=[]
temp_gammaband_coh_5E_E2E5=[]
temp_deltacoh_av30_5E_E2E5=[]
temp_thetacoh_av30_5E_E2E5=[]
temp_alphacoh_av30_5E_E2E5=[]
temp_betacoh_av30_5E_E2E5=[]
temp_gammacoh_av30_5E_E2E5=[]

# Five electrodes - 9th combination 
temp_patientID_5E_E3E5=[]
temp_deltaband_coh_5E_E3E5=[]
temp_thetaband_coh_5E_E3E5=[]
temp_alphaband_coh_5E_E3E5=[]
temp_betaband_coh_5E_E3E5=[]
temp_gammaband_coh_5E_E3E5=[]
temp_deltacoh_av30_5E_E3E5=[]
temp_thetacoh_av30_5E_E3E5=[]
temp_alphacoh_av30_5E_E3E5=[]
temp_betacoh_av30_5E_E3E5=[]
temp_gammacoh_av30_5E_E3E5=[]

# Five electrodes - 10th combination 
temp_patientID_5E_E4E5=[]
temp_deltaband_coh_5E_E4E5=[]
temp_thetaband_coh_5E_E4E5=[]
temp_alphaband_coh_5E_E4E5=[]
temp_betaband_coh_5E_E4E5=[]
temp_gammaband_coh_5E_E4E5=[]
temp_deltacoh_av30_5E_E4E5=[]
temp_thetacoh_av30_5E_E4E5=[]
temp_alphacoh_av30_5E_E4E5=[]
temp_betacoh_av30_5E_E4E5=[]
temp_gammacoh_av30_5E_E4E5=[]


# Six electrodes - first combination 
temp_patientID_6E_E1E2=[]
temp_deltaband_coh_6E_E1E2=[]
temp_thetaband_coh_6E_E1E2=[]
temp_alphaband_coh_6E_E1E2=[]
temp_betaband_coh_6E_E1E2=[]
temp_gammaband_coh_6E_E1E2=[]
temp_deltacoh_av30_6E_E1E2=[]
temp_thetacoh_av30_6E_E1E2=[]
temp_alphacoh_av30_6E_E1E2=[]
temp_betacoh_av30_6E_E1E2=[]
temp_gammacoh_av30_6E_E1E2=[]

# Six electrodes - second combination 
temp_patientID_6E_E1E3=[]
temp_deltaband_coh_6E_E1E3=[]
temp_thetaband_coh_6E_E1E3=[]
temp_alphaband_coh_6E_E1E3=[]
temp_betaband_coh_6E_E1E3=[]
temp_gammaband_coh_6E_E1E3=[]
temp_deltacoh_av30_6E_E1E3=[]
temp_thetacoh_av30_6E_E1E3=[]
temp_alphacoh_av30_6E_E1E3=[]
temp_betacoh_av30_6E_E1E3=[]
temp_gammacoh_av30_6E_E1E3=[]

# Six electrodes - third combination 
temp_patientID_6E_E2E3=[]
temp_deltaband_coh_6E_E2E3=[]
temp_thetaband_coh_6E_E2E3=[]
temp_alphaband_coh_6E_E2E3=[]
temp_betaband_coh_6E_E2E3=[]
temp_gammaband_coh_6E_E2E3=[]
temp_deltacoh_av30_6E_E2E3=[]
temp_thetacoh_av30_6E_E2E3=[]
temp_alphacoh_av30_6E_E2E3=[]
temp_betacoh_av30_6E_E2E3=[]
temp_gammacoh_av30_6E_E2E3=[]


# Six electrodes - fourth combination 
temp_patientID_6E_E1E4=[]
temp_deltaband_coh_6E_E1E4=[]
temp_thetaband_coh_6E_E1E4=[]
temp_alphaband_coh_6E_E1E4=[]
temp_betaband_coh_6E_E1E4=[]
temp_gammaband_coh_6E_E1E4=[]
temp_deltacoh_av30_6E_E1E4=[]
temp_thetacoh_av30_6E_E1E4=[]
temp_alphacoh_av30_6E_E1E4=[]
temp_betacoh_av30_6E_E1E4=[]
temp_gammacoh_av30_6E_E1E4=[]


# Six electrodes - fifth combination 
temp_patientID_6E_E2E4=[]
temp_deltaband_coh_6E_E2E4=[]
temp_thetaband_coh_6E_E2E4=[]
temp_alphaband_coh_6E_E2E4=[]
temp_betaband_coh_6E_E2E4=[]
temp_gammaband_coh_6E_E2E4=[]
temp_deltacoh_av30_6E_E2E4=[]
temp_thetacoh_av30_6E_E2E4=[]
temp_alphacoh_av30_6E_E2E4=[]
temp_betacoh_av30_6E_E2E4=[]
temp_gammacoh_av30_6E_E2E4=[]

# Six electrodes - sixth combination 
temp_patientID_6E_E3E4=[]
temp_deltaband_coh_6E_E3E4=[]
temp_thetaband_coh_6E_E3E4=[]
temp_alphaband_coh_6E_E3E4=[]
temp_betaband_coh_6E_E3E4=[]
temp_gammaband_coh_6E_E3E4=[]
temp_deltacoh_av30_6E_E3E4=[]
temp_thetacoh_av30_6E_E3E4=[]
temp_alphacoh_av30_6E_E3E4=[]
temp_betacoh_av30_6E_E3E4=[]
temp_gammacoh_av30_6E_E3E4=[]

# Six electrodes - seventh combination 
temp_patientID_6E_E1E5=[]
temp_deltaband_coh_6E_E1E5=[]
temp_thetaband_coh_6E_E1E5=[]
temp_alphaband_coh_6E_E1E5=[]
temp_betaband_coh_6E_E1E5=[]
temp_gammaband_coh_6E_E1E5=[]
temp_deltacoh_av30_6E_E1E5=[]
temp_thetacoh_av30_6E_E1E5=[]
temp_alphacoh_av30_6E_E1E5=[]
temp_betacoh_av30_6E_E1E5=[]
temp_gammacoh_av30_6E_E1E5=[]

# Six electrodes - 8th combination 
temp_patientID_6E_E2E5=[]
temp_deltaband_coh_6E_E2E5=[]
temp_thetaband_coh_6E_E2E5=[]
temp_alphaband_coh_6E_E2E5=[]
temp_betaband_coh_6E_E2E5=[]
temp_gammaband_coh_6E_E2E5=[]
temp_deltacoh_av30_6E_E2E5=[]
temp_thetacoh_av30_6E_E2E5=[]
temp_alphacoh_av30_6E_E2E5=[]
temp_betacoh_av30_6E_E2E5=[]
temp_gammacoh_av30_6E_E2E5=[]

# Six electrodes - 9th combination 
temp_patientID_6E_E3E5=[]
temp_deltaband_coh_6E_E3E5=[]
temp_thetaband_coh_6E_E3E5=[]
temp_alphaband_coh_6E_E3E5=[]
temp_betaband_coh_6E_E3E5=[]
temp_gammaband_coh_6E_E3E5=[]
temp_deltacoh_av30_6E_E3E5=[]
temp_thetacoh_av30_6E_E3E5=[]
temp_alphacoh_av30_6E_E3E5=[]
temp_betacoh_av30_6E_E3E5=[]
temp_gammacoh_av30_6E_E3E5=[]

# Six electrodes - 10th combination 
temp_patientID_6E_E4E5=[]
temp_deltaband_coh_6E_E4E5=[]
temp_thetaband_coh_6E_E4E5=[]
temp_alphaband_coh_6E_E4E5=[]
temp_betaband_coh_6E_E4E5=[]
temp_gammaband_coh_6E_E4E5=[]
temp_deltacoh_av30_6E_E4E5=[]
temp_thetacoh_av30_6E_E4E5=[]
temp_alphacoh_av30_6E_E4E5=[]
temp_betacoh_av30_6E_E4E5=[]
temp_gammacoh_av30_6E_E4E5=[]


# Six electrodes - 11th combination 
temp_patientID_6E_E1E6=[]
temp_deltaband_coh_6E_E1E6=[]
temp_thetaband_coh_6E_E1E6=[]
temp_alphaband_coh_6E_E1E6=[]
temp_betaband_coh_6E_E1E6=[]
temp_gammaband_coh_6E_E1E6=[]
temp_deltacoh_av30_6E_E1E6=[]
temp_thetacoh_av30_6E_E1E6=[]
temp_alphacoh_av30_6E_E1E6=[]
temp_betacoh_av30_6E_E1E6=[]
temp_gammacoh_av30_6E_E1E6=[]

# Six electrodes - 12th combination 
temp_patientID_6E_E2E6=[]
temp_deltaband_coh_6E_E2E6=[]
temp_thetaband_coh_6E_E2E6=[]
temp_alphaband_coh_6E_E2E6=[]
temp_betaband_coh_6E_E2E6=[]
temp_gammaband_coh_6E_E2E6=[]
temp_deltacoh_av30_6E_E2E6=[]
temp_thetacoh_av30_6E_E2E6=[]
temp_alphacoh_av30_6E_E2E6=[]
temp_betacoh_av30_6E_E2E6=[]
temp_gammacoh_av30_6E_E2E6=[]

# Six electrodes - 13th combination 
temp_patientID_6E_E3E6=[]
temp_deltaband_coh_6E_E3E6=[]
temp_thetaband_coh_6E_E3E6=[]
temp_alphaband_coh_6E_E3E6=[]
temp_betaband_coh_6E_E3E6=[]
temp_gammaband_coh_6E_E3E6=[]
temp_deltacoh_av30_6E_E3E6=[]
temp_thetacoh_av30_6E_E3E6=[]
temp_alphacoh_av30_6E_E3E6=[]
temp_betacoh_av30_6E_E3E6=[]
temp_gammacoh_av30_6E_E3E6=[]

# Six electrodes - 14th combination 
temp_patientID_6E_E4E6=[]
temp_deltaband_coh_6E_E4E6=[]
temp_thetaband_coh_6E_E4E6=[]
temp_alphaband_coh_6E_E4E6=[]
temp_betaband_coh_6E_E4E6=[]
temp_gammaband_coh_6E_E4E6=[]
temp_deltacoh_av30_6E_E4E6=[]
temp_thetacoh_av30_6E_E4E6=[]
temp_alphacoh_av30_6E_E4E6=[]
temp_betacoh_av30_6E_E4E6=[]
temp_gammacoh_av30_6E_E4E6=[]

# Six electrodes - 15th combination 
temp_patientID_6E_E5E6=[]
temp_deltaband_coh_6E_E5E6=[]
temp_thetaband_coh_6E_E5E6=[]
temp_alphaband_coh_6E_E5E6=[]
temp_betaband_coh_6E_E5E6=[]
temp_gammaband_coh_6E_E5E6=[]
temp_deltacoh_av30_6E_E5E6=[]
temp_thetacoh_av30_6E_E5E6=[]
temp_alphacoh_av30_6E_E5E6=[]
temp_betacoh_av30_6E_E5E6=[]
temp_gammacoh_av30_6E_E5E6=[]





# Power 2E
temp_P1_patientID_2E_E1=[]
temp_P1_delta_2E_E1=[]
temp_P1_theta_2E_E1=[]
temp_P1_alpha_2E_E1=[]
temp_P1_beta_2E_E1=[]
temp_P1_gamma_2E_E1=[]
temp_P1_patientID_2E_E2=[]
temp_P1_delta_2E_E2=[]
temp_P1_theta_2E_E2=[]
temp_P1_alpha_2E_E2=[]
temp_P1_beta_2E_E2=[]
temp_P1_gamma_2E_E2=[]

# Power 3E
temp_P1_patientID_3E_E1=[]
temp_P1_delta_3E_E1=[]
temp_P1_theta_3E_E1=[]
temp_P1_alpha_3E_E1=[]
temp_P1_beta_3E_E1=[]
temp_P1_gamma_3E_E1=[]
temp_P1_patientID_3E_E2=[]
temp_P1_delta_3E_E2=[]
temp_P1_theta_3E_E2=[]
temp_P1_alpha_3E_E2=[]
temp_P1_beta_3E_E2=[]
temp_P1_gamma_3E_E2=[]
temp_P1_patientID_3E_E3=[]
temp_P1_delta_3E_E3=[]
temp_P1_theta_3E_E3=[]
temp_P1_alpha_3E_E3=[]
temp_P1_beta_3E_E3=[]
temp_P1_gamma_3E_E3=[]

# Power 4E
temp_P1_patientID_4E_E1=[]
temp_P1_delta_4E_E1=[]
temp_P1_theta_4E_E1=[]
temp_P1_alpha_4E_E1=[]
temp_P1_beta_4E_E1=[]
temp_P1_gamma_4E_E1=[]
temp_P1_patientID_4E_E2=[]
temp_P1_delta_4E_E2=[]
temp_P1_theta_4E_E2=[]
temp_P1_alpha_4E_E2=[]
temp_P1_beta_4E_E2=[]
temp_P1_gamma_4E_E2=[]
temp_P1_patientID_4E_E3=[]
temp_P1_delta_4E_E3=[]
temp_P1_theta_4E_E3=[]
temp_P1_alpha_4E_E3=[]
temp_P1_beta_4E_E3=[]
temp_P1_gamma_4E_E3=[]
temp_P1_patientID_4E_E4=[]
temp_P1_delta_4E_E4=[]
temp_P1_theta_4E_E4=[]
temp_P1_alpha_4E_E4=[]
temp_P1_beta_4E_E4=[]
temp_P1_gamma_4E_E4=[]

# Power 5E
temp_P1_patientID_5E_E1=[]
temp_P1_delta_5E_E1=[]
temp_P1_theta_5E_E1=[]
temp_P1_alpha_5E_E1=[]
temp_P1_beta_5E_E1=[]
temp_P1_gamma_5E_E1=[]
temp_P1_patientID_5E_E2=[]
temp_P1_delta_5E_E2=[]
temp_P1_theta_5E_E2=[]
temp_P1_alpha_5E_E2=[]
temp_P1_beta_5E_E2=[]
temp_P1_gamma_5E_E2=[]
temp_P1_patientID_5E_E3=[]
temp_P1_delta_5E_E3=[]
temp_P1_theta_5E_E3=[]
temp_P1_alpha_5E_E3=[]
temp_P1_beta_5E_E3=[]
temp_P1_gamma_5E_E3=[]
temp_P1_patientID_5E_E4=[]
temp_P1_delta_5E_E4=[]
temp_P1_theta_5E_E4=[]
temp_P1_alpha_5E_E4=[]
temp_P1_beta_5E_E4=[]
temp_P1_gamma_5E_E4=[]
temp_P1_patientID_5E_E5=[]
temp_P1_delta_5E_E5=[]
temp_P1_theta_5E_E5=[]
temp_P1_alpha_5E_E5=[]
temp_P1_beta_5E_E5=[]
temp_P1_gamma_5E_E5=[]

# Power 6E
temp_P1_patientID_6E_E1=[]
temp_P1_delta_6E_E1=[]
temp_P1_theta_6E_E1=[]
temp_P1_alpha_6E_E1=[]
temp_P1_beta_6E_E1=[]
temp_P1_gamma_6E_E1=[]
temp_P1_patientID_6E_E2=[]
temp_P1_delta_6E_E2=[]
temp_P1_theta_6E_E2=[]
temp_P1_alpha_6E_E2=[]
temp_P1_beta_6E_E2=[]
temp_P1_gamma_6E_E2=[]
temp_P1_patientID_6E_E3=[]
temp_P1_delta_6E_E3=[]
temp_P1_theta_6E_E3=[]
temp_P1_alpha_6E_E3=[]
temp_P1_beta_6E_E3=[]
temp_P1_gamma_6E_E3=[]
temp_P1_patientID_6E_E4=[]
temp_P1_delta_6E_E4=[]
temp_P1_theta_6E_E4=[]
temp_P1_alpha_6E_E4=[]
temp_P1_beta_6E_E4=[]
temp_P1_gamma_6E_E4=[]
temp_P1_patientID_6E_E5=[]
temp_P1_delta_6E_E5=[]
temp_P1_theta_6E_E5=[]
temp_P1_alpha_6E_E5=[]
temp_P1_beta_6E_E5=[]
temp_P1_gamma_6E_E5=[]
temp_P1_patientID_6E_E6=[]
temp_P1_delta_6E_E6=[]
temp_P1_theta_6E_E6=[]
temp_P1_alpha_6E_E6=[]
temp_P1_beta_6E_E6=[]
temp_P1_gamma_6E_E6=[]


# 30 second coherence 
temp_delta30=[]
temp_theta30=[]
temp_alpha30=[]
temp_beta30=[]
temp_gamma30=[]


temp_deltacoh_av30_2E_E1E2=[]
temp_thetacoh_av30_2E_E1E2=[]
temp_alphacoh_av30_2E_E1E2=[]
temp_betacoh_av30_2E_E1E2=[]
temp_gammacoh_av30_2E_E1E2=[]

temp_deltacoh_av30_3E_E1E2=[]
temp_thetacoh_av30_3E_E1E2=[]
temp_alphacoh_av30_3E_E1E2=[]
temp_betacoh_av30_3E_E1E2=[]
temp_gammacoh_av30_3E_E1E2=[]

temp_deltacoh_av30_3E_E1E3=[]
temp_thetacoh_av30_3E_E1E3=[]
temp_alphacoh_av30_3E_E1E3=[]
temp_betacoh_av30_3E_E1E3=[]
temp_gammacoh_av30_3E_E1E3=[]

temp_deltacoh_av30_3E_E2E3=[]
temp_thetacoh_av30_3E_E2E3=[]
temp_alphacoh_av30_3E_E2E3=[]
temp_betacoh_av30_3E_E2E3=[]
temp_gammacoh_av30_3E_E2E3=[]


# Loading list with outliers 
Outliers=pd.read_csv('/scratch/users/s184063/RBD_Features/Outliers_RBD.csv')
Bad_channels=Outliers['Bad signals']
print('Outlier dataframe')
print(Bad_channels)


# Looping over all EDF files in folder 
edf_files_list = list_files_in_folder(input_path)

edf_files_list=sorted(edf_files_list)

# Make output path to scratch !!!!!
error_dict=[]

print("EDF files in the folder:")
for edf_file in edf_files_list:
    print(edf_file)

    try:
    
        # Loading EDF file and extracting signals, signal headers and headers for the file
        signals, signal_headers, header = plib.highlevel.read_edf(edf_file)

        # Extracting patientID and visiting number
        numbers_found = os.path.basename(edf_file)
        print("Numbers found in the filename:", numbers_found)

        filename, file_extension = os.path.splitext(numbers_found)

        print('filename')
        print(filename)

        numbers_found=filename

        patientID=filename

        print('PatientID')
        # Skipping the first part of the filename to extract the real patientID 
        patientID_temp=filename[21:] #restructuredfile_RBD_82001_(1)
        patientID=patientID_temp[:-4] 
        print(patientID)

        

        # Extracting the indices for the EEG signals in the data
        # This 'labels_to_find' variable should be corrected for each dataset 
        labels_to_find = ['F3M2','F4M1','C3M2','C4M1', 'O1M2','O2M1']
        indices = get_indices(signal_headers, labels_to_find)
        #print(indices)
        
        
    
        #Looping over the possible labels 
        for label in labels_to_find:
            print(f"The label '{label}' is at index {indices[label]}")
                            
                        
            # Extracting the electrode indexes 
            if label=='F3M2':
                F3M2_index_trial=indices[label]
                print('F3M2 label trial')
                print(F3M2_index_trial)
                print(type(F3M2_index_trial))
                        
                        
                if type(F3M2_index_trial)==int:
                    F3M2_index = F3M2_index_trial
                    print('F3M2_index true')

                    print("F3M2 variable exists.")
                    F3M2_signal=signals[F3M2_index]
                    F3M2_signalheader=signal_headers[F3M2_index]
                    print('F3M2 defined')
                    print(type(F3M2_signal))
                    print(len(F3M2_signal))

                    # file:restructuredfile_RBD_82010_(1).EDF_Electrode:F3M2
                    check_outlier=f'file:restructuredfile_RBD_{patientID}_(1).EDF_Electrode:F3M2'
                    print(check_outlier)
                    print(type(check_outlier))

                    for g in range(len(Bad_channels)):
                        if check_outlier == Bad_channels[g]:
                            print('Bad channel - from outlier document ')
                            print(Bad_channels[g])

                            print('Channel generated')
                            print(f"file:restructuredfile_RBD_'{patientID}'_(1).EDF_Electrode:F3M2")
                            print('Bad channel - F3M2 was deleted')

                            del F3M2_signal, F3M2_index, F3M2_index_trial
                    

                            
                else:
                    print("The variable F3M2 does not exist.")
                            


            # Extracting the electrode indexes 
            if label=='F4M1':
                F4M1_index_trial=indices[label]
                print('F4M1 label trial')
                print(F4M1_index_trial)
                print(type(F4M1_index_trial))
                        
                        
                if type(F4M1_index_trial)==int:
                    F4M1_index = F4M1_index_trial
                    print('F4M1_index true')

                    print("F4M1 variable exists.")
                    F4M1_signal=signals[F4M1_index]
                    F4M1_signalheader=signal_headers[F4M1_index]
                    print('F4M1 defined')
                    print(type(F4M1_signal))
                    print(len(F4M1_signal))

                    # file:restructuredfile_RBD_82010_(1).EDF_Electrode:F4M1
                    check_outlier=f'file:restructuredfile_RBD_{patientID}_(1).EDF_Electrode:F4M1'
                    print(check_outlier)
                    print(type(check_outlier))

                    for g in range(len(Bad_channels)):
                        if check_outlier == Bad_channels[g]:
                            print('Bad channel - from outlier document ')
                            print(Bad_channels[g])

                            print('Channel generated')
                            print(f"file:restructuredfile_RBD_'{patientID}'_(1).EDF_Electrode:F4M1")
                            print('Bad channel - F4M1 was deleted')

                            del F4M1_signal, F4M1_index, F4M1_index_trial

            
                else:
                    print("The variable F4M1 does not exist.")
                            

            # Extracting the electrode indexes 
            if label=='C3M2':
                C3M2_index_trial=indices[label]
                print('C3M2 label trial')
                print(C3M2_index_trial)
                print(type(C3M2_index_trial))
                        
                        
                if type(C3M2_index_trial)==int:
                    C3M2_index = C3M2_index_trial
                    print('C3M2_index true')
                    print(type(C3M2_index))


                    print("C3M2 variable exists.")
                    C3M2_signal=signals[C3M2_index]
                    C3M2_signalheader=signal_headers[C3M2_index]
                    print('C3M2 defined')
                    print(type(C3M2_signal))
                    print(len(C3M2_signal))

                    
                    # file:restructuredfile_RBD_82010_(1).EDF_Electrode:C3M2
                    check_outlier=f'file:restructuredfile_RBD_{patientID}_(1).EDF_Electrode:C3M2'
                    print(check_outlier)
                    print(type(check_outlier))

                    for g in range(len(Bad_channels)):
                        if check_outlier == Bad_channels[g]:
                            print('Bad channel - from outlier document ')
                            print(Bad_channels[g])

                            print('Channel generated')
                            print(f"file:restructuredfile_RBD_'{patientID}'_(1).EDF_Electrode:C3M2")
                            print('Bad channel - C3M2 was deleted')

                            del C3M2_signal, C3M2_index, C3M2_index_trial
                              
                else:
                    print("The variable C3M2 does not exist.")
                    

            # Extracting the electrode indexes 
            if label=='C4M1':
                C4M1_index_trial=indices[label]
                print('C4M1 label trial')
                print(C4M1_index_trial)
                print(type(C4M1_index_trial))
                        
                        
                if type(C4M1_index_trial)==int:
                    C4M1_index = C4M1_index_trial
                    print('C4M1_index true')

                    print("C4M1 variable exists.")
                    C4M1_signal=signals[C4M1_index]
                    C4M1_signalheader=signal_headers[C4M1_index]
                    print('C4M1 defined')
                    print(type(C4M1_signal))
                    print(len(C4M1_signal))

                    # file:restructuredfile_RBD_82010_(1).EDF_Electrode:C4M1
                    check_outlier=f'file:restructuredfile_RBD_{patientID}_(1).EDF_Electrode:C4M1'
                    print(check_outlier)
                    print(type(check_outlier))

                    for g in range(len(Bad_channels)):
                        if check_outlier == Bad_channels[g]:
                            
                            print('Bad channel - from outlier document ')
                            print(Bad_channels[g])

                            print('Channel generated')
                            print(f"file:restructuredfile_RBD_'{patientID}'_(1).EDF_Electrode:C4M1")
                            print('Bad channel - C4M1 was deleted')

                            del C4M1_signal, C4M1_index, C4M1_index_trial
                            
                else:
                    print("The variable C4M1 does not exist.")
                            
                    
            # Extracting the electrode indexes 
            if label=='O2M1':
                O2M1_index_trial=indices[label]
                print('O2M1 label trial')
                print(O2M1_index_trial)
                print(type(O2M1_index_trial))
                        
                        
                if type(O2M1_index_trial)==int:
                    O2M1_index = O2M1_index_trial
                    print('O2M1_index true')

                    print("O2M1 variable exists.")
                    O2M1_signal=signals[O2M1_index]
                    O2M1_signalheader=signal_headers[O2M1_index]
                    print('O2M1 defined')
                    print(type(O2M1_signal))
                    print(len(O2M1_signal))

                    # file:restructuredfile_RBD_82010_(1).EDF_Electrode:O2M1
                    check_outlier=f'file:restructuredfile_RBD_{patientID}_(1).EDF_Electrode:O2M1'
                    print(check_outlier)
                    print(type(check_outlier))

                    for g in range(len(Bad_channels)):
                        if check_outlier == Bad_channels[g]:
                            
                            print('Bad channel - from outlier document ')
                            print(Bad_channels[g])

                            print('Channel generated')
                            print(f"file:restructuredfile_RBD_'{patientID}'_(1).EDF_Electrode:O2M1")
                            print('Bad channel - O2M1 was deleted')

                            del O2M1_signal, O2M1_index, O2M1_index_trial
                            
                else:
                    print("The variable O2M1 does not exist.")
                            

            # Extracting the electrode indexes 
            if label=='O1M2':
                O1M2_index_trial=indices[label]
                print('O1M2 label trial')
                print(O1M2_index_trial)
                print(type(O1M2_index_trial))
                        
                        
                if type(O1M2_index_trial)==int:
                    O1M2_index = O1M2_index_trial
                    print('O1M2_index true')

                    print("O1M2 variable exists.")
                    O1M2_signal=signals[O1M2_index]
                    O1M2_signalheader=signal_headers[O1M2_index]
                    print('O1M2 defined')
                    print(type(O1M2_signal))
                    print(len(O1M2_signal))

                    # file:restructuredfile_RBD_82010_(1).EDF_Electrode:O1M2
                    check_outlier=f'file:restructuredfile_RBD_{patientID}_(1).EDF_Electrode:O1M2'
                    print(check_outlier)
                    print(type(check_outlier))

                    for g in range(len(Bad_channels)):
                        if check_outlier == Bad_channels[g]:
                            
                            print('Bad channel - from outlier document ')
                            print(Bad_channels[g])

                            print('Channel generated')
                            print(f"file:restructuredfile_RBD_'{patientID}'_(1).EDF_Electrode:O1M2")
                            print('Bad channel - O1M2 was deleted')

                            del O1M2_signal, O1M2_index, O1M2_index_trial

                            
                else:
                    print("The variable O1M2 does not exist.")


        # Deleting indexes to make sure no errors are made mixing up patients 
        if 'F3M2_index_trial' in locals():
            del F3M2_index_trial

        if 'F4M1_index_trial' in locals():
            del F4M1_index_trial

        if 'C3M2_index_trial' in locals():
            del C3M2_index_trial

        if 'C4M1_index_trial' in locals():
            del C4M1_index_trial

        if 'O1M2_index_trial' in locals():
            del O1M2_index_trial
        
        if 'O2M1_index_trial' in locals():
            del O2M1_index_trial


        # Packing electrode combinations based on the correct index 
        if 'F3M2_index' in locals() and 'F4M1_index' in locals() and 'C3M2_index' in locals() and 'C4M1_index' in locals() and 'O1M2_index' in locals() and 'O2M1_index' in locals(): 
                    
            print('The combination: F3M2, F4M1, C3M2, C4M1, O1M2, O2M1 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            Electrodes=['F3M2','F4M1','C3M2','C4M1','O1M2','O2M1']
            Signals=np.vstack((F3M2_signal,F4M1_signal,C3M2_signal,C4M1_signal,O1M2_signal,O2M1_signal))

            del F3M2_index,F4M1_index,C3M2_index,C4M1_index,O1M2_index,O2M1_index

        
        # Packing electrode combinations based on the correct index 
        if 'F3M2_index' in locals() and 'F4M1_index' in locals() and 'C4M1_index' in locals() and 'O1M2_index' in locals() : 
                    
            print('The combination: F3M2, F4M1, C4M1, O1M2 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            Electrodes=['F3M2','F4M1','C4M1','O1M2']
            Signals=np.vstack((F3M2_signal,F4M1_signal,C4M1_signal,O1M2_signal))

            del F3M2_index,F4M1_index,C4M1_index,O1M2_index


        # Packing electrode combinations based on the correct index 
        if 'F3M2_index' in locals() and 'C3M2_index' in locals() and 'C4M1_index' in locals(): 
                    
            print('The combination: F3M2, C3M2, C4M1 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            Electrodes=['F3M2','C3M2','C4M1']
            Signals=np.vstack((F3M2_signal,C3M2_signal,C4M1_signal))

            del F3M2_index,C3M2_index,C4M1_index


        # Packing electrode combinations based on the correct index 
        if 'F3M2_index' in locals() and 'C3M2_index' in locals() and 'C4M1_index' in locals() and 'O1M2_index' in locals() and 'O2M1_index' in locals(): 
                    
            print('The combination: F3M2, C3M2, C4M1, O1M2, O2M1 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            Electrodes=['F3M2','C3M2','C4M1','O1M2','O2M1']
            Signals=np.vstack((F3M2_signal,C3M2_signal,C4M1_signal,O1M2_signal,O2M1_signal))

            del F3M2_index,C3M2_index,C4M1_index,O1M2_index,O2M1_index


        # Packing electrode combinations based on the correct index 
        if 'F3M2_index' in locals() and 'F4M1_index' in locals() and 'C3M2_index' in locals() and 'C4M1_index' in locals() and 'O1M2_index' in locals(): 
                    
            print('The combination: F3M2, F4M1, C3M2, C4M1, O1M2 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            Electrodes=['F3M2','F4M1','C3M2','C4M1','O1M2']
            Signals=np.vstack((F3M2_signal,F4M1_signal,C3M2_signal,C4M1_signal,O1M2_signall))

            del F3M2_index,F4M1_index,C3M2_index,C4M1_index,O1M2_index

        
        # Packing electrode combinations based on the correct index 
        if 'F3M2_index' in locals() and 'F4M1_index' in locals() and 'C3M2_index' in locals() and 'C4M1_index' in locals() and 'O2M1_index' in locals(): 
                    
            print('The combination: F3M2, F4M1, C3M2, C4M1, O2M1 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            Electrodes=['F3M2','F4M1','C3M2','C4M1','O2M1']
            Signals=np.vstack((F3M2_signal,F4M1_signal,C3M2_signal,C4M1_signal,O2M1_signal))

            del F3M2_index,F4M1_index,C3M2_index,C4M1_index,O2M1_index

        # Packing electrode combinations based on the correct index 
        if 'F3M2_index' in locals()  and 'C3M2_index' in locals()  and 'O1M2_index' in locals(): 
                    
            print('The combination: F3M2, C3M2, O1M2 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            Electrodes=['F3M2','C3M2','O1M2']
            Signals=np.vstack((F3M2_signal,C3M2_signal,O1M2_signal))

            del F3M2_index,C3M2_index,O1M2_index



        if 'C3M2_index' in locals() and 'C4M1_index' in locals() and 'O1M2_index' in locals() and 'O2M1_index' in locals(): 
                    
            print('The combination: C3M2, C4M1, O1M2, O2M1 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            Electrodes=['C3M2','C4M1','O1M2','O2M1']
            Signals=np.vstack((C3M2_signal,C4M1_signal,O1M2_signal,O2M1_signal))


            del C3M2_index,C4M1_index,O1M2_index,O2M1_index

        if 'C3M2_index' in locals() and 'C4M1_index' in locals() and 'O2M1_index' in locals(): 
                    
            print('The combination: C3M2, C4M1, O2M1 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            Electrodes=['C3M2','C4M1','O2M1'] 
            Signals=np.vstack((C3M2_signal,C4M1_signal,O2M1_signal))

            del C3M2_index,C4M1_index,O2M1_index


        if 'C3M2_index' in locals() and 'C4M1_index' in locals() and 'O1M2_index' in locals(): 
                    
            print('The combination: C3M2, C4M1, O1M2 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            Electrodes=['C3M2','C4M1','O1M2']
            Signals=np.vstack((C3M2_signal,C4M1_signal,O1M2_signal))

            del C3M2_index,C4M1_index,O1M2_index


        if 'C3M2_index' in locals() and 'C4M1_index' in locals(): 
                    
            print('The combination: C3M2, C4M1 exists ')

            # Stacking values having a specific order (frontal, central, occipital)
            Electrodes=['C3M2','C4M1']
            Signals=np.vstack((C3M2_signal,C4M1_signal))

            del C3M2_index,C4M1_index

        


        ### Loop over electrodes to get one band power analysis per electrode ####
        
        for loop_factor in range(len(Electrodes)): 
            
            print('Bandpower loop')
            print(loop_factor)
            
            print('Electrodes')
            print(Electrodes)

            print('Signals matrix information')
            print(Signals.shape)
            print(type(Signals))

            Electrode_for_naming=extract_letters_and_numbers(Electrodes[loop_factor])
            print(Electrode_for_naming)


            
            ######### Band power analysis ##################
            # Calling pre-processing function for a single electrode 
            print('Signal for pre-processing')
            Electrode_single=Signals[loop_factor]
            print(type(Electrode_single))
            #print(Electrode_single)


            signal_new_single, fs_new_single, time_filtered_HP_single = preprocessing(Electrode_single)#, f'{Electrode_for_naming}_signalheader')



            # Creating bandpass filters for EEG frequency bands (theta, delta, alpha, beta, gamma)
            sos_delta, sos_theta, sos_alpha, sos_beta, sos_gamma=bandpass_frequency_band()
                    

            # Apply the filters
            # Applying bandpass filter to signal using filtfilt in order to make zero phase filtering 

            # Signal 1
            time_filtered_delta1=scipy.signal.sosfiltfilt(sos_delta, time_filtered_HP_single, axis=-1, padtype='odd', padlen=None)
            time_filtered_theta1=scipy.signal.sosfiltfilt(sos_theta, time_filtered_HP_single, axis=-1, padtype='odd', padlen=None)
            time_filtered_alpha1=scipy.signal.sosfiltfilt(sos_alpha, time_filtered_HP_single, axis=-1, padtype='odd', padlen=None)
            time_filtered_beta1=scipy.signal.sosfiltfilt(sos_beta, time_filtered_HP_single, axis=-1, padtype='odd', padlen=None)
            time_filtered_gamma1=scipy.signal.sosfiltfilt(sos_gamma, time_filtered_HP_single, axis=-1, padtype='odd', padlen=None)


            '''
            # Magnitude spectrum dB of highpass filtered time signal 1
            fig, axs = matplotlib.pyplot.subplots(2, 3, figsize=(15, 10))
            fig.suptitle("Magnitude spectra bandpass filtered - Channel 1")
            axs[0 , 0].magnitude_spectrum(time_filtered_delta1,128,scale='dB')
            #axs[0 , 0].set_xlabel('Frequency [Hz]') 
            axs[0 , 0].set_ylabel('Magnitude [dB]') 
            axs[0 , 0].set_title('Delta filtered')
                
            axs[0 , 1].magnitude_spectrum(time_filtered_theta1,128,scale='dB')
            #axs[0 , 1].set_xlabel('Frequency [Hz]') 
            axs[0 , 1].set_ylabel('Magnitude [dB]') 
            axs[0 , 1].set_title('Theta filtered')

            axs[0 , 2].magnitude_spectrum(time_filtered_alpha1,128,scale='dB')
            #axs[0 , 2].set_xlabel('Frequency [Hz]') 
            axs[0 , 2].set_ylabel('Magnitude [dB]') 
            axs[0 , 2].set_title('Alpha filtered')

            axs[1 , 0].magnitude_spectrum(time_filtered_beta1,128,scale='dB')
            axs[1 , 0].set_xlabel('Frequency [Hz]') 
            axs[1 , 0].set_ylabel('Magnitude [dB]') 
            axs[1 , 0].set_title('Beta filtered')

            axs[1 , 1].magnitude_spectrum(time_filtered_gamma1,128,scale='dB')
            axs[1 , 1].set_xlabel('Frequency [Hz]') 
            axs[1 , 1].set_ylabel('Magnitude [dB]') 
            axs[1 , 1].set_title('Gamma filtered')
            matplotlib.pyplot.tight_layout()
            matplotlib.pyplot.show()
            '''
        

            ### relative power for bandpass filtered signals ###

            # Signal 1
            print('Signal 1 - bandpass')
            P1_delta=relative_power_for_frequencyband(time_filtered_delta1,time_filtered_HP_single)
            P1_theta=relative_power_for_frequencyband(time_filtered_theta1,time_filtered_HP_single)
            P1_alpha=relative_power_for_frequencyband(time_filtered_alpha1,time_filtered_HP_single)
            P1_beta=relative_power_for_frequencyband(time_filtered_beta1,time_filtered_HP_single)
            P1_gamma=relative_power_for_frequencyband(time_filtered_gamma1,time_filtered_HP_single)

            P1_patientID=patientID

            # Number of electrodes 
            if len(Electrodes) ==2:

                if Electrodes[0]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_2E_E1.append(P1_patientID)
                    temp_P1_delta_2E_E1.append(P1_delta)
                    temp_P1_theta_2E_E1.append(P1_theta)
                    temp_P1_alpha_2E_E1.append(P1_alpha)
                    temp_P1_beta_2E_E1.append(P1_beta)
                    temp_P1_gamma_2E_E1.append(P1_gamma)

                    # Stack values
                    P1_patientID_2E_E1=np.stack(temp_P1_patientID_2E_E1)
                    P1_delta_2E_E1=np.stack(temp_P1_delta_2E_E1)
                    P1_theta_2E_E1=np.stack(temp_P1_theta_2E_E1)
                    P1_alpha_2E_E1=np.stack(temp_P1_alpha_2E_E1)
                    P1_beta_2E_E1=np.stack(temp_P1_beta_2E_E1)
                    P1_gamma_2E_E1=np.stack(temp_P1_gamma_2E_E1)

                    # Packing data in dictonary 
                    Bandpower_values_2E_E1 = {
                        'PatientID': P1_patientID_2E_E1,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_2E_E1.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_2E_E1.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_2E_E1.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_2E_E1.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_2E_E1.tolist(),

                        }
                    
                    print('Bandpower 2E was packed - E1')
                    print(Bandpower_values_2E_E1)

                elif Electrodes[1]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_2E_E2.append(P1_patientID)
                    temp_P1_delta_2E_E2.append(P1_delta)
                    temp_P1_theta_2E_E2.append(P1_theta)
                    temp_P1_alpha_2E_E2.append(P1_alpha)
                    temp_P1_beta_2E_E2.append(P1_beta)
                    temp_P1_gamma_2E_E2.append(P1_gamma)

                    # Stack values
                    P1_patientID_2E_E2=np.stack(temp_P1_patientID_2E_E2)
                    P1_delta_2E_E2=np.stack(temp_P1_delta_2E_E2)
                    P1_theta_2E_E2=np.stack(temp_P1_theta_2E_E2)
                    P1_alpha_2E_E2=np.stack(temp_P1_alpha_2E_E2)
                    P1_beta_2E_E2=np.stack(temp_P1_beta_2E_E2)
                    P1_gamma_2E_E2=np.stack(temp_P1_gamma_2E_E2)

                    # Packing data in dictonary 
                    Bandpower_values_2E_E2 = {
                        'PatientID': P1_patientID_2E_E2,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_2E_E2.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_2E_E2.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_2E_E2.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_2E_E2.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_2E_E2.tolist(),

                        }
                    
                    print('Bandpower 2E was packed - E2')
                    print(Bandpower_values_2E_E2)


            # Number of electrodes 
            if len(Electrodes) ==3:
                
                if Electrodes[0]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_3E_E1.append(P1_patientID)
                    temp_P1_delta_3E_E1.append(P1_delta)
                    temp_P1_theta_3E_E1.append(P1_theta)
                    temp_P1_alpha_3E_E1.append(P1_alpha)
                    temp_P1_beta_3E_E1.append(P1_beta)
                    temp_P1_gamma_3E_E1.append(P1_gamma)

                    # Stack values
                    P1_patientID_3E_E1=np.stack(temp_P1_patientID_3E_E1)
                    P1_delta_3E_E1=np.stack(temp_P1_delta_3E_E1)
                    P1_theta_3E_E1=np.stack(temp_P1_theta_3E_E1)
                    P1_alpha_3E_E1=np.stack(temp_P1_alpha_3E_E1)
                    P1_beta_3E_E1=np.stack(temp_P1_beta_3E_E1)
                    P1_gamma_3E_E1=np.stack(temp_P1_gamma_3E_E1)

                    # Packing data in dictonary 
                    Bandpower_values_3E_E1 = {
                        'PatientID': P1_patientID_3E_E1,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_3E_E1.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_3E_E1.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_3E_E1.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_3E_E1.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_3E_E1.tolist(),

                        }
                    
                    print('Bandpower 3E was packed - E1')
                    print(Bandpower_values_3E_E1)

                elif Electrodes[1]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_3E_E2.append(P1_patientID)
                    temp_P1_delta_3E_E2.append(P1_delta)
                    temp_P1_theta_3E_E2.append(P1_theta)
                    temp_P1_alpha_3E_E2.append(P1_alpha)
                    temp_P1_beta_3E_E2.append(P1_beta)
                    temp_P1_gamma_3E_E2.append(P1_gamma)

                    # Stack values
                    P1_patientID_3E_E2=np.stack(temp_P1_patientID_3E_E2)
                    P1_delta_3E_E2=np.stack(temp_P1_delta_3E_E2)
                    P1_theta_3E_E2=np.stack(temp_P1_theta_3E_E2)
                    P1_alpha_3E_E2=np.stack(temp_P1_alpha_3E_E2)
                    P1_beta_3E_E2=np.stack(temp_P1_beta_3E_E2)
                    P1_gamma_3E_E2=np.stack(temp_P1_gamma_3E_E2)

                    # Packing data in dictonary 
                    Bandpower_values_3E_E2 = {
                        'PatientID': P1_patientID_3E_E2,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_3E_E2.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_3E_E2.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_3E_E2.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_3E_E2.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_3E_E2.tolist(),

                        }
                    
                    print('Bandpower 3E was packed - E2')
                    print(Bandpower_values_3E_E2)
            
            
                elif Electrodes[2]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_3E_E3.append(P1_patientID)
                    temp_P1_delta_3E_E3.append(P1_delta)
                    temp_P1_theta_3E_E3.append(P1_theta)
                    temp_P1_alpha_3E_E3.append(P1_alpha)
                    temp_P1_beta_3E_E3.append(P1_beta)
                    temp_P1_gamma_3E_E3.append(P1_gamma)

                    # Stack values
                    P1_patientID_3E_E3=np.stack(temp_P1_patientID_3E_E3)
                    P1_delta_3E_E3=np.stack(temp_P1_delta_3E_E3)
                    P1_theta_3E_E3=np.stack(temp_P1_theta_3E_E3)
                    P1_alpha_3E_E3=np.stack(temp_P1_alpha_3E_E3)
                    P1_beta_3E_E3=np.stack(temp_P1_beta_3E_E3)
                    P1_gamma_3E_E3=np.stack(temp_P1_gamma_3E_E3)

                    # Packing data in dictonary 
                    Bandpower_values_3E_E3 = {
                        'PatientID': P1_patientID_3E_E3,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_3E_E3.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_3E_E3.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_3E_E3.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_3E_E3.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_3E_E3.tolist(),

                        }
                    
                    print('Bandpower 3E was packed - E3')
                    print(Bandpower_values_3E_E3)

            # Number of electrodes 
            if len(Electrodes) ==4:
                
                if Electrodes[0]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_4E_E1.append(P1_patientID)
                    temp_P1_delta_4E_E1.append(P1_delta)
                    temp_P1_theta_4E_E1.append(P1_theta)
                    temp_P1_alpha_4E_E1.append(P1_alpha)
                    temp_P1_beta_4E_E1.append(P1_beta)
                    temp_P1_gamma_4E_E1.append(P1_gamma)

                    # Stack values
                    P1_patientID_4E_E1=np.stack(temp_P1_patientID_4E_E1)
                    P1_delta_4E_E1=np.stack(temp_P1_delta_4E_E1)
                    P1_theta_4E_E1=np.stack(temp_P1_theta_4E_E1)
                    P1_alpha_4E_E1=np.stack(temp_P1_alpha_4E_E1)
                    P1_beta_4E_E1=np.stack(temp_P1_beta_4E_E1)
                    P1_gamma_4E_E1=np.stack(temp_P1_gamma_4E_E1)

                    # Packing data in dictonary 
                    Bandpower_values_4E_E1 = {
                        'PatientID': P1_patientID_4E_E1,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_4E_E1.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_4E_E1.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_4E_E1.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_4E_E1.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_4E_E1.tolist(),

                        }
                    
                    print('Bandpower 4E was packed - E1')
                    print(Bandpower_values_4E_E1)

                elif Electrodes[1]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_4E_E2.append(P1_patientID)
                    temp_P1_delta_4E_E2.append(P1_delta)
                    temp_P1_theta_4E_E2.append(P1_theta)
                    temp_P1_alpha_4E_E2.append(P1_alpha)
                    temp_P1_beta_4E_E2.append(P1_beta)
                    temp_P1_gamma_4E_E2.append(P1_gamma)

                    # Stack values
                    P1_patientID_4E_E2=np.stack(temp_P1_patientID_4E_E2)
                    P1_delta_4E_E2=np.stack(temp_P1_delta_4E_E2)
                    P1_theta_4E_E2=np.stack(temp_P1_theta_4E_E2)
                    P1_alpha_4E_E2=np.stack(temp_P1_alpha_4E_E2)
                    P1_beta_4E_E2=np.stack(temp_P1_beta_4E_E2)
                    P1_gamma_4E_E2=np.stack(temp_P1_gamma_4E_E2)

                    # Packing data in dictonary 
                    Bandpower_values_4E_E2 = {
                        'PatientID': P1_patientID_4E_E2,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_4E_E2.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_4E_E2.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_4E_E2.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_4E_E2.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_4E_E2.tolist(),

                        }
                    
                    print('Bandpower 4E was packed - E2')
                    print(Bandpower_values_4E_E2)
            
            
                elif Electrodes[2]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_4E_E3.append(P1_patientID)
                    temp_P1_delta_4E_E3.append(P1_delta)
                    temp_P1_theta_4E_E3.append(P1_theta)
                    temp_P1_alpha_4E_E3.append(P1_alpha)
                    temp_P1_beta_4E_E3.append(P1_beta)
                    temp_P1_gamma_4E_E3.append(P1_gamma)

                    # Stack values
                    P1_patientID_4E_E3=np.stack(temp_P1_patientID_4E_E3)
                    P1_delta_4E_E3=np.stack(temp_P1_delta_4E_E3)
                    P1_theta_4E_E3=np.stack(temp_P1_theta_4E_E3)
                    P1_alpha_4E_E3=np.stack(temp_P1_alpha_4E_E3)
                    P1_beta_4E_E3=np.stack(temp_P1_beta_4E_E3)
                    P1_gamma_4E_E3=np.stack(temp_P1_gamma_4E_E3)

                    # Packing data in dictonary 
                    Bandpower_values_4E_E3 = {
                        'PatientID': P1_patientID_4E_E3,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_4E_E3.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_4E_E3.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_4E_E3.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_4E_E3.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_4E_E3.tolist(),

                        }
                    
                    print('Bandpower 4E was packed - E3')
                    print(Bandpower_values_4E_E3)

                elif Electrodes[3]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_4E_E4.append(P1_patientID)
                    temp_P1_delta_4E_E4.append(P1_delta)
                    temp_P1_theta_4E_E4.append(P1_theta)
                    temp_P1_alpha_4E_E4.append(P1_alpha)
                    temp_P1_beta_4E_E4.append(P1_beta)
                    temp_P1_gamma_4E_E4.append(P1_gamma)

                    # Stack values
                    P1_patientID_4E_E4=np.stack(temp_P1_patientID_4E_E4)
                    P1_delta_4E_E4=np.stack(temp_P1_delta_4E_E4)
                    P1_theta_4E_E4=np.stack(temp_P1_theta_4E_E4)
                    P1_alpha_4E_E4=np.stack(temp_P1_alpha_4E_E4)
                    P1_beta_4E_E4=np.stack(temp_P1_beta_4E_E4)
                    P1_gamma_4E_E4=np.stack(temp_P1_gamma_4E_E4)

                    # Packing data in dictonary 
                    Bandpower_values_4E_E4 = {
                        'PatientID': P1_patientID_4E_E4,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_4E_E4.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_4E_E4.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_4E_E4.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_4E_E4.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_4E_E4.tolist(),

                        }
                    
                    print('Bandpower 4E was packed - E4')
                    print(Bandpower_values_4E_E4)

            # Number of electrodes 
            if len(Electrodes) ==5:
                
                if Electrodes[0]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_5E_E1.append(P1_patientID)
                    temp_P1_delta_5E_E1.append(P1_delta)
                    temp_P1_theta_5E_E1.append(P1_theta)
                    temp_P1_alpha_5E_E1.append(P1_alpha)
                    temp_P1_beta_5E_E1.append(P1_beta)
                    temp_P1_gamma_5E_E1.append(P1_gamma)

                    # Stack values
                    P1_patientID_5E_E1=np.stack(temp_P1_patientID_5E_E1)
                    P1_delta_5E_E1=np.stack(temp_P1_delta_5E_E1)
                    P1_theta_5E_E1=np.stack(temp_P1_theta_5E_E1)
                    P1_alpha_5E_E1=np.stack(temp_P1_alpha_5E_E1)
                    P1_beta_5E_E1=np.stack(temp_P1_beta_5E_E1)
                    P1_gamma_5E_E1=np.stack(temp_P1_gamma_5E_E1)

                    # Packing data in dictonary 
                    Bandpower_values_5E_E1 = {
                        'PatientID': P1_patientID_5E_E1,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_5E_E1.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_5E_E1.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_5E_E1.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_5E_E1.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_5E_E1.tolist(),

                        }
                    
                    print('Bandpower 5E was packed - E1')
                    print(Bandpower_values_5E_E1)

                elif Electrodes[1]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_5E_E2.append(P1_patientID)
                    temp_P1_delta_5E_E2.append(P1_delta)
                    temp_P1_theta_5E_E2.append(P1_theta)
                    temp_P1_alpha_5E_E2.append(P1_alpha)
                    temp_P1_beta_5E_E2.append(P1_beta)
                    temp_P1_gamma_5E_E2.append(P1_gamma)

                    # Stack values
                    P1_patientID_5E_E2=np.stack(temp_P1_patientID_5E_E2)
                    P1_delta_5E_E2=np.stack(temp_P1_delta_5E_E2)
                    P1_theta_5E_E2=np.stack(temp_P1_theta_5E_E2)
                    P1_alpha_5E_E2=np.stack(temp_P1_alpha_5E_E2)
                    P1_beta_5E_E2=np.stack(temp_P1_beta_5E_E2)
                    P1_gamma_5E_E2=np.stack(temp_P1_gamma_5E_E2)

                    # Packing data in dictonary 
                    Bandpower_values_5E_E2 = {
                        'PatientID': P1_patientID_5E_E2,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_5E_E2.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_5E_E2.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_5E_E2.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_5E_E2.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_5E_E2.tolist(),

                        }
                    
                    print('Bandpower 5E was packed - E2')
                    print(Bandpower_values_5E_E2)
            
            
                elif Electrodes[2]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_5E_E3.append(P1_patientID)
                    temp_P1_delta_5E_E3.append(P1_delta)
                    temp_P1_theta_5E_E3.append(P1_theta)
                    temp_P1_alpha_5E_E3.append(P1_alpha)
                    temp_P1_beta_5E_E3.append(P1_beta)
                    temp_P1_gamma_5E_E3.append(P1_gamma)

                    # Stack values
                    P1_patientID_5E_E3=np.stack(temp_P1_patientID_5E_E3)
                    P1_delta_5E_E3=np.stack(temp_P1_delta_5E_E3)
                    P1_theta_5E_E3=np.stack(temp_P1_theta_5E_E3)
                    P1_alpha_5E_E3=np.stack(temp_P1_alpha_5E_E3)
                    P1_beta_5E_E3=np.stack(temp_P1_beta_5E_E3)
                    P1_gamma_5E_E3=np.stack(temp_P1_gamma_5E_E3)

                    # Packing data in dictonary 
                    Bandpower_values_5E_E3 = {
                        'PatientID': P1_patientID_5E_E3,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_5E_E3.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_5E_E3.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_5E_E3.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_5E_E3.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_5E_E3.tolist(),

                        }
                    
                    print('Bandpower 5E was packed - E3')
                    print(Bandpower_values_5E_E3)

                elif Electrodes[3]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_5E_E4.append(P1_patientID)
                    temp_P1_delta_5E_E4.append(P1_delta)
                    temp_P1_theta_5E_E4.append(P1_theta)
                    temp_P1_alpha_5E_E4.append(P1_alpha)
                    temp_P1_beta_5E_E4.append(P1_beta)
                    temp_P1_gamma_5E_E4.append(P1_gamma)

                    # Stack values
                    P1_patientID_5E_E4=np.stack(temp_P1_patientID_5E_E4)
                    P1_delta_5E_E4=np.stack(temp_P1_delta_5E_E4)
                    P1_theta_5E_E4=np.stack(temp_P1_theta_5E_E4)
                    P1_alpha_5E_E4=np.stack(temp_P1_alpha_5E_E4)
                    P1_beta_5E_E4=np.stack(temp_P1_beta_5E_E4)
                    P1_gamma_5E_E4=np.stack(temp_P1_gamma_5E_E4)

                    # Packing data in dictonary 
                    Bandpower_values_5E_E4 = {
                        'PatientID': P1_patientID_5E_E4,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_5E_E4.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_5E_E4.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_5E_E4.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_5E_E4.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_5E_E4.tolist(),

                        }
                    
                    print('Bandpower 5E was packed - E4')
                    print(Bandpower_values_5E_E4)


                elif Electrodes[4]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_5E_E5.append(P1_patientID)
                    temp_P1_delta_5E_E5.append(P1_delta)
                    temp_P1_theta_5E_E5.append(P1_theta)
                    temp_P1_alpha_5E_E5.append(P1_alpha)
                    temp_P1_beta_5E_E5.append(P1_beta)
                    temp_P1_gamma_5E_E5.append(P1_gamma)

                    # Stack values
                    P1_patientID_5E_E5=np.stack(temp_P1_patientID_5E_E5)
                    P1_delta_5E_E5=np.stack(temp_P1_delta_5E_E5)
                    P1_theta_5E_E5=np.stack(temp_P1_theta_5E_E5)
                    P1_alpha_5E_E5=np.stack(temp_P1_alpha_5E_E5)
                    P1_beta_5E_E5=np.stack(temp_P1_beta_5E_E5)
                    P1_gamma_5E_E5=np.stack(temp_P1_gamma_5E_E5)

                    # Packing data in dictonary 
                    Bandpower_values_5E_E5 = {
                        'PatientID': P1_patientID_5E_E5,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_5E_E5.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_5E_E5.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_5E_E5.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_5E_E5.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_5E_E5.tolist(),

                        }
                    
                    print('Bandpower 5E was packed - E5')
                    print(Bandpower_values_5E_E5)

            # Number of electrodes 
            if len(Electrodes) ==6:
                
                if Electrodes[0]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_6E_E1.append(P1_patientID)
                    temp_P1_delta_6E_E1.append(P1_delta)
                    temp_P1_theta_6E_E1.append(P1_theta)
                    temp_P1_alpha_6E_E1.append(P1_alpha)
                    temp_P1_beta_6E_E1.append(P1_beta)
                    temp_P1_gamma_6E_E1.append(P1_gamma)

                    # Stack values
                    P1_patientID_6E_E1=np.stack(temp_P1_patientID_6E_E1)
                    P1_delta_6E_E1=np.stack(temp_P1_delta_6E_E1)
                    P1_theta_6E_E1=np.stack(temp_P1_theta_6E_E1)
                    P1_alpha_6E_E1=np.stack(temp_P1_alpha_6E_E1)
                    P1_beta_6E_E1=np.stack(temp_P1_beta_6E_E1)
                    P1_gamma_6E_E1=np.stack(temp_P1_gamma_6E_E1)

                    # Packing data in dictonary 
                    Bandpower_values_6E_E1 = {
                        'PatientID': P1_patientID_6E_E1,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_6E_E1.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_6E_E1.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_6E_E1.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_6E_E1.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_6E_E1.tolist(),

                        }
                    
                    print('Bandpower 6E was packed - E1')
                    print(Bandpower_values_6E_E1)

                elif Electrodes[1]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_6E_E2.append(P1_patientID)
                    temp_P1_delta_6E_E2.append(P1_delta)
                    temp_P1_theta_6E_E2.append(P1_theta)
                    temp_P1_alpha_6E_E2.append(P1_alpha)
                    temp_P1_beta_6E_E2.append(P1_beta)
                    temp_P1_gamma_6E_E2.append(P1_gamma)

                    # Stack values
                    P1_patientID_6E_E2=np.stack(temp_P1_patientID_6E_E2)
                    P1_delta_6E_E2=np.stack(temp_P1_delta_6E_E2)
                    P1_theta_6E_E2=np.stack(temp_P1_theta_6E_E2)
                    P1_alpha_6E_E2=np.stack(temp_P1_alpha_6E_E2)
                    P1_beta_6E_E2=np.stack(temp_P1_beta_6E_E2)
                    P1_gamma_6E_E2=np.stack(temp_P1_gamma_6E_E2)

                    # Packing data in dictonary 
                    Bandpower_values_6E_E2 = {
                        'PatientID': P1_patientID_6E_E2,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_6E_E2.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_6E_E2.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_6E_E2.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_6E_E2.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_6E_E2.tolist(),

                        }
                    
                    print('Bandpower 6E was packed - E2')
                    print(Bandpower_values_6E_E2)
            
            
                elif Electrodes[2]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_6E_E3.append(P1_patientID)
                    temp_P1_delta_6E_E3.append(P1_delta)
                    temp_P1_theta_6E_E3.append(P1_theta)
                    temp_P1_alpha_6E_E3.append(P1_alpha)
                    temp_P1_beta_6E_E3.append(P1_beta)
                    temp_P1_gamma_6E_E3.append(P1_gamma)

                    # Stack values
                    P1_patientID_6E_E3=np.stack(temp_P1_patientID_6E_E3)
                    P1_delta_6E_E3=np.stack(temp_P1_delta_6E_E3)
                    P1_theta_6E_E3=np.stack(temp_P1_theta_6E_E3)
                    P1_alpha_6E_E3=np.stack(temp_P1_alpha_6E_E3)
                    P1_beta_6E_E3=np.stack(temp_P1_beta_6E_E3)
                    P1_gamma_6E_E3=np.stack(temp_P1_gamma_6E_E3)

                    # Packing data in dictonary 
                    Bandpower_values_6E_E3 = {
                        'PatientID': P1_patientID_6E_E3,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_6E_E3.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_6E_E3.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_6E_E3.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_6E_E3.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_6E_E3.tolist(),

                        }
                    
                    print('Bandpower 6E was packed - E3')
                    print(Bandpower_values_6E_E3)

                elif Electrodes[3]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_6E_E4.append(P1_patientID)
                    temp_P1_delta_6E_E4.append(P1_delta)
                    temp_P1_theta_6E_E4.append(P1_theta)
                    temp_P1_alpha_6E_E4.append(P1_alpha)
                    temp_P1_beta_6E_E4.append(P1_beta)
                    temp_P1_gamma_6E_E4.append(P1_gamma)

                    # Stack values
                    P1_patientID_6E_E4=np.stack(temp_P1_patientID_6E_E4)
                    P1_delta_6E_E4=np.stack(temp_P1_delta_6E_E4)
                    P1_theta_6E_E4=np.stack(temp_P1_theta_6E_E4)
                    P1_alpha_6E_E4=np.stack(temp_P1_alpha_6E_E4)
                    P1_beta_6E_E4=np.stack(temp_P1_beta_6E_E4)
                    P1_gamma_6E_E4=np.stack(temp_P1_gamma_6E_E4)

                    # Packing data in dictonary 
                    Bandpower_values_6E_E4 = {
                        'PatientID': P1_patientID_6E_E4,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_6E_E4.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_6E_E4.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_6E_E4.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_6E_E4.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_6E_E4.tolist(),

                        }
                    
                    print('Bandpower 6E was packed - E4')
                    print(Bandpower_values_6E_E4)


                elif Electrodes[4]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_6E_E5.append(P1_patientID)
                    temp_P1_delta_6E_E5.append(P1_delta)
                    temp_P1_theta_6E_E5.append(P1_theta)
                    temp_P1_alpha_6E_E5.append(P1_alpha)
                    temp_P1_beta_6E_E5.append(P1_beta)
                    temp_P1_gamma_6E_E5.append(P1_gamma)

                    # Stack values
                    P1_patientID_6E_E5=np.stack(temp_P1_patientID_6E_E5)
                    P1_delta_6E_E5=np.stack(temp_P1_delta_6E_E5)
                    P1_theta_6E_E5=np.stack(temp_P1_theta_6E_E5)
                    P1_alpha_6E_E5=np.stack(temp_P1_alpha_6E_E5)
                    P1_beta_6E_E5=np.stack(temp_P1_beta_6E_E5)
                    P1_gamma_6E_E5=np.stack(temp_P1_gamma_6E_E5)

                    # Packing data in dictonary 
                    Bandpower_values_6E_E5 = {
                        'PatientID': P1_patientID_6E_E5,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_6E_E5.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_6E_E5.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_6E_E5.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_6E_E5.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_6E_E5.tolist(),

                        }
                    
                    print('Bandpower 6E was packed - E5')
                    print(Bandpower_values_6E_E5)


                elif Electrodes[5]==Electrodes[loop_factor]:
            
                    temp_P1_patientID_6E_E6.append(P1_patientID)
                    temp_P1_delta_6E_E6.append(P1_delta)
                    temp_P1_theta_6E_E6.append(P1_theta)
                    temp_P1_alpha_6E_E6.append(P1_alpha)
                    temp_P1_beta_6E_E6.append(P1_beta)
                    temp_P1_gamma_6E_E6.append(P1_gamma)

                    # Stack values
                    P1_patientID_6E_E6=np.stack(temp_P1_patientID_6E_E6)
                    P1_delta_6E_E6=np.stack(temp_P1_delta_6E_E6)
                    P1_theta_6E_E6=np.stack(temp_P1_theta_6E_E6)
                    P1_alpha_6E_E6=np.stack(temp_P1_alpha_6E_E6)
                    P1_beta_6E_E6=np.stack(temp_P1_beta_6E_E6)
                    P1_gamma_6E_E6=np.stack(temp_P1_gamma_6E_E6)

                    # Packing data in dictonary 
                    Bandpower_values_6E_E6 = {
                        'PatientID': P1_patientID_6E_E6,
                        'P1_delta_'+str(Electrode_for_naming): P1_delta_6E_E6.tolist(),
                        'P1_theta_'+str(Electrode_for_naming): P1_theta_6E_E6.tolist(),
                        'P1_alpha_'+str(Electrode_for_naming): P1_alpha_6E_E6.tolist(),
                        'P1_beta_'+str(Electrode_for_naming): P1_beta_6E_E6.tolist(),
                        'P1_gamma_'+str(Electrode_for_naming): P1_gamma_6E_E6.tolist(),

                        }
                    
                    print('Bandpower 6E was packed - E6')
                    print(Bandpower_values_6E_E6)

        # Deleting variables to release memory
        del Signals



        # Making array to loop over for all electrode combinations 
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
            
        
            # Checking the electrodes loaded 
            Electrode_name1 = E_combinations[d][0] #"C3M2"  # Replace with the consistent part of the first array's name
            Electrode_name2 = E_combinations[d][1] #"O1M2"  # Replace with the consistent part of the second array's name

            # Defining electrode combination for naming the CSV files in the end 
            Electrode_combination_naming = [Electrode_name1, Electrode_name2]
            Electrode_combination_naming = extract_letters_and_numbers(Electrode_combination_naming)
            print('Electrode combination')
            print(Electrode_combination_naming)

            ############## Defining electrode1 and electrode2 ##################
            # Up to 15 combinatiosn are possible depending on the amount of electrodes available
            # 6 electrodes --> 15 combinations 
            # 5 electrodes --> 10 combinations 
            # 4 electrodes --> 6 combinations 
            # 3 electrodes --> 3 combinations 
            # 2 electrodes --> 1 combination 

            # Combination 1
            if Electrode_name1 =='F3M2' and Electrode_name2=='F4M1':
                Electrode1=F3M2_signal
                Signalheader1=F3M2_signalheader
                print('F3M2 is electrode 1')
                #print(Electrode1)
                #print(F3M2_signal)

                Electrode2=F4M1_signal
                Signalheader2=F4M1_signalheader
                print('F4M1 is electrode 2')
                #print(Electrode2)
                #print(F4M1_signal)
                del Electrode_name1, Electrode_name2

            # Combination 2
            elif Electrode_name1 =='F3M2' and Electrode_name2=='C3M2':
                Electrode1=F3M2_signal
                Signalheader1=F3M2_signalheader
                print('F3M2 is electrode 1')
                #print(Electrode1)
                #print(F3M2_signal)

                Electrode2=C3M2_signal
                Signalheader2=C3M2_signalheader
                print('C3M2 is electrode 2')
                #print(Electrode2)
                #print(C3M2_signal)
                del Electrode_name1, Electrode_name2


            # Combination 3 
            elif Electrode_name1 =='F3M2' and Electrode_name2=='C4M1':
                Electrode1=F3M2_signal
                Signalheader1=F3M2_signalheader
                print('F3M2 is electrode 1')
                #print(Electrode1)
                #print(F3M2_signal)

                Electrode2=C4M1_signal
                Signalheader2=C4M1_signalheader
                print('C4M1 is electrode 2')
                #print(Electrode2)
                #print(C4M1_signal)
                del Electrode_name1, Electrode_name2

            # Combination 4
            elif Electrode_name1 =='F3M2' and Electrode_name2=='O1M2':
                Electrode1=F3M2_signal
                Signalheader1=F3M2_signalheader
                print('F3M2 is electrode 1')
                #print(Electrode1)
                #print(F3M2_signal)

                Electrode2=O1M2_signal
                Signalheader2=O1M2_signalheader
                print('O1M2 is electrode 2')
                #print(Electrode2)
                #print(O1M2_signal)
                del Electrode_name1, Electrode_name2

            # Combination 5 
            elif Electrode_name1 =='F3M2' and Electrode_name2=='O2M1':
                Electrode1=F3M2_signal
                Signalheader1=F3M2_signalheader
                print('F3M2 is electrode 1')
                #print(Electrode1)
                #print(F3M2_signal)

                Electrode2=O2M1_signal
                Signalheader2=O2M1_signalheader
                print('O2M1 is electrode 2')
                #print(Electrode2)
                #print(O2M1_signal)
                del Electrode_name1, Electrode_name2


            # Combination 6 
            elif Electrode_name1 =='F4M1' and Electrode_name2=='C3M2':
                Electrode1=F4M1_signal
                Signalheader1=F4M1_signalheader
                print('F4M1 is electrode 1')
                #print(Electrode1)
                #print(F4M1_signal)

                Electrode2=C3M2_signal
                Signalheader2=C3M2_signalheader
                print('C3M2 is electrode 2')
                #print(Electrode2)
                #print(C3M2_signal)
                del Electrode_name1, Electrode_name2

            # Combination 7
            elif Electrode_name1 =='F4M1' and Electrode_name2=='C4M1':
                Electrode1=F4M1_signal
                Signalheader1=F4M1_signalheader
                print('F4M1 is electrode 1')
                #print(Electrode1)
                #print(F4M1_signal)

                Electrode2=C4M1_signal
                Signalheader2=C4M1_signalheader
                print('C4M1 is electrode 2')
                #print(Electrode2)
                #print(C4M1_signal)
                del Electrode_name1, Electrode_name2
            
            # Combination 8 
            elif Electrode_name1 =='F4M1' and Electrode_name2=='O1M2':
                Electrode1=F4M1_signal
                Signalheader1=F4M1_signalheader
                print('F4M1 is electrode 1')
                #print(Electrode1)
                #print(F4M1_signal)

                Electrode2=O1M2_signal
                Signalheader2=O1M2_signalheader
                print('O1M2 is electrode 2')
                #print(Electrode2)
                #print(O1M2_signal)
                del Electrode_name1, Electrode_name2

            # Combination 9 
            elif Electrode_name1 =='F4M1' and Electrode_name2=='O2M1':
                Electrode1=F4M1_signal
                Signalheader1=F4M1_signalheader
                print('F4M1 is electrode 1')
                #print(Electrode1)
                #print(F4M1_signal)

                Electrode2=O2M1_signal
                Signalheader2=O2M1_signalheader
                print('O2M1 is electrode 2')
                #print(Electrode2)
                #print(O2M1_signal)
                del Electrode_name1, Electrode_name2
        
            # Combination 10 
            elif Electrode_name1 =='C3M2' and Electrode_name2=='C4M1':
                Electrode1=C3M2_signal
                Signalheader1=C3M2_signalheader
                print('C3M2 is electrode 1')
                #print(Electrode1)
                #print(C3M2_signal)

                Electrode2=C4M1_signal
                Signalheader2=C4M1_signalheader
                print('C4M1 is electrode 2')
                #print(Electrode2)
                #print(C4M1_signal)
                del Electrode_name1, Electrode_name2
            

            # Combination 11
            elif Electrode_name1 =='C3M2' and Electrode_name2=='O1M2':
                Electrode1=C3M2_signal
                Signalheader1=C3M2_signalheader
                print('C3M2 is electrode 1')
                #print(Electrode1)
                #print(C3M2_signal)

                Electrode2=O1M2_signal
                Signalheader2=O1M2_signalheader
                print('O1M2 is electrode 2')
                #print(Electrode2)
                #print(O1M2_signal)
                del Electrode_name1, Electrode_name2

            # Combination 12 
            elif Electrode_name1 =='C3M2' and Electrode_name2=='O2M1':
                Electrode1=C3M2_signal
                Signalheader1=C3M2_signalheader
                print('C3M2 is electrode 1')
                #print(Electrode1)
                #print(C3M2_signal)

                Electrode2=O2M1_signal
                Signalheader2=O2M1_signalheader
                print('O2M1 is electrode 2')
                #print(Electrode2)
                #print(O2M1_signal)
                del Electrode_name1, Electrode_name2

            # Combination 13 
            elif Electrode_name1 =='C4M1' and Electrode_name2=='O1M2':
                Electrode1=C4M1_signal
                Signalheader1=C4M1_signalheader
                print('C4M1 is electrode 1')
                #print(Electrode1)
                #print(C4M1_signal)

                Electrode2=O1M2_signal
                Signalheader2=O1M2_signalheader
                print('O1M2 is electrode 2')
                #print(Electrode2)
                #print(O1M2_signal)
                del Electrode_name1, Electrode_name2
            
            # Combination 14 
            elif Electrode_name1 =='C4M1' and Electrode_name2=='O2M1':
                Electrode1=C4M1_signal
                Signalheader1=C4M1_signalheader
                print('C4M1 is electrode 1')
                #print(Electrode1)
                #print(C4M1_signal)

                Electrode2=O2M1_signal
                Signalheader2=O2M1_signalheader
                print('O2M1 is electrode 2')
                #print(Electrode2)
                #print(O2M1_signal)
                del Electrode_name1, Electrode_name2
            
            # Combination 15 
            elif Electrode_name1 =='O1M2' and Electrode_name2=='O2M1':
                Electrode1=O1M2_signal
                Signalheader1=O1M2_signalheader
                print('O1M2 is electrode 1')
                #print(Electrode1)
                #print(O1M2_signal)

                Electrode2=O2M1_signal
                Signalheader2=O2M1_signalheader
                print('O2M1 is electrode 2')
                #print(Electrode2)
                #print(O2M1_signal)

                del Electrode_name1, Electrode_name2
            

            ##### Pre-processing the two electrodes #####################

            # Calling pre-processing function for channel 1 
            signal_new_chan1, fs_new_chan1, time_filtered_HP_chan1 = preprocessing(Electrode1)#, Signalheader1)

            # Calling pre-processing function for channel 2
            signal_new_chan2, fs_new_chan2, time_filtered_HP_chan2 = preprocessing(Electrode2)#,Signalheader2)

            # Deleting to release memory
            del Electrode1, Electrode2


            ####### Normal coherence analysis #########

            # Comparing the power spectral densities between two channels 
            # Coherence shows the correlation between two signals as a function of frequency
        
            # Using coherence function in python 
            #Coherence=(abs(Pxy_den**2))/(Pxx_den*Pyy_den)
            f, Cxy = scipy.signal.coherence(time_filtered_HP_chan1, time_filtered_HP_chan2, fs_new_chan1, nperseg=1024)
            '''
            matplotlib,pyplot.semilogy(f, Cxy)
            matplotlib,pyplot.title('Coherence of time signal channel 1 and 2')
            matplotlib,pyplot.xlabel('frequency [Hz]')
            matplotlib,pyplot.ylabel('Coherence')
            matplotlib,pyplot.show()
            '''

            ##### Extracting coherence features ######
            # Average coherence value for each EEG frequency band 
            # Calling coherence extracting feature function 
            deltaband_coh, thetaband_coh, alphaband_coh,betaband_coh,gammaband_coh = coherence_features(Cxy,f)

            # Deleting to release memory
            del Cxy, f

            ##### Coherence for 30 second intervals and average coherence for each frequency band ####

            # Overview of algorithm in steps:

            # 1) divide both time_filtered signal 1 and 2 into 30 seonc intervals - in a loop 
            # 2) load the signals into the scipy.signal.coherence function to calculate coherence for the 30 seconds 
            # 3) Use the 'coherence_features' function to extract features for 30 second intervals
            # 4) pack the features in a matrix (theta, delta, alpha, beta, gamma) columns x rows = number of 30 sec intervals 
            # 5) Calculate average over each column (average of each frequency band) - should end up giving 5 values 

        
            # Step 1) 

            # calculate the amount of samples that would be 30 seconds 
            fs=128 #Hz
            time_30s_calculate=fs*30 # fs is the sampling frequency (number of samples per second) and we would like the amount of samples for 30 seconds
            idx_30=time_30s_calculate
            
            
            # Indexing for 30 seconds intervals using the amount of samples for 30 seconds 'idx_30'
            for i in range(0,len(time_filtered_HP_chan1),time_30s_calculate):
                
                # indexing in the two signals for 30 second intervals 
                intervals_30_signal1=time_filtered_HP_chan1[i:i+idx_30] # channel 1
                intervals_30_signal2=time_filtered_HP_chan2[i:i+idx_30] # channel 2 
                
                # Step 2) 
                # Calculating coherence for 30 second intervals 
                f_30sec, Cxy_30sec = scipy.signal.coherence(intervals_30_signal1, intervals_30_signal2, fs, nperseg=1024)

                # Step 3) 
                # Extracting coherence featues for 30 second intervals 
                deltaband_coh_30sec, thetaband_coh_30sec, alphaband_coh_30sec,betaband_coh_30sec,gammaband_coh_30sec = coherence_features(Cxy_30sec,f_30sec)

                # Deleting to clear memory 
                del f_30sec, Cxy_30sec

                # Step 4) 
                # Saving the values in a temporary variable
                temp_delta30.append(deltaband_coh_30sec)
                temp_theta30.append(thetaband_coh_30sec)
                temp_alpha30.append(alphaband_coh_30sec)
                temp_beta30.append(betaband_coh_30sec)
                temp_gamma30.append(gammaband_coh_30sec)

                # Stacking the temporary variables 
                delta_coh30=np.stack(temp_delta30)
                theta_coh30=np.stack(temp_theta30)
                alpha_coh30=np.stack(temp_alpha30)
                beta_coh30=np.stack(temp_beta30)
                gamma_coh30=np.stack(temp_gamma30)


            # Step 5) 
            # Calculating average over one columns 
            delta_coh_av_30=np.average(delta_coh30)
            #print(delta_coh_av_30)

            # Calculating average over one column
            theta_coh_av_30=np.average(theta_coh30)
            #print(theta_coh_av_30)

            # Calculating average over one column
            alpha_coh_av_30=np.average(alpha_coh30)
            #print(alpha_coh_av_30)

            # Calculating average over one column 
            beta_coh_av_30=np.average(beta_coh30)
            #print(beta_coh_av_30)

            # Calculating average over one column
            gamma_coh_av_30=np.average(gamma_coh30)
            #print(gamma_coh_av_30)
                
            
            # Only two electrodes with one combination of electrodes 
            if len(E_combinations) ==1:
                    
                print('Temporary E-combinations length 1')

                # saving the values from the function 
                temp_patientID_2E_E1E2.append(patientID)
                temp_deltaband_coh_2E_E1E2.append(deltaband_coh)
                temp_thetaband_coh_2E_E1E2.append(thetaband_coh)
                temp_alphaband_coh_2E_E1E2.append(alphaband_coh)
                temp_betaband_coh_2E_E1E2.append(betaband_coh)
                temp_gammaband_coh_2E_E1E2.append(gammaband_coh)
                temp_deltacoh_av30_2E_E1E2.append(delta_coh_av_30)
                temp_thetacoh_av30_2E_E1E2.append(theta_coh_av_30)
                temp_alphacoh_av30_2E_E1E2.append(alpha_coh_av_30)
                temp_betacoh_av30_2E_E1E2.append(beta_coh_av_30)
                temp_gammacoh_av30_2E_E1E2.append(gamma_coh_av_30)

                # stacking variables 
                patientID_stacked_2E_E1E2=np.stack(temp_patientID_2E_E1E2)
                delta_coh_2E_E1E2=np.stack(temp_deltaband_coh_2E_E1E2)
                theta_coh_2E_E1E2=np.stack(temp_thetaband_coh_2E_E1E2)
                alpha_coh_2E_E1E2=np.stack(temp_alphaband_coh_2E_E1E2)
                beta_coh_2E_E1E2=np.stack(temp_betaband_coh_2E_E1E2)
                gamma_coh_2E_E1E2=np.stack(temp_gammaband_coh_2E_E1E2)
                deltacoh_av30_2E_E1E2=np.stack(temp_deltacoh_av30_2E_E1E2)
                thetacoh_av30_2E_E1E2=np.stack(temp_thetacoh_av30_2E_E1E2)
                alphacoh_av30_2E_E1E2=np.stack(temp_alphacoh_av30_2E_E1E2)
                betacoh_av30_2E_E1E2=np.stack(temp_betacoh_av30_2E_E1E2)
                gammacoh_av30_2E_E1E2=np.stack(temp_gammacoh_av30_2E_E1E2)

                # Packing data in dictonary 
                Coherence_values_2E_E1E2 = {
                    'PatientID_2E': patientID_stacked_2E_E1E2,
                    'Delta_coh_'+str(Electrode_combination_naming): delta_coh_2E_E1E2.tolist(),
                    'Theta_coh_'+str(Electrode_combination_naming): theta_coh_2E_E1E2.tolist(),
                    'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_2E_E1E2.tolist(),
                    'Beta_coh_'+str(Electrode_combination_naming): beta_coh_2E_E1E2.tolist(),
                    'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_2E_E1E2.tolist(),
                    'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_2E_E1E2.tolist(),
                    'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_2E_E1E2.tolist(),
                    'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_2E_E1E2.tolist(),
                    'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_2E_E1E2.tolist(),
                    'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_2E_E1E2.tolist(),
                    }
                
                print('Coherence 2E values dictonary')
                #print(Coherence_values_2E_E1E2)

            # Three electrodes are present giving three combinations 
            elif len(E_combinations) ==3:
                    
                    
                print('Temporary E-combinations length 3')

                # Storing the first combination E1E2_3
                if E_combinations[0]==E_combinations[d]:

                    print('E_combinations[0] chosen for three electrodes first combination')
                    print(E_combinations[0])
                        
                    # Filling out temporary variables: 
                    temp_patientID_3E_E1E2.append(patientID)
                    temp_deltaband_coh_3E_E1E2.append(deltaband_coh)
                    temp_thetaband_coh_3E_E1E2.append(thetaband_coh)
                    temp_alphaband_coh_3E_E1E2.append(alphaband_coh)
                    temp_betaband_coh_3E_E1E2.append(betaband_coh)
                    temp_gammaband_coh_3E_E1E2.append(gammaband_coh)    
                    temp_deltacoh_av30_3E_E1E2.append(delta_coh_av_30)
                    temp_thetacoh_av30_3E_E1E2.append(theta_coh_av_30)
                    temp_alphacoh_av30_3E_E1E2.append(alpha_coh_av_30)
                    temp_betacoh_av30_3E_E1E2.append(beta_coh_av_30)
                    temp_gammacoh_av30_3E_E1E2.append(gamma_coh_av_30)

                    # stacking variables 
                    patientID_stacked_3E_E1E2=np.stack(temp_patientID_3E_E1E2)
                    delta_coh_3E_E1E2=np.stack(temp_deltaband_coh_3E_E1E2)
                    theta_coh_3E_E1E2=np.stack(temp_thetaband_coh_3E_E1E2)
                    alpha_coh_3E_E1E2=np.stack(temp_alphaband_coh_3E_E1E2)
                    beta_coh_3E_E1E2=np.stack(temp_betaband_coh_3E_E1E2)
                    gamma_coh_3E_E1E2=np.stack(temp_gammaband_coh_3E_E1E2)
                    deltacoh_av30_3E_E1E2=np.stack(temp_deltacoh_av30_3E_E1E2)
                    thetacoh_av30_3E_E1E2=np.stack(temp_thetacoh_av30_3E_E1E2)
                    alphacoh_av30_3E_E1E2=np.stack(temp_alphacoh_av30_3E_E1E2)
                    betacoh_av30_3E_E1E2=np.stack(temp_betacoh_av30_3E_E1E2)
                    gammacoh_av30_3E_E1E2=np.stack(temp_gammacoh_av30_3E_E1E2)

                    # Packing data in dictonary 
                    Coherence_values_3E_E1E2 = {
                        'PatientID': patientID_stacked_3E_E1E2,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_3E_E1E2.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_3E_E1E2.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_3E_E1E2.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_3E_E1E2.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_3E_E1E2.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_3E_E1E2.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_3E_E1E2.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_3E_E1E2.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_3E_E1E2.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_3E_E1E2.tolist(),
                        }
                    
                    print('Coherence 3E values dictonary - first combination')
                    #print(Electrode_combination_naming)
                    #print(Coherence_values_3E_E1E2)

                    
                    # Storing the second combination E1E3_3
                elif E_combinations[1]==E_combinations[d]:
                    
                    # Filling out temporary variables: 
                    temp_patientID_3E_E1E3.append(patientID)
                    temp_deltaband_coh_3E_E1E3.append(deltaband_coh)
                    temp_thetaband_coh_3E_E1E3.append(thetaband_coh)
                    temp_alphaband_coh_3E_E1E3.append(alphaband_coh)
                    temp_betaband_coh_3E_E1E3.append(betaband_coh)
                    temp_gammaband_coh_3E_E1E3.append(gammaband_coh)
                    temp_deltacoh_av30_3E_E1E3.append(delta_coh_av_30)
                    temp_thetacoh_av30_3E_E1E3.append(theta_coh_av_30)
                    temp_alphacoh_av30_3E_E1E3.append(alpha_coh_av_30)
                    temp_betacoh_av30_3E_E1E3.append(beta_coh_av_30)
                    temp_gammacoh_av30_3E_E1E3.append(gamma_coh_av_30)


                        
                    # stacking variables 
                    patientID_stacked_3E_E1E3=np.stack(temp_patientID_3E_E1E3)
                    delta_coh_3E_E1E3=np.stack(temp_deltaband_coh_3E_E1E3)
                    theta_coh_3E_E1E3=np.stack(temp_thetaband_coh_3E_E1E3)
                    alpha_coh_3E_E1E3=np.stack(temp_alphaband_coh_3E_E1E3)
                    beta_coh_3E_E1E3=np.stack(temp_betaband_coh_3E_E1E3)
                    gamma_coh_3E_E1E3=np.stack(temp_gammaband_coh_3E_E1E3)
                    deltacoh_av30_3E_E1E3=np.stack(temp_deltacoh_av30_3E_E1E3)
                    thetacoh_av30_3E_E1E3=np.stack(temp_thetacoh_av30_3E_E1E3)
                    alphacoh_av30_3E_E1E3=np.stack(temp_alphacoh_av30_3E_E1E3)
                    betacoh_av30_3E_E1E3=np.stack(temp_betacoh_av30_3E_E1E3)
                    gammacoh_av30_3E_E1E3=np.stack(temp_gammacoh_av30_3E_E1E3)

                    # Packing data in dictonary 
                    Coherence_values_3E_E1E3 = {
                        'PatientID': patientID_stacked_3E_E1E3,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_3E_E1E3.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_3E_E1E3.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_3E_E1E3.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_3E_E1E3.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_3E_E1E3.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_3E_E1E3.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_3E_E1E3.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_3E_E1E3.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_3E_E1E3.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_3E_E1E3.tolist(),
                        }
                    
                    print('Coherence 3E values dictonary - second combination')
                    #print(Electrode_combination_naming)
                    #print(Coherence_values_3E_E1E3)

                    # Storing the third combination E2E3_3
                elif E_combinations[2]==E_combinations[d]:

                    # Filling out temporary variables: 
                    temp_patientID_3E_E2E3.append(patientID)
                    temp_deltaband_coh_3E_E2E3.append(deltaband_coh)
                    temp_thetaband_coh_3E_E2E3.append(thetaband_coh)
                    temp_alphaband_coh_3E_E2E3.append(alphaband_coh)
                    temp_betaband_coh_3E_E2E3.append(betaband_coh)
                    temp_gammaband_coh_3E_E2E3.append(gammaband_coh)
                    temp_deltacoh_av30_3E_E2E3.append(delta_coh_av_30)
                    temp_thetacoh_av30_3E_E2E3.append(theta_coh_av_30)
                    temp_alphacoh_av30_3E_E2E3.append(alpha_coh_av_30)
                    temp_betacoh_av30_3E_E2E3.append(beta_coh_av_30)
                    temp_gammacoh_av30_3E_E2E3.append(gamma_coh_av_30)
                        
                    # stacking variables 
                    patientID_stacked_3E_E2E3=np.stack(temp_patientID_3E_E2E3)
                    delta_coh_3E_E2E3=np.stack(temp_deltaband_coh_3E_E2E3)
                    theta_coh_3E_E2E3=np.stack(temp_thetaband_coh_3E_E2E3)
                    alpha_coh_3E_E2E3=np.stack(temp_alphaband_coh_3E_E2E3)
                    beta_coh_3E_E2E3=np.stack(temp_betaband_coh_3E_E2E3)
                    gamma_coh_3E_E2E3=np.stack(temp_gammaband_coh_3E_E2E3)
                    deltacoh_av30_3E_E2E3=np.stack(temp_deltacoh_av30_3E_E2E3)
                    thetacoh_av30_3E_E2E3=np.stack(temp_thetacoh_av30_3E_E2E3)
                    alphacoh_av30_3E_E2E3=np.stack(temp_alphacoh_av30_3E_E2E3)
                    betacoh_av30_3E_E2E3=np.stack(temp_betacoh_av30_3E_E2E3)
                    gammacoh_av30_3E_E2E3=np.stack(temp_gammacoh_av30_3E_E2E3)

                    # Packing data in dictonary 
                    Coherence_values_3E_E2E3 = {
                        'PatientID': patientID_stacked_3E_E2E3,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_3E_E2E3.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_3E_E2E3.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_3E_E2E3.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_3E_E2E3.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_3E_E2E3.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_3E_E2E3.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_3E_E2E3.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_3E_E2E3.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_3E_E2E3.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_3E_E2E3.tolist(),
                        }
                    
                    print('Coherence 3E values dictonary - third combination')
                    #print(Electrode_combination_naming)
                    #print(Coherence_values_3E_E3E2)
                
            
            # 4 electrodes will give 6 combinations 

            elif len(E_combinations) ==6:
                
                    
                print('Temporary E-combinations length 6')

                # Storing the first combination 4E_E1E2
                if E_combinations[0]==E_combinations[d]:

                    print('E_combinations[0] chosen for four electrodes first combination')
                    print(E_combinations[0])
                        
                    # Filling out temporary variables: 
                    temp_patientID_4E_E1E2.append(patientID)
                    temp_deltaband_coh_4E_E1E2.append(deltaband_coh)
                    temp_thetaband_coh_4E_E1E2.append(thetaband_coh)
                    temp_alphaband_coh_4E_E1E2.append(alphaband_coh)
                    temp_betaband_coh_4E_E1E2.append(betaband_coh)
                    temp_gammaband_coh_4E_E1E2.append(gammaband_coh)
                    temp_deltacoh_av30_4E_E1E2.append(delta_coh_av_30)
                    temp_thetacoh_av30_4E_E1E2.append(theta_coh_av_30)
                    temp_alphacoh_av30_4E_E1E2.append(alpha_coh_av_30)
                    temp_betacoh_av30_4E_E1E2.append(beta_coh_av_30)
                    temp_gammacoh_av30_4E_E1E2.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_4E_E1E2=np.stack(temp_patientID_4E_E1E2)
                    delta_coh_4E_E1E2=np.stack(temp_deltaband_coh_4E_E1E2)
                    theta_coh_4E_E1E2=np.stack(temp_thetaband_coh_4E_E1E2)
                    alpha_coh_4E_E1E2=np.stack(temp_alphaband_coh_4E_E1E2)
                    beta_coh_4E_E1E2=np.stack(temp_betaband_coh_4E_E1E2)
                    gamma_coh_4E_E1E2=np.stack(temp_gammaband_coh_4E_E1E2)
                    deltacoh_av30_4E_E1E2=np.stack(temp_deltacoh_av30_4E_E1E2)
                    thetacoh_av30_4E_E1E2=np.stack(temp_thetacoh_av30_4E_E1E2)
                    alphacoh_av30_4E_E1E2=np.stack(temp_alphacoh_av30_4E_E1E2)
                    betacoh_av30_4E_E1E2=np.stack(temp_betacoh_av30_4E_E1E2)
                    gammacoh_av30_4E_E1E2=np.stack(temp_gammacoh_av30_4E_E1E2)

                    # Packing data in dictonary 
                    Coherence_values_4E_E1E2 = {
                        'PatientID': patientID_stacked_4E_E1E2,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_4E_E1E2.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_4E_E1E2.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_4E_E1E2.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_4E_E1E2.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_4E_E1E2.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_4E_E1E2.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_4E_E1E2.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_4E_E1E2.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_4E_E1E2.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_4E_E1E2.tolist(),
                        }
                        
                    
                    print('Patient dictionary 4E_E1E2 - three electrodes, first combination')
                    #print(Coherence_values_4E_E1E2)
                    
                # Storing the second combination 4E_E1E3
                elif E_combinations[1]==E_combinations[d]:

                    print('E_combinations[1] chosen for four electrodes second combination')
                    print(E_combinations[1])

                    # Filling out temporary variables: 
                    temp_patientID_4E_E1E3.append(patientID)
                    temp_deltaband_coh_4E_E1E3.append(deltaband_coh)
                    temp_thetaband_coh_4E_E1E3.append(thetaband_coh)
                    temp_alphaband_coh_4E_E1E3.append(alphaband_coh)
                    temp_betaband_coh_4E_E1E3.append(betaband_coh)
                    temp_gammaband_coh_4E_E1E3.append(gammaband_coh)
                    temp_deltacoh_av30_4E_E1E3.append(delta_coh_av_30)
                    temp_thetacoh_av30_4E_E1E3.append(theta_coh_av_30)
                    temp_alphacoh_av30_4E_E1E3.append(alpha_coh_av_30)
                    temp_betacoh_av30_4E_E1E3.append(beta_coh_av_30)
                    temp_gammacoh_av30_4E_E1E3.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_4E_E1E3=np.stack(temp_patientID_4E_E1E3)
                    delta_coh_4E_E1E3=np.stack(temp_deltaband_coh_4E_E1E3)
                    theta_coh_4E_E1E3=np.stack(temp_thetaband_coh_4E_E1E3)
                    alpha_coh_4E_E1E3=np.stack(temp_alphaband_coh_4E_E1E3)
                    beta_coh_4E_E1E3=np.stack(temp_betaband_coh_4E_E1E3)
                    gamma_coh_4E_E1E3=np.stack(temp_gammaband_coh_4E_E1E3)
                    deltacoh_av30_4E_E1E3=np.stack(temp_deltacoh_av30_4E_E1E3)
                    thetacoh_av30_4E_E1E3=np.stack(temp_thetacoh_av30_4E_E1E3)
                    alphacoh_av30_4E_E1E3=np.stack(temp_alphacoh_av30_4E_E1E3)
                    betacoh_av30_4E_E1E3=np.stack(temp_betacoh_av30_4E_E1E3)
                    gammacoh_av30_4E_E1E3=np.stack(temp_gammacoh_av30_4E_E1E3)

                    # Packing data in dictonary 
                    Coherence_values_4E_E1E3 = {
                        'PatientID': patientID_stacked_4E_E1E3,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_4E_E1E3.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_4E_E1E3.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_4E_E1E3.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_4E_E1E3.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_4E_E1E3.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_4E_E1E3.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_4E_E1E3.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_4E_E1E3.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_4E_E1E3.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_4E_E1E3.tolist(),
                        }
                        
                    print('Patient dictionary 4E_E1E3 - four electrodes, second combination')
                    #print(Coherence_values_4E_E1E3)

                # Storing the third combination 4E_E2E3
                elif E_combinations[2]==E_combinations[d]:

                    print('E_combinations[2] chosen for four electrodes third combination')
                    print(E_combinations[2])

                    # Filling out temporary variables: 
                        
                    temp_patientID_4E_E2E3.append(patientID)
                    temp_deltaband_coh_4E_E2E3.append(deltaband_coh)
                    temp_thetaband_coh_4E_E2E3.append(thetaband_coh)
                    temp_alphaband_coh_4E_E2E3.append(alphaband_coh)
                    temp_betaband_coh_4E_E2E3.append(betaband_coh)
                    temp_gammaband_coh_4E_E2E3.append(gammaband_coh)
                    temp_deltacoh_av30_4E_E2E3.append(delta_coh_av_30)
                    temp_thetacoh_av30_4E_E2E3.append(theta_coh_av_30)
                    temp_alphacoh_av30_4E_E2E3.append(alpha_coh_av_30)
                    temp_betacoh_av30_4E_E2E3.append(beta_coh_av_30)
                    temp_gammacoh_av30_4E_E2E3.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_4E_E2E3=np.stack(temp_patientID_4E_E2E3)
                    delta_coh_4E_E2E3=np.stack(temp_deltaband_coh_4E_E2E3)
                    theta_coh_4E_E2E3=np.stack(temp_thetaband_coh_4E_E2E3)
                    alpha_coh_4E_E2E3=np.stack(temp_alphaband_coh_4E_E2E3)
                    beta_coh_4E_E2E3=np.stack(temp_betaband_coh_4E_E2E3)
                    gamma_coh_4E_E2E3=np.stack(temp_gammaband_coh_4E_E2E3)
                    deltacoh_av30_4E_E2E3=np.stack(temp_deltacoh_av30_4E_E2E3)
                    thetacoh_av30_4E_E2E3=np.stack(temp_thetacoh_av30_4E_E2E3)
                    alphacoh_av30_4E_E2E3=np.stack(temp_alphacoh_av30_4E_E2E3)
                    betacoh_av30_4E_E2E3=np.stack(temp_betacoh_av30_4E_E2E3)
                    gammacoh_av30_4E_E2E3=np.stack(temp_gammacoh_av30_4E_E2E3)

                    # Packing data in dictonary 
                    Coherence_values_4E_E2E3 = {
                        'PatientID': patientID_stacked_4E_E2E3,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_4E_E2E3.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_4E_E2E3.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_4E_E2E3.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_4E_E2E3.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_4E_E2E3.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_4E_E2E3.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_4E_E2E3.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_4E_E2E3.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_4E_E2E3.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_4E_E2E3.tolist(),
                        }
                        
                    print('Patient dictionary 4E_E2E3 - four electrodes, third combination')
                    #print(Coherence_values_4E_E2E3)

                        
                # Storing the fourth combination 4E_E1E4
                elif E_combinations[3]==E_combinations[d]:

                    print('E_combinations[3] chosen for four electrodes fourth combination')
                    print(E_combinations[3])

                    # Filling out temporary variables: 
                    temp_patientID_4E_E1E4.append(patientID)
                    temp_deltaband_coh_4E_E1E4.append(deltaband_coh)
                    temp_thetaband_coh_4E_E1E4.append(thetaband_coh)
                    temp_alphaband_coh_4E_E1E4.append(alphaband_coh)
                    temp_betaband_coh_4E_E1E4.append(betaband_coh)
                    temp_gammaband_coh_4E_E1E4.append(gammaband_coh)
                    temp_deltacoh_av30_4E_E1E4.append(delta_coh_av_30)
                    temp_thetacoh_av30_4E_E1E4.append(theta_coh_av_30)
                    temp_alphacoh_av30_4E_E1E4.append(alpha_coh_av_30)
                    temp_betacoh_av30_4E_E1E4.append(beta_coh_av_30)
                    temp_gammacoh_av30_4E_E1E4.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_4E_E1E4=np.stack(temp_patientID_4E_E1E4)
                    delta_coh_4E_E1E4=np.stack(temp_deltaband_coh_4E_E1E4)
                    theta_coh_4E_E1E4=np.stack(temp_thetaband_coh_4E_E1E4)
                    alpha_coh_4E_E1E4=np.stack(temp_alphaband_coh_4E_E1E4)
                    beta_coh_4E_E1E4=np.stack(temp_betaband_coh_4E_E1E4)
                    gamma_coh_4E_E1E4=np.stack(temp_gammaband_coh_4E_E1E4)
                    deltacoh_av30_4E_E1E4=np.stack(temp_deltacoh_av30_4E_E1E4)
                    thetacoh_av30_4E_E1E4=np.stack(temp_thetacoh_av30_4E_E1E4)
                    alphacoh_av30_4E_E1E4=np.stack(temp_alphacoh_av30_4E_E1E4)
                    betacoh_av30_4E_E1E4=np.stack(temp_betacoh_av30_4E_E1E4)
                    gammacoh_av30_4E_E1E4=np.stack(temp_gammacoh_av30_4E_E1E4)

                    # Packing data in dictonary 
                    Coherence_values_4E_E1E4 = {
                        'PatientID': patientID_stacked_4E_E1E4,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_4E_E1E4.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_4E_E1E4.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_4E_E1E4.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_4E_E1E4.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_4E_E1E4.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_4E_E1E4.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_4E_E1E4.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_4E_E1E4.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_4E_E1E4.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_4E_E1E4.tolist(),
                        }
                        
                    print('Patient dictionary 4E_E1E4 - four electrodes, fourth combination')
                    #print(Coherence_values_4E_E1E4)
                    
                # Storing the fourth combination 4E_E2E4
                elif E_combinations[4]==E_combinations[d]:

                    print('E_combinations[4] chosen for four electrodes fifth combination')
                    print(E_combinations[4])

                    # Filling out temporary variables: 
                    temp_patientID_4E_E2E4.append(patientID)
                    temp_deltaband_coh_4E_E2E4.append(deltaband_coh)
                    temp_thetaband_coh_4E_E2E4.append(thetaband_coh)
                    temp_alphaband_coh_4E_E2E4.append(alphaband_coh)
                    temp_betaband_coh_4E_E2E4.append(betaband_coh)
                    temp_gammaband_coh_4E_E2E4.append(gammaband_coh)
                    temp_deltacoh_av30_4E_E2E4.append(delta_coh_av_30)
                    temp_thetacoh_av30_4E_E2E4.append(theta_coh_av_30)
                    temp_alphacoh_av30_4E_E2E4.append(alpha_coh_av_30)
                    temp_betacoh_av30_4E_E2E4.append(beta_coh_av_30)
                    temp_gammacoh_av30_4E_E2E4.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_4E_E2E4=np.stack(temp_patientID_4E_E2E4)
                    delta_coh_4E_E2E4=np.stack(temp_deltaband_coh_4E_E2E4)
                    theta_coh_4E_E2E4=np.stack(temp_thetaband_coh_4E_E2E4)
                    alpha_coh_4E_E2E4=np.stack(temp_alphaband_coh_4E_E2E4)
                    beta_coh_4E_E2E4=np.stack(temp_betaband_coh_4E_E2E4)
                    gamma_coh_4E_E2E4=np.stack(temp_gammaband_coh_4E_E2E4)
                    deltacoh_av30_4E_E2E4=np.stack(temp_deltacoh_av30_4E_E2E4)
                    thetacoh_av30_4E_E2E4=np.stack(temp_thetacoh_av30_4E_E2E4)
                    alphacoh_av30_4E_E2E4=np.stack(temp_alphacoh_av30_4E_E2E4)
                    betacoh_av30_4E_E2E4=np.stack(temp_betacoh_av30_4E_E2E4)
                    gammacoh_av30_4E_E2E4=np.stack(temp_gammacoh_av30_4E_E2E4)

                    # Packing data in dictonary 
                    Coherence_values_4E_E2E4 = {
                        'PatientID': patientID_stacked_4E_E2E4,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_4E_E2E4.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_4E_E2E4.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_4E_E2E4.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_4E_E2E4.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_4E_E2E4.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_4E_E2E4.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_4E_E2E4.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_4E_E2E4.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_4E_E2E4.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_4E_E2E4.tolist(),
                        }
                        
                    print('Patient dictionary 4E_E2E4 - four electrodes, fifth combination')
                    #print(Coherence_values_4E_E2E4)

                    
                # Storing the fourth combination 4E_E3E4
                elif E_combinations[5]==E_combinations[d]:

                    print('E_combinations[5] chosen for four electrodes sixth combination')
                    print(E_combinations[5])

                    # Filling out temporary variables: 
                    temp_patientID_4E_E3E4.append(patientID)
                    temp_deltaband_coh_4E_E3E4.append(deltaband_coh)
                    temp_thetaband_coh_4E_E3E4.append(thetaband_coh)
                    temp_alphaband_coh_4E_E3E4.append(alphaband_coh)
                    temp_betaband_coh_4E_E3E4.append(betaband_coh)
                    temp_gammaband_coh_4E_E3E4.append(gammaband_coh)
                    temp_deltacoh_av30_4E_E3E4.append(delta_coh_av_30)
                    temp_thetacoh_av30_4E_E3E4.append(theta_coh_av_30)
                    temp_alphacoh_av30_4E_E3E4.append(alpha_coh_av_30)
                    temp_betacoh_av30_4E_E3E4.append(beta_coh_av_30)
                    temp_gammacoh_av30_4E_E3E4.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_4E_E3E4=np.stack(temp_patientID_4E_E3E4)
                    delta_coh_4E_E3E4=np.stack(temp_deltaband_coh_4E_E3E4)
                    theta_coh_4E_E3E4=np.stack(temp_thetaband_coh_4E_E3E4)
                    alpha_coh_4E_E3E4=np.stack(temp_alphaband_coh_4E_E3E4)
                    beta_coh_4E_E3E4=np.stack(temp_betaband_coh_4E_E3E4)
                    gamma_coh_4E_E3E4=np.stack(temp_gammaband_coh_4E_E3E4)
                    deltacoh_av30_4E_E3E4=np.stack(temp_deltacoh_av30_4E_E3E4)
                    thetacoh_av30_4E_E3E4=np.stack(temp_thetacoh_av30_4E_E3E4)
                    alphacoh_av30_4E_E3E4=np.stack(temp_alphacoh_av30_4E_E3E4)
                    betacoh_av30_4E_E3E4=np.stack(temp_betacoh_av30_4E_E3E4)
                    gammacoh_av30_4E_E3E4=np.stack(temp_gammacoh_av30_4E_E3E4)

                    # Packing data in dictonary 
                    Coherence_values_4E_E3E4 = {
                        'PatientID': patientID_stacked_4E_E3E4,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_4E_E3E4.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_4E_E3E4.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_4E_E3E4.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_4E_E3E4.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_4E_E3E4.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_4E_E3E4.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_4E_E3E4.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_4E_E3E4.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_4E_E3E4.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_4E_E3E4.tolist(),
                        }
                        
                    print('Patient dictionary 4E_E3E4 - four electrodes, sixth combination')
                    #print(Coherence_values_4E_E3E4)


            # 5 electrodes will give 10 combinations 

            elif len(E_combinations) ==10:
                
                    
                print('Temporary E-combinations length 10')

                # Storing the first combination E1E2_5
                if E_combinations[0]==E_combinations[d]:

                    print('E_combinations[0] chosen for five electrodes first combination')
                    print(E_combinations[0])
                        
                        
                    # Filling out temporary variables: 
                    temp_patientID_5E_E1E2.append(patientID)
                    temp_deltaband_coh_5E_E1E2.append(deltaband_coh)
                    temp_thetaband_coh_5E_E1E2.append(thetaband_coh)
                    temp_alphaband_coh_5E_E1E2.append(alphaband_coh)
                    temp_betaband_coh_5E_E1E2.append(betaband_coh)
                    temp_gammaband_coh_5E_E1E2.append(gammaband_coh)
                    temp_deltacoh_av30_5E_E1E2.append(delta_coh_av_30)
                    temp_thetacoh_av30_5E_E1E2.append(theta_coh_av_30)
                    temp_alphacoh_av30_5E_E1E2.append(alpha_coh_av_30)
                    temp_betacoh_av30_5E_E1E2.append(beta_coh_av_30)
                    temp_gammacoh_av30_5E_E1E2.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_5E_E1E2=np.stack(temp_patientID_5E_E1E2)
                    delta_coh_5E_E1E2=np.stack(temp_deltaband_coh_5E_E1E2)
                    theta_coh_5E_E1E2=np.stack(temp_thetaband_coh_5E_E1E2)
                    alpha_coh_5E_E1E2=np.stack(temp_alphaband_coh_5E_E1E2)
                    beta_coh_5E_E1E2=np.stack(temp_betaband_coh_5E_E1E2)
                    gamma_coh_5E_E1E2=np.stack(temp_gammaband_coh_5E_E1E2)
                    deltacoh_av30_5E_E1E2=np.stack(temp_deltacoh_av30_5E_E1E2)
                    thetacoh_av30_5E_E1E2=np.stack(temp_thetacoh_av30_5E_E1E2)
                    alphacoh_av30_5E_E1E2=np.stack(temp_alphacoh_av30_5E_E1E2)
                    betacoh_av30_5E_E1E2=np.stack(temp_betacoh_av30_5E_E1E2)
                    gammacoh_av30_5E_E1E2=np.stack(temp_gammacoh_av30_5E_E1E2)

                    # Packing data in dictonary 
                    Coherence_values_5E_E1E2 = {
                        'PatientID': patientID_stacked_5E_E1E2,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_5E_E1E2.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_5E_E1E2.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_5E_E1E2.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_5E_E1E2.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_5E_E1E2.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_5E_E1E2.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_5E_E1E2.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_5E_E1E2.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_5E_E1E2.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_5E_E1E2.tolist(),
                        }
                    
                    print('Patient dictionary E1E2_5 - five electrodes, first combination')
                    #print(Coherence_values_5E_E1E2)
                    
                # Storing the second combination E1E3_5
                elif E_combinations[1]==E_combinations[d]:

                    print('E_combinations[1] chosen for five electrodes second combination')
                    print(E_combinations[1])

                        
                    # Filling out temporary variables: 
                    temp_patientID_5E_E1E3.append(patientID)
                    temp_deltaband_coh_5E_E1E3.append(deltaband_coh)
                    temp_thetaband_coh_5E_E1E3.append(thetaband_coh)
                    temp_alphaband_coh_5E_E1E3.append(alphaband_coh)
                    temp_betaband_coh_5E_E1E3.append(betaband_coh)
                    temp_gammaband_coh_5E_E1E3.append(gammaband_coh)
                    temp_deltacoh_av30_5E_E1E3.append(delta_coh_av_30)
                    temp_thetacoh_av30_5E_E1E3.append(theta_coh_av_30)
                    temp_alphacoh_av30_5E_E1E3.append(alpha_coh_av_30)
                    temp_betacoh_av30_5E_E1E3.append(beta_coh_av_30)
                    temp_gammacoh_av30_5E_E1E3.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_5E_E1E3=np.stack(temp_patientID_5E_E1E3)
                    delta_coh_5E_E1E3=np.stack(temp_deltaband_coh_5E_E1E3)
                    theta_coh_5E_E1E3=np.stack(temp_thetaband_coh_5E_E1E3)
                    alpha_coh_5E_E1E3=np.stack(temp_alphaband_coh_5E_E1E3)
                    beta_coh_5E_E1E3=np.stack(temp_betaband_coh_5E_E1E3)
                    gamma_coh_5E_E1E3=np.stack(temp_gammaband_coh_5E_E1E3)
                    deltacoh_av30_5E_E1E3=np.stack(temp_deltacoh_av30_5E_E1E3)
                    thetacoh_av30_5E_E1E3=np.stack(temp_thetacoh_av30_5E_E1E3)
                    alphacoh_av30_5E_E1E3=np.stack(temp_alphacoh_av30_5E_E1E3)
                    betacoh_av30_5E_E1E3=np.stack(temp_betacoh_av30_5E_E1E3)
                    gammacoh_av30_5E_E1E3=np.stack(temp_gammacoh_av30_5E_E1E3)

                    # Packing data in dictonary 
                    Coherence_values_5E_E1E3 = {
                        'PatientID': patientID_stacked_5E_E1E3,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_5E_E1E3.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_5E_E1E3.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_5E_E1E3.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_5E_E1E3.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_5E_E1E3.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_5E_E1E3.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_5E_E1E3.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_5E_E1E3.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_5E_E1E3.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_5E_E1E3.tolist(),
                        }
                        
                    print('Patient dictionary E1E3_5 - five electrodes, second combination')
                    #print(Coherence_values_5E_E1E3)

                # Storing the third combination E2E3_5
                elif E_combinations[2]==E_combinations[d]:

                    print('E_combinations[2] chosen for five electrodes third combination')
                    print(E_combinations[2])

                        
                    # Filling out temporary variables: 
                    temp_patientID_5E_E2E3.append(patientID)
                    temp_deltaband_coh_5E_E2E3.append(deltaband_coh)
                    temp_thetaband_coh_5E_E2E3.append(thetaband_coh)
                    temp_alphaband_coh_5E_E2E3.append(alphaband_coh)
                    temp_betaband_coh_5E_E2E3.append(betaband_coh)
                    temp_gammaband_coh_5E_E2E3.append(gammaband_coh)
                    temp_deltacoh_av30_5E_E2E3.append(delta_coh_av_30)
                    temp_thetacoh_av30_5E_E2E3.append(theta_coh_av_30)
                    temp_alphacoh_av30_5E_E2E3.append(alpha_coh_av_30)
                    temp_betacoh_av30_5E_E2E3.append(beta_coh_av_30)
                    temp_gammacoh_av30_5E_E2E3.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_5E_E2E3=np.stack(temp_patientID_5E_E2E3)
                    delta_coh_5E_E2E3=np.stack(temp_deltaband_coh_5E_E2E3)
                    theta_coh_5E_E2E3=np.stack(temp_thetaband_coh_5E_E2E3)
                    alpha_coh_5E_E2E3=np.stack(temp_alphaband_coh_5E_E2E3)
                    beta_coh_5E_E2E3=np.stack(temp_betaband_coh_5E_E2E3)
                    gamma_coh_5E_E2E3=np.stack(temp_gammaband_coh_5E_E2E3)
                    deltacoh_av30_5E_E2E3=np.stack(temp_deltacoh_av30_5E_E2E3)
                    thetacoh_av30_5E_E2E3=np.stack(temp_thetacoh_av30_5E_E2E3)
                    alphacoh_av30_5E_E2E3=np.stack(temp_alphacoh_av30_5E_E2E3)
                    betacoh_av30_5E_E2E3=np.stack(temp_betacoh_av30_5E_E2E3)
                    gammacoh_av30_5E_E2E3=np.stack(temp_gammacoh_av30_5E_E2E3)

                    # Packing data in dictonary 
                    Coherence_values_5E_E2E3 = {
                        'PatientID': patientID_stacked_5E_E2E3,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_5E_E2E3.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_5E_E2E3.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_5E_E2E3.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_5E_E2E3.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_5E_E2E3.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_5E_E2E3.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_5E_E2E3.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_5E_E2E3.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_5E_E2E3.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_5E_E2E3.tolist(),
                        }
                    
                    print('Patient dictionary E2E3_5 - five electrodes, third combination')
                    #print(Coherence_values_5E_E2E3)

                        
                # Storing the fourth combination E1E4_5
                elif E_combinations[3]==E_combinations[d]:

                    print('E_combinations[3] chosen for five electrodes fourth combination')
                    print(E_combinations[3])

                        
                    # Filling out temporary variables: 
                    temp_patientID_5E_E1E4.append(patientID)
                    temp_deltaband_coh_5E_E1E4.append(deltaband_coh)
                    temp_thetaband_coh_5E_E1E4.append(thetaband_coh)
                    temp_alphaband_coh_5E_E1E4.append(alphaband_coh)
                    temp_betaband_coh_5E_E1E4.append(betaband_coh)
                    temp_gammaband_coh_5E_E1E4.append(gammaband_coh)
                    temp_deltacoh_av30_5E_E1E4.append(delta_coh_av_30)
                    temp_thetacoh_av30_5E_E1E4.append(theta_coh_av_30)
                    temp_alphacoh_av30_5E_E1E4.append(alpha_coh_av_30)
                    temp_betacoh_av30_5E_E1E4.append(beta_coh_av_30)
                    temp_gammacoh_av30_5E_E1E4.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_5E_E1E4=np.stack(temp_patientID_5E_E1E4)
                    delta_coh_5E_E1E4=np.stack(temp_deltaband_coh_5E_E1E4)
                    theta_coh_5E_E1E4=np.stack(temp_thetaband_coh_5E_E1E4)
                    alpha_coh_5E_E1E4=np.stack(temp_alphaband_coh_5E_E1E4)
                    beta_coh_5E_E1E4=np.stack(temp_betaband_coh_5E_E1E4)
                    gamma_coh_5E_E1E4=np.stack(temp_gammaband_coh_5E_E1E4)
                    deltacoh_av30_5E_E1E4=np.stack(temp_deltacoh_av30_5E_E1E4)
                    thetacoh_av30_5E_E1E4=np.stack(temp_thetacoh_av30_5E_E1E4)
                    alphacoh_av30_5E_E1E4=np.stack(temp_alphacoh_av30_5E_E1E4)
                    betacoh_av30_5E_E1E4=np.stack(temp_betacoh_av30_5E_E1E4)
                    gammacoh_av30_5E_E1E4=np.stack(temp_gammacoh_av30_5E_E1E4)

                    # Packing data in dictonary 
                    Coherence_values_5E_E1E4 = {
                        'PatientID': patientID_stacked_5E_E1E4,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_5E_E1E4.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_5E_E1E4.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_5E_E1E4.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_5E_E1E4.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_5E_E1E4.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_5E_E1E4.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_5E_E1E4.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_5E_E1E4.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_5E_E1E4.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_5E_E1E4.tolist(),
                        }
                    
                    print('Patient dictionary E1E4_5 - five electrodes, fourth combination')
                    #print(Coherence_values_5E_E1E4)
                    
                # Storing the fourth combination E2E4_5
                elif E_combinations[4]==E_combinations[d]:

                    print('E_combinations[4] chosen for five electrodes fifth combination')
                    print(E_combinations[4])

                        
                    # Filling out temporary variables: 
                    temp_patientID_5E_E2E4.append(patientID)
                    temp_deltaband_coh_5E_E2E4.append(deltaband_coh)
                    temp_thetaband_coh_5E_E2E4.append(thetaband_coh)
                    temp_alphaband_coh_5E_E2E4.append(alphaband_coh)
                    temp_betaband_coh_5E_E2E4.append(betaband_coh)
                    temp_gammaband_coh_5E_E2E4.append(gammaband_coh)
                    temp_deltacoh_av30_5E_E2E4.append(delta_coh_av_30)
                    temp_thetacoh_av30_5E_E2E4.append(theta_coh_av_30)
                    temp_alphacoh_av30_5E_E2E4.append(alpha_coh_av_30)
                    temp_betacoh_av30_5E_E2E4.append(beta_coh_av_30)
                    temp_gammacoh_av30_5E_E2E4.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_5E_E2E4=np.stack(temp_patientID_5E_E2E4)
                    delta_coh_5E_E2E4=np.stack(temp_deltaband_coh_5E_E2E4)
                    theta_coh_5E_E2E4=np.stack(temp_thetaband_coh_5E_E2E4)
                    alpha_coh_5E_E2E4=np.stack(temp_alphaband_coh_5E_E2E4)
                    beta_coh_5E_E2E4=np.stack(temp_betaband_coh_5E_E2E4)
                    gamma_coh_5E_E2E4=np.stack(temp_gammaband_coh_5E_E2E4)
                    deltacoh_av30_5E_E2E4=np.stack(temp_deltacoh_av30_5E_E2E4)
                    thetacoh_av30_5E_E2E4=np.stack(temp_thetacoh_av30_5E_E2E4)
                    alphacoh_av30_5E_E2E4=np.stack(temp_alphacoh_av30_5E_E2E4)
                    betacoh_av30_5E_E2E4=np.stack(temp_betacoh_av30_5E_E2E4)
                    gammacoh_av30_5E_E2E4=np.stack(temp_gammacoh_av30_5E_E2E4)

                    # Packing data in dictonary 
                    Coherence_values_5E_E2E4 = {
                        'PatientID': patientID_stacked_5E_E2E4,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_5E_E2E4.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_5E_E2E4.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_5E_E2E4.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_5E_E2E4.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_5E_E2E4.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_5E_E2E4.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_5E_E2E4.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_5E_E2E4.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_5E_E2E4.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_5E_E2E4.tolist(),
                        }
                    
                    print('Patient dictionary E2E4_5 - five electrodes, fifth combination')
                    #print(Coherence_values_5E_E2E4)

                    
                # Storing the sixth combination E3E4_5
                elif E_combinations[5]==E_combinations[d]:

                    print('E_combinations[5] chosen for five electrodes sixth combination')
                    print(E_combinations[5])

                        
                    # Filling out temporary variables: 
                    temp_patientID_5E_E3E4.append(patientID)
                    temp_deltaband_coh_5E_E3E4.append(deltaband_coh)
                    temp_thetaband_coh_5E_E3E4.append(thetaband_coh)
                    temp_alphaband_coh_5E_E3E4.append(alphaband_coh)
                    temp_betaband_coh_5E_E3E4.append(betaband_coh)
                    temp_gammaband_coh_5E_E3E4.append(gammaband_coh)
                    temp_deltacoh_av30_5E_E3E4.append(delta_coh_av_30)
                    temp_thetacoh_av30_5E_E3E4.append(theta_coh_av_30)
                    temp_alphacoh_av30_5E_E3E4.append(alpha_coh_av_30)
                    temp_betacoh_av30_5E_E3E4.append(beta_coh_av_30)
                    temp_gammacoh_av30_5E_E3E4.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_5E_E3E4=np.stack(temp_patientID_5E_E3E4)
                    delta_coh_5E_E3E4=np.stack(temp_deltaband_coh_5E_E3E4)
                    theta_coh_5E_E3E4=np.stack(temp_thetaband_coh_5E_E3E4)
                    alpha_coh_5E_E3E4=np.stack(temp_alphaband_coh_5E_E3E4)
                    beta_coh_5E_E3E4=np.stack(temp_betaband_coh_5E_E3E4)
                    gamma_coh_5E_E3E4=np.stack(temp_gammaband_coh_5E_E3E4)
                    deltacoh_av30_5E_E3E4=np.stack(temp_deltacoh_av30_5E_E3E4)
                    thetacoh_av30_5E_E3E4=np.stack(temp_thetacoh_av30_5E_E3E4)
                    alphacoh_av30_5E_E3E4=np.stack(temp_alphacoh_av30_5E_E3E4)
                    betacoh_av30_5E_E3E4=np.stack(temp_betacoh_av30_5E_E3E4)
                    gammacoh_av30_5E_E3E4=np.stack(temp_gammacoh_av30_5E_E3E4)

                    # Packing data in dictonary 
                    Coherence_values_5E_E3E4 = {
                        'PatientID': patientID_stacked_5E_E3E4,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_5E_E3E4.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_5E_E3E4.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_5E_E3E4.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_5E_E3E4.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_5E_E3E4.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_5E_E3E4.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_5E_E3E4.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_5E_E3E4.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_5E_E3E4.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_5E_E3E4.tolist(),
                        }
                    
                    print('Patient dictionary E3E4_5 - five electrodes, sixth combination')
                    #print(Coherence_values_5E_E3E4)
                    
                # Storing the seventh combination E1E5_5
                elif E_combinations[6]==E_combinations[d]:

                    print('E_combinations[6] chosen for five electrodes seventh combination')
                    print(E_combinations[6])

                        
                    # Filling out temporary variables: 
                    temp_patientID_5E_E1E5.append(patientID)
                    temp_deltaband_coh_5E_E1E5.append(deltaband_coh)
                    temp_thetaband_coh_5E_E1E5.append(thetaband_coh)
                    temp_alphaband_coh_5E_E1E5.append(alphaband_coh)
                    temp_betaband_coh_5E_E1E5.append(betaband_coh)
                    temp_gammaband_coh_5E_E1E5.append(gammaband_coh)
                    temp_deltacoh_av30_5E_E1E5.append(delta_coh_av_30)
                    temp_thetacoh_av30_5E_E1E5.append(theta_coh_av_30)
                    temp_alphacoh_av30_5E_E1E5.append(alpha_coh_av_30)
                    temp_betacoh_av30_5E_E1E5.append(beta_coh_av_30)
                    temp_gammacoh_av30_5E_E1E5.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_5E_E1E5=np.stack(temp_patientID_5E_E1E5)
                    delta_coh_5E_E1E5=np.stack(temp_deltaband_coh_5E_E1E5)
                    theta_coh_5E_E1E5=np.stack(temp_thetaband_coh_5E_E1E5)
                    alpha_coh_5E_E1E5=np.stack(temp_alphaband_coh_5E_E1E5)
                    beta_coh_5E_E1E5=np.stack(temp_betaband_coh_5E_E1E5)
                    gamma_coh_5E_E1E5=np.stack(temp_gammaband_coh_5E_E1E5)
                    deltacoh_av30_5E_E1E5=np.stack(temp_deltacoh_av30_5E_E1E5)
                    thetacoh_av30_5E_E1E5=np.stack(temp_thetacoh_av30_5E_E1E5)
                    alphacoh_av30_5E_E1E5=np.stack(temp_alphacoh_av30_5E_E1E5)
                    betacoh_av30_5E_E1E5=np.stack(temp_betacoh_av30_5E_E1E5)
                    gammacoh_av30_5E_E1E5=np.stack(temp_gammacoh_av30_5E_E1E5)

                    # Packing data in dictonary 
                    Coherence_values_5E_E1E5 = {
                        'PatientID': patientID_stacked_5E_E1E5,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_5E_E1E5.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_5E_E1E5.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_5E_E1E5.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_5E_E1E5.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_5E_E1E5.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_5E_E1E5.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_5E_E1E5.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_5E_E1E5.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_5E_E1E5.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_5E_E1E5.tolist(),
                        }
                    
                    print('Patient dictionary E1E5_5 - five electrodes, seventh combination')
                    #print(Coherence_values_5E_E1E5)


                # Storing the combination E2E5_5
                elif E_combinations[7]==E_combinations[d]:

                    print('E_combinations[7] chosen for five electrodes eight combination')
                    print(E_combinations[7])

                        
                    # Filling out temporary variables: 
                    temp_patientID_5E_E2E5.append(patientID)
                    temp_deltaband_coh_5E_E2E5.append(deltaband_coh)
                    temp_thetaband_coh_5E_E2E5.append(thetaband_coh)
                    temp_alphaband_coh_5E_E2E5.append(alphaband_coh)
                    temp_betaband_coh_5E_E2E5.append(betaband_coh)
                    temp_gammaband_coh_5E_E2E5.append(gammaband_coh)
                    temp_deltacoh_av30_5E_E2E5.append(delta_coh_av_30)
                    temp_thetacoh_av30_5E_E2E5.append(theta_coh_av_30)
                    temp_alphacoh_av30_5E_E2E5.append(alpha_coh_av_30)
                    temp_betacoh_av30_5E_E2E5.append(beta_coh_av_30)
                    temp_gammacoh_av30_5E_E2E5.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_5E_E2E5=np.stack(temp_patientID_5E_E2E5)
                    delta_coh_5E_E2E5=np.stack(temp_deltaband_coh_5E_E2E5)
                    theta_coh_5E_E2E5=np.stack(temp_thetaband_coh_5E_E2E5)
                    alpha_coh_5E_E2E5=np.stack(temp_alphaband_coh_5E_E2E5)
                    beta_coh_5E_E2E5=np.stack(temp_betaband_coh_5E_E2E5)
                    gamma_coh_5E_E2E5=np.stack(temp_gammaband_coh_5E_E2E5)
                    deltacoh_av30_5E_E2E5=np.stack(temp_deltacoh_av30_5E_E2E5)
                    thetacoh_av30_5E_E2E5=np.stack(temp_thetacoh_av30_5E_E2E5)
                    alphacoh_av30_5E_E2E5=np.stack(temp_alphacoh_av30_5E_E2E5)
                    betacoh_av30_5E_E2E5=np.stack(temp_betacoh_av30_5E_E2E5)
                    gammacoh_av30_5E_E2E5=np.stack(temp_gammacoh_av30_5E_E2E5)

                    # Packing data in dictonary 
                    Coherence_values_5E_E2E5 = {
                        'PatientID': patientID_stacked_5E_E2E5,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_5E_E2E5.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_5E_E2E5.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_5E_E2E5.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_5E_E2E5.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_5E_E2E5.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_5E_E2E5.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_5E_E2E5.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_5E_E2E5.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_5E_E2E5.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_5E_E2E5.tolist(),
                        }
                    
                    print('Patient dictionary E2E5_5 - five electrodes, 8th combination')
                    #print(Coherence_values_5E_E2E5)
                
                # Storing the fourth combination E3E5_5
                elif E_combinations[8]==E_combinations[d]:

                    print('E_combinations[8] chosen for five electrodes 9th combination')
                    print(E_combinations[8])

                        
                    # Filling out temporary variables: 
                    temp_patientID_5E_E3E5.append(patientID)
                    temp_deltaband_coh_5E_E3E5.append(deltaband_coh)
                    temp_thetaband_coh_5E_E3E5.append(thetaband_coh)
                    temp_alphaband_coh_5E_E3E5.append(alphaband_coh)
                    temp_betaband_coh_5E_E3E5.append(betaband_coh)
                    temp_gammaband_coh_5E_E3E5.append(gammaband_coh)
                    temp_deltacoh_av30_5E_E3E5.append(delta_coh_av_30)
                    temp_thetacoh_av30_5E_E3E5.append(theta_coh_av_30)
                    temp_alphacoh_av30_5E_E3E5.append(alpha_coh_av_30)
                    temp_betacoh_av30_5E_E3E5.append(beta_coh_av_30)
                    temp_gammacoh_av30_5E_E3E5.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_5E_E3E5=np.stack(temp_patientID_5E_E3E5)
                    delta_coh_5E_E3E5=np.stack(temp_deltaband_coh_5E_E3E5)
                    theta_coh_5E_E3E5=np.stack(temp_thetaband_coh_5E_E3E5)
                    alpha_coh_5E_E3E5=np.stack(temp_alphaband_coh_5E_E3E5)
                    beta_coh_5E_E3E5=np.stack(temp_betaband_coh_5E_E3E5)
                    gamma_coh_5E_E3E5=np.stack(temp_gammaband_coh_5E_E3E5)
                    deltacoh_av30_5E_E3E5=np.stack(temp_deltacoh_av30_5E_E3E5)
                    thetacoh_av30_5E_E3E5=np.stack(temp_thetacoh_av30_5E_E3E5)
                    alphacoh_av30_5E_E3E5=np.stack(temp_alphacoh_av30_5E_E3E5)
                    betacoh_av30_5E_E3E5=np.stack(temp_betacoh_av30_5E_E3E5)
                    gammacoh_av30_5E_E3E5=np.stack(temp_gammacoh_av30_5E_E3E5)

                    # Packing data in dictonary 
                    Coherence_values_5E_E3E5 = {
                        'PatientID': patientID_stacked_5E_E3E5,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_5E_E3E5.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_5E_E3E5.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_5E_E3E5.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_5E_E3E5.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_5E_E3E5.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_5E_E3E5.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_5E_E3E5.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_5E_E3E5.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_5E_E3E5.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_5E_E3E5.tolist(),
                        }
                    
                    print('Patient dictionary E3E5_5 - five electrodes, 9th combination')
                    #print(Coherence_values_5E_E3E5)

                    
                # Storing the fourth combination E4E5_5
                elif E_combinations[9]==E_combinations[d]:

                    print('E_combinations[9] chosen for five electrodes 10th combination')
                    print(E_combinations[9])

                        
                    # Filling out temporary variables: 
                    temp_patientID_5E_E4E5.append(patientID)
                    temp_deltaband_coh_5E_E4E5.append(deltaband_coh)
                    temp_thetaband_coh_5E_E4E5.append(thetaband_coh)
                    temp_alphaband_coh_5E_E4E5.append(alphaband_coh)
                    temp_betaband_coh_5E_E4E5.append(betaband_coh)
                    temp_gammaband_coh_5E_E4E5.append(gammaband_coh)
                    temp_deltacoh_av30_5E_E4E5.append(delta_coh_av_30)
                    temp_thetacoh_av30_5E_E4E5.append(theta_coh_av_30)
                    temp_alphacoh_av30_5E_E4E5.append(alpha_coh_av_30)
                    temp_betacoh_av30_5E_E4E5.append(beta_coh_av_30)
                    temp_gammacoh_av30_5E_E4E5.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_5E_E4E5=np.stack(temp_patientID_5E_E4E5)
                    delta_coh_5E_E4E5=np.stack(temp_deltaband_coh_5E_E4E5)
                    theta_coh_5E_E4E5=np.stack(temp_thetaband_coh_5E_E4E5)
                    alpha_coh_5E_E4E5=np.stack(temp_alphaband_coh_5E_E4E5)
                    beta_coh_5E_E4E5=np.stack(temp_betaband_coh_5E_E4E5)
                    gamma_coh_5E_E4E5=np.stack(temp_gammaband_coh_5E_E4E5)
                    deltacoh_av30_5E_E4E5=np.stack(temp_deltacoh_av30_5E_E4E5)
                    thetacoh_av30_5E_E4E5=np.stack(temp_thetacoh_av30_5E_E4E5)
                    alphacoh_av30_5E_E4E5=np.stack(temp_alphacoh_av30_5E_E4E5)
                    betacoh_av30_5E_E4E5=np.stack(temp_betacoh_av30_5E_E4E5)
                    gammacoh_av30_5E_E4E5=np.stack(temp_gammacoh_av30_5E_E4E5)

                    # Packing data in dictonary 
                    Coherence_values_5E_E4E5 = {
                        'PatientID': patientID_stacked_5E_E4E5,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_5E_E4E5.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_5E_E4E5.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_5E_E4E5.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_5E_E4E5.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_5E_E4E5.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_5E_E4E5.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_5E_E4E5.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_5E_E4E5.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_5E_E4E5.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_5E_E4E5.tolist(),
                        }
                    
                    print('Patient dictionary E4E5_5 - five electrodes, 10th combination')
                    #print(Coherence_values_5E_E4E5)


            elif len(E_combinations) ==15:
                        
                            
                print('Temporary E-combinations length 15')

                # Storing the first combination E1E2_6
                if E_combinations[0]==E_combinations[d]:

                    print('E_combinations[0] chosen for six electrodes first combination')
                    print(E_combinations[0])
                            
                    # Filling out temporary variables: 
                    temp_patientID_6E_E1E2.append(patientID)
                    temp_deltaband_coh_6E_E1E2.append(deltaband_coh)
                    temp_thetaband_coh_6E_E1E2.append(thetaband_coh)
                    temp_alphaband_coh_6E_E1E2.append(alphaband_coh)
                    temp_betaband_coh_6E_E1E2.append(betaband_coh)
                    temp_gammaband_coh_6E_E1E2.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E1E2.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E1E2.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E1E2.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E1E2.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E1E2.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E1E2=np.stack(temp_patientID_6E_E1E2)
                    delta_coh_6E_E1E2=np.stack(temp_deltaband_coh_6E_E1E2)
                    theta_coh_6E_E1E2=np.stack(temp_thetaband_coh_6E_E1E2)
                    alpha_coh_6E_E1E2=np.stack(temp_alphaband_coh_6E_E1E2)
                    beta_coh_6E_E1E2=np.stack(temp_betaband_coh_6E_E1E2)
                    gamma_coh_6E_E1E2=np.stack(temp_gammaband_coh_6E_E1E2)
                    deltacoh_av30_6E_E1E2=np.stack(temp_deltacoh_av30_6E_E1E2)
                    thetacoh_av30_6E_E1E2=np.stack(temp_thetacoh_av30_6E_E1E2)
                    alphacoh_av30_6E_E1E2=np.stack(temp_alphacoh_av30_6E_E1E2)
                    betacoh_av30_6E_E1E2=np.stack(temp_betacoh_av30_6E_E1E2)
                    gammacoh_av30_6E_E1E2=np.stack(temp_gammacoh_av30_6E_E1E2)

                    # Packing data in dictonary 
                    Coherence_values_6E_E1E2 = {
                        'PatientID': patientID_stacked_6E_E1E2,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E1E2.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E1E2.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E1E2.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E1E2.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E1E2.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E1E2.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E1E2.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E1E2.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E1E2.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E1E2.tolist(),
                        }
                    
                    print('Patient dictionary E1E2_6 - six electrodes, first combination')
                    #print(Coherence_values_6E_E1E2)
                    
                # Storing the second combination E1E3_6
                elif E_combinations[1]==E_combinations[d]:

                    print('E_combinations[1] chosen for six electrodes second combination')
                    print(E_combinations[1])

                    # Filling out temporary variables: 
                    temp_patientID_6E_E1E3.append(patientID)
                    temp_deltaband_coh_6E_E1E3.append(deltaband_coh)
                    temp_thetaband_coh_6E_E1E3.append(thetaband_coh)
                    temp_alphaband_coh_6E_E1E3.append(alphaband_coh)
                    temp_betaband_coh_6E_E1E3.append(betaband_coh)
                    temp_gammaband_coh_6E_E1E3.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E1E3.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E1E3.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E1E3.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E1E3.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E1E3.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E1E3=np.stack(temp_patientID_6E_E1E3)
                    delta_coh_6E_E1E3=np.stack(temp_deltaband_coh_6E_E1E3)
                    theta_coh_6E_E1E3=np.stack(temp_thetaband_coh_6E_E1E3)
                    alpha_coh_6E_E1E3=np.stack(temp_alphaband_coh_6E_E1E3)
                    beta_coh_6E_E1E3=np.stack(temp_betaband_coh_6E_E1E3)
                    gamma_coh_6E_E1E3=np.stack(temp_gammaband_coh_6E_E1E3)
                    deltacoh_av30_6E_E1E3=np.stack(temp_deltacoh_av30_6E_E1E3)
                    thetacoh_av30_6E_E1E3=np.stack(temp_thetacoh_av30_6E_E1E3)
                    alphacoh_av30_6E_E1E3=np.stack(temp_alphacoh_av30_6E_E1E3)
                    betacoh_av30_6E_E1E3=np.stack(temp_betacoh_av30_6E_E1E3)
                    gammacoh_av30_6E_E1E3=np.stack(temp_gammacoh_av30_6E_E1E3)

                    # Packing data in dictonary 
                    Coherence_values_6E_E1E3 = {
                        'PatientID': patientID_stacked_6E_E1E3,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E1E3.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E1E3.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E1E3.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E1E3.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E1E3.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E1E3.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E1E3.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E1E3.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E1E3.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E1E3.tolist(),
                        }
                    
                    print('Patient dictionary E1E3_6 - six electrodes, second combination')
                    #print(Coherence_values_6E_E1E3)

                # Storing the third combination E2E3_6
                elif E_combinations[2]==E_combinations[d]:

                    print('E_combinations[2] chosen for six electrodes third combination')
                    print(E_combinations[2])

                    # Filling out temporary variables: 
                    temp_patientID_6E_E2E3.append(patientID)
                    temp_deltaband_coh_6E_E2E3.append(deltaband_coh)
                    temp_thetaband_coh_6E_E2E3.append(thetaband_coh)
                    temp_alphaband_coh_6E_E2E3.append(alphaband_coh)
                    temp_betaband_coh_6E_E2E3.append(betaband_coh)
                    temp_gammaband_coh_6E_E2E3.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E2E3.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E2E3.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E2E3.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E2E3.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E2E3.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E2E3=np.stack(temp_patientID_6E_E2E3)
                    delta_coh_6E_E2E3=np.stack(temp_deltaband_coh_6E_E2E3)
                    theta_coh_6E_E2E3=np.stack(temp_thetaband_coh_6E_E2E3)
                    alpha_coh_6E_E2E3=np.stack(temp_alphaband_coh_6E_E2E3)
                    beta_coh_6E_E2E3=np.stack(temp_betaband_coh_6E_E2E3)
                    gamma_coh_6E_E2E3=np.stack(temp_gammaband_coh_6E_E2E3)
                    deltacoh_av30_6E_E2E3=np.stack(temp_deltacoh_av30_6E_E2E3)
                    thetacoh_av30_6E_E2E3=np.stack(temp_thetacoh_av30_6E_E2E3)
                    alphacoh_av30_6E_E2E3=np.stack(temp_alphacoh_av30_6E_E2E3)
                    betacoh_av30_6E_E2E3=np.stack(temp_betacoh_av30_6E_E2E3)
                    gammacoh_av30_6E_E2E3=np.stack(temp_gammacoh_av30_6E_E2E3)

                    # Packing data in dictonary 
                    Coherence_values_6E_E2E3 = {
                        'PatientID': patientID_stacked_6E_E2E3,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E2E3.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E2E3.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E2E3.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E2E3.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E2E3.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E2E3.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E2E3.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E2E3.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E2E3.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E2E3.tolist(),
                        }
                    
                    print('Patient dictionary E2E3_6 - six electrodes, third combination')
                    #print(Coherence_values_6E_E2E3)

                        
                # Storing the fourth combination E1E4_6
                elif E_combinations[3]==E_combinations[d]:

                    print('E_combinations[3] chosen for six electrodes fourth combination')
                    print(E_combinations[3])

                    # Filling out temporary variables: 
                    temp_patientID_6E_E1E4.append(patientID)
                    temp_deltaband_coh_6E_E1E4.append(deltaband_coh)
                    temp_thetaband_coh_6E_E1E4.append(thetaband_coh)
                    temp_alphaband_coh_6E_E1E4.append(alphaband_coh)
                    temp_betaband_coh_6E_E1E4.append(betaband_coh)
                    temp_gammaband_coh_6E_E1E4.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E1E4.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E1E4.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E1E4.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E1E4.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E1E4.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E1E4=np.stack(temp_patientID_6E_E1E4)
                    delta_coh_6E_E1E4=np.stack(temp_deltaband_coh_6E_E1E4)
                    theta_coh_6E_E1E4=np.stack(temp_thetaband_coh_6E_E1E4)
                    alpha_coh_6E_E1E4=np.stack(temp_alphaband_coh_6E_E1E4)
                    beta_coh_6E_E1E4=np.stack(temp_betaband_coh_6E_E1E4)
                    gamma_coh_6E_E1E4=np.stack(temp_gammaband_coh_6E_E1E4)
                    deltacoh_av30_6E_E1E4=np.stack(temp_deltacoh_av30_6E_E1E4)
                    thetacoh_av30_6E_E1E4=np.stack(temp_thetacoh_av30_6E_E1E4)
                    alphacoh_av30_6E_E1E4=np.stack(temp_alphacoh_av30_6E_E1E4)
                    betacoh_av30_6E_E1E4=np.stack(temp_betacoh_av30_6E_E1E4)
                    gammacoh_av30_6E_E1E4=np.stack(temp_gammacoh_av30_6E_E1E4)

                    # Packing data in dictonary 
                    Coherence_values_6E_E1E4 = {
                        'PatientID': patientID_stacked_6E_E1E4,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E1E4.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E1E4.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E1E4.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E1E4.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E1E4.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E1E4.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E1E4.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E1E4.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E1E4.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E1E4.tolist(),
                        }
                    
                    print('Patient dictionary E1E4_6 - six electrodes, fourth combination')
                    #print(Coherence_values_6E_E1E4)
                    
                # Storing the fourth combination E2E4_6
                elif E_combinations[4]==E_combinations[d]:

                    print('E_combinations[4] chosen for six electrodes fifth combination')
                    print(E_combinations[4])

                        # Filling out temporary variables: 
                    temp_patientID_6E_E2E4.append(patientID)
                    temp_deltaband_coh_6E_E2E4.append(deltaband_coh)
                    temp_thetaband_coh_6E_E2E4.append(thetaband_coh)
                    temp_alphaband_coh_6E_E2E4.append(alphaband_coh)
                    temp_betaband_coh_6E_E2E4.append(betaband_coh)
                    temp_gammaband_coh_6E_E2E4.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E2E4.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E2E4.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E2E4.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E2E4.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E2E4.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E2E4=np.stack(temp_patientID_6E_E2E4)
                    delta_coh_6E_E2E4=np.stack(temp_deltaband_coh_6E_E2E4)
                    theta_coh_6E_E2E4=np.stack(temp_thetaband_coh_6E_E2E4)
                    alpha_coh_6E_E2E4=np.stack(temp_alphaband_coh_6E_E2E4)
                    beta_coh_6E_E2E4=np.stack(temp_betaband_coh_6E_E2E4)
                    gamma_coh_6E_E2E4=np.stack(temp_gammaband_coh_6E_E2E4)
                    deltacoh_av30_6E_E2E4=np.stack(temp_deltacoh_av30_6E_E2E4)
                    thetacoh_av30_6E_E2E4=np.stack(temp_thetacoh_av30_6E_E2E4)
                    alphacoh_av30_6E_E2E4=np.stack(temp_alphacoh_av30_6E_E2E4)
                    betacoh_av30_6E_E2E4=np.stack(temp_betacoh_av30_6E_E2E4)
                    gammacoh_av30_6E_E2E4=np.stack(temp_gammacoh_av30_6E_E2E4)

                    # Packing data in dictonary 
                    Coherence_values_6E_E2E4 = {
                        'PatientID': patientID_stacked_6E_E2E4,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E2E4.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E2E4.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E2E4.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E2E4.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E2E4.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E2E4.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E2E4.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E2E4.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E2E4.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E2E4.tolist(),
                        }
                    
                    print('Patient dictionary E2E4_6 - six electrodes, fifth combination')
                    #print(Coherence_values_6E_E2E4)

                    
                # Storing the sixth combination E3E4_6
                elif E_combinations[5]==E_combinations[d]:

                    print('E_combinations[5] chosen for six electrodes sixth combination')
                    print(E_combinations[5])

                        # Filling out temporary variables: 
                    temp_patientID_6E_E3E4.append(patientID)
                    temp_deltaband_coh_6E_E3E4.append(deltaband_coh)
                    temp_thetaband_coh_6E_E3E4.append(thetaband_coh)
                    temp_alphaband_coh_6E_E3E4.append(alphaband_coh)
                    temp_betaband_coh_6E_E3E4.append(betaband_coh)
                    temp_gammaband_coh_6E_E3E4.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E3E4.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E3E4.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E3E4.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E3E4.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E3E4.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E3E4=np.stack(temp_patientID_6E_E3E4)
                    delta_coh_6E_E3E4=np.stack(temp_deltaband_coh_6E_E3E4)
                    theta_coh_6E_E3E4=np.stack(temp_thetaband_coh_6E_E3E4)
                    alpha_coh_6E_E3E4=np.stack(temp_alphaband_coh_6E_E3E4)
                    beta_coh_6E_E3E4=np.stack(temp_betaband_coh_6E_E3E4)
                    gamma_coh_6E_E3E4=np.stack(temp_gammaband_coh_6E_E3E4)
                    deltacoh_av30_6E_E3E4=np.stack(temp_deltacoh_av30_6E_E3E4)
                    thetacoh_av30_6E_E3E4=np.stack(temp_thetacoh_av30_6E_E3E4)
                    alphacoh_av30_6E_E3E4=np.stack(temp_alphacoh_av30_6E_E3E4)
                    betacoh_av30_6E_E3E4=np.stack(temp_betacoh_av30_6E_E3E4)
                    gammacoh_av30_6E_E3E4=np.stack(temp_gammacoh_av30_6E_E3E4)

                    # Packing data in dictonary 
                    Coherence_values_6E_E3E4 = {
                        'PatientID': patientID_stacked_6E_E3E4,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E3E4.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E3E4.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E3E4.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E3E4.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E3E4.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E3E4.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E3E4.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E3E4.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E3E4.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E3E4.tolist(),
                        }
                    
                    print('Patient dictionary E3E4_6 - six electrodes, sixth combination')
                    #print(Coherence_values_6E_E3E4)
                    
                # Storing the seventh combination E1E5_6
                elif E_combinations[6]==E_combinations[d]:

                    print('E_combinations[6] chosen for six electrodes seventh combination')
                    print(E_combinations[6])

                        # Filling out temporary variables: 
                    temp_patientID_6E_E1E5.append(patientID)
                    temp_deltaband_coh_6E_E1E5.append(deltaband_coh)
                    temp_thetaband_coh_6E_E1E5.append(thetaband_coh)
                    temp_alphaband_coh_6E_E1E5.append(alphaband_coh)
                    temp_betaband_coh_6E_E1E5.append(betaband_coh)
                    temp_gammaband_coh_6E_E1E5.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E1E5.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E1E5.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E1E5.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E1E5.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E1E5.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E1E5=np.stack(temp_patientID_6E_E1E5)
                    delta_coh_6E_E1E5=np.stack(temp_deltaband_coh_6E_E1E5)
                    theta_coh_6E_E1E5=np.stack(temp_thetaband_coh_6E_E1E5)
                    alpha_coh_6E_E1E5=np.stack(temp_alphaband_coh_6E_E1E5)
                    beta_coh_6E_E1E5=np.stack(temp_betaband_coh_6E_E1E5)
                    gamma_coh_6E_E1E5=np.stack(temp_gammaband_coh_6E_E1E5)
                    deltacoh_av30_6E_E1E5=np.stack(temp_deltacoh_av30_6E_E1E5)
                    thetacoh_av30_6E_E1E5=np.stack(temp_thetacoh_av30_6E_E1E5)
                    alphacoh_av30_6E_E1E5=np.stack(temp_alphacoh_av30_6E_E1E5)
                    betacoh_av30_6E_E1E5=np.stack(temp_betacoh_av30_6E_E1E5)
                    gammacoh_av30_6E_E1E5=np.stack(temp_gammacoh_av30_6E_E1E5)

                    # Packing data in dictonary 
                    Coherence_values_6E_E1E5 = {
                        'PatientID': patientID_stacked_6E_E1E5,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E1E5.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E1E5.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E1E5.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E1E5.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E1E5.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E1E5.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E1E5.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E1E5.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E1E5.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E1E5.tolist(),
                        }
                    
                    print('Patient dictionary E1E5_6 - six electrodes, seventh combination')
                    #print(Coherence_values_6E_E1E5)


                # Storing the combination E2E5_6
                elif E_combinations[7]==E_combinations[d]:

                    print('E_combinations[7] chosen for six electrodes eight combination')
                    print(E_combinations[7])

                        # Filling out temporary variables: 
                    temp_patientID_6E_E2E5.append(patientID)
                    temp_deltaband_coh_6E_E2E5.append(deltaband_coh)
                    temp_thetaband_coh_6E_E2E5.append(thetaband_coh)
                    temp_alphaband_coh_6E_E2E5.append(alphaband_coh)
                    temp_betaband_coh_6E_E2E5.append(betaband_coh)
                    temp_gammaband_coh_6E_E2E5.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E2E5.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E2E5.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E2E5.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E2E5.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E2E5.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E2E5=np.stack(temp_patientID_6E_E2E5)
                    delta_coh_6E_E2E5=np.stack(temp_deltaband_coh_6E_E2E5)
                    theta_coh_6E_E2E5=np.stack(temp_thetaband_coh_6E_E2E5)
                    alpha_coh_6E_E2E5=np.stack(temp_alphaband_coh_6E_E2E5)
                    beta_coh_6E_E2E5=np.stack(temp_betaband_coh_6E_E2E5)
                    gamma_coh_6E_E2E5=np.stack(temp_gammaband_coh_6E_E2E5)
                    deltacoh_av30_6E_E2E5=np.stack(temp_deltacoh_av30_6E_E2E5)
                    thetacoh_av30_6E_E2E5=np.stack(temp_thetacoh_av30_6E_E2E5)
                    alphacoh_av30_6E_E2E5=np.stack(temp_alphacoh_av30_6E_E2E5)
                    betacoh_av30_6E_E2E5=np.stack(temp_betacoh_av30_6E_E2E5)
                    gammacoh_av30_6E_E2E5=np.stack(temp_gammacoh_av30_6E_E2E5)

                    # Packing data in dictonary 
                    Coherence_values_6E_E2E5 = {
                        'PatientID': patientID_stacked_6E_E2E5,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E2E5.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E2E5.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E2E5.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E2E5.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E2E5.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E2E5.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E2E5.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E2E5.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E2E5.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E2E5.tolist(),
                        }
                    
                    print('Patient dictionary E2E5_6 - six electrodes, 8th combination')
                    #print(Coherence_values_6E_E2E5)
                
                # Storing the fourth combination E3E5_6
                elif E_combinations[8]==E_combinations[d]:

                    print('E_combinations[8] chosen for six electrodes 9th combination')
                    print(E_combinations[8])

                        # Filling out temporary variables: 
                    temp_patientID_6E_E3E5.append(patientID)
                    temp_deltaband_coh_6E_E3E5.append(deltaband_coh)
                    temp_thetaband_coh_6E_E3E5.append(thetaband_coh)
                    temp_alphaband_coh_6E_E3E5.append(alphaband_coh)
                    temp_betaband_coh_6E_E3E5.append(betaband_coh)
                    temp_gammaband_coh_6E_E3E5.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E3E5.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E3E5.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E3E5.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E3E5.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E3E5.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E3E5=np.stack(temp_patientID_6E_E3E5)
                    delta_coh_6E_E3E5=np.stack(temp_deltaband_coh_6E_E3E5)
                    theta_coh_6E_E3E5=np.stack(temp_thetaband_coh_6E_E3E5)
                    alpha_coh_6E_E3E5=np.stack(temp_alphaband_coh_6E_E3E5)
                    beta_coh_6E_E3E5=np.stack(temp_betaband_coh_6E_E3E5)
                    gamma_coh_6E_E3E5=np.stack(temp_gammaband_coh_6E_E3E5)
                    deltacoh_av30_6E_E3E5=np.stack(temp_deltacoh_av30_6E_E3E5)
                    thetacoh_av30_6E_E3E5=np.stack(temp_thetacoh_av30_6E_E3E5)
                    alphacoh_av30_6E_E3E5=np.stack(temp_alphacoh_av30_6E_E3E5)
                    betacoh_av30_6E_E3E5=np.stack(temp_betacoh_av30_6E_E3E5)
                    gammacoh_av30_6E_E3E5=np.stack(temp_gammacoh_av30_6E_E3E5)

                    # Packing data in dictonary 
                    Coherence_values_6E_E3E5 = {
                        'PatientID': patientID_stacked_6E_E3E5,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E3E5.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E3E5.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E3E5.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E3E5.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E3E5.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E3E5.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E3E5.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E3E5.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E3E5.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E3E5.tolist(),
                        }
                    
                    print('Patient dictionary E3E5_6 - six electrodes, 9th combination')
                    #print(Coherence_values_6E_E3E5)

                    
                # Storing the fourth combination E4E5_6
                elif E_combinations[9]==E_combinations[d]:

                    print('E_combinations[9] chosen for six electrodes 10th combination')
                    print(E_combinations[9])

                        # Filling out temporary variables: 
                    temp_patientID_6E_E4E5.append(patientID)
                    temp_deltaband_coh_6E_E4E5.append(deltaband_coh)
                    temp_thetaband_coh_6E_E4E5.append(thetaband_coh)
                    temp_alphaband_coh_6E_E4E5.append(alphaband_coh)
                    temp_betaband_coh_6E_E4E5.append(betaband_coh)
                    temp_gammaband_coh_6E_E4E5.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E4E5.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E4E5.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E4E5.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E4E5.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E4E5.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E4E5=np.stack(temp_patientID_6E_E4E5)
                    delta_coh_6E_E4E5=np.stack(temp_deltaband_coh_6E_E4E5)
                    theta_coh_6E_E4E5=np.stack(temp_thetaband_coh_6E_E4E5)
                    alpha_coh_6E_E4E5=np.stack(temp_alphaband_coh_6E_E4E5)
                    beta_coh_6E_E4E5=np.stack(temp_betaband_coh_6E_E4E5)
                    gamma_coh_6E_E4E5=np.stack(temp_gammaband_coh_6E_E4E5)
                    deltacoh_av30_6E_E4E5=np.stack(temp_deltacoh_av30_6E_E4E5)
                    thetacoh_av30_6E_E4E5=np.stack(temp_thetacoh_av30_6E_E4E5)
                    alphacoh_av30_6E_E4E5=np.stack(temp_alphacoh_av30_6E_E4E5)
                    betacoh_av30_6E_E4E5=np.stack(temp_betacoh_av30_6E_E4E5)
                    gammacoh_av30_6E_E4E5=np.stack(temp_gammacoh_av30_6E_E4E5)

                    # Packing data in dictonary 
                    Coherence_values_6E_E4E5 = {
                        'PatientID': patientID_stacked_6E_E4E5,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E4E5.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E4E5.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E4E5.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E4E5.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E4E5.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E4E5.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E4E5.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E4E5.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E4E5.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E4E5.tolist(),
                        }
                    
                    print('Patient dictionary E4E5_6 - six electrodes, 10th combination')
                    #print(Coherence_values_6E_E4E5)

                    
                # Storing the fourth combination E1E6_6
                elif E_combinations[10]==E_combinations[d]:

                    print('E_combinations[10] chosen for six electrodes 11th combination')
                    print(E_combinations[10])

                        # Filling out temporary variables: 
                    temp_patientID_6E_E1E6.append(patientID)
                    temp_deltaband_coh_6E_E1E6.append(deltaband_coh)
                    temp_thetaband_coh_6E_E1E6.append(thetaband_coh)
                    temp_alphaband_coh_6E_E1E6.append(alphaband_coh)
                    temp_betaband_coh_6E_E1E6.append(betaband_coh)
                    temp_gammaband_coh_6E_E1E6.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E1E6.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E1E6.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E1E6.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E1E6.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E1E6.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E1E6=np.stack(temp_patientID_6E_E1E6)
                    delta_coh_6E_E1E6=np.stack(temp_deltaband_coh_6E_E1E6)
                    theta_coh_6E_E1E6=np.stack(temp_thetaband_coh_6E_E1E6)
                    alpha_coh_6E_E1E6=np.stack(temp_alphaband_coh_6E_E1E6)
                    beta_coh_6E_E1E6=np.stack(temp_betaband_coh_6E_E1E6)
                    gamma_coh_6E_E1E6=np.stack(temp_gammaband_coh_6E_E1E6)
                    deltacoh_av30_6E_E1E6=np.stack(temp_deltacoh_av30_6E_E1E6)
                    thetacoh_av30_6E_E1E6=np.stack(temp_thetacoh_av30_6E_E1E6)
                    alphacoh_av30_6E_E1E6=np.stack(temp_alphacoh_av30_6E_E1E6)
                    betacoh_av30_6E_E1E6=np.stack(temp_betacoh_av30_6E_E1E6)
                    gammacoh_av30_6E_E1E6=np.stack(temp_gammacoh_av30_6E_E1E6)

                    # Packing data in dictonary 
                    Coherence_values_6E_E1E6 = {
                        'PatientID': patientID_stacked_6E_E1E6,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E1E6.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E1E6.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E1E6.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E1E6.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E1E6.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E1E6.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E1E6.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E1E6.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E1E6.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E1E6.tolist(),
                        }
                    
                    print('Patient dictionary E1E6_6 - six electrodes, 11th combination')
                    #print(Coherence_values_6E_E1E6)

                    
                # Storing the 12th combination E2E6_6
                elif E_combinations[11]==E_combinations[d]:

                    print('E_combinations[11] chosen for six electrodes 12th combination')
                    print(E_combinations[11])

                        # Filling out temporary variables: 
                    temp_patientID_6E_E2E6.append(patientID)
                    temp_deltaband_coh_6E_E2E6.append(deltaband_coh)
                    temp_thetaband_coh_6E_E2E6.append(thetaband_coh)
                    temp_alphaband_coh_6E_E2E6.append(alphaband_coh)
                    temp_betaband_coh_6E_E2E6.append(betaband_coh)
                    temp_gammaband_coh_6E_E2E6.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E2E6.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E2E6.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E2E6.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E2E6.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E2E6.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E2E6=np.stack(temp_patientID_6E_E2E6)
                    delta_coh_6E_E2E6=np.stack(temp_deltaband_coh_6E_E2E6)
                    theta_coh_6E_E2E6=np.stack(temp_thetaband_coh_6E_E2E6)
                    alpha_coh_6E_E2E6=np.stack(temp_alphaband_coh_6E_E2E6)
                    beta_coh_6E_E2E6=np.stack(temp_betaband_coh_6E_E2E6)
                    gamma_coh_6E_E2E6=np.stack(temp_gammaband_coh_6E_E2E6)
                    deltacoh_av30_6E_E2E6=np.stack(temp_deltacoh_av30_6E_E2E6)
                    thetacoh_av30_6E_E2E6=np.stack(temp_thetacoh_av30_6E_E2E6)
                    alphacoh_av30_6E_E2E6=np.stack(temp_alphacoh_av30_6E_E2E6)
                    betacoh_av30_6E_E2E6=np.stack(temp_betacoh_av30_6E_E2E6)
                    gammacoh_av30_6E_E2E6=np.stack(temp_gammacoh_av30_6E_E2E6)

                    # Packing data in dictonary 
                    Coherence_values_6E_E2E6 = {
                        'PatientID': patientID_stacked_6E_E2E6,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E2E6.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E2E6.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E2E6.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E2E6.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E2E6.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E2E6.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E2E6.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E2E6.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E2E6.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E2E6.tolist(),
                        }
                    
                    print('Patient dictionary E2E6_6 - six electrodes, 12th combination')
                    #print(Coherence_values_6E_E2E6)
                    
                    
                # Storing the fourth combination E3E6_6
                elif E_combinations[12]==E_combinations[d]:

                    print('E_combinations[12] chosen for six electrodes 13th combination')
                    print(E_combinations[12])

                        # Filling out temporary variables: 
                    temp_patientID_6E_E3E6.append(patientID)
                    temp_deltaband_coh_6E_E3E6.append(deltaband_coh)
                    temp_thetaband_coh_6E_E3E6.append(thetaband_coh)
                    temp_alphaband_coh_6E_E3E6.append(alphaband_coh)
                    temp_betaband_coh_6E_E3E6.append(betaband_coh)
                    temp_gammaband_coh_6E_E3E6.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E3E6.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E3E6.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E3E6.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E3E6.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E3E6.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E3E6=np.stack(temp_patientID_6E_E3E6)
                    delta_coh_6E_E3E6=np.stack(temp_deltaband_coh_6E_E3E6)
                    theta_coh_6E_E3E6=np.stack(temp_thetaband_coh_6E_E3E6)
                    alpha_coh_6E_E3E6=np.stack(temp_alphaband_coh_6E_E3E6)
                    beta_coh_6E_E3E6=np.stack(temp_betaband_coh_6E_E3E6)
                    gamma_coh_6E_E3E6=np.stack(temp_gammaband_coh_6E_E3E6)
                    deltacoh_av30_6E_E3E6=np.stack(temp_deltacoh_av30_6E_E3E6)
                    thetacoh_av30_6E_E3E6=np.stack(temp_thetacoh_av30_6E_E3E6)
                    alphacoh_av30_6E_E3E6=np.stack(temp_alphacoh_av30_6E_E3E6)
                    betacoh_av30_6E_E3E6=np.stack(temp_betacoh_av30_6E_E3E6)
                    gammacoh_av30_6E_E3E6=np.stack(temp_gammacoh_av30_6E_E3E6)

                    # Packing data in dictonary 
                    Coherence_values_6E_E3E6 = {
                        'PatientID': patientID_stacked_6E_E3E6,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E3E6.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E3E6.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E3E6.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E3E6.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E3E6.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E3E6.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E3E6.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E3E6.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E3E6.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E3E6.tolist(),
                        }
                    
                    print('Patient dictionary E3E6_6 - six electrodes, 13th combination')
                    #print(Coherence_values_6E_E3E6)



                # Storing the fourth combination E4E6_6
                elif E_combinations[13]==E_combinations[d]:

                    print('E_combinations[13] chosen for six electrodes 14th combination')
                    print(E_combinations[13])

                        # Filling out temporary variables: 
                    temp_patientID_6E_E4E6.append(patientID)
                    temp_deltaband_coh_6E_E4E6.append(deltaband_coh)
                    temp_thetaband_coh_6E_E4E6.append(thetaband_coh)
                    temp_alphaband_coh_6E_E4E6.append(alphaband_coh)
                    temp_betaband_coh_6E_E4E6.append(betaband_coh)
                    temp_gammaband_coh_6E_E4E6.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E4E6.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E4E6.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E4E6.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E4E6.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E4E6.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E4E6=np.stack(temp_patientID_6E_E4E6)
                    delta_coh_6E_E4E6=np.stack(temp_deltaband_coh_6E_E4E6)
                    theta_coh_6E_E4E6=np.stack(temp_thetaband_coh_6E_E4E6)
                    alpha_coh_6E_E4E6=np.stack(temp_alphaband_coh_6E_E4E6)
                    beta_coh_6E_E4E6=np.stack(temp_betaband_coh_6E_E4E6)
                    gamma_coh_6E_E4E6=np.stack(temp_gammaband_coh_6E_E4E6)
                    deltacoh_av30_6E_E4E6=np.stack(temp_deltacoh_av30_6E_E4E6)
                    thetacoh_av30_6E_E4E6=np.stack(temp_thetacoh_av30_6E_E4E6)
                    alphacoh_av30_6E_E4E6=np.stack(temp_alphacoh_av30_6E_E4E6)
                    betacoh_av30_6E_E4E6=np.stack(temp_betacoh_av30_6E_E4E6)
                    gammacoh_av30_6E_E4E6=np.stack(temp_gammacoh_av30_6E_E4E6)

                    # Packing data in dictonary 
                    Coherence_values_6E_E4E6 = {
                        'PatientID': patientID_stacked_6E_E4E6,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E4E6.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E4E6.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E4E6.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E4E6.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E4E6.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E4E6.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E4E6.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E4E6.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E4E6.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E4E6.tolist(),
                        }
                    
                    print('Patient dictionary E4E6_6 - six electrodes, 14th combination')
                    #print(Coherence_values_6E_E4E6)
                    
                    
                # Storing the fourth combination E5E6_6
                elif E_combinations[14]==E_combinations[d]:

                    print('E_combinations[14] chosen for six electrodes 15th combination')
                    print(E_combinations[14])

                        # Filling out temporary variables: 
                    temp_patientID_6E_E5E6.append(patientID)
                    temp_deltaband_coh_6E_E5E6.append(deltaband_coh)
                    temp_thetaband_coh_6E_E5E6.append(thetaband_coh)
                    temp_alphaband_coh_6E_E5E6.append(alphaband_coh)
                    temp_betaband_coh_6E_E5E6.append(betaband_coh)
                    temp_gammaband_coh_6E_E5E6.append(gammaband_coh)
                    temp_deltacoh_av30_6E_E5E6.append(delta_coh_av_30)
                    temp_thetacoh_av30_6E_E5E6.append(theta_coh_av_30)
                    temp_alphacoh_av30_6E_E5E6.append(alpha_coh_av_30)
                    temp_betacoh_av30_6E_E5E6.append(beta_coh_av_30)
                    temp_gammacoh_av30_6E_E5E6.append(gamma_coh_av_30)
                            
                    # stacking variables 
                    patientID_stacked_6E_E5E6=np.stack(temp_patientID_6E_E5E6)
                    delta_coh_6E_E5E6=np.stack(temp_deltaband_coh_6E_E5E6)
                    theta_coh_6E_E5E6=np.stack(temp_thetaband_coh_6E_E5E6)
                    alpha_coh_6E_E5E6=np.stack(temp_alphaband_coh_6E_E5E6)
                    beta_coh_6E_E5E6=np.stack(temp_betaband_coh_6E_E5E6)
                    gamma_coh_6E_E5E6=np.stack(temp_gammaband_coh_6E_E5E6)
                    deltacoh_av30_6E_E5E6=np.stack(temp_deltacoh_av30_6E_E5E6)
                    thetacoh_av30_6E_E5E6=np.stack(temp_thetacoh_av30_6E_E5E6)
                    alphacoh_av30_6E_E5E6=np.stack(temp_alphacoh_av30_6E_E5E6)
                    betacoh_av30_6E_E5E6=np.stack(temp_betacoh_av30_6E_E5E6)
                    gammacoh_av30_6E_E5E6=np.stack(temp_gammacoh_av30_6E_E5E6)

                    # Packing data in dictonary 
                    Coherence_values_6E_E5E6 = {
                        'PatientID': patientID_stacked_6E_E5E6,
                        'Delta_coh_'+str(Electrode_combination_naming): delta_coh_6E_E5E6.tolist(),
                        'Theta_coh_'+str(Electrode_combination_naming): theta_coh_6E_E5E6.tolist(),
                        'Alpha_coh_'+str(Electrode_combination_naming): alpha_coh_6E_E5E6.tolist(),
                        'Beta_coh_'+str(Electrode_combination_naming): beta_coh_6E_E5E6.tolist(),
                        'Gamma_coh_'+str(Electrode_combination_naming): gamma_coh_6E_E5E6.tolist(),
                        'Deltacoh_av30s'+str(Electrode_combination_naming): deltacoh_av30_6E_E5E6.tolist(),
                        'Thetacoh_av30s'+str(Electrode_combination_naming): thetacoh_av30_6E_E5E6.tolist(),
                        'Alphacoh_av30s'+str(Electrode_combination_naming): alphacoh_av30_6E_E5E6.tolist(),
                        'Betacoh_av30s'+str(Electrode_combination_naming): betacoh_av30_6E_E5E6.tolist(),
                        'Gammacoh_av30s'+str(Electrode_combination_naming): gammacoh_av30_6E_E5E6.tolist(),
                        }
                    
                    print('Patient dictionary E5E6_6 - six electrodes, 15th combination')
                    #print(Coherence_values_6E_E5E6)

    except Exception as e:
        print(f"An error occurred: {e}. Moving on to the next iteration...")
        error_dict.append({'error':e,'filename':edf_file})
        
        # Delete variables in locals - to prevent errors

        if 'F3M2_index_trial' in locals():
            del F3M2_index_trial

        if 'F4M1_index_trial' in locals():
            del F4M1_index_trial

        if 'C3M2_index_trial' in locals():
            del C3M2_index_trial

        if 'C4M1_index_trial' in locals():
            del C4M1_index_trial

        if 'O1M2_index_trial' in locals():
            del O1M2_index_trial
        
        if 'O2M1_index_trial' in locals():
            del O2M1_index_trial

        if 'Signals' in locals():
            del Signals

       
        if 'F3M2_index' in locals(): 
            del F3M2_index

        if 'F4M1_index' in locals(): 
            del F4M1_index
        
        if 'C3M2_index' in locals():
            del C3M2_index

        if 'C4M1_index' in locals(): 
            del C4M1_index
        
        if 'O1M2_index' in locals(): 
            del O1M2_index
        
        if 'O2M1_index' in locals(): 
            del O2M1_index




### Creating merge functions for merging dataframe ############

# Merging the dataframes containing the three combinations of electrodes 
def Merge15(dict1, dict2, dict3, dict4, dict5 ,dict6, dict7, dict8, dict9, dict10, dict11, dict12, dict13, dict14, dict15):
    res = {**dict1, **dict2, **dict3, **dict4, **dict5, **dict6, **dict7, **dict8, **dict9, **dict10, **dict11, **dict12, **dict13, **dict14, **dict15}
    return res


def Merge11(dict1, dict2, dict3, dict4, dict5 ,dict6, dict7, dict8, dict9, dict10, dict11):
    res = {**dict1, **dict2, **dict3, **dict4, **dict5, **dict6, **dict7, **dict8, **dict9, **dict10, **dict11}
    return res


def Merge10(dict1, dict2, dict3, dict4, dict5 ,dict6, dict7, dict8, dict9, dict10):
    res = {**dict1, **dict2, **dict3, **dict4, **dict5, **dict6, **dict7, **dict8, **dict9, **dict10}
    return res


def Merge7(dict1, dict2, dict3, dict4, dict5 ,dict6, dict7):
    res = {**dict1, **dict2, **dict3, **dict4, **dict5, **dict6, **dict7}
    return res

def Merge6(dict1, dict2, dict3, dict4, dict5 ,dict6):
    res = {**dict1, **dict2, **dict3, **dict4, **dict5, **dict6}
    return res

def Merge5(dict1, dict2, dict3, dict4, dict5):
    res = {**dict1, **dict2, **dict3, **dict4, **dict5}
    return res


def Merge4(dict1, dict2, dict3,dict4):
    res = {**dict1, **dict2, **dict3, **dict4}
    return res

def Merge3(dict1, dict2, dict3):
    res = {**dict1, **dict2, **dict3}
    return res

def Merge2(dict1, dict2):
    res = {**dict1, **dict2}
    return res




######################## 6 electrodes ####################################
if 'Bandpower_values_6E_E1' in locals():
    Bandpower_6E_full_dict=Merge6(Bandpower_values_6E_E1, Bandpower_values_6E_E2, Bandpower_values_6E_E3, Bandpower_values_6E_E4, Bandpower_values_6E_E5, Bandpower_values_6E_E6)
    BP_6E=pd.DataFrame(Bandpower_6E_full_dict)
    BP_6E.to_csv(f"/scratch/users/s184063/RBD_Features/Bandpower_1_10_6electrodes.csv", index=False)
    print('Bandpower 6e were generated')   
    del Bandpower_values_6E_E1

if 'Coherence_values_6E_E1E2' in locals():
    # Generating dictonary for the merged dictornary 
    full_dict_6E = Merge15(Coherence_values_6E_E1E2,Coherence_values_6E_E1E3,Coherence_values_6E_E2E3,Coherence_values_6E_E1E4,Coherence_values_6E_E2E4,Coherence_values_6E_E3E4,Coherence_values_6E_E1E5,Coherence_values_6E_E2E5,Coherence_values_6E_E3E5,Coherence_values_6E_E4E5,Coherence_values_6E_E1E6,Coherence_values_6E_E2E6,Coherence_values_6E_E3E6,Coherence_values_6E_E4E6,Coherence_values_6E_E5E6)
    print('Full dictornary 6 electrodes')

    # Generating full data frame for the 6 electrode data 
    full_dataframe_6E=pd.DataFrame(full_dict_6E)
    full_dataframe_6E.to_csv(f"/scratch/users/s184063/RBD_Features/Coherence_patient_1_10_6electrodes.csv", index=False) # change filename using os
    print('cvs file was generated - 6E')
    del Coherence_values_6E_E1E2
############################################################################



######################## 5 electrodes ####################################
if 'Bandpower_values_5E_E1' in locals():
    Bandpower_5E_full_dict=Merge5(Bandpower_values_5E_E1, Bandpower_values_5E_E2, Bandpower_values_5E_E3, Bandpower_values_5E_E4, Bandpower_values_5E_E5)
    BP_5E=pd.DataFrame(Bandpower_5E_full_dict)
    BP_5E.to_csv(f"/scratch/users/s184063/RBD_Features/Bandpower_1_10_5electrodes.csv", index=False)
    print('Bandpower 5e were generated')    
    del Bandpower_values_5E_E1

if 'Coherence_values_5E_E1E2' in locals():
    # Generating dictonary for the merged dictornary 
    full_dict_5E = Merge10(Coherence_values_5E_E1E2,Coherence_values_5E_E1E3,Coherence_values_5E_E2E3,Coherence_values_5E_E1E4,Coherence_values_5E_E2E4,Coherence_values_5E_E3E4,Coherence_values_5E_E1E5,Coherence_values_5E_E2E5,Coherence_values_5E_E3E5,Coherence_values_5E_E4E5)
    print('Full dictornary 5 electrodes')

    # Generating full data frame for the 5 electrode data 
    full_dataframe_5E=pd.DataFrame(full_dict_5E)

    full_dataframe_5E.to_csv(f"/scratch/users/s184063/RBD_Features/Coherence_patient_1_10_5electrodes.csv", index=False) # change filename using os
    print('csv with 5e where generated')
    del Coherence_values_5E_E1E2
############################################################################




######################## 4 electrodes ####################################
if 'Bandpower_values_4E_E1' in locals():
    Bandpower_4E_full_dict=Merge4(Bandpower_values_4E_E1, Bandpower_values_4E_E2, Bandpower_values_4E_E3, Bandpower_values_4E_E4)
    BP_4E=pd.DataFrame(Bandpower_4E_full_dict)
    BP_4E.to_csv(f"/scratch/users/s184063/RBD_Features/Bandpower_1_10_4electrodes.csv", index=False)
    print('Bandpower 4e were generated')
    del Bandpower_values_4E_E1

if 'Coherence_values_4E_E1E2' in locals():
    # Generating dictonary for the merged dictornary 
    full_dict_4E = Merge6(Coherence_values_4E_E1E2,Coherence_values_4E_E1E3,Coherence_values_4E_E2E3,Coherence_values_4E_E1E4,Coherence_values_4E_E2E4,Coherence_values_4E_E3E4)
    print('Full dictornary 4 electrodes')

    # Generating full data frame for the 4 electrode data 
    full_dataframe_4E=pd.DataFrame(full_dict_4E)

    full_dataframe_4E.to_csv(f"/scratch/users/s184063/RBD_Features/Coherence_patient_1_10_4electrodes.csv", index=False) # change filename using os
    print('csv file with 4 electrodes where generated')
    
    del Coherence_values_4E_E1E2
############################################################################




######################## 3 electrodes ####################################
if 'Bandpower_values_3E_E1' in locals():
    Bandpower_3E_full_dict=Merge3(Bandpower_values_3E_E1, Bandpower_values_3E_E2, Bandpower_values_3E_E3)
    BP_3E=pd.DataFrame(Bandpower_3E_full_dict)
    BP_3E.to_csv(f"/scratch/users/s184063/RBD_Features/Bandpower_1_10_3electrodes.csv", index=False)
    print('Bandpower 3e generated csv file')
    del Bandpower_values_3E_E1

if 'Coherence_values_3E_E1E2' in locals():
    # Generating dictonary for the merged dictornary 
    full_dict_3E = Merge3(Coherence_values_3E_E1E2,Coherence_values_3E_E1E3,Coherence_values_3E_E2E3)
    print('Full dictornary 3 electrodes')

    # Generating full data frame for the 3 electrode data 
    full_dataframe_3E=pd.DataFrame(full_dict_3E)
    full_dataframe_3E.to_csv(f"/scratch/users/s184063/RBD_Features/Coherence_patient_1_10_3electrodes.csv", index=False) # change filename using os
    print('csv file 3e were generated')
    del Coherence_values_3E_E1E2
############################################################################



##################### 2 electrodes #########################################
if 'Bandpower_values_2E_E1' in locals():
    Bandpower_2E_full_dict=Merge2(Bandpower_values_2E_E1, Bandpower_values_2E_E2)
    BP_2E=pd.Dataframe(Bandpower_2E_full_dict)
    BP_2E.to_csv(f"/scratch/users/s184063/RBD_Features/Bandpower_1_10_2electrodes.csv", index=False)
    print('BP 2E csv generated')
    del Bandpower_values_2E_E1


if 'Coherence_values_2E_E1E2' in locals():
    # Generating full data frame for the 2 electrode data 
    print('Generating dataframe and CSV file for 2-electrode data')
    full_dataframe_2E=pd.DataFrame(Coherence_values_2E_E1E2) 
    full_dataframe_2E.to_csv(f"/scratch/users/s184063/RBD_Features/Coherence_patient_1_10_2electrodes.csv", index=False) # change filename using os
    print('csv file 2E generated')
    del Coherence_values_2E_E1E2
############################################################################


print('Done')





        





                        


                
                    


                





        
    





    



    






