# Load standard packages
import numpy as np
from numpy import shape, vstack
import pandas as pd
import copy
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
# Load signal processing packages
import scipy # Signal processing 
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.signal import resample_poly, firwin, butter, bilinear, periodogram, welch, filtfilt, sosfiltfilt
# For processing EDF files 
import imblearn
from imblearn.metrics import specificity_score
import pyedflib 
from pyedflib import highlevel # Extra packages for EDF
import pyedflib as plib
# Load wavelet package 
import pywt 
from pywt import wavedec
import sys
import itertools 
from itertools import combinations
import xgboost
from xgboost import XGBClassifier
import sklearn
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, auc
from sklearn.inspection import permutation_importance
import random
# Using sys function to import 'My_functions_script'
sys.path.insert(0, 'C:/Users/natas/Documents/Master thesis code')

# Import My_functions_script
from My_functions_script_China import list_files_in_folder, preprocessing, bandpass_frequency_band, inverse_wavelet_5_levels, relative_power_for_frequencyband, coherence_features, extract_numbers_from_filename, extract_letters_and_numbers, split_string_by_length

### Loading data ###########################
df_fullnight=pd.read_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/All_Coherence_fullnight_features_China_patientsandcontrols.csv') # full night
df_1part=pd.read_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/All_Coherence_1part_features_China_patientsandcontrols.csv') # part 1 
df_2part=pd.read_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/All_Coherence_2part_features_China_patientsandcontrols.csv') # part 2 
df_3part=pd.read_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/All_Coherence_3part_features_China_patientsandcontrols.csv') # part 3
df_4part=pd.read_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/All_Coherence_4part_features_China_patientsandcontrols.csv') # part 4




# Combining all coherence features 

# Dropping the first patienID folder for most of the files in order to make a concatenation of all files 
df_1part=df_1part.drop(['PatientID','Dx','Sex','Cohort'],axis=1)
df_2part=df_2part.drop(['PatientID','Dx','Sex','Cohort'],axis=1)
df_3part=df_3part.drop(['PatientID','Dx','Sex','Cohort'],axis=1)
df_4part=df_4part.drop(['PatientID','Dx','Sex','Cohort'],axis=1)

df_1part=df_1part.add_suffix('_1part')
df_2part=df_2part.add_suffix('_2part')
df_3part=df_3part.add_suffix('_3part')
df_4part=df_4part.add_suffix('_4part')

print(df_1part)


# cropping dataframes
#print(df_1part)
df_fullnight_edited = df_fullnight.iloc[:,1:556]
df_part1_edited=df_1part.iloc[:,1:556]
df_part2_edited=df_2part.iloc[:,1:556]
df_part3_edited=df_3part.iloc[:,1:556]
df_part4_edited=df_4part.iloc[:,1:556]


print(df_part1_edited)

# Combining the dataframes. Only one column contains patientID
df_coherence=pd.concat([df_fullnight,df_part1_edited,df_part2_edited,df_part3_edited,df_part4_edited],axis=1)
df_coherence.to_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/Coherence_All_combined_China.csv')

df_coherence_model=df_coherence # defining the dataset for a separate coherence model 
print(df_coherence)




# Correlation 
df_correlation=pd.read_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/All_Correlation_features_China_patientsandcontrols.csv')
print(df_correlation)
df_correlation_model=df_correlation # defining the dataset for a separate correlatiom model 


### All combined coherence and correlation ####

df_coherence=df_coherence.drop(['PatientID','Dx','Sex','Cohort'],axis=1)
length=df_coherence.shape[1]
df_coherence_cropped=df_coherence.iloc[:,1:length]
print(df_coherence_cropped)

all_features=pd.concat([df_correlation,df_coherence_cropped],axis=1)
all_features.to_csv('C:/Users/natas/Documents/Master thesis code/All features/China dataset/Final_All_features_China_coherence_and_correlation.csv')

print(all_features)




############### Enable the model you want to run ################################
df_combined=df_correlation_model
#df_combined=df_coherence_model
#df_combined=all_features
#################################################################################

df_NT1=df_combined[df_combined['Dx'] == 'NT1']


df_patients=pd.concat([df_NT1])

print('df patients')
print(df_patients)

df_controls=df_combined[df_combined['Dx'] == 'Control']

print('df controls')
print(df_controls)

df_combined=pd.concat([df_patients,df_controls])


df_for_use=copy.deepcopy(df_combined)
############################################


df_shuffled = df_for_use.sample(frac=1).reset_index(drop=True)
df_for_use=df_shuffled

####### Preparing X and Y ########################
df_X=df_for_use.drop(['PatientID','Dx','Sex','Cohort'], axis=1)
print(df_X)


# Creating values 0 = value, 1 = NaN
df_X_baseline=1*np.isnan(df_X)
print(df_X_baseline)


# Encoding the categorical columns for the X matrix

# Encoding categorical columns (Dx, Sex, patientID and cohort)
categorical_columns = df_for_use[['Sex','Dx']].columns.tolist()
print('Categorical columns')
print(categorical_columns)


#Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Apply one-hot encoding to the categorical columns
one_hot_encoded = encoder.fit_transform(df_for_use[categorical_columns])

#Create a DataFrame with the one-hot encoded columns
#We use get_feature_names_out() to get the column names for the encoded data
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

print(one_hot_df)



########## Y_variable ##########################
# Onehot data - encoded into 1 and 0 
Y_variable=one_hot_df[['Dx_NT1']] # This is RBD labels
print(Y_variable)

###############################################
one_hot_df=one_hot_df.drop(['Dx_NT1','Dx_Control'],axis=1)
print('One_hot_df')
print(one_hot_df)

# Concatenate the one-hot encoded dataframe with the original dataframe
X_matrix = pd.concat([df_X_baseline, one_hot_df], axis=1)

# Cropping X_matrix to not include the indexes 
datalength = df_combined.shape[1]
X_matrix=X_matrix.iloc[:,1:datalength]
print('Dataframe')
print(X_matrix)
print(type(X_matrix))

X_matrix_df=X_matrix

# Converting X_matrix and Y-variable into np.arrays
X_matrix=pd.DataFrame.to_numpy(X_matrix)

print('np.array')
print(X_matrix)
print(X_matrix.shape)


print('Y-variable pd.dataframe')
print(Y_variable)
Y_variable=pd.DataFrame.to_numpy(Y_variable)
print('np.array Y ')
print(Y_variable)


######################## Running the model #################################################
# create model (classifier) instance 
classifier = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=0.5, objective='binary:logistic')

# stratified K-fold (takes percentage of classes into account, when splitting the data)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
skf.get_n_splits(X_matrix, Y_variable)

print('Stratified K-fold')
print(skf)

train_split=[]
test_split=[]
temp_precision=[]
temp_recall=[]
temp_f1_score=[]
temp_accuracy=[]
temp_specificity=[]
y_pred_concatenated=[]
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# Defining figure 
fig, ax = plt.subplots(figsize=(6, 6))

#StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(X_matrix, Y_variable)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

    # Saving the train and test split 
    train_split.append(train_index)
    test_split.append(test_index)


    # fitting the model 
    classifier.fit(X_matrix[train_index], Y_variable[train_index])


    ###########    FEATURE IMPORTANCE #################
    # get feature importances
    # For a forrest this is called mean decrease in impurity
    importances = classifier.feature_importances_

    temp_column_name=[]
    # print them out
    for i, importance in enumerate(importances):
        #print(f"Feature {i}: {importance}")

        if importance > 0:
            # Indexing in X to find the name of the features 
            column_name = X_matrix_df.columns[i]

            temp_column_name.append(column_name)

    print('Important features')
    print(temp_column_name)

    
    ### Permutation on the test set #######
    #result = permutation_importance(
    #    classifier, X_matrix[test_index], Y_variable[test_index], n_repeats=10, random_state=42, n_jobs=2
    #)

    #sorted_importances_idx = result.importances_mean.argsort()
    #importances = pd.DataFrame(
    #    result.importances[sorted_importances_idx].T,
    #    columns=X_matrix_df.columns[sorted_importances_idx],
    #)
    #print('Pertumation test set')
    #print(importances)
    #ax = importances.plot.box(vert=False, whis=10)
    #ax.set_title("Permutation Importances (test set), RBD model")
    #ax.axvline(x=0, color="k", linestyle="--")
    #ax.set_xlabel("Decrease in accuracy score")
    #ax.figure.tight_layout()
    #plt.show()
    

    ###########################################################################


    ##### PREDICTION  ####################################
    # Predicting probabilities with the trained model 
    y_pred=classifier.predict_proba(X_matrix[test_index])
    print(len(y_pred))
    print(y_pred)

    # Saving the predictions in the same order as the test split 
    y_pred_concatenated.append(y_pred)

    # Converting the probabilities to 1 and 0 
    # This is used for the metrics - using only one of the columns (first one - it is one-hot encoded)
    y_pred_binary=np.round(y_pred)
    #print(y_pred_binary[:,1])
    #print(Y_variable[test_index])

    ############### PERFORMANCE METRICS ########################################
    # Calculating metrics
    confusion_matrix_metrics=sklearn.metrics.confusion_matrix(Y_variable[test_index],y_pred_binary[:,1]) ### Talk about this part!!!!!!!!!!
    print('Confusion matrix')
    print(confusion_matrix_metrics)

    classificationreport=classification_report(Y_variable[test_index], y_pred_binary[:,1])
    print(classificationreport)

    precision = sklearn.metrics.precision_score(Y_variable[test_index], y_pred_binary[:,1])

    recall = sklearn.metrics.recall_score(Y_variable[test_index], y_pred_binary[:,1])

    f1_score= sklearn.metrics.f1_score(Y_variable[test_index], y_pred_binary[:,1])

    accuracy=sklearn.metrics.accuracy_score(Y_variable[test_index], y_pred_binary[:,1])

    specificity=imblearn.metrics.specificity_score(Y_variable[test_index], y_pred_binary[:,1])

    # Storing all values 

    temp_precision.append(precision)
    temp_recall.append(recall)
    temp_f1_score.append(f1_score)
    temp_accuracy.append(accuracy)
    temp_specificity.append(specificity)

    print('Precision')
    print(precision)
    print('Recall')
    print(recall)
    print('F1 score')
    print(f1_score)
    print('Accuracy')
    print(accuracy)
    print('Specificity')
    print(specificity)

 


    ##### Plotting Precision and Recall curve ############

    # Does not work, because the indexes are np.array, but the function wants a df i think 
    
    # Presicion / recall curve
    
    #display = PrecisionRecallDisplay.from_estimator(
    #    classifier, X_matrix[test_index], Y_variable[test_index], name="BostedDecisionTree", plot_chance_level=True
    #)
    #_ = display.ax_.set_title("2-class Precision-Recall curve, RBD model")

    # To display the plot
    #plt.show()

    # Plotting ROC curve 
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X_matrix[test_index],
        Y_variable[test_index],
        name=f"ROC fold {i}",
        alpha=0.3,
        lw=1,
        ax=ax,
        plot_chance_level=(i == n_splits - 1),
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    



### Plotting ROC curve #############
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Mean ROC curve with variability\n(Positive label NT1, dataset=Baseline Sleep stage synchronization features)",
)
ax.legend(loc="lower right")
plt.show()


#### Average performance metrics #########

# Stacking values using np.stack 
precision_all = np.stack(temp_precision,axis=0) 
recall_all = np.stack(temp_recall,axis=0)
f1_score_all= np.stack(temp_f1_score,axis=0)
accuracy_all=np.stack(temp_accuracy,axis=0)
specificity_all=np.stack(temp_specificity,axis=0)

# Mean and std for metrics 
precision_mean=np.mean(precision_all)
precision_std=np.std(precision_all)


recall_mean=np.mean(recall_all)
recall_std=np.std(recall_all)

f1_score_mean=np.mean(f1_score_all)
f1_score_std=np.std(f1_score_all)

accuracy_mean=np.mean(accuracy_all)
accuracy_std=np.std(accuracy_all)


specificity_mean=np.mean(specificity_all)
specificity_std=np.std(specificity_all)


print('All metrics')
print('Precision mean and std')
print(precision_mean)
print(precision_std)
print('Recall mean and std')
print(recall_mean)
print(recall_std)
print('F1 score mean and std')
print(f1_score_mean)
print(f1_score_std)
print('Accuracy mean and std')
print(accuracy_mean)
print(accuracy_std)
print('Specificity mean and std')
print(specificity_mean)
print(specificity_std)

