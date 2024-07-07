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

import seaborn as sns
import ast
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
df_shuffled = df_for_use.sample(frac=1).reset_index(drop=True)
df_for_use=df_shuffled

####### Preparing X and Y ########################
df_X=df_for_use.drop(['PatientID','Sex','Dx','Cohort'], axis=1)
print(df_X)

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

#print(one_hot_df)
#female=one_hot_df['Sex_F1_M2_2']
#print(sum(female))


########## Y_variable ##########################
# Onehot data - encoded into 1 and 0 
Y_variable=one_hot_df[['Dx_NT1']] # This is RBD labels
print(Y_variable)
###############################################
one_hot_df=one_hot_df.drop(['Dx_NT1','Dx_Control'],axis=1)
print('One_hot_df')
print(one_hot_df)

# Concatenate the one-hot encoded dataframe with the original dataframe
# Concatenate the one-hot encoded dataframe with the original dataframe
X_matrix = pd.concat([df_X, one_hot_df], axis=1)

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
temp_all_importance=[]


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

        
        temp_column_name.append(importance)

        #if importance > 0:
        #    # Indexing in X to find the name of the features 
        #    column_name = X_matrix_df.columns[i]
        #
        #    temp_column_name.append(column_name)
            

    print('Important features')
    print(len(temp_column_name))
    # This variable should be saved and stacked outside of the fold !!!!
    temp_all_importance.append(temp_column_name)
    
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
    title=f"Mean ROC curve with variability\n(Positive label NT1, dataset=Sleep stage synchronization features)",
)
ax.legend(loc="lower right")
plt.show()
plt.clf()


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


'''
##### Feature importance ##################

# Stacking feature importance from each fold (5 folds)
importance_features=np.stack(temp_all_importance,axis=1) # matrix dimensions (4277,5) = 4227 features and 5 folds 
print('Feature importance matrix')
print(importance_features)
print(importance_features.shape)

# Summing over the rows 
# End result should be an array with 4227 elements 
importance_array=np.sum(importance_features,axis=1)

# Checking calculations
print('Importance array')
print(importance_array)
print(importance_array.shape)


sorted_features=sorted(importance_array,reverse=True)
print(type(sorted_features))
sorted_features=np.stack(sorted_features,axis=0)
print(type(sorted_features))
print(sorted_features)
print(len(sorted_features))



##### Overall 10 most important features ############


# This plot is made for the dataset with all features
temp_feature=[]
temp_value=[]
for h in range(10): 


    # Finding the indexes for the sorted features in the 'importance_array', 
    # that is not sorted. This is done to get the original index in order to find the name of the correct feature
    indexes_importance=np.where(sorted_features[h]==importance_array)
    print(indexes_importance)
    print(type(indexes_importance))

    # Indexing and finding the feature name and value 
    feature_value=importance_array[indexes_importance]
    print(feature_value)
    print(type(feature_value))

    # Collecting feature values 
    temp_value.append(feature_value)


    # Extracting the feature from dataframe
    feature = X_matrix_df.columns[indexes_importance]

    # Collecting features 
    temp_feature.append(feature[0]) # saving the features for boxplotting 



# Saving values gathered in the loop 
print('list with features')
print(temp_feature)
print(type(temp_feature))

print('feature value')
print(temp_value)
values=np.stack(temp_value,axis=1)
print(values)

# Creating dataframe with values for plotting 
df_plt=pd.DataFrame(values, columns=temp_feature)
print(df_plt)


boxplot_features=temp_feature 
plt.suptitle('10 most important features',fontsize=20)
plt.subplot(1, 1, 1) # Adjust these numbers as per your requirement.
sns.barplot(data=df_plt,errorbar=None)
plt.xticks(rotation=45, ha='right',fontsize=14)
#plt.title('')
plt.tight_layout()
plt.show()   
plt.clf()





########### Coherence feature plot ################################

# These plots should use the dataset for coherence 

# Number of coherence features = 2777 columns 
# Extract only from coherence features - divide X_matrix_df up into coherence and correlation features when indexing in the feature_importance_array
# The full dataset was made the following way: all_features=pd.concat([df_correlation,df_coherence_cropped],axis=1) 

print(df_coherence) 



# Category plots 

# Category 1) Frequency band plots (delta, theta, alpha, beta, gamma) = 5 barplots 

# should find the names in the df (done)
# use these to find the values in 'importance_array' indexes 
# e.g. sum all importance values having something with alpha in the name 
# run the rest of the loop from above 


temp_idx_delta_coherence=[]
temp_idx_theta_coherence=[]
temp_idx_alpha_coherence=[]
temp_idx_beta_coherence=[]
temp_idx_gamma_coherence=[]



#### Delta ##########

# Extracting all coherence features with 'delta' in the name
delta=df_coherence.filter(like='delta', axis=1) 
print(delta.shape[1])
print(delta.shape)
print(delta)


# In this loop the indexes of the delta matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(delta.shape[1]):
    
    # Indexing in delta category df and getting feature names 
    idx=delta.columns[i]

    # Using the feature names from 'delta' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_delta=X_matrix_df.columns.get_loc(idx)

    temp_idx_delta_coherence.append(original_idx_delta)


print('List with indexes for categorized delta features')
#
print(temp_idx_delta_coherence)
delta_coherence=np.stack(temp_idx_delta_coherence,axis=0)
print(delta_coherence)

print(delta_coherence.shape[0])
print(delta_coherence.shape)
print(importance_array.shape[0])

# Indexing in the importance array to get the delta coherence importance values and summing the values to one value for this category
delta_importance=np.sum(importance_array[delta_coherence])

print(delta_importance)

#### Theta ###########

# Extracting all coherence features with 'delta' in the name
theta=df_coherence.filter(like='theta', axis=1) 
print(theta.shape[1])
print(theta)


# In this loop the indexes of the theta matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(theta.shape[1]):
    
    # Indexing in delta category df and getting feature names 
    idx=theta.columns[i]

    # Using the feature names from 'theta' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_theta=X_matrix_df.columns.get_loc(idx)

    temp_idx_theta_coherence.append(original_idx_theta)


print('List with indexes for categorized theta features')
#print(temp_idx_theta_coherence)
theta_coherence=np.stack(temp_idx_theta_coherence,axis=0)

# Indexing in the importance array to get the theta coherence importance values and summing the values to one value for this category
theta_importance=np.sum(importance_array[theta_coherence])

print(theta_importance)


#### alpha ###########

# Extracting all coherence features with 'delta' in the name
alpha=df_coherence.filter(like='alpha', axis=1) 
print(alpha.shape[1])
print(alpha)


# In this loop the indexes of the delta matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(alpha.shape[1]):
    
    # Indexing in delta category df and getting feature names 
    idx=alpha.columns[i]

    # Using the feature names from 'delta' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_alpha=X_matrix_df.columns.get_loc(idx)

    temp_idx_alpha_coherence.append(original_idx_alpha)


print('List with indexes for categorized alpha features')
#print(temp_idx_alpha_coherence)
alpha_coherence=np.stack(temp_idx_alpha_coherence,axis=0)

# Indexing in the importance array to get the alpha coherence importance values and summing the values to one value for this category
alpha_importance=np.sum(importance_array[alpha_coherence])

print(alpha_importance)



#### beta ###########

# Extracting all coherence features with 'delta' in the name
beta=df_coherence.filter(like='beta', axis=1) 
print(beta.shape[1])
print(beta)


# In this loop the indexes of the delta matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(beta.shape[1]):
    
    # Indexing in delta category df and getting feature names 
    idx=beta.columns[i]

    # Using the feature names from 'delta' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_beta=X_matrix_df.columns.get_loc(idx)

    temp_idx_beta_coherence.append(original_idx_beta)


print('List with indexes for categorized beta features')
#print(temp_idx_beta_coherence)
beta_coherence=np.stack(temp_idx_beta_coherence,axis=0)

# Indexing in the importance array to get the beta coherence importance values and summing the values to one value for this category
beta_importance=np.sum(importance_array[beta_coherence])

print(beta_importance)





#### gamma ###########

# Extracting all coherence features with 'delta' in the name
gamma=df_coherence.filter(like='gamma', axis=1) 
print(gamma.shape[1])
print(gamma)


# In this loop the indexes of the delta matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(gamma.shape[1]):
    
    # Indexing in delta category df and getting feature names 
    idx=gamma.columns[i]

    # Using the feature names from 'delta' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_gamma=X_matrix_df.columns.get_loc(idx)

    temp_idx_gamma_coherence.append(original_idx_gamma)


print('List with indexes for categorized gamma features')
#print(temp_idx_gamma_coherence)
gamma_coherence=np.stack(temp_idx_gamma_coherence,axis=0)

# Indexing in the importance array to get the gamma coherence importance values and summing the values to one value for this category
gamma_importance=np.sum(importance_array[gamma_coherence])



print(gamma_importance)



### Gathering the frequency importance values in a df for plotting ########

# Saving values gathered for delta, theta. alpha, beta and gamma
print('list with features - frequencies')

frequency_values_list=[delta_importance, theta_importance, alpha_importance, beta_importance, gamma_importance]

features_frequencies=np.stack(frequency_values_list)
features_frequencies=np.reshape(features_frequencies,(1,5)) # transposing to reshape
print(features_frequencies)
print(type(features_frequencies))
print(features_frequencies.shape)

# column names 
frequency_columns=['Delta','Theta','Alpha','Beta','Gamma']
print(type(frequency_columns))
print(frequency_columns)


# Creating dataframe with values for plotting 
df_plt_frequency=pd.DataFrame(features_frequencies, columns=frequency_columns)
print(df_plt_frequency)


boxplot_features=frequency_columns
plt.suptitle('Frequency feature importance',fontsize=20)
plt.subplot(1, 1, 1) # Adjust these numbers as per your requirement.
sns.barplot(data=df_plt_frequency,errorbar=None)
plt.xticks(rotation=45, ha='right',fontsize=16)
#plt.title('')
plt.tight_layout()
plt.show()   
plt.clf()




# Category 2) Sleep stages (wake, N1, N2, N3, REM) = 5 barplots 


temp_idx_wake_coherence=[]
temp_idx_N1_coherence=[]
temp_idx_N2_coherence=[]
temp_idx_N3_coherence=[]
temp_idx_REM_coherence=[]

#### wake ##########

# Extracting all coherence features with 'wake' in the name
wake=df_coherence.filter(like='Wake', axis=1) 
print(wake.shape[1])
print(wake)


# In this loop the indexes of the wake matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(wake.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=wake.columns[i]

    # Using the feature names from 'wake' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_wake=X_matrix_df.columns.get_loc(idx)

    temp_idx_wake_coherence.append(original_idx_wake)


print('List with indexes for categorized wake features')
#
# print(temp_idx_wake_coherence)
wake_coherence=np.stack(temp_idx_wake_coherence,axis=0)

print(wake_coherence.shape[0])

# Indexing in the importance array to get the wake coherence importance values and summing the values to one value for this category
wake_importance=np.sum(importance_array[wake_coherence])

print(wake_importance)




#### N1 ##########

# Extracting all coherence features with 'N1' in the name
N1=df_coherence.filter(like='N1', axis=1) 
print(N1.shape[1])
print(N1)


# In this loop the indexes of the N1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(N1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=N1.columns[i]

    # Using the feature names from 'N1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_N1=X_matrix_df.columns.get_loc(idx)

    temp_idx_N1_coherence.append(original_idx_N1)


print('List with indexes for categorized N1 features')
#
# print(temp_idx_N1_coherence)
N1_coherence=np.stack(temp_idx_N1_coherence,axis=0)

print(N1_coherence.shape[0])

# Indexing in the importance array to get the N1 coherence importance values and summing the values to one value for this category
N1_importance=np.sum(importance_array[N1_coherence])

print(N1_importance)





#### N2 ##########

# Extracting all coherence features with 'N2' in the name
N2=df_coherence.filter(like='N2', axis=1) 
print(N2.shape[1])
print(N2)


# In this loop the indexes of the N2 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(N2.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=N2.columns[i]

    # Using the feature names from 'N1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_N2=X_matrix_df.columns.get_loc(idx)

    temp_idx_N2_coherence.append(original_idx_N2)


print('List with indexes for categorized N2 features')
#print(temp_idx_N2_coherence)
N2_coherence=np.stack(temp_idx_N2_coherence,axis=0)

print(N2_coherence.shape[0])

# Indexing in the importance array to get the N2 coherence importance values and summing the values to one value for this category
N2_importance=np.sum(importance_array[N2_coherence])

print(N2_importance)





#### N3 ##########

# Extracting all coherence features with 'N3' in the name
N3=df_coherence.filter(like='N3', axis=1) 
print(N3.shape[1])
print(N3)


# In this loop the indexes of the N3 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(N3.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=N3.columns[i]

    # Using the feature names from 'N3' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_N3=X_matrix_df.columns.get_loc(idx)

    temp_idx_N3_coherence.append(original_idx_N3)


print('List with indexes for categorized N3 features')
#print(temp_idx_N3_coherence)
N3_coherence=np.stack(temp_idx_N3_coherence,axis=0)

print(N3_coherence.shape[0])

# Indexing in the importance array to get the N3 coherence importance values and summing the values to one value for this category
N3_importance=np.sum(importance_array[N3_coherence])

print(N3_importance)




#### REM ##########

# Extracting all coherence features with 'REM' in the name
REM=df_coherence.filter(like='REM', axis=1) 
print(REM.shape[1])
print(REM)


# In this loop the indexes of the REM matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(REM.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=REM.columns[i]

    # Using the feature names from 'REM' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_REM=X_matrix_df.columns.get_loc(idx)

    temp_idx_REM_coherence.append(original_idx_REM)


print('List with indexes for categorized REM features')
#print(temp_idx_REM_coherence)
REM_coherence=np.stack(temp_idx_REM_coherence,axis=0)

print(REM_coherence.shape[0])

# Indexing in the importance array to get the REM coherence importance values and summing the values to one value for this category
REM_importance=np.sum(importance_array[REM_coherence])

print(REM_importance)



### Gathering the frequency importance values in a df for plotting ########

# Saving values gathered for delta, theta. alpha, beta and gamma
print('list with features - frequencies')

sleepstage_values_list=[wake_importance, N1_importance, N2_importance, N3_importance, REM_importance]

features_sleepstage=np.stack(sleepstage_values_list)
features_sleepstage=np.reshape(features_sleepstage,(1,5)) # reshape
print(features_sleepstage)
print(type(features_sleepstage))
print(features_sleepstage.shape)

# column names 
sleepstage_columns=['Wake','N1','N2','N3','REM']
print(type(sleepstage_columns))
print(sleepstage_columns)


# Creating dataframe with values for plotting 
df_plt_sleepstage=pd.DataFrame(features_sleepstage, columns=sleepstage_columns)
print(df_plt_sleepstage)


boxplot_features=sleepstage_columns
plt.suptitle('Sleep stage feature importance',fontsize=20)
plt.subplot(1, 1, 1) # Adjust these numbers as per your requirement.
sns.barplot(data=df_plt_sleepstage,errorbar=None)
plt.xticks(rotation=45, ha='right',fontsize=16)
#plt.title('')
plt.tight_layout()
plt.show()   
plt.clf()


# Category 3) 15 combinations of electrodes = 15 barplots 

# All electrode combinations: 

# F3M2F4M1 (done)
# F3M2C3M2 (done)
# F3M2C4M1 (done)
# F3M2O2M1 (done)
# F3M2O1M2 (done)
# F4M1C3M2 (done)
# F4M1C4M1 (done)
# F4M1O2M1 (done)
# F4M1O1M2 (done)
# C3M2C4M1 (done)
# C3M2O1M2 (done)
# C3M2O2M1 (done)
# C4M1O2M1 (done)
# C4M1O1M2 (done)
# O1M2O2M1 (done)


temp_idx_F3M2F4M1_coherence=[]
temp_idx_F3M2C3M2_coherence=[]
temp_idx_F3M2C4M1_coherence=[]
temp_idx_F3M2O2M1_coherence=[]
temp_idx_F3M2O1M2_coherence=[]
temp_idx_F4M1C3M2_coherence=[]
temp_idx_F4M1C4M1_coherence=[]
temp_idx_F4M1O2M1_coherence=[]
temp_idx_F4M1O1M2_coherence=[]
temp_idx_C3M2C4M1_coherence=[]
temp_idx_C3M2O1M2_coherence=[]
temp_idx_C3M2O2M1_coherence=[]
temp_idx_C4M1O2M1_coherence=[]
temp_idx_C4M1O1M2_coherence=[]
temp_idx_O1M2O2M1_coherence=[]

#### F3M2F4M1 ##########

# Extracting all coherence features with 'F3M2F4M1' in the name
F3M2F4M1=df_coherence.filter(like='F3M2F4M1', axis=1) 
print(F3M2F4M1.shape[1])
print(F3M2F4M1)


# In this loop the indexes of the F3M2F4M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F3M2F4M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F3M2F4M1.columns[i]

    # Using the feature names from 'F3M2F4M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F3M2F4M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_F3M2F4M1_coherence.append(original_idx_F3M2F4M1)


print('List with indexes for categorized F3M2F4M1 features')
#print(temp_idx_F3M2F4M1_coherence)
F3M2F4M1_coherence=np.stack(temp_idx_F3M2F4M1_coherence,axis=0)

print(F3M2F4M1_coherence.shape[0])

# Indexing in the importance array to get the F3M2F4M1 coherence importance values and summing the values to one value for this category
F3M2F4M1_importance=np.sum(importance_array[F3M2F4M1_coherence])

print(F3M2F4M1_importance)





#### F3M2C3M2 ##########

# Extracting all coherence features with 'F3M2C3M2' in the name
F3M2C3M2=df_coherence.filter(like='F3M2C3M2', axis=1) 
print(F3M2C3M2.shape[1])
print(F3M2C3M2)


# In this loop the indexes of the F3M2C3M2 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F3M2C3M2.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F3M2C3M2.columns[i]

    # Using the feature names from 'F3M2C3M2' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F3M2C3M2=X_matrix_df.columns.get_loc(idx)

    temp_idx_F3M2C3M2_coherence.append(original_idx_F3M2C3M2)


print('List with indexes for categorized F3M2C3M2 features')
#print(temp_idx_F3M2C3M2_coherence)
F3M2C3M2_coherence=np.stack(temp_idx_F3M2C3M2_coherence,axis=0)

print(F3M2C3M2_coherence.shape[0])

# Indexing in the importance array to get the F3M2C3M2 coherence importance values and summing the values to one value for this category
F3M2C3M2_importance=np.sum(importance_array[F3M2C3M2_coherence])

print(F3M2C3M2_importance)





#### F3M2C4M1 ##########

# Extracting all coherence features with 'F3M2C4M1' in the name
F3M2C4M1=df_coherence.filter(like='F3M2C4M1', axis=1) 
print(F3M2C4M1.shape[1])
print(F3M2C4M1)


# In this loop the indexes of the F3M2C4M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F3M2C4M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F3M2C4M1.columns[i]

    # Using the feature names from 'F3M2C4M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F3M2C4M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_F3M2C4M1_coherence.append(original_idx_F3M2C4M1)


print('List with indexes for categorized F3M2C4M1 features')
#print(temp_idx_F3M2C4M1_coherence)
F3M2C4M1_coherence=np.stack(temp_idx_F3M2C4M1_coherence,axis=0)

print(F3M2C4M1_coherence.shape[0])

# Indexing in the importance array to get the F3M2C4M1 coherence importance values and summing the values to one value for this category
F3M2C4M1_importance=np.sum(importance_array[F3M2C4M1_coherence])

print(F3M2C4M1_importance)






#### F3M2O2M1 ##########

# Extracting all coherence features with 'F3M2O2M1' in the name
F3M2O2M1=df_coherence.filter(like='F3M2O2M1', axis=1) 
print(F3M2O2M1.shape[1])
print(F3M2O2M1)


# In this loop the indexes of the F3M2O2M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F3M2O2M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F3M2O2M1.columns[i]

    # Using the feature names from 'F3M2O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F3M2O2M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_F3M2O2M1_coherence.append(original_idx_F3M2O2M1)


print('List with indexes for categorized F3M2O2M1 features')
#print(temp_idx_F3M2O2M1_coherence)
F3M2O2M1_coherence=np.stack(temp_idx_F3M2O2M1_coherence,axis=0)

print(F3M2O2M1_coherence.shape[0])

# Indexing in the importance array to get the F3M2O2M1 coherence importance values and summing the values to one value for this category
F3M2O2M1_importance=np.sum(importance_array[F3M2O2M1_coherence])

print(F3M2O2M1_importance)






#### F3M2O1M2 ##########

# Extracting all coherence features with 'F3M2O1M2' in the name
F3M2O1M2=df_coherence.filter(like='F3M2O1M2', axis=1) 
print(F3M2O1M2.shape[1])
print(F3M2O1M2)


# In this loop the indexes of the F3M2O1M2 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F3M2O1M2.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F3M2O1M2.columns[i]

    # Using the feature names from 'F3M2O1M2' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F3M2O1M2=X_matrix_df.columns.get_loc(idx)

    temp_idx_F3M2O1M2_coherence.append(original_idx_F3M2O1M2)


print('List with indexes for categorized F3M2O1M2 features')
#print(temp_idx_F3M2O1M2_coherence)
F3M2O1M2_coherence=np.stack(temp_idx_F3M2O1M2_coherence,axis=0)

print(F3M2O1M2_coherence.shape[0])

# Indexing in the importance array to get the F3M2O1M2 coherence importance values and summing the values to one value for this category
F3M2O1M2_importance=np.sum(importance_array[F3M2O1M2_coherence])

print(F3M2O1M2_importance)




#### F4M1C3M2 ##########

# Extracting all coherence features with 'F4M1C3M2' in the name
F4M1C3M2=df_coherence.filter(like='F4M1C3M2', axis=1) 
print(F4M1C3M2.shape[1])
print(F4M1C3M2)


# In this loop the indexes of the F4M1C3M2 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F4M1C3M2.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F4M1C3M2.columns[i]

    # Using the feature names from 'F4M1C3M2' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F4M1C3M2=X_matrix_df.columns.get_loc(idx)

    temp_idx_F4M1C3M2_coherence.append(original_idx_F4M1C3M2)


print('List with indexes for categorized F4M1C3M2 features')
#print(temp_idx_F4M1C3M2_coherence)
F4M1C3M2_coherence=np.stack(temp_idx_F4M1C3M2_coherence,axis=0)

print(F4M1C3M2_coherence.shape[0])

# Indexing in the importance array to get the F4M1C3M2 coherence importance values and summing the values to one value for this category
F4M1C3M2_importance=np.sum(importance_array[F4M1C3M2_coherence])

print(F4M1C3M2_importance)





#### F4M1C4M1 ##########

# Extracting all coherence features with 'F4M1C4M1' in the name
F4M1C4M1=df_coherence.filter(like='F4M1C4M1', axis=1) 
print(F4M1C4M1.shape[1])
print(F4M1C4M1)


# In this loop the indexes of the F4M1C4M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F4M1C4M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F4M1C4M1.columns[i]

    # Using the feature names from 'F4M1C4M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F4M1C4M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_F4M1C4M1_coherence.append(original_idx_F4M1C4M1)


print('List with indexes for categorized F4M1C4M1 features')
#print(temp_idx_F4M1C4M1_coherence)
F4M1C4M1_coherence=np.stack(temp_idx_F4M1C4M1_coherence,axis=0)

print(F4M1C4M1_coherence.shape[0])

# Indexing in the importance array to get the F4M1C4M1 coherence importance values and summing the values to one value for this category
F4M1C4M1_importance=np.sum(importance_array[F4M1C4M1_coherence])

print(F4M1C4M1_importance)





#### F4M1O2M1 ##########

# Extracting all coherence features with 'F4M1O2M1' in the name
F4M1O2M1=df_coherence.filter(like='F4M1O2M1', axis=1) 
print(F4M1O2M1.shape[1])
print(F4M1O2M1)


# In this loop the indexes of the F4M1O2M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F4M1O2M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F4M1O2M1.columns[i]

    # Using the feature names from 'F4M1O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F4M1O2M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_F4M1O2M1_coherence.append(original_idx_F4M1O2M1)


print('List with indexes for categorized F4M1O2M1 features')
#print(temp_idx_F4M1O2M1_coherence)
F4M1O2M1_coherence=np.stack(temp_idx_F4M1O2M1_coherence,axis=0)

print(F4M1O2M1_coherence.shape[0])

# Indexing in the importance array to get the F4M1O2M1 coherence importance values and summing the values to one value for this category
F4M1O2M1_importance=np.sum(importance_array[F4M1O2M1_coherence])

print(F4M1O2M1_importance)





#### F4M1O1M2 ##########

# Extracting all coherence features with 'F4M1O1M2' in the name
F4M1O1M2=df_coherence.filter(like='F4M1O1M2', axis=1) 
print(F4M1O1M2.shape[1])
print(F4M1O1M2)


# In this loop the indexes of the F4M1O1M2 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F4M1O1M2.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F4M1O1M2.columns[i]

    # Using the feature names from 'F4M1O1M2' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F4M1O1M2=X_matrix_df.columns.get_loc(idx)

    temp_idx_F4M1O1M2_coherence.append(original_idx_F4M1O1M2)


print('List with indexes for categorized F4M1O1M2 features')
#print(temp_idx_F4M1O1M2_coherence)
F4M1O1M2_coherence=np.stack(temp_idx_F4M1O1M2_coherence,axis=0)

print(F4M1O1M2_coherence.shape[0])

# Indexing in the importance array to get the F4M1O1M2 coherence importance values and summing the values to one value for this category
F4M1O1M2_importance=np.sum(importance_array[F4M1O1M2_coherence])

print(F4M1O1M2_importance)




#### C3M2C4M1 ##########

# Extracting all coherence features with 'C3M2C4M1' in the name
C3M2C4M1=df_coherence.filter(like='C3M2C4M1', axis=1) 
print(C3M2C4M1.shape[1])
print(C3M2C4M1)


# In this loop the indexes of the C3M2C4M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(C3M2C4M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=C3M2C4M1.columns[i]

    # Using the feature names from 'C3M2C4M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_C3M2C4M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_C3M2C4M1_coherence.append(original_idx_C3M2C4M1)


print('List with indexes for categorized C3M2C4M1 features')
#print(temp_idx_C3M2C4M1_coherence)
C3M2C4M1_coherence=np.stack(temp_idx_C3M2C4M1_coherence,axis=0)

print(C3M2C4M1_coherence.shape[0])

# Indexing in the importance array to get the C3M2C4M1 coherence importance values and summing the values to one value for this category
C3M2C4M1_importance=np.sum(importance_array[C3M2C4M1_coherence])

print(C3M2C4M1_importance)




#### C3M2O1M2 ##########

# Extracting all coherence features with 'C3M2O1M2' in the name
C3M2O1M2=df_coherence.filter(like='C3M2O1M2', axis=1) 
print(C3M2O1M2.shape[1])
print(C3M2O1M2)


# In this loop the indexes of the C3M2O1M2 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(C3M2O1M2.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=C3M2O1M2.columns[i]

    # Using the feature names from 'C3M2O1M2' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_C3M2O1M2=X_matrix_df.columns.get_loc(idx)

    temp_idx_C3M2O1M2_coherence.append(original_idx_C3M2O1M2)


print('List with indexes for categorized C3M2O1M2 features')
#print(temp_idx_C3M2O1M2_coherence)
C3M2O1M2_coherence=np.stack(temp_idx_C3M2O1M2_coherence,axis=0)

print(C3M2O1M2_coherence.shape[0])

# Indexing in the importance array to get the C3M2O1M2 coherence importance values and summing the values to one value for this category
C3M2O1M2_importance=np.sum(importance_array[C3M2O1M2_coherence])

print(C3M2O1M2_importance)





#### C3M2O2M1 ##########

# Extracting all coherence features with 'C3M2O2M1' in the name
C3M2O2M1=df_coherence.filter(like='C3M2O2M1', axis=1) 
print(C3M2O2M1.shape[1])
print(C3M2O2M1)


# In this loop the indexes of the C3M2O2M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(C3M2O2M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=C3M2O2M1.columns[i]

    # Using the feature names from 'C3M2O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_C3M2O2M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_C3M2O2M1_coherence.append(original_idx_C3M2O2M1)


print('List with indexes for categorized C3M2O2M1 features')
#print(temp_idx_C3M2O2M1_coherence)
C3M2O2M1_coherence=np.stack(temp_idx_C3M2O2M1_coherence,axis=0)

print(C3M2O2M1_coherence.shape[0])

# Indexing in the importance array to get the C3M2O2M1 coherence importance values and summing the values to one value for this category
C3M2O2M1_importance=np.sum(importance_array[C3M2O2M1_coherence])

print(C3M2O2M1_importance)






#### C4M1O2M1 ##########

# Extracting all coherence features with 'C4M1O2M1' in the name
C4M1O2M1=df_coherence.filter(like='C4M1O2M1', axis=1) 
print(C4M1O2M1.shape[1])
print(C4M1O2M1)


# In this loop the indexes of the C4M1O2M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(C4M1O2M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=C4M1O2M1.columns[i]

    # Using the feature names from 'C4M1O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_C4M1O2M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_C4M1O2M1_coherence.append(original_idx_C4M1O2M1)


print('List with indexes for categorized C4M1O2M1 features')
#print(temp_idx_C4M1O2M1_coherence)
C4M1O2M1_coherence=np.stack(temp_idx_C4M1O2M1_coherence,axis=0)

print(C4M1O2M1_coherence.shape[0])

# Indexing in the importance array to get the C4M1O2M1 coherence importance values and summing the values to one value for this category
C4M1O2M1_importance=np.sum(importance_array[C4M1O2M1_coherence])

print(C4M1O2M1_importance)






#### C4M1O1M2 ##########

# Extracting all coherence features with 'C4M1O1M2' in the name
C4M1O1M2=df_coherence.filter(like='C4M1O1M2', axis=1) 
print(C4M1O1M2.shape[1])
print(C4M1O1M2)


# In this loop the indexes of the C4M1O1M2 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(C4M1O1M2.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=C4M1O1M2.columns[i]

    # Using the feature names from 'C4M1O1M2' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_C4M1O1M2=X_matrix_df.columns.get_loc(idx)

    temp_idx_C4M1O1M2_coherence.append(original_idx_C4M1O1M2)


print('List with indexes for categorized C4M1O1M2 features')
#print(temp_idx_C4M1O1M2_coherence)
C4M1O1M2_coherence=np.stack(temp_idx_C4M1O1M2_coherence,axis=0)

print(C4M1O1M2_coherence.shape[0])

# Indexing in the importance array to get the C4M1O1M2 coherence importance values and summing the values to one value for this category
C4M1O1M2_importance=np.sum(importance_array[C4M1O1M2_coherence])

print(C4M1O1M2_importance)






#### O1M2O2M1 ##########

# Extracting all coherence features with 'O1M2O2M1' in the name
O1M2O2M1=df_coherence.filter(like='O1M2O2M1', axis=1) 
print(O1M2O2M1.shape[1])
print(O1M2O2M1)


# In this loop the indexes of the O1M2O2M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(O1M2O2M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=O1M2O2M1.columns[i]

    # Using the feature names from 'O1M2O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_O1M2O2M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_O1M2O2M1_coherence.append(original_idx_O1M2O2M1)


print('List with indexes for categorized O1M2O2M1 features')
#print(temp_idx_O1M2O2M1_coherence)
O1M2O2M1_coherence=np.stack(temp_idx_O1M2O2M1_coherence,axis=0)

print(O1M2O2M1_coherence.shape[0])

# Indexing in the importance array to get the O1M2O2M1 coherence importance values and summing the values to one value for this category
O1M2O2M1_importance=np.sum(importance_array[O1M2O2M1_coherence])

print(O1M2O2M1_importance)






### Gathering the frequency importance values in a df for plotting ########

# Saving values gathered for electrodes
print('list with features - electrodes')

electrodes_values_list=[F3M2F4M1_importance, F3M2C3M2_importance, F3M2C4M1_importance,F3M2O2M1_importance,F3M2O1M2_importance,F4M1C3M2_importance,F4M1C4M1_importance,F4M1O2M1_importance, F4M1O1M2_importance,C3M2C4M1_importance, C3M2O1M2_importance, C3M2O2M1_importance, C4M1O2M1_importance,C4M1O1M2_importance,O1M2O2M1_importance]
features_electrodes=np.stack(electrodes_values_list)
features_electrodes=np.reshape(features_electrodes,(1,15)) # reshape
print(features_electrodes)
print(type(features_electrodes))
print(features_electrodes.shape)




# column names 
electrodes_columns=['F3M2F4M1','F3M2C3M2','F3M2C4M1','F3M2O2M1','F3M2O1M2','F4M1C3M2','F4M1C4M1','F4M1O2M1','F4M1O1M2','C3M2C4M1','C3M2O1M2','C3M2O2M1','C4M1O2M1','C4M1O1M2','O1M2O2M1']
print(type(electrodes_columns))
#print(electrodes_columns)


# Creating dataframe with values for plotting 
df_plt_electrodes=pd.DataFrame(features_electrodes, columns=electrodes_columns)
print(df_plt_electrodes)


boxplot_features=electrodes_columns
plt.suptitle('Coherence electrode feature importance',fontsize=20)
plt.subplot(1, 1, 1) # Adjust these numbers as per your requirement.
sns.barplot(data=df_plt_electrodes,errorbar=None)
plt.xticks(rotation=45, ha='right',fontsize=16)
#plt.title('')
plt.tight_layout()
plt.show()   
plt.clf()

del F3M2F4M1_importance, F3M2C3M2_importance, F3M2C4M1_importance,F3M2O2M1_importance,F3M2O1M2_importance,F4M1C3M2_importance,F4M1C4M1_importance,F4M1O2M1_importance, F4M1O1M2_importance,C3M2C4M1_importance, C3M2O1M2_importance, C3M2O2M1_importance, C4M1O2M1_importance,C4M1O1M2_importance,O1M2O2M1_importance



########## Correlation feature plots ############################

# Category plots 

# Category 1) Epoch 1, 3, 5, 15, 30 


temp_idx_epocssize_1_correlation=[]
temp_idx_epocssize_3_correlation=[]
temp_idx_epocssize_5_correlation=[]
temp_idx_epocssize_15_correlation=[]
temp_idx_epocssize_30_correlation=[]

#### Epoch 1s ##########

# Extracting all coherence features with 'Epoch 1s' in the name
epocssize_1=df_correlation.filter(like='epocssize_1_', axis=1) 
print(epocssize_1.shape[1])
print(epocssize_1)


# In this loop the indexes of the epocssize_1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(epocssize_1.shape[1]):
    
    # Indexing in epocssize_1 category df and getting feature names 
    idx=epocssize_1.columns[i]

    # Using the feature names from 'epocssize_1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_epocssize_1=X_matrix_df.columns.get_loc(idx)

    temp_idx_epocssize_1_correlation.append(original_idx_epocssize_1)


print('List with indexes for categorized epocssize_1 features')
#print(temp_idx_epocssize_1_correlation)
epocssize_1_correlation=np.stack(temp_idx_epocssize_1_correlation,axis=0)

print(epocssize_1_correlation.shape[0])

# Indexing in the importance array to get the epocssize_1 correlation importance values and summing the values to one value for this category
epocssize_1_importance=np.sum(importance_array[epocssize_1_correlation])

print(epocssize_1_importance)




#### Epoch 3s ##########

# Extracting all coherence features with 'Epoch 3s' in the name
epocssize_3=df_correlation.filter(like='epocssize_3_', axis=1) 
print(epocssize_3.shape[1])
print(epocssize_3)


# In this loop the indexes of the epocssize_3 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(epocssize_3.shape[1]):
    
    # Indexing in epocssize_1 category df and getting feature names 
    idx=epocssize_3.columns[i]

    # Using the feature names from 'epocssize_1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_epocssize_3=X_matrix_df.columns.get_loc(idx)

    temp_idx_epocssize_3_correlation.append(original_idx_epocssize_3)


print('List with indexes for categorized epocssize_3 features')
#print(temp_idx_epocssize_3_correlation)
epocssize_3_correlation=np.stack(temp_idx_epocssize_3_correlation,axis=0)

print(epocssize_3_correlation.shape[0])

# Indexing in the importance array to get the epocssize_3 correlation importance values and summing the values to one value for this category
epocssize_3_importance=np.sum(importance_array[epocssize_3_correlation])

print(epocssize_3_importance)



#### Epoch 5s ##########

# Extracting all coherence features with 'Epoch 1s' in the name
epocssize_5=df_correlation.filter(like='epocssize_5', axis=1) 
print(epocssize_5.shape[1])
print(epocssize_5)


# In this loop the indexes of the epocssize_5 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(epocssize_5.shape[1]):
    
    # Indexing in epocssize_1 category df and getting feature names 
    idx=epocssize_5.columns[i]

    # Using the feature names from 'epocssize_1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_epocssize_5=X_matrix_df.columns.get_loc(idx)

    temp_idx_epocssize_5_correlation.append(original_idx_epocssize_5)


print('List with indexes for categorized epocssize_5 features')
#print(temp_idx_epocssize_5_correlation)
epocssize_5_correlation=np.stack(temp_idx_epocssize_5_correlation,axis=0)

print(epocssize_5_correlation.shape[0])

# Indexing in the importance array to get the epocssize_1 correlation importance values and summing the values to one value for this category
epocssize_5_importance=np.sum(importance_array[epocssize_5_correlation])

print(epocssize_5_importance)



#### Epoch 15s ##########

# Extracting all coherence features with 'Epoch 15s' in the name
epocssize_15=df_correlation.filter(like='epocssize_15', axis=1) 
print(epocssize_15.shape[1])
print(epocssize_15)


# In this loop the indexes of the epocssize_15 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(epocssize_15.shape[1]):
    
    # Indexing in epocssize_15 category df and getting feature names 
    idx=epocssize_15.columns[i]

    # Using the feature names from 'epocssize_15' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_epocssize_15=X_matrix_df.columns.get_loc(idx)

    temp_idx_epocssize_15_correlation.append(original_idx_epocssize_15)


print('List with indexes for categorized epocssize_15 features')
#print(temp_idx_epocssize_15_correlation)
epocssize_15_correlation=np.stack(temp_idx_epocssize_15_correlation,axis=0)

print(epocssize_15_correlation.shape[0])

# Indexing in the importance array to get the epocssize_15 correlation importance values and summing the values to one value for this category
epocssize_15_importance=np.sum(importance_array[epocssize_15_correlation])

print(epocssize_15_importance)



#### Epoch 30s ##########

# Extracting all coherence features with 'Epoch 30s' in the name
epocssize_30=df_correlation.filter(like='epocssize_30', axis=1) 
print(epocssize_30.shape[1])
print(epocssize_30)


# In this loop the indexes of the epocssize_30 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(epocssize_30.shape[1]):
    
    # Indexing in epocssize_1 category df and getting feature names 
    idx=epocssize_30.columns[i]

    # Using the feature names from 'epocssize_30' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_epocssize_30=X_matrix_df.columns.get_loc(idx)

    temp_idx_epocssize_30_correlation.append(original_idx_epocssize_30)


print('List with indexes for categorized epocssize_30 features')
#print(temp_idx_epocssize_30_correlation)
epocssize_30_correlation=np.stack(temp_idx_epocssize_30_correlation,axis=0)

print(epocssize_30_correlation.shape[0])

# Indexing in the importance array to get the epocssize_30 correlation importance values and summing the values to one value for this category
epocssize_30_importance=np.sum(importance_array[epocssize_30_correlation])

print(epocssize_30_importance)




### Gathering the frequency importance values in a df for plotting ########

# Saving values gathered for electrodes
print('list with features - electrodes')

epoch_values_list=[epocssize_1_importance, epocssize_3_importance, epocssize_5_importance, epocssize_15_importance, epocssize_30_importance]
features_epoch=np.stack(epoch_values_list)
features_epoch=np.reshape(features_epoch,(1,5)) # reshape
print(features_epoch)
print(type(features_epoch))
print(features_epoch.shape)




# column names 
epoch_columns=['Epoch 1s', 'Epoch 3s','Epoch 5s','Epoch 15s','Epoch 30s']
print(type(epoch_columns))
#print(electrodes_columns)


# Creating dataframe with values for plotting 
df_plt_epochsize=pd.DataFrame(features_epoch, columns=epoch_columns)
print(df_plt_epochsize)


boxplot_features=epoch_columns
plt.suptitle('Correlation epoch size feature importance',fontsize=20)
plt.subplot(1, 1, 1) # Adjust these numbers as per your requirement.
sns.barplot(data=df_plt_epochsize,errorbar=None)
plt.xticks(rotation=45, ha='right',fontsize=16)
#plt.title('')
plt.tight_layout()
plt.show()   
plt.clf()






# Category 2) 15 combinations of electrodes = 15 barplots 

# All electrode combinations: 

# F3M2F4M1 (done)
# F3M2C3M2 (done)
# F3M2C4M1 (done)
# F3M2O2M1 (done)
# F3M2O1M2 (done)
# F4M1C3M2 (done)
# F4M1C4M1 (done)
# F4M1O2M1 (done)
# F4M1O1M2 (done)
# C3M2C4M1 (done)
# C3M2O1M2 (done)
# C3M2O2M1 (done)
# C4M1O2M1 (done)
# C4M1O1M2 (done)
# O1M2O2M1 (done)


temp_idx_F3M2F4M1_correlation=[]
temp_idx_F3M2C3M2_correlation=[]
temp_idx_F3M2C4M1_correlation=[]
temp_idx_F3M2O2M1_correlation=[]
temp_idx_F3M2O1M2_correlation=[]
temp_idx_F4M1C3M2_correlation=[]
temp_idx_F4M1C4M1_correlation=[]
temp_idx_F4M1O2M1_correlation=[]
temp_idx_F4M1O1M2_correlation=[]
temp_idx_C3M2C4M1_correlation=[]
temp_idx_C3M2O1M2_correlation=[]
temp_idx_C3M2O2M1_correlation=[]
temp_idx_C4M1O2M1_correlation=[]
temp_idx_C4M1O1M2_correlation=[]
temp_idx_O1M2O2M1_correlation=[]

#### F3M2F4M1 ##########

# Extracting all correlation features with 'F3M2F4M1' in the name
F3M2F4M1=df_correlation.filter(like='F3M2F4M1', axis=1) 
print(F3M2F4M1.shape[1])
print(F3M2F4M1)


# In this loop the indexes of the F3M2F4M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F3M2F4M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F3M2F4M1.columns[i]

    # Using the feature names from 'F3M2F4M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F3M2F4M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_F3M2F4M1_correlation.append(original_idx_F3M2F4M1)


print('List with indexes for categorized F3M2F4M1 features')
#print(temp_idx_F3M2F4M1_correlation)
F3M2F4M1_correlation=np.stack(temp_idx_F3M2F4M1_coherence,axis=0)

print(F3M2F4M1_correlation.shape[0])

# Indexing in the importance array to get the F3M2F4M1 correlation importance values and summing the values to one value for this category
F3M2F4M1_importance=np.sum(importance_array[F3M2F4M1_correlation])

print(F3M2F4M1_importance)





#### F3M2C3M2 ##########

# Extracting all correlation features with 'F3M2C3M2' in the name
F3M2C3M2=df_correlation.filter(like='F3M2C3M2', axis=1) 
print(F3M2C3M2.shape[1])
print(F3M2C3M2)


# In this loop the indexes of the F3M2C3M2 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F3M2C3M2.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F3M2C3M2.columns[i]

    # Using the feature names from 'F3M2C3M2' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F3M2C3M2=X_matrix_df.columns.get_loc(idx)

    temp_idx_F3M2C3M2_correlation.append(original_idx_F3M2C3M2)


print('List with indexes for categorized F3M2C3M2 features')
#print(temp_idx_F3M2C3M2_coherence)
F3M2C3M2_correlation=np.stack(temp_idx_F3M2C3M2_correlation,axis=0)

print(F3M2C3M2_correlation.shape[0])

# Indexing in the importance array to get the F3M2C3M2 correlation importance values and summing the values to one value for this category
F3M2C3M2_importance=np.sum(importance_array[F3M2C3M2_correlation])

print(F3M2C3M2_importance)





#### F3M2C4M1 ##########

# Extracting all correlation features with 'F3M2C4M1' in the name
F3M2C4M1=df_correlation.filter(like='F3M2C4M1', axis=1) 
print(F3M2C4M1.shape[1])
print(F3M2C4M1)


# In this loop the indexes of the F3M2C4M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F3M2C4M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F3M2C4M1.columns[i]

    # Using the feature names from 'F3M2C4M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F3M2C4M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_F3M2C4M1_correlation.append(original_idx_F3M2C4M1)


print('List with indexes for categorized F3M2C4M1 features')
#print(temp_idx_F3M2C4M1_coherence)
F3M2C4M1_correlation=np.stack(temp_idx_F3M2C4M1_correlation,axis=0)

print(F3M2C4M1_correlation.shape[0])

# Indexing in the importance array to get the F3M2C4M1 correlation importance values and summing the values to one value for this category
F3M2C4M1_importance=np.sum(importance_array[F3M2C4M1_correlation])

print(F3M2C4M1_importance)






#### F3M2O2M1 ##########

# Extracting all correlation features with 'F3M2O2M1' in the name
F3M2O2M1=df_correlation.filter(like='F3M2O2M1', axis=1) 
print(F3M2O2M1.shape[1])
print(F3M2O2M1)


# In this loop the indexes of the F3M2O2M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F3M2O2M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F3M2O2M1.columns[i]

    # Using the feature names from 'F3M2O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F3M2O2M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_F3M2O2M1_correlation.append(original_idx_F3M2O2M1)


print('List with indexes for categorized F3M2O2M1 features')
#print(temp_idx_F3M2O2M1_correlation)
F3M2O2M1_correlation=np.stack(temp_idx_F3M2O2M1_correlation,axis=0)

print(F3M2O2M1_correlation.shape[0])

# Indexing in the importance array to get the F3M2O2M1 correlation importance values and summing the values to one value for this category
F3M2O2M1_importance=np.sum(importance_array[F3M2O2M1_correlation])

print(F3M2O2M1_importance)






#### F3M2O1M2 ##########

# Extracting all correlation features with 'F3M2O1M2' in the name
F3M2O1M2=df_correlation.filter(like='F3M2O1M2', axis=1) 
print(F3M2O1M2.shape[1])
print(F3M2O1M2)


# In this loop the indexes of the F3M2O1M2 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F3M2O1M2.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F3M2O1M2.columns[i]

    # Using the feature names from 'F3M2O1M2' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F3M2O1M2=X_matrix_df.columns.get_loc(idx)

    temp_idx_F3M2O1M2_correlation.append(original_idx_F3M2O1M2)


print('List with indexes for categorized F3M2O1M2 features')
#print(temp_idx_F3M2O1M2_correlation)
F3M2O1M2_correlation=np.stack(temp_idx_F3M2O1M2_correlation,axis=0)

print(F3M2O1M2_correlation.shape[0])

# Indexing in the importance array to get the F3M2O1M2 correlation importance values and summing the values to one value for this category
F3M2O1M2_importance=np.sum(importance_array[F3M2O1M2_correlation])

print(F3M2O1M2_importance)




#### F4M1C3M2 ##########

# Extracting all correlation features with 'F4M1C3M2' in the name
F4M1C3M2=df_correlation.filter(like='F4M1C3M2', axis=1) 
print(F4M1C3M2.shape[1])
print(F4M1C3M2)


# In this loop the indexes of the F4M1C3M2 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F4M1C3M2.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F4M1C3M2.columns[i]

    # Using the feature names from 'F4M1C3M2' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F4M1C3M2=X_matrix_df.columns.get_loc(idx)

    temp_idx_F4M1C3M2_correlation.append(original_idx_F4M1C3M2)


print('List with indexes for categorized F4M1C3M2 features')
#print(temp_idx_F4M1C3M2_correlation)
F4M1C3M2_correlation=np.stack(temp_idx_F4M1C3M2_correlation,axis=0)

print(F4M1C3M2_correlation.shape[0])

# Indexing in the importance array to get the F4M1C3M2 correlation importance values and summing the values to one value for this category
F4M1C3M2_importance=np.sum(importance_array[F4M1C3M2_correlation])

print(F4M1C3M2_importance)





#### F4M1C4M1 ##########

# Extracting all correlation features with 'F4M1C4M1' in the name
F4M1C4M1=df_correlation.filter(like='F4M1C4M1', axis=1) 
print(F4M1C4M1.shape[1])
print(F4M1C4M1)


# In this loop the indexes of the F4M1C4M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F4M1C4M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F4M1C4M1.columns[i]

    # Using the feature names from 'F4M1C4M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F4M1C4M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_F4M1C4M1_correlation.append(original_idx_F4M1C4M1)


print('List with indexes for categorized F4M1C4M1 features')
#print(temp_idx_F4M1C4M1_correlation)
F4M1C4M1_correlation=np.stack(temp_idx_F4M1C4M1_correlation,axis=0)

print(F4M1C4M1_correlation.shape[0])

# Indexing in the importance array to get the F4M1C4M1 correlation importance values and summing the values to one value for this category
F4M1C4M1_importance=np.sum(importance_array[F4M1C4M1_correlation])

print(F4M1C4M1_importance)





#### F4M1O2M1 ##########

# Extracting all correlation features with 'F4M1O2M1' in the name
F4M1O2M1=df_correlation.filter(like='F4M1O2M1', axis=1) 
print(F4M1O2M1.shape[1])
print(F4M1O2M1)


# In this loop the indexes of the F4M1O2M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F4M1O2M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F4M1O2M1.columns[i]

    # Using the feature names from 'F4M1O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F4M1O2M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_F4M1O2M1_correlation.append(original_idx_F4M1O2M1)


print('List with indexes for categorized F4M1O2M1 features')
#print(temp_idx_F4M1O2M1_correlation)
F4M1O2M1_correlation=np.stack(temp_idx_F4M1O2M1_correlation,axis=0)

print(F4M1O2M1_correlation.shape[0])

# Indexing in the importance array to get the F4M1O2M1 correlation importance values and summing the values to one value for this category
F4M1O2M1_importance=np.sum(importance_array[F4M1O2M1_correlation])

print(F4M1O2M1_importance)





#### F4M1O1M2 ##########

# Extracting all correlation features with 'F4M1O1M2' in the name
F4M1O1M2=df_correlation.filter(like='F4M1O1M2', axis=1) 
print(F4M1O1M2.shape[1])
print(F4M1O1M2)


# In this loop the indexes of the F4M1O1M2 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(F4M1O1M2.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=F4M1O1M2.columns[i]

    # Using the feature names from 'F4M1O1M2' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_F4M1O1M2=X_matrix_df.columns.get_loc(idx)

    temp_idx_F4M1O1M2_correlation.append(original_idx_F4M1O1M2)


print('List with indexes for categorized F4M1O1M2 features')
#print(temp_idx_F4M1O1M2_correlation)
F4M1O1M2_correlation=np.stack(temp_idx_F4M1O1M2_correlation,axis=0)

print(F4M1O1M2_correlation.shape[0])

# Indexing in the importance array to get the F4M1O1M2 correlation importance values and summing the values to one value for this category
F4M1O1M2_importance=np.sum(importance_array[F4M1O1M2_correlation])

print(F4M1O1M2_importance)




#### C3M2C4M1 ##########

# Extracting all correlation features with 'C3M2C4M1' in the name
C3M2C4M1=df_correlation.filter(like='C3M2C4M1', axis=1) 
print(C3M2C4M1.shape[1])
print(C3M2C4M1)


# In this loop the indexes of the C3M2C4M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(C3M2C4M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=C3M2C4M1.columns[i]

    # Using the feature names from 'C3M2C4M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_C3M2C4M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_C3M2C4M1_correlation.append(original_idx_C3M2C4M1)


print('List with indexes for categorized C3M2C4M1 features')
#print(temp_idx_C3M2C4M1_correlation)
C3M2C4M1_correlation=np.stack(temp_idx_C3M2C4M1_correlation,axis=0)

print(C3M2C4M1_correlation.shape[0])

# Indexing in the importance array to get the C3M2C4M1 correlation importance values and summing the values to one value for this category
C3M2C4M1_importance=np.sum(importance_array[C3M2C4M1_correlation])

print(C3M2C4M1_importance)




#### C3M2O1M2 ##########

# Extracting all correlation features with 'C3M2O1M2' in the name
C3M2O1M2=df_correlation.filter(like='C3M2O1M2', axis=1) 
print(C3M2O1M2.shape[1])
print(C3M2O1M2)


# In this loop the indexes of the C3M2O1M2 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(C3M2O1M2.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=C3M2O1M2.columns[i]

    # Using the feature names from 'C3M2O1M2' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_C3M2O1M2=X_matrix_df.columns.get_loc(idx)

    temp_idx_C3M2O1M2_correlation.append(original_idx_C3M2O1M2)


print('List with indexes for categorized C3M2O1M2 features')
#print(temp_idx_C3M2O1M2_correlation)
C3M2O1M2_correlation=np.stack(temp_idx_C3M2O1M2_correlation,axis=0)

print(C3M2O1M2_correlation.shape[0])

# Indexing in the importance array to get the C3M2O1M2 correlation importance values and summing the values to one value for this category
C3M2O1M2_importance=np.sum(importance_array[C3M2O1M2_correlation])

print(C3M2O1M2_importance)





#### C3M2O2M1 ##########

# Extracting all correlation features with 'C3M2O2M1' in the name
C3M2O2M1=df_correlation.filter(like='C3M2O2M1', axis=1) 
print(C3M2O2M1.shape[1])
print(C3M2O2M1)


# In this loop the indexes of the C3M2O2M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(C3M2O2M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=C3M2O2M1.columns[i]

    # Using the feature names from 'C3M2O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_C3M2O2M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_C3M2O2M1_correlation.append(original_idx_C3M2O2M1)


print('List with indexes for categorized C3M2O2M1 features')
#print(temp_idx_C3M2O2M1_correlation)
C3M2O2M1_correlation=np.stack(temp_idx_C3M2O2M1_correlation,axis=0)

print(C3M2O2M1_correlation.shape[0])

# Indexing in the importance array to get the C3M2O2M1 correlation importance values and summing the values to one value for this category
C3M2O2M1_importance=np.sum(importance_array[C3M2O2M1_correlation])

print(C3M2O2M1_importance)






#### C4M1O2M1 ##########

# Extracting all correlation features with 'C4M1O2M1' in the name
C4M1O2M1=df_correlation.filter(like='C4M1O2M1', axis=1) 
print(C4M1O2M1.shape[1])
print(C4M1O2M1)


# In this loop the indexes of the C4M1O2M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(C4M1O2M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=C4M1O2M1.columns[i]

    # Using the feature names from 'C4M1O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_C4M1O2M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_C4M1O2M1_correlation.append(original_idx_C4M1O2M1)


print('List with indexes for categorized C4M1O2M1 features')
#print(temp_idx_C4M1O2M1_correlation)
C4M1O2M1_correlation=np.stack(temp_idx_C4M1O2M1_correlation,axis=0)

print(C4M1O2M1_correlation.shape[0])

# Indexing in the importance array to get the C4M1O2M1 correlation importance values and summing the values to one value for this category
C4M1O2M1_importance=np.sum(importance_array[C4M1O2M1_correlation])

print(C4M1O2M1_importance)






#### C4M1O1M2 ##########

# Extracting all correlation features with 'C4M1O1M2' in the name
C4M1O1M2=df_correlation.filter(like='C4M1O1M2', axis=1) 
print(C4M1O1M2.shape[1])
print(C4M1O1M2)


# In this loop the indexes of the C4M1O1M2 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(C4M1O1M2.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=C4M1O1M2.columns[i]

    # Using the feature names from 'C4M1O1M2' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_C4M1O1M2=X_matrix_df.columns.get_loc(idx)

    temp_idx_C4M1O1M2_correlation.append(original_idx_C4M1O1M2)


print('List with indexes for categorized C4M1O1M2 features')
#print(temp_idx_C4M1O1M2_correlation)
C4M1O1M2_correlation=np.stack(temp_idx_C4M1O1M2_correlation,axis=0)

print(C4M1O1M2_correlation.shape[0])

# Indexing in the importance array to get the C4M1O1M2 correlation importance values and summing the values to one value for this category
C4M1O1M2_importance=np.sum(importance_array[C4M1O1M2_correlation])

print(C4M1O1M2_importance)






#### O1M2O2M1 ##########

# Extracting all correlation features with 'O1M2O2M1' in the name
O1M2O2M1=df_correlation.filter(like='O1M2O2M1', axis=1) 
print(O1M2O2M1.shape[1])
print(O1M2O2M1)


# In this loop the indexes of the O1M2O2M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(O1M2O2M1.shape[1]):
    
    # Indexing in wake category df and getting feature names 
    idx=O1M2O2M1.columns[i]

    # Using the feature names from 'O1M2O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_O1M2O2M1=X_matrix_df.columns.get_loc(idx)

    temp_idx_O1M2O2M1_correlation.append(original_idx_O1M2O2M1)


print('List with indexes for categorized O1M2O2M1 features')
#print(temp_idx_O1M2O2M1_correlation)
O1M2O2M1_correlation=np.stack(temp_idx_O1M2O2M1_correlation,axis=0)

print(O1M2O2M1_correlation.shape[0])

# Indexing in the importance array to get the O1M2O2M1 correlation importance values and summing the values to one value for this category
O1M2O2M1_importance=np.sum(importance_array[O1M2O2M1_correlation])

print(O1M2O2M1_importance)






### Gathering the frequency importance values in a df for plotting ########

# Saving values gathered for electrodes
print('list with features - electrodes')

electrodes_values_list=[F3M2F4M1_importance, F3M2C3M2_importance, F3M2C4M1_importance,F3M2O2M1_importance,F3M2O1M2_importance,F4M1C3M2_importance,F4M1C4M1_importance,F4M1O2M1_importance, F4M1O1M2_importance,C3M2C4M1_importance, C3M2O1M2_importance, C3M2O2M1_importance, C4M1O2M1_importance,C4M1O1M2_importance,O1M2O2M1_importance]
features_electrodes=np.stack(electrodes_values_list)
features_electrodes=np.reshape(features_electrodes,(1,15)) # reshape
print(features_electrodes)
print(type(features_electrodes))
print(features_electrodes.shape)




# column names 
electrodes_columns=['F3M2F4M1','F3M2C3M2','F3M2C4M1','F3M2O2M1','F3M2O1M2','F4M1C3M2','F4M1C4M1','F4M1O2M1','F4M1O1M2','C3M2C4M1','C3M2O1M2','C3M2O2M1','C4M1O2M1','C4M1O1M2','O1M2O2M1']
print(type(electrodes_columns))
print(electrodes_columns)


# Creating dataframe with values for plotting 
df_plt_electrodes=pd.DataFrame(features_electrodes, columns=electrodes_columns)
print(df_plt_electrodes)


boxplot_features=electrodes_columns
plt.suptitle('Correlation electrode feature importance',fontsize=20)
plt.subplot(1, 1, 1) # Adjust these numbers as per your requirement.
sns.barplot(data=df_plt_electrodes,errorbar=None)
plt.xticks(rotation=45, ha='right',fontsize=16)
#plt.title('')
plt.tight_layout()
plt.show()   
plt.clf()



# Category 3) correlation, correlation diff, dtw, dtw diff 


######### Extracting 'corr' ###########


temp_corr_search=[]

# Because of some challenges with the name for this specific category it is done in a more complicated way

Electrodes=['F3M2','F4M1','C3M2','C4M1','O1M2','O2M1']

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

    # Extracting all coherence features with 'corr' in the name
    wake_search=df_correlation.filter(like='Wake_'+str(Electrode_combination_naming), axis=1) 

    # In this loop the indexes of the O1M2O2M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
    for i in range(wake_search.shape[1]):
    
        # Indexing in wake category df and getting feature names 
        idx=wake_search.columns[i]

        # Using the feature names from 'O1M2O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
        # This is done to get the indexes for each feature in the 'importance_array' 
        original_idx_wake_corr=X_matrix_df.columns.get_loc(idx)

        temp_corr_search.append(original_idx_wake_corr)


    # Extracting all coherence features with 'corr' in the name
    N1_search=df_correlation.filter(like='N1_'+str(Electrode_combination_naming), axis=1) 
    
    # In this loop the indexes of the O1M2O2M1 matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
    for i in range(N1_search.shape[1]):
    
        # Indexing in wake category df and getting feature names 
        idx=N1_search.columns[i]

        # Using the feature names from 'O1M2O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
        # This is done to get the indexes for each feature in the 'importance_array' 
        original_idx_N1_corr=X_matrix_df.columns.get_loc(idx)

        temp_corr_search.append(original_idx_N1_corr)



    # Extracting all coherence features with 'corr' in the name
    N2_search=df_correlation.filter(like='N2_'+str(Electrode_combination_naming), axis=1) 
    
    # In this loop the indexes of the N2_corr matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
    for i in range(N2_search.shape[1]):
    
        # Indexing in wake category df and getting feature names 
        idx=N2_search.columns[i]

        # Using the feature names from 'O1M2O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
        # This is done to get the indexes for each feature in the 'importance_array' 
        original_idx_N2_corr=X_matrix_df.columns.get_loc(idx)

        temp_corr_search.append(original_idx_N2_corr)


    # Extracting all coherence features with 'corr' in the name
    N3_search=df_correlation.filter(like='N3_'+str(Electrode_combination_naming), axis=1) 
    
    
    # In this loop the indexes of the N3_corr matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
    for i in range(N3_search.shape[1]):
    
        # Indexing in wake category df and getting feature names 
        idx=N3_search.columns[i]

        # Using the feature names from 'O1M2O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
        # This is done to get the indexes for each feature in the 'importance_array' 
        original_idx_N3_corr=X_matrix_df.columns.get_loc(idx)

        temp_corr_search.append(original_idx_N3_corr)



    # Extracting all coherence features with 'corr' in the name
    REM_search=df_correlation.filter(like='REM_'+str(Electrode_combination_naming), axis=1) 
    
    
    # In this loop the indexes of the N2_corr matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
    for i in range(REM_search.shape[1]):
    
        # Indexing in wake category df and getting feature names 
        idx=REM_search.columns[i]

        # Using the feature names from 'O1M2O2M1' category dataframe as indexes to the extract the original indexes from X_matrix_df
        # This is done to get the indexes for each feature in the 'importance_array' 
        original_idx_REM_corr=X_matrix_df.columns.get_loc(idx)

        temp_corr_search.append(original_idx_REM_corr)


print('All wake corr for all electrodes')
print(temp_corr_search)
print(len(temp_corr_search))




print('List with indexes for categorized corr features')
#print(temp_corr_search)
corr=np.stack(temp_corr_search,axis=0)

print(corr.shape[0])

# Indexing in the importance array to get the O1M2O2M1 correlation importance values and summing the values to one value for this category
corr_importance=np.sum(importance_array[corr])

print(corr_importance)



########## diff corr ###############

# Extracting all coherence features with 'Epoch 1s' in the name
diff_corr=df_correlation.filter(like='_corrdiff_', axis=1) 
print(diff_corr.shape[1])
print(diff_corr)

temp_idx_diff_corr_correlation=[]

# In this loop the indexes of the diff_corr matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(diff_corr.shape[1]):
    
    # Indexing in epocssize_1 category df and getting feature names 
    idx=diff_corr.columns[i]

    # Using the feature names from 'diff_corr' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_diff_corr=X_matrix_df.columns.get_loc(idx)

    temp_idx_diff_corr_correlation.append(original_idx_diff_corr)


print('List with indexes for categorized diff corr features')
#print(temp_idx_epocssize_1_correlation)
diff_corr_correlation=np.stack(temp_idx_diff_corr_correlation,axis=0)

print(diff_corr_correlation.shape[0])

# Indexing in the importance array to get the epocssize_1 correlation importance values and summing the values to one value for this category
diff_corr_importance=np.sum(importance_array[diff_corr_correlation])

print(diff_corr_importance)







########## dtw ###############

temp_idx_dtw_correlation=[]


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


    # Extracting all coherence features with 'Epoch 1s' in the name
    dtw=df_correlation.filter(like='_dwt_'+str(Electrode_combination_naming), axis=1) 
    print(dtw.shape[1])
    print(dtw)

    

    # In this loop the indexes of the dtw matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
    for i in range(dtw.shape[1]):
        
        # Indexing in epocssize_1 category df and getting feature names 
        idx=dtw.columns[i]

        # Using the feature names from 'dtw' category dataframe as indexes to the extract the original indexes from X_matrix_df
        # This is done to get the indexes for each feature in the 'importance_array' 
        original_idx_dtw=X_matrix_df.columns.get_loc(idx)

        temp_idx_dtw_correlation.append(original_idx_dtw)





print('List with indexes for categorized diff corr features')
#print(temp_idx_epocssize_1_correlation)
dtw_correlation=np.stack(temp_idx_dtw_correlation,axis=0)

print(dtw_correlation.shape[0])

# Indexing in the importance array to get the epocssize_1 correlation importance values and summing the values to one value for this category
dtw_importance=np.sum(importance_array[dtw_correlation])

print(dtw_importance)




########## diff dtw ###############

# Extracting all coherence features with 'Epoch 1s' in the name
diff_dtw=df_correlation.filter(like='_dwt_diff_', axis=1) 
print(diff_dtw.shape[0])
print(diff_dtw)

temp_idx_diff_dtw_correlation=[]

# In this loop the indexes of the diff_corr matrix and the original X_matrix_df are matched to find the indexes that should be used for extracting the importance values 
for i in range(diff_dtw.shape[1]):
    
    # Indexing in epocssize_1 category df and getting feature names 
    idx=diff_dtw.columns[i]

    # Using the feature names from 'diff_dtw' category dataframe as indexes to the extract the original indexes from X_matrix_df
    # This is done to get the indexes for each feature in the 'importance_array' 
    original_idx_diff_dtw=X_matrix_df.columns.get_loc(idx)

    temp_idx_diff_dtw_correlation.append(original_idx_diff_dtw)


print('List with indexes for categorized diff dtw features')
#print(temp_idx_epocssize_1_correlation)
diff_dtw_correlation=np.stack(temp_idx_diff_dtw_correlation,axis=0)

print(diff_dtw_correlation.shape[0])

# Indexing in the importance array to get the epocssize_1 correlation importance values and summing the values to one value for this category
diff_dtw_importance=np.sum(importance_array[diff_dtw_correlation])

print(diff_dtw_importance)





### Gathering the frequency importance values in a df for plotting ########

# Saving values gathered for electrodes
print('list with features - electrodes')

method_values_list=[corr_importance, diff_corr_importance,dtw_importance,diff_dtw_importance]
feature_method=np.stack(method_values_list)
feature_method=np.reshape(feature_method,(1,4)) # reshape
print(feature_method)
print(type(feature_method))
print(feature_method.shape)




# column names 
method_columns=['Corr.','Diff. and corr.','DTW','Diff. DTW']

print(type(method_columns))
print(method_columns)


# Creating dataframe with values for plotting 
df_plt_method=pd.DataFrame(feature_method, columns=method_columns)
print(df_plt_method)


boxplot_features=method_columns
plt.suptitle('Method feature importance',fontsize=20)
plt.subplot(1, 1, 1) # Adjust these numbers as per your requirement.
sns.barplot(data=df_plt_method,errorbar=None)
plt.xticks(rotation=45, ha='right',fontsize=16)
#plt.title('')
plt.tight_layout()
plt.show()   
plt.clf()



'''