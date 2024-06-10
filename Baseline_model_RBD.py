# Load standard packages
import numpy as np
from numpy import shape, vstack
import pandas as pd
import copy
import matplotlib
import matplotlib.pyplot as plt
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
import xgboost
from xgboost import XGBClassifier
import sklearn
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, auc
from sklearn.inspection import permutation_importance


# Using sys function to import 'My_functions_script'
sys.path.insert(0, 'C:/Users/natas/Documents/Master thesis code')

# Import My_functions_script
from My_functions_script import list_files_in_folder, preprocessing, bandpass_frequency_band, inverse_wavelet_5_levels, relative_power_for_frequencyband, coherence_features, extract_numbers_from_filename, extract_letters_and_numbers, split_string_by_length

### Loading data ###########################
#Loading data frame 

# Coherence
#df_combined=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/RBD controls/Coherence_features_combined_fullnight_RBD_and_controls.csv')
#df_combined=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/RBD controls/Coherence_features_combined_1partnight_RBDandcontrols.csv')
#df_combined=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/RBD controls/Coherence_features_combined_2partnight_RBDandcontrols.csv')
#df_combined=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/RBD controls/Coherence_features_combined_3partnight_RBDandcontrols.csv')
df_combined=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Coherence features/RBD controls/Coherence_features_combined_4partnight_RBDandcontrols.csv')


# Correlation 
#df_combined=pd.read_csv('C:/Users/natas/Documents/Master thesis code/Correlation Features/RBD/All_correlation_features_combined_RBDandcontrols.csv')




df_iRBD=df_combined[df_combined['Dianosis'] == 'I']
#df_PD=df_combined[df_combined['Dianosis'] == 'P']
#df_PD_D=df_combined[df_combined['Dianosis'] == 'D']

df_patients=pd.concat([df_iRBD])#,df_PD,df_PD_D])

print('df patients')
print(df_patients)

df_controls=df_combined[df_combined['Dianosis'] == 'Control']

print('df controls')
print(df_controls)

df_combined=pd.concat([df_patients,df_controls])

df_for_use=copy.deepcopy(df_combined)
############################################


df_shuffled = df_for_use.sample(frac=1).reset_index(drop=True)
df_for_use=df_shuffled

####### Preparing X and Y ########################
df_X=df_for_use.drop(['PatientID','Sex_F1_M2','Age','Dianosis'], axis=1)
print(df_X)


# Creating values 0 = value, 1 = NaN
df_X_baseline=1*np.isnan(df_X)
print(df_X_baseline)


# Encoding the categorical columns for the X matrix

# Encoding categorical columns (Dx, Sex, patientID and cohort)
categorical_columns = df_for_use[['Sex_F1_M2','Dianosis']].columns.tolist()
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
Y_variable=one_hot_df[['Dianosis_I']] # This is RBD labels
print(Y_variable)

###############################################
one_hot_df=one_hot_df.drop(['Dianosis_I','Dianosis_Control'],axis=1)
print('One_hot_df')
print(one_hot_df)

# Concatenate the one-hot encoded dataframe with the original dataframe
X_matrix = pd.concat([df_X_baseline, one_hot_df], axis=1)

# Cropping X_matrix to not include the indexes 
X_matrix=X_matrix.iloc[:,1:558]
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
classifier = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')

# stratified K-fold (takes percentage of classes into account, when splitting the data)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
skf.get_n_splits(X_matrix, Y_variable)

print('Stratified K-fold')
print(skf)

train_split=[]
test_split=[]
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

    '''
    ### Permutation on the test set #######
    result = permutation_importance(
        classifier, X_matrix[test_index], Y_variable[test_index], n_repeats=10, random_state=42, n_jobs=2
    )

    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=X_matrix_df.columns[sorted_importances_idx],
    )
    print('Pertumation test set')
    print(importances)
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances (test set), RBD model")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    plt.show()
    '''
    ###########################################################################


    ##### PREDICTION  ####################################
    # Predicting probabilities with the trained model 
    y_pred=classifier.predict_proba(X_matrix[test_index])
    print(len(y_pred))

    # Saving the predictions in the same order as the test split 
    y_pred_concatenated.append(y_pred)

    # Converting the probabilities to 1 and 0 
    # This is used for the metrics - using only one of the columns (first one - it is one-hot encoded)
    y_pred_binary=np.round(y_pred)
    #print(y_pred_binary[:,0])
    #print(Y_variable[test_index])

    ############### PERFORMANCE METRICS ########################################
    # Calculating metrics
    confusion_matrix_metrics=sklearn.metrics.confusion_matrix(Y_variable[test_index],y_pred_binary[:,0])
    print('Confusion matrix')
    print(confusion_matrix_metrics)

    classificationreport=classification_report(Y_variable[test_index], y_pred_binary[:,0])
    print(classificationreport)


    ##### Plotting Precision and Recall curve ############

    # Presicion / recall curve
    '''
    display = PrecisionRecallDisplay.from_estimator(
        classifier, X_matrix[test_index], Y_variable[test_index], name="BostedDecisionTree", plot_chance_level=True
    )
    _ = display.ax_.set_title("2-class Precision-Recall curve, RBD model")

    # To display the plot
    plt.show()
    '''

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
    title=f"Mean ROC curve with variability\n(Positive label RBD_I')",
)
ax.legend(loc="lower right")
plt.show()




############# Trash code from here ##################################


#y_pred_all=np.stack(y_pred_concatenated)


#print('stacked y')
#print(y_pred_all,axis=1)





# fit model
#bst.fit(X_train, y_train)
# make predictions
#preds = bst.predict(X_test)

#print('Information about predictions')
#print(preds)
#print(type(preds))
#print(len(preds))

########### 5-fold crossvalidation #####################
'''

y_pred = cross_val_predict(bst, X_matrix, Y_variable, cv=5)#  cv=np.min([sum(Y_variable == 0), sum(Y_variable == 1)]), method='predict_proba')

print(y_pred)
print(len(y_pred))
print(Y_variable)
print(len(Y_variable))
########################## Calculate performance metrics #########################
# accuracy, F1-score, confusion matrix 

accuracy_metric=sklearn.metrics.accuracy_score(Y_variable,y_pred) # y_true, y_predicted
print('Accuracy')
print(accuracy_metric)

f1_score_metric=sklearn.metrics.f1_score(Y_variable,y_pred)
print('f1-score')
print(f1_score_metric)

confusion_matrix_metrics=sklearn.metrics.confusion_matrix(Y_variable,y_pred)
print('Confusion matrix')
print(confusion_matrix_metrics)

classificationreport=classification_report(Y_variable, y_pred)
print(classificationreport)

'''