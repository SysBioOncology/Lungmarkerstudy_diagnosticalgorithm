# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression_pipeline import logistic_regression_pipeline
from ROC_curve_with_confidence_interval import ROC_curves_with_confidence_interval


df_input = pd.read_excel('Example_data_train_model.xlsx')

#log-10 scale the protein TMs and cell-free DNA concentrations
X = pd.DataFrame()
names_log10_var = ['CA125','CA15.3','CEA','CYFRA 21-1','HE4','NSE','proGRP','SCCA','cfDNA']
for i in range(0,len(names_log10_var)):
    X[names_log10_var[i]] = np.log10(df_input.loc[:,names_log10_var[i]])

names_nolog10_var = ['ctDNA','Age','Sex']
for j in range(0,len(names_nolog10_var)):
    X[names_nolog10_var[j]] = df_input.loc[:,names_nolog10_var[j]]

y_primary = df_input.loc[:,'LC']
y_nsclc = df_input.loc[:,'NSCLC']
y_sclc = df_input.loc[:,'SCLC']

#Define the classification problem that will be addressed by the model (choose one of the 3 problems below)
problem = 'LC' #no LC vs. LC (with PPV >= 98%)
#problem = 'NSCLC' #no LC + SCLC vs. NSCLC (with PPV >= 95%)
#problem = 'SCLC' #no LC + NSCLC vs. SCLC (with PPV >= 95%)

if problem == 'LC':
    names_classes = ['No lung cancer','Primary lung carcinoma']
    y = y_primary
    ppv_aim = 0.98
elif problem == 'NSCLC':
    names_classes = ['No lung cancer + SCLC','NSCLC']
    y = y_nsclc    
    ppv_aim = 0.95
elif problem == 'SCLC':
    names_classes = ['No lung cancer + NSCLC','SCLC']    
    y = y_sclc
    ppv_aim = 0.95

#Define what input variables will be used in the model
X = X.loc[:,['CA125','CA15.3','CEA','CYFRA 21-1','HE4','NSE','proGRP','SCCA','cfDNA','Age','ctDNA','Sex']]
 
#Names of the input variables
names_TMs = list(X)
#Define continuous variables (cnt_var), as these will be standardized before logistic regression
cnt_var = ['CA125','CA15.3','CEA','CYFRA 21-1','HE4','NSE','proGRP','SCCA','cfDNA','Age']
solver = 'saga'

#Run the logistic regression pipeline to retrieve the output    
[performances_per_threshold_val, performance_val, predicted_prob, predicted_class, 
percentage_class_one, y_pred_val_percv, y_pred_class_percv, performances_per_threshold_train, 
performances_train, predicted_prob_train, val_indices, probabilities, coefficients, prob_thresholds, 
logregs, scalers] = logistic_regression_pipeline(X, y, names_TMs, cnt_var, names_classes, solver)

#performances_per_threshold_val: performance metrics per probability threshold evaluated (thresholds can be found in variable 'probabilities')
#performance_val: performance metrics evaluated at 'optimal' probability threshold (where PPV >= 98% or 95% for training set)
#predicted_prob: predicted probabilities per patient (row) while in the validation set for the 200 repetitions (column)
#predicted_class: predicted class (0 or 1) per patient (row) while in the validation set for the 200 repetitions (column)
#percentage_class_one: overall percentage of 200 repetitions that the patient was predicted in class 1
#y_pred_val_percv : predicted probabilities for the validation set per cross-validation fold (5-folds with 200 repetitions)
#y_pred_class_percv: predicted class for the validation set per cross-validation fold (5-folds with 200 repetitions)
#performances_per_threshold_train: performance metrics per probability threshold (thresholds can be found in variable 'probabilities')
#performances_train: performance metrics evaluated at 'optimal' probability threshold (where PPV >= 98% or 95% for training set)
#predicted_prob_train: predicted probabilities per patient (row) while in the training set for 4 out of 5-folds x 200 repetitions
#val_indices: index of the patient in the validation set, per 5-folds x 200 repetitions
#probabilities: all probability thresholds used in 'performances_per_threshold_...'
#coefficients: estimated coefficients by the logistic regression model
#prob_thresholds: 'optimal' probability threshold where PPV >= 98% or 95% could be achieved in the training set
#logregs: save the trained logistic regression models for all 5-folds x 200 repetitions
#scalers: save the trained StandardScaler for all 5-folds x 200 repetitions

#The pre-set PPV could not always be met in the training set. If this is the case,
#determine the performance of the validation set only for the training sets where the
#pre-set PPV was met. Also save the folds of other variables where this 
#criteria was met. 
n_splits = len(logregs)
if sum(sum(performances_per_threshold_train[:,2,:]>=ppv_aim)>0) < n_splits:
    logregs_above_thresh = []
    scalers_above_thresh = []
    prob_thresholds_above_thresh = []
    val_indices_above_thresh = []
    y_pred_class_percv_above_thresh = []
    y_pred_val_percv_above_thresh = []
    for i in range(0,len(logregs)):
        if sum(performances_per_threshold_train[:,2,i]>=ppv_aim)>0:
            logregs_above_thresh.append(logregs[i])
            scalers_above_thresh.append(scalers[i])
            prob_thresholds_above_thresh.append(prob_thresholds[i])
            val_indices_above_thresh.append(val_indices[i])
            y_pred_class_percv_above_thresh.append(y_pred_class_percv[i])
            y_pred_val_percv_above_thresh.append(y_pred_val_percv[i])

    
    performance_val_above_thresh = performance_val[sum(performances_per_threshold_train[:,2,:]>=ppv_aim)>0]
    print('CV-folds where training set could meet pre-set PPV: %3.1f%%' %(len(performance_val_above_thresh)/len(performance_val)*100.0))
    print('Performances for these CV-folds:')
    for i in ['Sens val','Spec val','PPV val','NPV val','AUC val']:
        print('%s: %3.3f (%3.3f - %3.3f)' %(i,performance_val_above_thresh.loc[:,i].median(), performance_val_above_thresh.loc[:,i].quantile(q = 0.25),performance_val_above_thresh.loc[:,i].quantile(q = 0.75)))

    #Save the predicted probabilities and classes only for the CV-folds where the training
    #set met the pre-set PPV criterium
    predicted_class_above_thresh = np.ones((len(y), n_splits))*np.nan
    predicted_prob_above_thresh = np.ones((len(y), n_splits))*np.nan
    
    for i in range(0,len(val_indices_above_thresh)):
        predicted_class_above_thresh[val_indices_above_thresh[i],i] = y_pred_class_percv_above_thresh[i]
        predicted_prob_above_thresh[val_indices_above_thresh[i],i] = y_pred_val_percv_above_thresh[i]
    
    #Compute the percentage that a patient was classified as class one for these CV-folds
    percentage_class_one_above_thresh = []
    for i in range(0,len(percentage_class_one)):
        percentage_class_one_above_thresh.append(sum(predicted_class_above_thresh[i,:] == 1)/(n_splits-sum(np.isnan(predicted_class_above_thresh[i,:])))*100.0)

    #Save the coefficients of the models where the PPV criterium was met
    coefficients_above_thresh = coefficients[sum(performances_per_threshold_train[:,2,:]>=ppv_aim)>0]
    
#Plot the average ROC-curve, computed using vertical averaging
[mean_fprs, mean_tprs, std_tprs, tprs_upper, tprs_lower, 
 median_auc, lower_iqr_auc, upper_iqr_auc, aucs] = ROC_curves_with_confidence_interval(performances_per_threshold_val, np)

plt.figure()
plt.plot(mean_fprs,mean_tprs, label = 'AUC = %3.2f (%3.2f - %3.2f)' %(median_auc, lower_iqr_auc, upper_iqr_auc))
plt.fill_between(mean_fprs, tprs_lower, tprs_upper, alpha=.2)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.tight_layout()








