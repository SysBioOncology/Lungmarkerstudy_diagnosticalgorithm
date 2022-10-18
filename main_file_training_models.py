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
elif problem == 'NSCLC':
    names_classes = ['No lung cancer + SCLC','NSCLC']
    y = y_nsclc
elif problem == 'SCLC':
    names_classes = ['No lung cancer + NSCLC','SCLC']    
    y = y_sclc

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


#Plot the average ROC-curve, computed using vertical averaging
[mean_fprs, mean_tprs, std_tprs, tprs_upper, tprs_lower, 
 median_auc, lower_iqr_auc, upper_iqr_auc, aucs] = ROC_curves_with_confidence_interval(performances_per_threshold_val, np)

plt.figure()
plt.plot(mean_fprs,mean_tprs, label = 'AUC = %3.2f (%3.2f - %3.2f)' %(median_auc, lower_iqr_auc, upper_iqr_auc))
plt.fill_between(mean_fprs, tprs_lower, tprs_upper, alpha=.2)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.tight_layout()








