# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
import numpy as np
from logistic_regression_pipeline_predictnewpatients import logistic_regression_pipeline_predictnewpatient
from ROC_curve_with_confidence_interval import *
import matplotlib.pyplot as plt

#Main file to run the models with own data
#df_input = pd.read_excel('Example_data_run_models.xlsx', sheet_name = 'With output')
df_input = pd.read_excel('Example_data_run_models.xlsx', sheet_name = 'Without output')


#log-10 scale the protein TMs and cell-free DNA concentrations
X = pd.DataFrame()
names_log10_var = ['CA125','CA15.3','CEA','CYFRA 21-1','NSE','proGRP','cfDNA']
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

#Define the input data available: protein TMs or the combination of protein and DNA
input_combination = 'protein TMs'
#input_combination = 'protein + DNA TMs'


#Based on the chosen classification problem + combination of input variables, 
#specify the input dataframe and call the trained models (logregs), standard scalers (scalers) 
#and probability thresholds used to define the classes (prob_threshold)
if problem == 'LC':
    names_classes = ['No lung cancer','Primary lung carcinoma']
    y = y_primary
    if input_combination == 'protein TMs':
        X = X.loc[:,['CYFRA 21-1','CEA','Age','Sex']]
        cnt_var = ['CYFRA 21-1','CEA','Age']
        logregs = pd.read_pickle('logregs_LC_protein_TMs.pkl')
        scalers = pd.read_pickle('scalers_LC_protein_TMs.pkl')
        prob_thresholds = pd.read_pickle('prob_thresholds_LC_protein_TMs.pkl')
    elif input_combination == 'protein + DNA TMs':
        X = X.loc[:,['CYFRA 21-1','CEA','cfDNA','ctDNA','Age','Sex']]
        cnt_var = ['CYFRA 21-1','CEA','Age','cfDNA']
        logregs = pd.read_pickle('logregs_LC_protein_and_DNA_TMs.pkl')
        scalers = pd.read_pickle('scalers_LC_protein_and_DNA_TMs.pkl')
        prob_thresholds = pd.read_pickle('prob_thresholds_LC_protein_and_DNA_TMs.pkl')
    
elif problem == 'NSCLC':
    names_classes = ['No lung cancer + SCLC','NSCLC']
    y = y_nsclc
    if input_combination == 'protein TMs':
        X = X.loc[:,['CEA','CYFRA 21-1','NSE','proGRP','Age','Sex']]
        cnt_var = ['CEA','CYFRA 21-1','NSE','proGRP','Age']
        logregs = pd.read_pickle('logregs_NSCLC_protein_TMs.pkl')
        scalers = pd.read_pickle('scalers_NSCLC_protein_TMs.pkl')
        prob_thresholds = pd.read_pickle('prob_thresholds_NSCLC_protein_TMs.pkl')
    elif input_combination == 'protein + DNA TMs':
        X = X.loc[:,['CEA','CYFRA 21-1','NSE','proGRP','Age','Sex','cfDNA','ctDNA']]
        cnt_var = ['CEA','CYFRA 21-1','NSE','proGRP','Age','cfDNA']
        logregs = pd.read_pickle('logregs_NSCLC_protein_and_DNA_TMs.pkl')
        scalers = pd.read_pickle('scalers_NSCLC_protein_and_DNA_TMs.pkl')
        prob_thresholds = pd.read_pickle('prob_thresholds_NSCLC_protein_and_DNA_TMs.pkl')
elif problem == 'SCLC':
    names_classes = ['No lung cancer + NSCLC','SCLC']    
    y = y_sclc
    if input_combination == 'protein TMs':
        X = X.loc[:,['CA125','CA15.3','CYFRA 21-1','NSE','proGRP','Age','Sex']]
        cnt_var = ['CA125','CA15.3','CYFRA 21-1','NSE','proGRP','Age']
        logregs = pd.read_pickle('logregs_SCLC_protein_TMs.pkl')
        scalers = pd.read_pickle('scalers_SCLC_protein_TMs.pkl')
        prob_thresholds = pd.read_pickle('prob_thresholds_SCLC_protein_TMs.pkl')

    elif input_combination == 'protein + DNA TMs':
        X = X.loc[:,['CA125','CA15.3','CYFRA 21-1','NSE','proGRP','Age','cfDNA','Sex','ctDNA']]
        cnt_var = ['CA125','CA15.3','CYFRA 21-1','NSE','proGRP','Age','cfDNA']
        logregs = pd.read_pickle('logregs_SCLC_protein_and_DNA_TMs.pkl')
        scalers = pd.read_pickle('scalers_SCLC_protein_and_DNA_TMs.pkl')
        prob_thresholds = pd.read_pickle('prob_thresholds_SCLC_protein_and_DNA_TMs.pkl')

names_TMs = list(X)

#Call the script used for predictions of new patients
[performances_per_threshold, performance, y_pred_all, y_classes_all, 
 percentage_class_one, probabilities] = logistic_regression_pipeline_predictnewpatient(X, y, names_TMs, cnt_var, names_classes, logregs, scalers, prob_thresholds)

#If the diagnoses were provided, plot the average ROC-curve, computed using vertical averaging
[mean_fprs, mean_tprs, std_tprs, tprs_upper, tprs_lower, 
 median_auc, lower_iqr_auc, upper_iqr_auc, aucs] = ROC_curves_with_confidence_interval(performances_per_threshold, np)

plt.figure()
plt.plot(mean_fprs,mean_tprs, label = 'AUC = %3.2f (%3.2f - %3.2f)' %(median_auc, lower_iqr_auc, upper_iqr_auc))
plt.fill_between(mean_fprs, tprs_lower, tprs_upper, alpha=.2)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.tight_layout()



