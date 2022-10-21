# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression_pipeline_RFE import logistic_regression_pipeline_RFE
import seaborn as sns

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


#Define what input variables will be used in the model, only the protein TMs were used for RFE
X = X.loc[:,['CA125','CA15.3','CEA','CYFRA 21-1','HE4','NSE','proGRP','SCCA']]
 
#Names of the input variables
names_TMs = list(X)
#Define continuous variables (cnt_var), as these will be standardized before logistic regression
cnt_var = ['CA125','CA15.3','CEA','CYFRA 21-1','HE4','NSE','proGRP','SCCA']
solver = 'saga'

#Define the number of features that will be selected in the end
#This can be a list of multiple numbers, but can also be a list with one number (see comment)
n_features_to_select = [1,2,3,4,5,6,7,8]
#n_features_to_select = [1]

performances_per_threshold_val_nfeatures = []
performance_val_nfeatures = []
predicted_prob_nfeatures = []
predicted_class_nfeatures = []
percentage_class_one_nfeatures = []
y_pred_val_percv_nfeatures = []
y_pred_class_per_cv_nfeatures = []
performances_per_threshold_train_nfeatures = []
performances_train_nfeatures = []
predicted_prob_train_nfeatures = []
val_indices_nfeatures = []
coefficients_nfeatures = []
prob_thresholds_nfeatures = []
logregs_nfeatures = []
scalers_nfeatures = []
ranking_nfeatures = []
selected_features_all_nfeatures = []

logregs_above_thresh_nfeatures = []
scalers_above_thresh_nfeatures = []
prob_thresholds_above_thresh_nfeatures = []
val_indices_above_thresh_nfeatures = []
y_pred_class_percv_above_thresh_nfeatures = []
y_pred_val_percv_above_thresh_nfeatures = []
selected_features_all_above_thresh_nfeatures = []
ranking_above_thresh_nfeatures = []
predicted_class_above_thresh_nfeatures = []
predicted_prob_above_thresh_nfeatures = []
performance_val_above_thresh_nfeatures = []
percentage_class_one_above_thresh_nfeatures = []
cvfolds_above_thresh = []


for n_features in n_features_to_select:
    print('Fitting model with %d features' %n_features)
    [performances_per_threshold_val, performance_val, predicted_prob, predicted_class, 
    percentage_class_one, y_pred_val_percv, y_pred_class_percv, performances_per_threshold_train, 
    performances_train, predicted_prob_train, val_indices, probabilities, coefficients, 
    prob_thresholds, logregs, scalers, ranking, selected_features_all] = logistic_regression_pipeline_RFE(X, y, names_TMs, cnt_var, names_classes, solver, n_features)

    performances_per_threshold_val_nfeatures.append(performances_per_threshold_val)
    performance_val_nfeatures.append(performance_val)
    predicted_prob_nfeatures.append(predicted_prob)
    predicted_class_nfeatures.append(predicted_class)
    percentage_class_one_nfeatures.append(percentage_class_one)
    y_pred_val_percv_nfeatures.append(y_pred_val_percv)
    y_pred_class_per_cv_nfeatures.append(y_pred_class_percv)
    performances_per_threshold_train_nfeatures.append(performances_per_threshold_train)
    performances_train_nfeatures.append(performances_train)
    predicted_prob_train_nfeatures.append(predicted_prob_train)
    val_indices_nfeatures.append(val_indices)
    coefficients_nfeatures.append(coefficients)
    prob_thresholds_nfeatures.append(prob_thresholds)
    logregs_nfeatures.append(logregs)
    scalers_nfeatures.append(scalers)
    ranking_nfeatures.append(ranking)
    selected_features_all_nfeatures.append(selected_features_all)
    
    #The pre-set PPV could not always be met in the training set. If this is the case,
    #determine the performance of the validation set only for the training sets where the
    #pre-set PPV was met. Also save the folds of other variables where this 
    #criteria was met. 
    n_splits = len(logregs)
    logregs_above_thresh = []
    scalers_above_thresh = []
    prob_thresholds_above_thresh = []
    val_indices_above_thresh = []
    y_pred_class_percv_above_thresh = []
    y_pred_val_percv_above_thresh = []
    selected_features_all_above_thresh = []
    ranking_above_thresh = []
    for i in range(0,len(logregs)):
        if sum(performances_per_threshold_train[:,2,i]>=ppv_aim)>0:
            logregs_above_thresh.append(logregs[i])
            scalers_above_thresh.append(scalers[i])
            prob_thresholds_above_thresh.append(prob_thresholds[i])
            val_indices_above_thresh.append(val_indices[i])
            y_pred_class_percv_above_thresh.append(y_pred_class_percv[i])
            y_pred_val_percv_above_thresh.append(y_pred_val_percv[i])
    
    selected_features_all_above_thresh = selected_features_all.transpose()[sum(performances_per_threshold_train[:,2,:]>=ppv_aim)>0].transpose()
    ranking_above_thresh = ranking.transpose()[sum(performances_per_threshold_train[:,2,:]>=ppv_aim)>0].transpose()

    
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

    logregs_above_thresh_nfeatures.append(logregs_above_thresh)
    scalers_above_thresh_nfeatures.append(scalers_above_thresh)
    prob_thresholds_above_thresh_nfeatures.append(prob_thresholds_above_thresh)
    val_indices_above_thresh_nfeatures.append(val_indices_above_thresh)
    y_pred_class_percv_above_thresh_nfeatures.append(y_pred_class_percv_above_thresh)
    y_pred_val_percv_above_thresh_nfeatures.append(y_pred_val_percv_above_thresh)
    selected_features_all_above_thresh_nfeatures.append(selected_features_all_above_thresh)
    ranking_above_thresh_nfeatures.append(ranking_above_thresh)
    predicted_class_above_thresh_nfeatures.append(predicted_class_above_thresh)
    predicted_prob_above_thresh_nfeatures.append(predicted_prob_above_thresh)
    performance_val_above_thresh_nfeatures.append(performance_val_above_thresh)
    percentage_class_one_above_thresh_nfeatures.append(percentage_class_one_above_thresh)
    cvfolds_above_thresh.append(len(logregs_above_thresh)/n_splits*100.0)

    

    
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
#ranking: ranking of the variables (1 is the most informative and were selected in the end, n_features will be given ranking 1)
#selected_features_all: selected features per cross-validation fold (5-folds with 200 repetitions)


#Make a table with the median (IQR) performance metrics per number of selected features, but
#only for the CV-folds where the training set could achieve the pre-set PPV
performance_metrics_names = ['Sens val','Spec val','PPV val','NPV val','AUC val']
performance_metrics_table_above_thresh = pd.DataFrame()
performance_metrics_table_above_thresh['Number features'] = n_features_to_select
performance_metrics_table_above_thresh['CV folds PPV criteria'] = cvfolds_above_thresh

for j in range(0,len(performance_metrics_names)):
    performance_metric = []
    for i in range(0,len(n_features_to_select)):
        performance_metric.append('%3.2f (%3.2f-%3.2f)' %(performance_val_above_thresh_nfeatures[i].loc[:,performance_metrics_names[j]].median(),
                                                          performance_val_above_thresh_nfeatures[i].loc[:,performance_metrics_names[j]].quantile(q = 0.25), 
                                                          performance_val_above_thresh_nfeatures[i].loc[:,performance_metrics_names[j]].quantile(q = 0.75)))
    performance_metrics_table_above_thresh['%s' %performance_metrics_names[j]] = performance_metric

print(performance_metrics_table_above_thresh)

#Show the variables selected per number of selected features, only for the CV-folds where the training
#set could meet the pre-set PPV
selected_features_matrix_above_thresh = np.zeros((len(names_TMs), len(n_features_to_select)))

for i in range(0,len(n_features_to_select)):    
    selected_features_matrix_above_thresh[:,i] = sum(selected_features_all_above_thresh_nfeatures[i].transpose())/selected_features_all_above_thresh_nfeatures[i].shape[1]*100.0   

plt.figure()
sns.heatmap(selected_features_matrix_above_thresh, annot = True, fmt = ".0f", cmap="Blues")
plt.xticks(ticks =  np.linspace(0.5,len(n_features_to_select)-0.5, len(n_features_to_select)), labels = n_features_to_select)
plt.xlabel('Number of selected features')
plt.yticks(ticks =  np.linspace(0.5,len(names_TMs)-0.5,len(names_TMs)), labels = names_TMs, rotation = 0)
plt.ylabel('Input variables')
plt.tight_layout()


