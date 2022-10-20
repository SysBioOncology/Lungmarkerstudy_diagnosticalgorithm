# -*- coding: utf-8 -*-
"""
"""

def logistic_regression_pipeline_predictnewpatient(X, y, names_TMs, cnt_var, names_classes, logregs, scalers, prob_thresholds):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    import warnings
    import numpy as np
    from performance_metrics import performance_metrics
    
    plt.rcParams.update({'font.size': 14})
    warnings.filterwarnings("ignore") #Ignore all warnings
    
    n_splits = len(logregs)
    
    #Create lists to save all performance metrics per repetition
    aucs_val, sensitivity_val, specificity_val, PPVs_val, NPVs_val = [], [], [], [], []
    probabilities = np.linspace(0,1,101)
    y_pred_all = np.zeros((len(y), n_splits))
    y_classes_all = np.zeros((len(y), n_splits))    
    performances_per_threshold = np.zeros((len(probabilities),4,n_splits))

    
    #For-loop per trained model
    for j in range(0,n_splits):
        #Show progress by printing each 100th fold
        if j%100 == 0:
            print('Fold %d of %d' %(j, n_splits))
    
        
        #Standardize the continuous (_cnt) input data. 
        x_cnt = X.loc[:,cnt_var] 
        x_cat = X.drop(cnt_var, axis = 1)
        names = list(x_cnt)+(list(x_cat))
        
        #If the number of continuous variables > 0, scale these continuous variables
        #For scaling, the StandardScaler is was trained with the mean and std of the training set
        #Using this mean and std, standard scaling will be applied (z = (x - mean)/std)
        if len(cnt_var) > 0:
            scaler = scalers[j]
            X_cnt = scaler.transform(x_cnt)
        
        #Otherwise, add the empty columns
        else:
            X_cnt = x_cnt
        
        #Add the continuous and categorical variables together in one array    
        X_val = np.append(X_cnt,np.array(x_cat),axis = 1)

        #Call the trained logistic regression model        
        logreg=logregs[j]     
        
        #Predict the probabilities and classes of the new patients
        y_pred = logreg.predict_proba(X_val)[:,1]
        y_classes = (y_pred >= prob_thresholds[j][0]).astype(int)  
        y_pred_all[:,j] = y_pred
        y_classes_all[:,j] = y_classes
        
        #If the true output variable is available, determine the performance of the model on these new patients
        if sum(y.isna()) != len(y):
            y_true = np.array(y)
            sens_val, spec_val, PPV_val, NPV_val = performance_metrics(prob_thresholds[j], y_pred, y_true)
            sensitivity_val.append(sens_val)
            specificity_val.append(spec_val)
            PPVs_val.append(PPV_val)
            NPVs_val.append(NPV_val)
            roc_auc_val = roc_auc_score(y_true,y_pred)
            aucs_val.append(roc_auc_val)
            #Compute performance validation set at multiple probability thresholds
            for z in range(0,len(probabilities)):
                performances_per_threshold[z,:,j] = performance_metrics(probabilities[z],y_pred,y_true)
      
    #Save the performance metrics in a dataframe
    performance = pd.DataFrame(list(zip(sensitivity_val, specificity_val,aucs_val,PPVs_val,NPVs_val)),columns = ['Sens val','Spec val','AUC val','PPV val','NPV val'])    

    #Print the performances     
    print('Performance metrics:')
    for i in ['Sens val','Spec val','AUC val','PPV val','NPV val']:
        print('%s: %3.4f (%3.4f - %3.4f)' %(i, performance.loc[:,i].median(), performance.loc[:,i].quantile(q = 0.25), performance.loc[:,i].quantile(q = 0.75)))

    #Compute the percentage of models for which a patient is classified in class 1
    percentage_class_one = sum(y_classes_all.transpose())/n_splits*100   

    return(performances_per_threshold, performance, y_pred_all, y_classes_all, 
           percentage_class_one, probabilities)

