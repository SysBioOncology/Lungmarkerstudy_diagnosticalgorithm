# -*- coding: utf-8 -*-

def logistic_regression_pipeline_RFE(X, y, names_TMs, cnt_var, names_classes, solver, n_features_to_select):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    import warnings
    from sklearn.metrics import f1_score
    #from imblearn.metrics import geometric_mean_score
    import numpy as np
    from performance_metrics import performance_metrics
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.feature_selection import RFE
    
    plt.rcParams.update({'font.size': 14})
    warnings.filterwarnings("ignore") #Ignore all warnings
    
    #Convert output data to array
    y = np.array(y)
    
    #Define number of splits and repetitions for repeated statified K-fold cross-validation
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats = 200, random_state = 42)
    n_splits = skf.get_n_splits(X, y)
   
    #Create lists to save all performance metrics per split of the data
    aucs_val, sensitivity_val, specificity_val, PPVs_val, NPVs_val, f1_vals = [], [], [], [], [], []
    y_pred_class_percv, y_pred_val_percv = [], []
    parameters = []
    prob_thresholds = []
    val_indices = []
    logregs = []
    scalers = []
    j = 0
    probabilities = np.linspace(0,1,101)
    performances_per_threshold_val = np.zeros((len(probabilities),4,n_splits))
    performances_per_threshold_train = np.zeros((len(probabilities),4,n_splits))
    performances_train = np.zeros((n_splits,4))
    predicted_class = np.zeros((len(y),int(n_splits/5)))
    predicted_prob = np.zeros((len(y),int(n_splits/5)))
    predicted_prob_train = np.ones((len(y), int(n_splits)))*np.nan
    ranking = np.zeros((len(names_TMs), n_splits))
    selected_features_all = np.zeros((len(names_TMs),n_splits))

    #For-loop per cross-validation fold (5 folds * 200 repetitions)
    for train_index, val_index in skf.split(X, y): 
        #Show progress by printing each 10th fold
        if j%10 == 0:
            print('Fold %d of %d' %(j, n_splits))
        
        #Define the training and test set
        x_train_unscaled, x_val_unscaled = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index],y[val_index]
        val_indices.append(val_index)
        
        #Resample and standardize the training set (_train) and standardize the validation (_val) set. 
        #Standardization is only applied to the continuous (_cnt) variables and not to the categorical variables (_cat). 
        x_cnt_train = x_train_unscaled.loc[:,cnt_var] 
        x_cat_train = x_train_unscaled.drop(cnt_var, axis = 1)
        names = list(x_cnt_train)+(list(x_cat_train))
        x_cnt_val = x_val_unscaled.loc[:,cnt_var]
        x_cat_val = x_val_unscaled.drop(cnt_var, axis = 1) 
        
        #If the number of continuous variables > 0, scale these continuous variables
        #For scaling, the StandardScaler is determines the mean and std of the training set
        #Using this mean and std, standard scaling will be applied to both the training and test set (z = (x - mean)/std)
        
        if len(cnt_var) > 0:
            scaler = StandardScaler().fit(x_cnt_train)
            scalers.append(scaler)
            X_cnt_train = scaler.transform(x_cnt_train)
            X_cnt_val = scaler.transform(x_cnt_val)
        
        #Otherwise, add the empty columns
        else:
            X_cnt_train = x_cnt_train
            X_cnt_val = x_cnt_val
        
        #Add the continuous and categorical variables together in one array    
        X_train = np.append(X_cnt_train,np.array(x_cat_train),axis = 1)
        X_val = np.append(X_cnt_val,np.array(x_cat_val),axis = 1)
        

        
        logreg=LogisticRegression(solver = solver,penalty = 'none')        
        
        #Determine the least important features using recursive feature elimination
        #The least important feature (based on coefficient in logistic regression model
        #will be removed. This procedure will be repeated x times, until the 
        #remaining features equals n_features_to_select 
        rfe = RFE(estimator = logreg, step = 1, n_features_to_select = n_features_to_select)
        rfe.fit(X_train,y_train)
        X_train = rfe.transform(X_train)
        X_val = rfe.transform(X_val)
        selected_features_all[:,j] = rfe.support_
        ranking[:,j] = rfe.ranking_
        
        
        #Train the Logistic regression model by using the standardized input values and the output of the training set
        logreg.fit(X_train,y_train)
        logregs.append(logreg)
        
        #Predict the probabilities of the training set
        y_pred_train = logreg.predict_proba(X_train)[:,1]
        ytrue_train = y_train
        
        #Determine the performances (sensitivity, specificity, PPV and NPV) per probability threshold
        #The script performance_metrics.py is used for these computations
        for g in range(0,len(probabilities)):
            performances_per_threshold_train[g,:,j] = performance_metrics(probabilities[g], y_pred_train, ytrue_train)
        
        #Determine the minimum probability threshold at which the pre-specified PPV could still be achieved  
        #For NSCLC and SCLC, a pre-set PPV of 95% is aimed
        if names_classes[1] == 'NSCLC' or names_classes[1] == 'SCLC':
            #If a PPV of 95% could not be achieved, save the prob threshold at which the max PPV could be achieved
            #During post-processing, only the cross-validation folds at which PPV of 95% could be achieved were considered
            if np.max(performances_per_threshold_train[:,2,j]) < 0.95:
                prob_thresh_max = np.argmax(performances_per_threshold_train[:,2,j])/100
            else:
                prob_thresh_max = np.argwhere(performances_per_threshold_train[:,2,j] >= 0.95)[0]/100.0
        #For identification LC, a pre-set PPV of 98% is aimed
        elif names_classes[1] == 'Primary lung carcinoma':
            if np.max(performances_per_threshold_train[:,2,j]) < 0.98:
                prob_thresh_max = np.argmax(performances_per_threshold_train[:,2,j])/100
            else:
                prob_thresh_max = np.argwhere(performances_per_threshold_train[:,2,j] >= 0.98)[0]/100.0
        
        #For exclusion of LC, a NPV as high as possible would be required, preferably 100%. 
        #Determine the maximum probability threshold at which the maximum NPV could be achieved
        #During a posteriori analyses, it could be checked whether the training sets could 
        #achieve a NPV of 100%. 
        elif names_classes[1] == 'No lung carcinoma':
            NPV_train_max = 0.0
            prob_thresh_max = 0.0
            for prob_threshold in range(0,len(probabilities)):
                NPV_train = performances_per_threshold_train[prob_threshold,3,j]
                if NPV_train >= NPV_train_max:
                    NPV_train_max = NPV_train
                    prob_thresh_max = probabilities[prob_threshold]
        
        #Determine the performance of the training set at the 'optimal' probability threshold and save the probability threshold used
        performances_train[j,:] = performance_metrics(prob_thresh_max, y_pred_train, ytrue_train)
        prob_thresholds.append(prob_thresh_max)
                
        #Predict the probabilities and classes of the validation set, according to the set probability threshold
        y_pred_val = logreg.predict_proba(X_val)
        y_pred_class_val = (y_pred_val[:,1] >= prob_thresh_max).astype(int)   
        predicted_class[val_index,int(j/5)] = y_pred_class_val #Because of repeated 5-fold CV, each patient is per repetition (row) in 1/5 folds in the validation set 
        predicted_prob[val_index,int(j/5)] = y_pred_val[:,1]
        predicted_prob_train[train_index,j] = y_pred_train
        
        #Compute the performance metrics of the validation set at set probability threshold
        sens_val, spec_val, PPV_val, NPV_val = performance_metrics(prob_thresh_max, y_pred_val[:,1], y_val)
        f1_val = f1_score(y_val,y_pred_class_val)
        #Compute performance validation set at multiple probability thresholds
        for z in range(0,len(probabilities)):
            performances_per_threshold_val[z,:,j] = performance_metrics(probabilities[z],y_pred_val[:,1],y_val)
        
        #Retrieve the values of the coefficients (intercept + model coefficients)
        params = np.append(logreg.intercept_,logreg.coef_)
        parameters.append(params)
                    
        #Compute the ROC AUC, using standard function
        roc_auc_val = roc_auc_score(y_val,y_pred_val[:,1])
        aucs_val.append(roc_auc_val)

        #Save these performance metrics of the validation set
        sensitivity_val.append(sens_val)
        specificity_val.append(spec_val)
        PPVs_val.append(PPV_val)
        NPVs_val.append(NPV_val)
        f1_vals.append(f1_val)
        j = j+1

        #Save the predicted and real val outcomes
        y_pred_val_percv.append(y_pred_val[:,1].tolist())
        y_pred_class_percv.append(y_pred_class_val.tolist())
      
    #Save the performance metrics and coefficients in dataframes
    performance_val = pd.DataFrame(list(zip(sensitivity_val, specificity_val,aucs_val,PPVs_val,NPVs_val, f1_vals)),columns = ['Sens val','Spec val','AUC val','PPV val','NPV val', 'f1 val'])    
    coefficients = pd.DataFrame(data = parameters)

    #Print the performances of the validation sets (median (IQR))    
    print('Performance metrics - Validation set:')
    for i in ['Sens val','Spec val','AUC val','PPV val','NPV val', 'f1 val']:
        print('%s: %3.4f (%3.4f - %3.4f)' %(i, performance_val.loc[:,i].median(), performance_val.loc[:,i].quantile(q = 0.25), performance_val.loc[:,i].quantile(q = 0.75)))


    #Compute the percentage of cross-validation folds for which a patient is classified in class 1, while in the validation set
    percentage_class_one = sum(predicted_class.transpose())/(n_splits/5)*100   

    #return(prob_y_val, coefficients, probabilities, performance_val, performances_per_threshold_val, prob_thresholds, performances_train, predicted_class, predicted_prob, val_indices, performances_per_threshold_train, percentage_class_one, predicted_prob_train, logregs, scalers)

    return(performances_per_threshold_val, performance_val, predicted_prob, predicted_class, 
           percentage_class_one, y_pred_val_percv, y_pred_class_percv,
           performances_per_threshold_train, performances_train, predicted_prob_train, 
           val_indices, probabilities, coefficients, prob_thresholds, logregs, scalers, ranking, selected_features_all)

