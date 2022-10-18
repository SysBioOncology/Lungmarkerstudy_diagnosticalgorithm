# -*- coding: utf-8 -*-

#Code is based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

def ROC_curves_with_confidence_interval(performance_metrics_per_threshold_val, np):
    from sklearn.metrics import auc
    #Determine the ROC-curves using vertical averaging
    interp_tprs_all = np.zeros((101, performance_metrics_per_threshold_val.shape[2]))
    mean_fprs = np.linspace(0,1,101)
    aucs = []
    
    #Per cross-validation fold, determine the True Positive Rate (TPR) at set False Positive Rates 
    #using interpolation and the data retrieved per probability threshold
    for j in range(0,performance_metrics_per_threshold_val.shape[2]):    
        tprs = []
        fprs = []
        for i in range(0,performance_metrics_per_threshold_val.shape[0]):
            #Add all TPRs (= sensitivity) and FPRs (= 1-specificity) per probability threshold
            tpr = performance_metrics_per_threshold_val[i][0][j] 
            fpr = 1-performance_metrics_per_threshold_val[i][1][j]
            tprs.append(tpr)
            fprs.append(fpr)
        
        #Compute the AUC per cross-validation fold
        aucs.append(auc(fprs, tprs))
        #Interpolate the True Positive Rates at set False Positive Rates
        interp_tprs = np.interp(mean_fprs, fprs, tprs, period = 360)  
        interp_tprs[0] = 0.0
        interp_tprs_all[:,j] = interp_tprs  
    
    #Determine the mean and standard deviation of the True Positive Rates over all cross-validation folds
    mean_tprs = np.mean(interp_tprs_all, axis = 1)
    std_tprs = np.std(interp_tprs_all, axis=1)
    
    #Determine the mean +/- std, where 1 can be the max. value and 0 the min. value
    tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
    tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
    
    #Determine the median + IQR of the ROC-AUC
    median_auc = np.median(aucs)
    lower_iqr_auc = np.percentile(aucs, 25)
    upper_iqr_auc = np.percentile(aucs, 75)

    return [mean_fprs, mean_tprs, std_tprs, tprs_upper, tprs_lower, median_auc, lower_iqr_auc, upper_iqr_auc, aucs]
