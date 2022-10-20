# -*- coding: utf-8 -*-
"""
Determine the sensitivity, specificity, PPV and NPV
"""

def performance_metrics(probability_threshold,y_pred,y_true):
    import numpy as np
    TP = 0.0
    TN = 0.0
    FN = 0.0
    FP = 0.0
    for i in range(0,len(y_pred)):
        #For each patient, determine whether the predicted probability is below (negative) 
        #or above (positive) the set probability threshold. Compare with the true class
        #to categorize patient as true negative/positive or false negative/positive
        if np.round(y_pred[i],2) < probability_threshold:
            if y_true[i] == 0:
                TN += 1.0
            elif y_true[i] == 1:
                FN += 1.0
        elif np.round(y_pred[i],2) >= probability_threshold:
            if y_true[i] == 1:
                TP += 1.0
            elif y_true[i] == 0:
                FP += 1.0
    #Compute the sensitivity, specificity, PPV and NPV, according to common formulas
    #Set the NPV and PPV to zero when no patients have probabilities above/below the probability threshold
    #(and therefore the denumerator will become zero)
    if TP != 0:
        sensitivity = TP/(TP+FN)
        PPV = TP/(TP+FP)
    else:
        sensitivity = 0.0
        PPV = 0.0
    if TN != 0:
        specificity = TN/(TN+FP)
        NPV = TN/(TN+FN)
    else:
        specificity = 0.0
        NPV = 0.0
    
    
    return [sensitivity,specificity,PPV,NPV]


    
        
        
            
