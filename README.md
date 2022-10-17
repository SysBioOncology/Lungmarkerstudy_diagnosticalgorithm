# Lungmarkerstudy_diagnosticalgorithm

Code used in the paper 'Liquid biopsy-based decision support algorithms for diagnosis and subtyping of lung cancer' by Visser et al. 

In this paper, we used liquid biopsy data to train and validate logistic regression models for identification of **1) lung cancer (LC)**, **2) non-small-cell lung cancer (NSCLC)** and **3) small-cell lung cancer (SCLC)**. Cross-validation was performed to allow for validation of the model on patients who were not used for training of the model, i.e. the validation set. To give better insights in the clinical applicability of these models, the performance of the validation set was evaluated at a probability threshold at which a pre-specified PPV could be achieved in the training set. This pre-specified PPV was 98% for the LC model (1) and 95% for the NSCLC and SCLC models (2 and 3). 

Input data used was: 8 protein tumor markers (TMs) (CA125, CA15.3, CEA, CYFRA 21-1, HE4, NSE, proGRP and SCCA), 2 circulating tumor DNA (ctDNA) markers (concentration of cell-free DNA (wild-type *KRAS*) and presence of mutation in *KRAS*, *EGFR* or *BRAF*), age and sex. We determined the optimal combination of protein TMs for the 3 different classification tasks using Recursive Feature Elimination (RFE).

Output data used in the 3 different classification tasks were:
1) No LC (class 0) vs. LC (class 1)
2) No NSCLC (= no LC + SCLC) (class 0) vs. NSCLC (class 1)
3) No SCLC (= no LC + NSCLC) (class 0) vs. SCLC (class 1)

The code can be used for 2 purposes:
- Training models with the same logistic regression pipeline using own input features: files are stored in the folder *model_training*
 
  - Training using the 'optimal' set of markers

![Pipeline_logreg](Pipeline_logistic_regression.PNG)

  - The optimal set of protein TMs was determined using recursive feature elimination. This pipeline is similar to the previous logistic regression pipeline, except that the x most important features will be selected per cross-validation fold. These models were run for x = 1 to 8 (n_features_to_select) and the performances were compared to determine the best performance. 
  
![Pipeline_logreg_RFE](Pipeline_logistic_regression_RFE.png)  
  
- Applying the models from the articles to new patients for predicting of probabilities

