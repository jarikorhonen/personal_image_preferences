# -*- coding: utf-8 -*-
"""
@author: x
"""

# Load libraries
import pandas
from sklearn import ensemble
import random as rnd
import scipy.stats
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import preprocessing
from sklearn.multioutput import MultiOutputRegressor

import os
os.environ["MKL_NUM_THREADS"] = "1"

# Compute the accuracy of predicting the most preferred image among the
# six post-processed images
def best_prediction_accuracy(R_pred,R_validation,subitems):
    
    n_corr_pred=0
    n_pred=0
    for i in range(0,len(R_pred),subitems): 
        n_pred = n_pred + 1
        m1 = R_pred[i:i+subitems]
        m2 = R_validation[i:i+subitems]
        corr_pred = False
        for j in range(0,subitems):
            if m1[j]==max(m1) and m2[j]==max(m2):
                corr_pred = True
        if corr_pred==True:
            n_corr_pred = n_corr_pred + 1
    
    pred_accuracy = n_corr_pred/n_pred
    return pred_accuracy


# This function reads the ratings from the rating matrix
def readRatings():

    # Get rating matrix
    filepath = "./ceed2016_rating_matrix.csv"       
    df = pandas.read_csv(filepath, skiprows=[], header=None)
    array_R = np.array(df.values, dtype=float)  
    
    # Get number of items and users
    users = len(array_R[0,:])
    items = len(array_R[:,0])
    
    # Normalize ratings 
    min_arrR = 0 #np.min(array_R)
    max_arrR = 1 #np.max(array_R)
    for i in range(0,items):
        for u in range(0,users):
            array_R[i,u] = float((array_R[i,u]-min_arrR)/(max_arrR-min_arrR))
    
    return users, items, array_R, min_arrR, max_arrR


# This function initializes the user features randomly             
def initializeUserFeatures(users, num_user_feats):       
    
    rnd.seed(1)
    user_feat = []
    for i in range(0,users):  
        randvec = []
        for j in range(0,num_user_feats):
            randvec.append(rnd.random())
        user_feat.append(randvec)
        
    scaler = preprocessing.MinMaxScaler().fit(user_feat)
    user_feat = scaler.transform(user_feat)
    
    return user_feat


# This function reads item features from the feature file
def initializeItemFeatures(items, randfeat):
 
    if randfeat == False:            
        filepath = "./ceed2016_features.csv"    
        df = pandas.read_csv(filepath, skiprows=[], header=None)
        item_feat = np.array(df.values, dtype=float)
    
    else:    
        # Generate random features for items
        item_feats = 12
        item_feat = []
        for i in range(0,items):  
            randvec = []
            for j in range(0,item_feats):
                randvec.append(rnd.random())
            item_feat.append(randvec)
        scaler = preprocessing.MinMaxScaler().fit(item_feat)
        item_feat = scaler.transform(item_feat)
    
    return item_feat


# This function computes the baseline results
def computeBaselineResults(array_R, min_arrR, max_arrR, miss_item):
    
    # Initialize
    users = len(array_R[0,:])
    items = len(array_R[:,0])
    R_mean = []
    R_pred = []
    R_validation=[]
    subitems = 6
    
    item_vec = []
    for j in range(0,items,subitems):
        if j != miss_item:
            item_vec.append(j)
            
    # Loop through users and images to compute baseline predictions
    for u in range(0,users): 
        R_mean = []
        for j in range(0,subitems):
            item_vec_temp = [v+j for v in item_vec]    
            R_mean.extend([np.mean(array_R[item_vec_temp,u])])
        for i in range(0,items,subitems):
            if i == miss_item:
                r_vec = []
                for j in range(i,i+subitems):
                    r_vec.append(array_R[j,u])           
                R_pred.extend(R_mean)
                R_validation.extend(r_vec)
    
    return R_validation, R_pred 

# This function splits the dataset randomly into training and testing sets 
def computeProposedResults(array_R, min_arrR, max_arrR, miss_item, item_feat, user_feat, model):
    
    # Initialize
    users = len(array_R[0,:])
    items = len(array_R[:,0])
    F_train=[]
    F_validation=[]
    R_train=[]
    R_validation=[]  
    subitems = 6
    items = round(len(array_R[:,0])/subitems)
    num_item_feats = len(item_feat[0,:])
    num_user_feats = len(user_feat[0,:])
    
    # Allocate ratings and features to training and testing sets
    for i in range(0,items):
        for u in range(0,users):                
            f_vec = []
            r_vec = []
            for s in range(0,subitems):
                r_vec.append(array_R[i*subitems+s,u])                
                for j in range(0,num_item_feats):
                    f_vec.append(item_feat[i*subitems+s][j])               
            for j in range(0,num_user_feats):
                f_vec.append(user_feat[u][j])
            
            if i != miss_item:
                F_train.append(f_vec)   
                R_train.append(r_vec)
            else:
                F_validation.append(f_vec)
                R_validation.append(r_vec)

    # Train and validate the regression model             
    scaler = preprocessing.MinMaxScaler().fit(F_train)
    F_train = scaler.transform(F_train)
    F_validation = scaler.transform(F_validation)       
    model.fit(F_train,R_train)
    R_pred = model.predict(F_validation)   

    return R_validation, R_pred    


# This function predicts the ratings using the proposed scheme
def predict_R(randfeat, num_user_feats, model):
    
    
    users, items, array_R, min_arrR, max_arrR = readRatings()
    subitems = 6;
    user_feat = initializeUserFeatures(users, num_user_feats)
    item_feat = initializeItemFeatures(items, randfeat)
    R_val = []
    R_pre = []
    
    for i in range(0,np.int(np.floor(items/subitems))):
        r_val, r_pred = computeProposedResults(array_R, min_arrR, max_arrR, i, item_feat, user_feat, model)
        R_val.extend(r_val)
        R_pre.extend(r_pred)
        
    R_validation = [item for sublist in R_val for item in sublist]
    R_pred = [item for sublist in R_pre for item in sublist] 
    
    pr_pcc = scipy.stats.pearsonr(R_validation,R_pred)[0]
    pr_srcc = scipy.stats.spearmanr(R_validation,R_pred)[0]
    pr_rmse = np.sqrt(metrics.mean_squared_error(np.multiply(R_validation,(max_arrR-min_arrR)+min_arrR),
                                                 np.multiply(R_pred,(max_arrR-min_arrR)+min_arrR)))
    pr_bpa = best_prediction_accuracy(R_pred,R_validation,6)

    return pr_pcc, pr_srcc, pr_rmse, pr_bpa


# This function predicts the ratings using the baseline scheme
def predict_baseline_R():
    
    # Predict the ratings using the baseline scheme
    users, items, array_R, min_arrR, max_arrR = readRatings()
    subitems = 6;
    R_validation = []
    R_pred = []
    
    #print('** The results in test case',seednum,' **************************')
    for i in range(0,items,subitems):
        r_val, r_pred = computeBaselineResults(array_R, min_arrR, max_arrR, i)
        R_validation.extend(r_val)
        R_pred.extend(r_pred)
        
    pr_pcc = scipy.stats.pearsonr(R_validation,R_pred)[0]
    pr_srcc = scipy.stats.spearmanr(R_validation,R_pred)[0]
    pr_rmse = np.sqrt(metrics.mean_squared_error(np.multiply(R_validation,(max_arrR-min_arrR)+min_arrR),
                                                 np.multiply(R_pred,(max_arrR-min_arrR)+min_arrR)))
    pr_bpa = best_prediction_accuracy(R_pred,R_validation,6)

    return pr_pcc, pr_srcc, pr_rmse, pr_bpa    

    
# ===========================================================================
# Here starts the main part of the script
#
methoddescr=['Baseline','RandImFeats','Proposed']

out_file = open("./ceed2016_results_test_2.txt","w") 
out_file.write('=========================================================================\n')

  
# Loop through different methods
for method in range(0,3):
    full_SRCC = []
    full_PLCC = []
    full_RMSE = []
    full_PAPI = []
        
    if method < 2:
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_depth=None, random_state=None))
        modelname = 'Random Forest'
    if method > 1:
        model = MultiOutputRegressor(ensemble.GradientBoostingRegressor(n_estimators=100, min_samples_leaf=2, max_depth=5, random_state=8))
        modelname = 'Gradient Boosting'              
        
    if method==0:
        pr_pcc, pr_srcc, pr_rmse, pr_papi = predict_baseline_R()                
    if method>0:
        pr_pcc, pr_srcc, pr_rmse, pr_papi = predict_R(False, 20, model)

    full_PLCC.append(pr_pcc)
    full_SRCC.append(pr_srcc)
    full_RMSE.append(pr_rmse)
    full_PAPI.append(pr_papi)
    
    # Print the average results and standard deviations
    print('======================================================')
    print('Average results using',methoddescr[method%3],'method and',modelname,'model')
    print('PLCC: ',np.mean(full_PLCC),'( std:',np.std(full_PLCC),')')
    print('SRCC: ',np.mean(full_SRCC),'( std:',np.std(full_SRCC),')')
    print('RMSE: ',np.mean(full_RMSE),'( std:',np.std(full_RMSE),')')
    print('PAPI: ',np.mean(full_PAPI),'( std:',np.std(full_PAPI),')')
    print('======================================================')
        
    # write the same information in a file      
    out_file.write('Average results using %s method and %s model\n' % (methoddescr[method%3],modelname))
    out_file.write('PLCC: %1.3f (std: %1.4f)\n' % (np.mean(full_PLCC),np.std(full_PLCC)))
    out_file.write('SRCC: %1.3f (std: %1.4f)\n' % (np.mean(full_SRCC),np.std(full_SRCC)))
    out_file.write('RMSE: %1.3f (std: %1.3f)\n' % (np.mean(full_RMSE),np.std(full_RMSE)))
    out_file.write('PAPI: %1.3f (std: %1.3f)\n' % (np.mean(full_PAPI),np.std(full_PAPI)))
    out_file.write('=========================================================================\n')

out_file.close()

# The End