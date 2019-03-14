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


# This function trains and validates the regression model for predicting ratings
def predictRatings(array_R, min_arrR, max_arrR, item_feat, user_feat, mask, model): 

    # Make the training and validation sets based on mask
    F_train=[]
    F_validation=[]
    R_train=[]
    R_validation=[]  
    users = len(array_R[0,:])
    subitems = 6
    items = round(len(array_R[:,0])/subitems)
    num_item_feats = len(item_feat[0,:])
    num_user_feats = len(user_feat[0,:])
    
    subitem_list = range(0,subitems) #np.random.permutation(subitems)
    for u in range(0,users):                       
        for i in range(0,items):
            f_vec = []
            r_vec = []
            for s in subitem_list:
                r_vec.append(array_R[i*subitems+s,u])
                
                for j in range(0,num_item_feats):
                    f_vec.append(item_feat[i*subitems+s][j])   
            for j in range(0,num_user_feats):
                f_vec.append(user_feat[u][j])
            if mask[i*subitems,u]>0:
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
    R_validation = np.concatenate(R_validation)
    R_pred = np.concatenate(R_pred)
    R_train = np.concatenate(R_train)
    
    # Compute results    
    pr_pcc = scipy.stats.pearsonr(R_validation, R_pred)[0]
    pr_srcc = scipy.stats.spearmanr(R_validation,R_pred)[0]
    pr_rmse = np.sqrt(metrics.mean_squared_error(np.multiply(R_validation,(max_arrR-min_arrR)+min_arrR),np.multiply(R_pred,(max_arrR-min_arrR)+min_arrR)))
    pr_bpa = best_prediction_accuracy(R_pred,R_validation,subitems)   
    
    return R_pred, pr_pcc, pr_srcc, pr_rmse, pr_bpa


# This function reads the ratings from the rating matrix
def readRatings():

    # Get rating matrix
    filepath = "./ceed2016_rating_matrix.csv"       
    df = pandas.read_csv(filepath, skiprows=[], header=None)
    array_R = np.array(df.values, dtype=float)  
    
    # Get number of items and users
    users = len(array_R[0,:])
    items = len(array_R[:,0])
    
    # Normalize ratings from 0..100 to 0..1
    min_arrR = 0 #np.min(array_R)
    max_arrR = 1 #np.max(array_R)
    for i in range(0,items):
        for u in range(0,users):
            array_R[i,u] = float((array_R[i,u]-min_arrR)/(max_arrR-min_arrR))
    
    return users, items, array_R, min_arrR, max_arrR
 

# This function initializes the user features randomly           
def initializeUserFeatures(users, num_user_feats):       
    
    # Generate random features for users
    user_feat = []
    for i in range(0,users):  
        randvec = []
        for j in range(0,num_user_feats):
            randvec.append(rnd.random())
        user_feat.append(randvec)
        
    scaler = preprocessing.MinMaxScaler().fit(user_feat)
    user_feat = scaler.transform(user_feat)
    
    return user_feat


# This function initializes the item features, either by reading them
# from the feature file, or randomly for RandImFeats
def initializeItemFeatures(items, randfeat):
 
    if randfeat == False:        
        # Read image features from feature file        
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
def computeBaselineResults(array_R, min_arrR, max_arrR, mask):
    
    # Initialize
    users = len(array_R[0,:])
    items = len(array_R[:,0])
    R_mean = []
    R_pred = []
    R_train = []
    R_validation=[]
    subitems = 6
     
    # Compute baseline results
    for u in range(0,users): 
        for i in range(0,items):
            R_mean = np.sum(np.multiply(array_R[i,:],mask[i,:]))/np.sum(mask[i,:])
            if mask[i,u]>0:
                R_train.append(array_R[i,u])
            else:
                R_pred.append(R_mean)
                R_validation.append(array_R[i,u])
        
    
    # Compute statistics
    bl_pcc = scipy.stats.pearsonr(R_validation,R_pred)[0]
    bl_srcc = scipy.stats.spearmanr(R_validation,R_pred)[0]
    bl_rmse = np.sqrt(metrics.mean_squared_error(np.multiply(R_validation,(max_arrR-min_arrR)+min_arrR),
                                                 np.multiply(R_pred,(max_arrR-min_arrR)+min_arrR)))
    bl_papi = best_prediction_accuracy(R_pred,R_validation,subitems)
    
    return bl_pcc, bl_srcc, bl_rmse, bl_papi    
    

# This function splits the dataset randomly into training and testing sets   
def makeRandomSplit(items, subitems, users, split, seednum):
    
    # Make mask for ratings included in training     
    items = round(items/subitems)
    rnd.seed(seednum)
    row_sum = np.zeros(items)
    col_sum = np.zeros(users)
    
    # Make sure that there are at least one rating for each item and user
    while min(row_sum)==0 or min(col_sum)==0:
        
        row_sum = np.zeros(items*subitems)
        col_sum = np.zeros(users)
        mask = np.zeros((items*subitems,users))
        for i in range(0,items*subitems,subitems):
            for u in range(0,users):
                if rnd.random()<split:
                    for ii in range(i,i+subitems):
                        mask[ii,u]=1  
                        row_sum[ii] = row_sum[ii] + 1
                        col_sum[u] = col_sum[u] + 1    
              
    return mask
                         

# This function predicts the ratings using the proposed method   
def predict_R(seednum, split, randfeat, num_user_feats, model):
    
    rnd.seed(seednum)
    users, items, array_R, min_arrR, max_arrR = readRatings()
    user_feat = initializeUserFeatures(users, num_user_feats)
    item_feat = initializeItemFeatures(items, randfeat)
    mask = makeRandomSplit(items, 6, users, split, seednum)
    R_pred, pr_pcc, pr_srcc, pr_rmse, pr_papi = predictRatings(array_R, min_arrR, max_arrR, item_feat, user_feat, mask, model)

    return pr_pcc, pr_srcc, pr_rmse, pr_papi


# This function predicts the ratings using the proposed method  
def predict_baseline_R(seednum, split):
    
    # Predict the ratings using the baseline scheme
    rnd.seed(seednum)
    users, items, array_R, min_arrR, max_arrR = readRatings()
    mask = makeRandomSplit(items, 6, users, split, seednum)
    bl_pcc, bl_srcc, bl_rmse, bl_papi = computeBaselineResults(array_R, min_arrR, max_arrR, mask)
    
    return bl_pcc, bl_srcc, bl_rmse, bl_papi
    
# ===========================================================================
# Here starts the main part of the script
#
methoddescr=['Baseline','RandImFeats','RandImFeats','Proposed','Proposed']

out_file = open("./ceed2016_results_test1.txt","w") 
out_file.write('=========================================================================\n')

# Loop through different training dataset proportions
for split in [0.1,0.2,0.5,0.8]:
    PAPI_bl = []
    PAPI_bl_std = []
    PAPI_randim_rf = []
    PAPI_randim_rf_std = []
    PAPI_prop_rf = []
    PAPI_prop_rf_std = []
    PAPI_randim_gb = []
    PAPI_randim_gb_std = []
    PAPI_prop_gb = []
    PAPI_prop_gb_std = []
    
    # Loop through different methods
    for method in range(0,5):
        full_SRCC = []
        full_PLCC = []
        full_RMSE = []
        full_PAPI = []
                       
        if method == 0:
            modelname = 'None'
        if method == 2 or method == 4:
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, min_samples_leaf=1, max_depth=None, random_state=None))
            modelname = 'Random Forest'
        if method == 1 or method == 3:
            model = MultiOutputRegressor(ensemble.GradientBoostingRegressor(n_estimators=100, min_samples_leaf=1, max_depth=5, random_state=None))
            modelname = 'Gradient Boosting'              
            
        # Try 100 different splits
        for seednum in range(1,101):
            if method==0:
                pr_pcc, pr_srcc, pr_rmse, pr_papi = predict_baseline_R(seednum, split)                
            if method==1 or method==2:
                pr_pcc, pr_srcc, pr_rmse, pr_papi = predict_R(seednum, split, True, 20, model)
            if method==3 or method==4: 
                pr_pcc, pr_srcc, pr_rmse, pr_papi = predict_R(seednum, split, False, 20, model)
    
            full_PLCC.append(pr_pcc)
            full_SRCC.append(pr_srcc)
            full_RMSE.append(pr_rmse)
            full_PAPI.append(pr_papi)
        
        # Print the average results and standard deviations
        print('======================================================')
        print('Average results using',methoddescr[method],'method and',modelname,'model, split',split)
        print('PLCC: ',np.mean(full_PLCC),'( std:',np.std(full_PLCC),')')
        print('SRCC: ',np.mean(full_SRCC),'( std:',np.std(full_SRCC),')')
        print('RMSE: ',np.mean(full_RMSE),'( std:',np.std(full_RMSE),')')
        print('PAPI: ',np.mean(full_PAPI),'( std:',np.std(full_PAPI),')')
        print('======================================================')
            
        # write the same information in a file      
        out_file.write('Average results using %s method and %s model, split %1.1f\n' % (methoddescr[method],modelname,split))
        out_file.write('PLCC: %1.3f (std: %1.4f)\n' % (np.mean(full_PLCC),np.std(full_PLCC)))
        out_file.write('SRCC: %1.3f (std: %1.4f)\n' % (np.mean(full_SRCC),np.std(full_SRCC)))
        out_file.write('RMSE: %1.3f (std: %1.4f)\n' % (np.mean(full_RMSE),np.std(full_RMSE)))
        out_file.write('PAPI: %1.3f (std: %1.4f)\n' % (np.mean(full_PAPI),np.std(full_PAPI)))
        out_file.write('=========================================================================\n')

out_file.close()

# The End