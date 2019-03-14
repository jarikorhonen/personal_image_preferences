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

import os
os.environ["MKL_NUM_THREADS"] = "1"

# Regression model for user-item ratings trained and validated
def predictRatings(array_R, min_arrR, max_arrR, item_feat, user_feat, mask, model): 

    # Make the training and validation sets based on mask
    F_train=[]
    F_validation=[]
    R_train=[]
    R_validation=[]  
    users = len(array_R[0,:])
    items = len(array_R[:,0])
    num_item_feats = len(item_feat[0,:])
    num_user_feats = len(user_feat[0,:])
    
    # Loop through the user-item pairs to allocate them in training and validation sets
    for i in range(0,items): 
        for u in range(0,users):
            f_vec=[]
            for j in range(0,num_item_feats):
                f_vec.append(item_feat[i][j])   
            for j in range(0,num_user_feats):
                f_vec.append(user_feat[u][j])
            if mask[i,u]>0:
                F_train.append(f_vec)   
                R_train.append(array_R[i,u])
            else:
                F_validation.append(f_vec)
                R_validation.append(array_R[i,u])
                
    scaler = preprocessing.MinMaxScaler().fit(F_train)
    F_train = scaler.transform(F_train)
    F_validation = scaler.transform(F_validation)
       
    model.fit(F_train,R_train)
    R_pred = model.predict(F_validation)
        
    # Compute the prediction performance metrics
    pr_pcc = scipy.stats.pearsonr(R_validation, R_pred)[0]
    pr_srcc = scipy.stats.spearmanr(R_validation,R_pred)[0]
    pr_rmse = np.sqrt(metrics.mean_squared_error(np.multiply(R_validation,(max_arrR-min_arrR)+min_arrR),np.multiply(R_pred,(max_arrR-min_arrR)+min_arrR)))
    
    return R_pred, pr_pcc, pr_srcc, pr_rmse


# Ratings read and scaled from the rating matrix file
def readRatings(dataset):

    # Get rating matrix
    filepath = "./cid2013_rating_matrix_dataset_%d.csv" % dataset    
    df = pandas.read_csv(filepath, skiprows=[], header=None)
    array_R = np.array(df.values, dtype=float)  
    
    # Get number of items and users
    users = len(array_R[0,:])
    items = len(array_R[:,0])
    
    # Normalize ratings from 0..100 to 0..1
    min_arrR = 0 #np.min(array_R)
    max_arrR = 100 #np.max(array_R)
    for i in range(0,items):
        for u in range(0,users):
            array_R[i,u] = float((array_R[i,u]-min_arrR)/(max_arrR-min_arrR))
    
    return users, items, array_R, min_arrR, max_arrR
 
        
# User latent features initialized randomly   
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


# Item features initialized, either from a file or randomly
def initializeItemFeatures(items, randfeat, dataset):
 
    if randfeat == False:        
        # Read image features from feature file 
        filepath = "./cid2013_features_dataset_%d.csv" % dataset   
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


def computeBaselineResults(array_R, min_arrR, max_arrR, mask):
    
    # Get number of items and users
    users = len(array_R[0,:])
    items = len(array_R[:,0])
    
    # Compute the baseline prediction results
    R_mean = []
    R_pred = []
    R_train=[]
    R_validation=[]
        
    for i in range(0,items): 
        R_mean.append(np.sum(np.multiply(array_R[i,:],mask[i,:]))/np.sum(mask[i,:])) 
        for u in range(0,users):
            if mask[i,u]>0:
                R_train.append(array_R[i,u])
            else:
                R_pred.append(R_mean[i])
                R_validation.append(array_R[i,u])
        
    R_train = np.array(R_train)
    R_pred = np.array(R_pred)
    R_validation = np.array(R_validation)
    
    # Compute baseline performation metrics
    bl_pcc = scipy.stats.pearsonr(R_validation,R_pred)[0]
    bl_srcc = scipy.stats.spearmanr(R_validation,R_pred)[0]
    bl_rmse = np.sqrt(metrics.mean_squared_error(np.multiply(R_validation,(max_arrR-min_arrR)+min_arrR),
                                                 np.multiply(R_pred,(max_arrR-min_arrR)+min_arrR)))
    
    return bl_pcc, bl_srcc, bl_rmse    
    
    
def makeRandomSplit(items, users, split, seednum):
    
    # Make mask for ratings included in training     
    rnd.seed(seednum)
    
    row_sum = np.zeros(items)
    col_sum = np.zeros(users)
    
    # Make sure that there is at least one rating for each item and user
    while min(row_sum)<1 or min(col_sum)<1:
        
        row_sum = np.zeros(items)
        col_sum = np.zeros(users)
        mask = np.zeros((items,users))
        for i in range(0,items):
            for u in range(0,users):
                if rnd.random()<split:
                    mask[i,u]=1  
                    row_sum[i] = row_sum[i] + 1
                    col_sum[u] = col_sum[u] + 1
                    
    return mask
                         
   
def predict_R(seednum, split, randfeat, dataset, num_user_feats, model):
    
    # Predict the ratings using the proposed scheme
    rnd.seed(seednum)
    users, items, array_R, min_arrR, max_arrR = readRatings(dataset)
    user_feat = initializeUserFeatures(users, num_user_feats)
    item_feat = initializeItemFeatures(items, randfeat, dataset)
    mask = makeRandomSplit(items, users, split, seednum)
    
    #print('** The results in test case',seednum,' **************************')
    R_pred, pr_pcc, pr_srcc, pr_rmse = predictRatings(array_R, min_arrR, max_arrR, item_feat, user_feat, mask, model)

    return pr_pcc, pr_srcc, pr_rmse


def predict_baseline_R(seednum, split, dataset):
    
    # Predict the ratings using the baseline scheme
    rnd.seed(seednum)
    users, items, array_R, min_arrR, max_arrR = readRatings(dataset)
    mask = makeRandomSplit(items, users, split, seednum)
    bl_pcc, bl_srcc, bl_rmse = computeBaselineResults(array_R, min_arrR, max_arrR, mask)
    
    return bl_pcc, bl_srcc, bl_rmse
    
# ===========================================================================
# Here starts the main part of the script
#
methoddescr=['Baseline','RandImFeats','RandImFeats','Proposed','Proposed']

out_file = open("./cid2013_results_test_1.txt","w") 
out_file.write('=========================================================================\n')

# Loop through random splits where 0.1, 0.2, 0.5 or 0.8 of datapoints 
# allocated for training
for split in [0.1,0.2,0.5,0.8]:
    
    
    print('************** Split:',split)
    
    # Loop through all the six CID2013 datasets
    for dataset in range(1,7):
        
        # Loop through different methods
        for method in range(0,5):
            full_SRCC = []
            full_PLCC = []
            full_RMSE = []
            
            if method == 0:
                modelname = 'None'
            if method == 2 or method == 4:
                model = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, max_depth=None, random_state=None)
                modelname = 'Random Forest'
            if method == 1 or method == 3:
                model = ensemble.GradientBoostingRegressor(n_estimators=100, min_samples_leaf=2, max_depth=5, random_state=None)
                modelname = 'Gradient Boosting'               
            
            # Try 100 different splits
            for seednum in range(1,101):
                if method==0:
                    pr_pcc, pr_srcc, pr_rmse = predict_baseline_R(seednum, split, dataset)  
                if method==1 or method==2:
                    pr_pcc, pr_srcc, pr_rmse = predict_R(seednum, split, True, dataset, 20, model)
                if method==3 or method==4:
                    pr_pcc, pr_srcc, pr_rmse = predict_R(seednum, split, False, dataset, 20, model)
    
                full_PLCC.append(pr_pcc)
                full_SRCC.append(pr_srcc)
                full_RMSE.append(pr_rmse)
            '''
            # Print the average results and standard deviations
            print('======================================================')
            print('Average results for dataset',dataset,'using',methoddescr[method%3],'method and',modelname,'model and',split,'split')
            print('PLCC: ',np.mean(full_PLCC),'( std:',np.std(full_PLCC),')')
            print('SRCC: ',np.mean(full_SRCC),'( std:',np.std(full_SRCC),')')
            print('RMSE: ',np.mean(full_RMSE),'( std:',np.std(full_RMSE),')')
            print('======================================================')
            '''
            # write the same information in a file      
            out_file.write('Average results, dataset %d using %s method, %s model, split: %1.2f\n' % (dataset,methoddescr[method],modelname,split))
            out_file.write('PLCC: %1.3f (std: %1.4f)\n' % (np.mean(full_PLCC),np.std(full_PLCC)))
            out_file.write('SRCC: %1.3f (std: %1.4f)\n' % (np.mean(full_SRCC),np.std(full_SRCC)))
            out_file.write('RMSE: %2.1f (std: %1.3f)\n' % (np.mean(full_RMSE),np.std(full_RMSE)))
            out_file.write('=========================================================================\n')
                  
out_file.close()

# The End
