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

# Compute the results using the proposed method
def computeProposedResults(array_R, min_arrR, max_arrR, miss_item, item_feat, user_feat, model):
    
    # Intialize
    users = len(array_R[0,:])
    items = len(array_R[:,0])   
    R_train=[]
    R_validation=[]
    F_train=[]
    F_validation=[]
    num_item_feats = len(item_feat[0,:])
    num_user_feats = len(user_feat[0,:])
        
    # Loop through users and items to make training and testing sets
    for i in range(0,items): 
        for u in range(0,users):
            f_vec=[]
            for j in range(0,num_item_feats):
                f_vec.append(item_feat[i][j])
            for j in range(0,num_user_feats):
                f_vec.append(user_feat[u][j])            
            if i != miss_item:        
                R_train.append(array_R[i,u])
                F_train.append(f_vec)
            else:
                R_validation.append(array_R[i,u])
                F_validation.append(f_vec)
    
    # Train and validate the model             
    scaler = preprocessing.MinMaxScaler().fit(F_train)
    F_train = scaler.transform(F_train)
    F_validation = scaler.transform(F_validation)       
    model.fit(F_train,R_train)
    R_pred = model.predict(F_validation)    

    # Return the results
    return R_validation, R_pred    


# Compute the results using the baseline method
def computeBaselineResults(array_R, min_arrR, max_arrR, miss_item, item_feat, model):
    
    # Initialize
    users = len(array_R[0,:])
    items = len(array_R[:,0])
    R_train=[]
    R_validation=[]
    F_train=[]
    F_validation=[]
    num_item_feats = len(item_feat[0,:])
     
     # Loop through users and items to make training and testing sets
    for i in range(0,items): 
        f_vec=[]
        for j in range(0,num_item_feats):
            f_vec.append(item_feat[i][j])           
        if i != miss_item:        
            R_train.append(np.mean(array_R[i,:]))
            F_train.append(f_vec)
        else:
            for u in range(0,users):
                R_validation.append(array_R[i,u])
                F_validation.append(f_vec)
     
    # Train and validate the baseline model              
    scaler = preprocessing.MinMaxScaler().fit(F_train)
    F_train = scaler.transform(F_train)
    F_validation = scaler.transform(F_validation)       
    model.fit(F_train,R_train)
    R_pred = model.predict(F_validation)  
    
    return R_validation, R_pred    


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

# Item features read from a file
def initializeItemFeatures(items, dataset):
         
    # Read image features from feature file 
    filepath = "./cid2013_features_dataset_%d.csv" % dataset   
    df = pandas.read_csv(filepath, skiprows=[], header=None)
    item_feat = np.array(df.values, dtype=float)
    
    return item_feat


# Main function for the proposed method 
def predict_R(dataset, num_user_feats, model):
    
    # Predict the ratings using the proposed scheme
    users, items, array_R, min_arrR, max_arrR = readRatings(dataset)
    user_feat = initializeUserFeatures(users, num_user_feats)
    item_feat = initializeItemFeatures(items, dataset)
    R_validation = []
    R_pred = []
    
    # Loop through all the items and aggregate predicted data points
    for i in range(0,items):
        r_val, r_pred = computeProposedResults(array_R, min_arrR, max_arrR, i, item_feat, user_feat, model)
        R_validation.extend(r_val)
        R_pred.extend(r_pred)
     
    # Compute results
    pr_pcc = scipy.stats.pearsonr(R_validation,R_pred)[0]
    pr_srcc = scipy.stats.spearmanr(R_validation,R_pred)[0]
    pr_rmse = np.sqrt(metrics.mean_squared_error(np.multiply(R_validation,(max_arrR-min_arrR)+min_arrR),
                                                 np.multiply(R_pred,(max_arrR-min_arrR)+min_arrR)))

    return pr_pcc, pr_srcc, pr_rmse


# Main function for the baseline method 
def predict_baseline_R(dataset, model):
    
    # Predict the ratings using the baseline scheme
    users, items, array_R, min_arrR, max_arrR = readRatings(dataset)
    item_feat = initializeItemFeatures(items, dataset)
    R_validation = []
    R_pred = []
    
    # Loop through all the items and aggregate predicted data points
    for i in range(0,items):
        r_val, r_pred = computeBaselineResults(array_R, min_arrR, max_arrR, i, item_feat, model)
        R_validation.extend(r_val)
        R_pred.extend(r_pred)
     
    # Compute results
    bl_pcc = scipy.stats.pearsonr(R_validation,R_pred)[0]
    bl_srcc = scipy.stats.spearmanr(R_validation,R_pred)[0]
    bl_rmse = np.sqrt(metrics.mean_squared_error(np.multiply(R_validation,(max_arrR-min_arrR)+min_arrR),
                                                 np.multiply(R_pred,(max_arrR-min_arrR)+min_arrR)))
    
    return bl_pcc, bl_srcc, bl_rmse
  
    
# ===========================================================================
# Here starts the main part of the script
#
methoddescr=['Baseline','Proposed']

out_file = open("./cid2013_results_test_2.txt","w") 
out_file.write('=========================================================================\n')

# Loop through all the six CID2013 datasets
for dataset in range(1,7):
    
    # Loop through different methods
    for method in range(0,4):
      
        if method < 2:
            model = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, max_depth=None, random_state=None)
            modelname = 'Random Forest'
        if method > 1:
            model = ensemble.GradientBoostingRegressor(n_estimators=100, min_samples_leaf=2, max_depth=5, random_state=None)
            modelname = 'Gradient Boosting'     
                   
        if method%2==0:
            full_PLCC, full_SRCC, full_RMSE = predict_baseline_R(dataset, model)                
        if method%2==1:
            full_PLCC, full_SRCC, full_RMSE = predict_R(dataset, 20, model)

        '''
        # Print the average results and standard deviations
        print('======================================================')
        print('Average results for dataset',dataset,'using',methoddescr[method%2],'method and',modelname,'model')
        print('PLCC: ',np.mean(full_PLCC),'( std:',np.std(full_PLCC),')')
        print('SRCC: ',np.mean(full_SRCC),'( std:',np.std(full_SRCC),')')
        print('RMSE: ',np.mean(full_RMSE),'( std:',np.std(full_RMSE),')')
        print('======================================================')
        '''
        
        # write the same information in a file      
        out_file.write('Average results for dataset %d using %s method and %s model\n' % (dataset,methoddescr[method%2],modelname))
        out_file.write('PLCC: %1.4f\n' % (np.mean(full_PLCC)))
        out_file.write('SRCC: %1.4f\n' % (np.mean(full_SRCC)))
        out_file.write('RMSE: %2.2f\n' % (np.mean(full_RMSE)))
        out_file.write('=========================================================================\n')

out_file.close()

# The End