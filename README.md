# Assessment of personally perceived image quality

![Personal image preference estimation](https://github.com/jarikorhonen/personal_image_preferences/blob/master/cvpr2019.png)

Since there is large deviation in personal opinions and aesthetic standards concerning image quality, it is a challenge is to find the settings and post-processing techniques that fit to the individual users’ personal taste. In this study, we aim to predict the personally perceived image quality by combining classical image feature analysis and collaboration filtering approach known from the recommendation systems. More information will be available in the paper (accepted for publication in CVPR 2019).

## Instructions for reproducing the results in the paper:

There are two main phases for generating the results: feature extraction (implementing in Matlab) and regression (implemented in Python). Features will be stored in CSV file that is used by the Python scripts. The used image databases are not included in the supplementary material, but they need to be downloaded from the respective websites:

CVD2013: http://www.helsinki.fi/~tiovirta/Resources/CID2013/
CEED2016: https://data.mendeley.com/datasets/3hfzp6vwkm/3

The original subjective scores are available from the respective websites. To facilitate data processing, we have included CSV files with the user-item rating matrices in the similar format in the supplementary material:


#### CID2013:

cid2013_rating_matrix_dataset_1.csv

cid2013_rating_matrix_dataset_2.csv

cid2013_rating_matrix_dataset_3.csv

cid2013_rating_matrix_dataset_4.csv

cid2013_rating_matrix_dataset_5.csv

cid2013_rating_matrix_dataset_6.csv 


#### CEED2016:

rating_matrix_ceed2016.csv

### Feature extraction:

extract_image_features.m implements the image feature vector extraction. 

Usage: feature_vec = compute_image_features(filename)
Input: filename: path to the image file
Output: image feature vector (12 features)

extract_cid2013_features.m extracts the features for CID2013 database and stores the values in the files named as cid2013_features_dataset_X.csv, where X is the dataset number (1-6).

extract_ceed2016_features.m extracts the features for CEED2016 database and stores the values in the file named as ceed2016_features.csv.

### Regression:

To repeat the experiments described in the paper, the following Python scripts are be provided:

predict_ratings_cid2013_test1.py

predict_ratings_cid2013_test2.py

predict_ratings_ceed2013_test1.py

predict_ratings_ceed2013_test2.py

It is assumed that the relevant image features files and the rating matrix files are available for the scripts.





