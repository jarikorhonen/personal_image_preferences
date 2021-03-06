# Assessment of personally perceived image quality

![Personal image preference estimation](https://github.com/jarikorhonen/personal_image_preferences/blob/master/cvpr2019.png)

Since there is large deviation in personal opinions and aesthetic standards concerning image quality, it is a challenge is to find the settings and post-processing techniques that fit to the individual users’ personal taste. In this study, we aim to predict the personally perceived image quality by combining classical image feature analysis and collaboration filtering approach known from the recommendation systems. More information is available in the [CVPR'19 paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Korhonen_Assessing_Personally_Perceived_Image_Quality_via_Image_Features_and_Collaborative_CVPR_2019_paper.html).

## Instructions for reproducing the results in the paper:

There are two main phases for generating the results: feature extraction (implementing in Matlab) and regression (implemented in Python). Features will be stored in CSV file that is used by the Python scripts. The used image databases are not included in the supplementary material, but they need to be downloaded from the respective websites:

* CID2013: http://www.helsinki.fi/~tiovirta/Resources/CID2013/ *Not currently available*
* CEED2016: https://data.mendeley.com/datasets/3hfzp6vwkm/3

The original subjective scores are available from the respective websites. To facilitate data processing, we have included CSV files with the user-item rating matrices in the similar format in the supplementary material:

#### CID2013:

* cid2013_rating_matrix_dataset_1.csv
* cid2013_rating_matrix_dataset_2.csv
* cid2013_rating_matrix_dataset_3.csv
* cid2013_rating_matrix_dataset_4.csv
* cid2013_rating_matrix_dataset_5.csv
* cid2013_rating_matrix_dataset_6.csv 

#### CEED2016:

* rating_matrix_ceed2016.csv

### Feature extraction:

_extract_image_features.m_ implements the image feature vector extraction. 

__Usage:__ feature_vec = compute_image_features(filename)

__Input:__ filename: path to the image file

__Output:__ image feature vector (12 features)

_extract_cid2013_features.m_ extracts the features for CID2013 database and stores the values in the files named as _cid2013_features_dataset_X.csv_, where X is the dataset number (1-6).

_extract_ceed2016_features.m_ extracts the features for CEED2016 database and stores the values in the file named as _ceed2016_features.csv_.

### Regression:

To repeat the experiments described in the paper, the following Python scripts are be provided:

* predict_ratings_cid2013_test1.py
* predict_ratings_cid2013_test2.py
* predict_ratings_ceed2013_test1.py
* predict_ratings_ceed2013_test2.py

It is assumed that the relevant image features files and the rating matrix files are available for the scripts.


