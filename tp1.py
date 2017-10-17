# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 21:19:26 2017

@author: Daniel
"""



####### GUIDELINES #######
# (Done) Process the data correctly, including randomizing the order of the data points and standardizing the values.
# Determine the parameters with cross validation on two thirds of the data, leaving one third out for testing.
# For the regularization parameter of the logistic regression classifier, start with a C value of 1 and double it at each iteration for 20 iterations. Plot the errors against the logarithm of the C value.
# For the k value of the K-Nearest Neighbours classifier, test k values from 1 to 39 using odd values only.
# Use the same bandwidth value for all the Kernel Density Estimators in your Naive Bayes classifier, and try values from 0.01 to 1 with a step of 0.02.
# (Done) When splitting your data, for testing and for cross validation, use stratified sampling.
# (Done) Use 5 folds for cross validation
# Use the fraction of incorrect classifications as the measure of the error. This is equal to 1-accuracy, and the accuracy can be obtained with the score method of the logistic regression and KNN classifiers in Scikit-learn.
# For the NB classifier, you can implement your own measure of the accuracy or use the accuracy_score function in the metrics module.
# For comparing the classifiers, use McNemar's test with a 95% confidence interval




# Importing the libraries required
import numpy as np
from sklearn import preprocessing 
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression

#################################### Data #####################################

# Step 1 - Process Data
def input_data(file_name):
    """ Function to load, shuffle and standardize the data from the data file """
    mat = np.loadtxt(file_name, delimiter=',')
    np.random.shuffle(mat)
    features = preprocessing.scale(mat[:,0:3])
    class_labels = mat[:,4]
    return features, class_labels

features, class_labels = input_data("TP1-data.csv")

############################# Logistic Regression #############################

# Step 1 - Split data using stratified sampling to use on Cross Validation 
def split_data(features, class_labels, size):
    """ Split data into a stratified train set and test set. 
    The test set size is 1-train_size """
    X_r, X_t, Y_r, Y_t = train_test_split(features, class_labels, train_size = size, stratify = class_labels)
    return X_r, X_t, Y_r, Y_t

X_r, X_t, Y_r, Y_t = split_data(features, class_labels, 0.66)
    

# Step 2 - Get 5 folds to use on Cross Validation

def fold(Y_r, folds):
    kf = StratifiedKFold(Y_r, n_folds = folds)
    return kf

folds = 5    
kfolds = fold(Y_r, folds)
        
# Step 3 - Compute train and validation error
# TODO, im not sure how - For the regularization parameter of the logistic regression classifier, 
# start with a C value of 1 and double it at each iteration for 20 iterations.
# TODO - Plot the errors against the logarithm of the C value.
# - (Done) Use the fraction of incorrect classifications as the measure of the error.
# This is equal to 1-accuracy, and the accuracy can be obtained with the score 
# method of the logistic regression and KNN classifiers in Scikit-learn. 
def calc_fold(features, X, Y, train_subset, valid_subset, C=1e12):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=C, tol=1e-10)
    X_train = X[train_subset,:features]
    Y_train = Y[train_subset]
    X_valid = X[valid_subset,:features]
    Y_valid = Y[valid_subset]
    reg.fit(X_train, Y_train)
    # fraction of incorrect classifications
    # TODO not sure if this is correct
    train_error = 1-reg.score(X_train, Y_train) 
    valid_error = 1-reg.score(X_valid, Y_valid)
    return train_error, valid_error

def logRegression():
    for features in range(2, 5):
        train_error = valid_error = 0
        for train_subset, valid_subset in kfolds:
            r, v = calc_fold(features, X_r, Y_r, train_subset, valid_subset)
            train_error += r
            valid_error += v
            print(features, ':', train_error/folds, valid_error/folds)