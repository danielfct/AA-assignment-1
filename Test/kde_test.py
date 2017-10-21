# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 17:41:07 2017

@author: Andrea
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:40:15 2017

@author: Andrea
"""

# -*- coding: utf-8 -*-

#Loading the relevant libraries
import pandas as pd
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


############ FUNCTIONS TO PREPROCESS DATA #####################################
def reading_csv(filename):
    """This function reads a csv file a returns
    a data matrix and a vector labels.
    We are assuming the labels are in the last position."""
    data = pd.read_csv(filename)
    last_column_index= data.shape[1] - 1 #0-based numeration
    X= data.iloc[:,0:last_column_index]
    y= data.iloc[:,last_column_index:]
    return X, y


def split_training(data_matrix, label_vector, train_size, split_seed):
    """This function takes as an input the Data Matrix and the
    Label Vector we are working with, shuffles the data according
    to a Seed given as an input and splits according to the proportion
    decided by the user.
    It returns four elements: the training and test Data Matrix,
    the training and testing Label Vectors"""
    X_train, X_test, y_train, y_test= train_test_split(data_matrix, 
                                                       label_vector,
                                                       train_size= train_size,
                                                       random_state= split_seed,
                                                       shuffle= True,
                                                       stratify= label_vector)
    return X_train, X_test, y_train, y_test


def normalise(X_train, X_test):
    """This function performs a normalisation by subtracting the mean
    and then dividing by the standard deviation.
    It computes the mean value and standard deviation only on the training
    matrix and apply the same transformation to both matrices."""
    # Create a standard processor object
    scaler= StandardScaler()
    # Create an object to transform the data to fit standard processor
    X_scaled= scaler.fit_transform(X_train)
    # Run the normalizer on the dataframes
    X_train = pd.DataFrame(X_scaled)
    X_test= pd.DataFrame(scaler.transform(X_test))
    return X_train, X_test


def rename_columns(X_train, X_test, y_train, y_test, feature_names, label_name):
    """This function changes the name of the columns of training and
    data set and of the label vectors, using the same for the two.
    It assumes that Column_Names and Label_Name are tuple of strings of the
    right dimensions."""
    X_train.columns= feature_names
    y_train.columns= label_name
    X_test.columns= feature_names
    y_test.columns= label_name
    return X_train, X_test, y_train, y_test


def display_data(data_matrix, label_vector):
    """This function allows to visualise the data and print
    some information"""
    print("Printing the first ten rows...")
    print(data_matrix.head(10)) #printing the first ten rows
    print("\nPrinting summary statistics...")
    print(data_matrix.describe()) #Getting some information
    print("\nPrinting scatterplots considering the class")
    
    pd.plotting.scatter_matrix(data_matrix, alpha=0.8, figsize=(6, 6),
                               diagonal='kde', c= label_vector)
    plt.show()
    plt.close()


def preprocess_data(filename, train_size, split_seed, feature_names, label_name=['Class']):
    """Encapsulating all the work done insofar.
        Taking as input the filename, the relative
        size of the training set, the seed for the
        random reshuffling and the names we are willing to give to the columns"""
    X, y= reading_csv(filename)
    X_train, X_test, y_train, y_test= split_training(X, y, train_size, split_seed)
    X_train, X_test= normalise(X_train, X_test)
    X_train, X_test, y_train, y_test= rename_columns(X_train, X_test, y_train, y_test, feature_names, label_name)
    print("Display information for the training set.")
    display_data(X_train, y_train)
    return X_train, X_test, y_train, y_test


####### FUNCTIONS TO IMPLEMENT NAIVE BAYES ####################################
def log_priors(Y_train):
    """This function compute the logarithm of the priors by taking the ratio
    of classes in the training set"""
    first_class= np.sum(Y_train) / Y_train.shape[0]
    return np.log(first_class), np.log(1 - first_class)

def separate_classes(X, y):
    """This function separates a data matrix according to the class label.
    It returns the separated databases"""
    class_index= (y == 1)
    X_first_class= X.iloc[class_index,:]
    X_zero_class= X.iloc[~class_index,:]
    return X_first_class, X_zero_class

def log_likelihoods(K, h, X_train, Y_train, x_test):
    loglikelihood_first= np.zeros(1)
    loglikelihood_zero= np.zeros(1)
    n_dim= X_train.shape[1]
    for i in range(n_dim):
        X_train_first, X_train_zero= separate_classes(X_train, Y_train)
        kde_first= KernelDensity(bandwidth= h, kernel= K)
        kde_zero= KernelDensity(bandwidth= h, kernel= K)
        kde_first.fit(X_train_first)
        kde_zero.fit(X_train_zero)
        loglikelihood_first+= kde_first.score_samples(x_test.reshape(1, 4))
        loglikelihood_zero+= kde_zero.score_samples(x_test.reshape(1, 4))
    return loglikelihood_first, loglikelihood_zero

def bayes_classify_point(K, h, X_train, Y_train, x_test):
    log_prior_first, log_prior_zero= log_priors(Y_train)
    log_likelihood_first, loglikelihood_zero= log_likelihoods(K, h, X_train, Y_train, x_test)
    if (log_prior_first + log_likelihood_first) > (log_prior_zero + loglikelihood_zero):
        return 1
    else:
        return 0

def bayes_classify(K, h, X_train, Y_train, X_test):
    n_rows= X_test.shape[0]
    Y_predict= np.zeros(n_rows)
    for row in range(n_rows):
        X_test_classify= X_test.iloc[row, :].values.ravel()
        Y_predict[row]= bayes_classify_point(K, h, X_train, Y_train, X_test_classify)
    return Y_predict

def bayes_cv_with_bandwidth(K, h, X_train, y_train, kfolds, cv_seed):
    """This function computes the cv error for Naive Bayes."""
    skf = StratifiedKFold(n_splits= kfolds, random_state= cv_seed, shuffle= True)
    errors= []
    for train, test in skf.split(X_train, y_train.values.ravel()):
        y_testcv= y_train.iloc[test,:]
        X_testcv= X_train.iloc[test,:]
        y_traincv= y_train.iloc[train,:]
        X_traincv= X_train.iloc[train,:]
        y_predictedcv= bayes_classify(K, h, X_traincv, y_traincv.values.ravel(), X_testcv)

        errors.append(1 - accuracy_score(y_testcv, y_predictedcv))
        #test
        print(accuracy_score(y_testcv, y_predictedcv))
    return np.mean(errors), np.std(errors)
    
def bayes_cv(K, max_h, X_train, y_train, kfolds, cv_seed):
    cv_error= []
    for curr_bandwidth in np.arange(0.01, max_h, 0.01):
        print("Current Bandwidth %3.2f" % curr_bandwidth)
        curr_err, curr_std= bayes_cv_with_bandwidth(K, curr_bandwidth, X_train, y_train, kfolds, cv_seed)
        cv_error.append([curr_bandwidth, curr_err, curr_std])
    return np.array(cv_error)

def bayes_tuning(cv_error):
    index_best= cv_error[:,1].argmin()
    return cv_error[index_best, 0]

def bayes_test(K, h, X_train, Y_train, X_test, Y_test):
    Y_predict= bayes_classify(K, h, X_train, Y_train.values.ravel(), X_test)
    bayes_confusion_matrix= confusion_matrix(Y_test, Y_predict)
    return bayes_confusion_matrix, Y_predict
################# MAIN ########################################################
filename= 'TP1-data.csv'
feature_names= ['Variance', 'Skewness', 'Curtosis', 'Entropy']
train_size= 2./3.
split_seed= 10182017
kfolds= 5
cv_seed= 522224

    
# Load and preprocess data
X_train, X_test, y_train, y_test= preprocess_data(filename, train_size, split_seed, feature_names)
print(X_train.shape)
bayes_cv= bayes_cv('gaussian', 1, X_train, y_train, 5, cv_seed)
h= bayes_tuning(bayes_cv)
bayes_matrix, Y_predict= bayes_test('gaussian', h, X_train, y_train, X_test, y_test)
print(bayes_matrix)