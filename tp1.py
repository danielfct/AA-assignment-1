# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 03:13:58 2017

@author: Andrea
"""

#Loading the relevant libraries
import pandas as pd
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from tempfile import TemporaryFile


############ FUNCTIONS TO PREPROCESS DATA #####################################
def reading_csv(filename):
    """This function reads a csv file a returns
    a data matri and a vector labels.
    We are assuming the labels are in the last position."""
    data = pd.read_csv(filename)
    last_column_index= data.shape[1] - 1 #0-based numeration
    X= data.iloc[:,0:last_column_index]
    y= data.iloc[:,last_column_index:]
    return X, y

def split_training(Data_Matrix, Label_Vector, Train_Size, Seed):
    """This function takes as an input the Data Matrix and the
    Label Vector we are working with, shuffles the data according
    to a Seed given as an input and splits according to the proportion
    decided by the user.
    It returns four elements: the training and test Data Matrix,
    the training and testing Label Vectors"""
    X_train, X_test, y_train, y_test= train_test_split(Data_Matrix, Label_Vector,
                                                       train_size= Train_Size,
                                                       random_state= Seed,
                                                       shuffle= True,
                                                       stratify= Label_Vector)
    return X_train, X_test, y_train, y_test

def normalise(X_train, X_test):
    """This function performs a normalisation by subtracting the mean
    and then dividing by the standard deviation.
    It computes the mean value and standard deviation only on the training
    matrix and apply the same transformation to both matrices."""
    # Create a standard processor object
    scaler= StandardScaler()
    # Create an object to transform the data to fit standard processor
    x_scaled= scaler.fit_transform(X_train)
    # Run the normalizer on the dataframes
    X_train = pd.DataFrame(x_scaled)
    X_test= pd.DataFrame(scaler.transform(X_test))
    return X_train, X_test

def rename_columns(X_train, X_test, y_train, y_test, Column_Names, Label_Name):
    """This function changes the name of the columns of training and
    data set and of the label vectors, using the same for the two.
    It assumes that Column_Names and Label_Name are tuple of strings of the
    right dimensions."""
    X_train.columns= Column_Names
    y_train.columns= Label_Name
    X_test.columns= Column_Names
    y_test.columns= Label_Name
    return X_train, X_test, y_train, y_test

def display_data(Data_Matrix, Label_Vector):
    """This function allows to visualise the data and print
    some information"""
    print("Printing the first ten rows...")
    print(Data_Matrix.head(10)) #printing the first ten rows
    print("\nPrinting summary statistics...")
    print(Data_Matrix.describe()) #Getting some information
    print("\nPrinting scatterplots considering the class")
    
    plt.figure(0)
    pd.plotting.scatter_matrix(Data_Matrix, alpha=0.8, figsize=(6, 6),
                               diagonal='kde', c= Label_Vector)
    plt.show()
    plt.close()


def preprocess_data(filename, Train_Size, Seed, Column_Names, Label_Name=['Class']):
    """Encapsulating all the work done insofar.
        Taking as input the filename, the relative
        size of the training set, the seed for the
        random reshuffling and the names we are willing to give to the columns"""
    X, y= reading_csv(filename)
    X_train, X_test, y_train, y_test= split_training(X, y, Train_Size, Seed)
    X_train, X_test= normalise(X_train, X_test)
    X_train, X_test, y_train, y_test= rename_columns(X_train, X_test, y_train, y_test, Column_Names, Label_Name)
    print("Display information for the training set.")
    display_data(X_train, y_train)
    return X_train, X_test, y_train, y_test


############### FUNCTIONS TO COMPUTE THE LOGISTIC REGRESSION ##################
def logistic_regression_tuning(X_train, y_train, cv_seed, iteration):
    """This function trains the hyperparamater of the linear regression
    doing a 5-fold stratified CV with the provided Training data and labels.
    The hyperparameter is obtained by doubling at every iteration the previous one.
    We start with 1 as the first value. The model is returned.
    We set a seed, given as input."""
    CV_data= []
    for i in range(0, iteration):
        inverse_penalisation= pow(2,i)
        logistic= linear_model.LogisticRegression(penalty='l2',
                                                  C= inverse_penalisation)
        logistic.fit(X_train, y_train.values.ravel())
        cv_eval= model_selection.cross_val_score(logistic, X_train,
                                                 y_train.values.ravel(),
                                                 cv=StratifiedKFold(n_splits=5,random_state= cv_seed,shuffle=True))
        CV_data.append([inverse_penalisation, np.mean(cv_eval), np.std(cv_eval), logistic.score(X_train, y_train)])
    CV_data= pd.DataFrame(CV_data)
    CV_data.columns= ["InvPenalisation", "CVAccuracy", "CVStd", "TrainAccuracy"]
    x= np.log(CV_data[["InvPenalisation"]])
    y_cv= CV_data[["CVAccuracy"]]
    y_train= CV_data[["TrainAccuracy"]]
    
    plt.figure(1)
    plt.plot(x, 1 - y_cv, label= "CV Error")
    plt.plot(x, 1 - y_train, label= "Train Error")
    plt.xlabel("Logarithm of Inverse Penalisation")
    plt.ylabel("Error")
    plt.title("L2 Logistic Regression Tuning")
    plt.legend()
    plt.show()
    plt.close()
    return CV_data
    
def logistic_fitting(X_train, y_train, CV_data, cv_seed):
    """This function fits the logistic regression considering
    the best value obtained in tuning. It returns the model.
    We set a seed, given as input."""
    index= CV_data['CVAccuracy'].idxmax()
    inverse_penalisation= CV_data['InvPenalisation'][index]
    logistic= linear_model.LogisticRegression(penalty='l2',
                                                  C= inverse_penalisation)
    logistic.fit(X_train, y_train.values.ravel())
    cv_eval= model_selection.cross_val_score(logistic, X_train,
                                                 y_train.values.ravel(),
                                                 cv=StratifiedKFold(n_splits=5,random_state= cv_seed,shuffle=True))
    print("\nLogistic Regression (L2) Tuning:\n\tCV ErrorMean: %3.4f\t Std: %3.4f" % (1 - np.average(cv_eval), np.std(cv_eval))) 
    return logistic
    
def logistic_regression_training(X_train, y_train, cv_seed, iteration):
    """This function wraps the tuning and fitting of the logistic regression.
    It return the model. We set a seed, given as input."""
    CV_data= logistic_regression_tuning(X_train, y_train, cv_seed, iteration)
    logistic= logistic_fitting(X_train, y_train, CV_data, cv_seed)
    return logistic

def logistic_testing(X_test, y_test, logistic):
    """This function tests the logistic regression provided
    on a given test set. It return the confusion matrix."""
    y_pred= logistic.predict(X_test)
    logistic_confusion_matrix= confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = logistic_confusion_matrix.ravel()
    print("\nLogistic Regression (L2) Testing:")
    print("\tTrue Negative: %d" % tn)
    print("\tFalse Positive: %d" % fp)
    print("\tFalse Negative: %d" % fn)
    print("\tTrue Positive: %d" % tp)
    print("Test Error: \t%3.4f" % (1 - accuracy_score(y_test, y_pred)))
    return logistic_confusion_matrix


########### FUNCTIONS TO COMPUTE THE kNN ######################################
def knn_tuning(X_train, y_train, cv_seed, maximum_k):
    """This function trains the k-parameter of the kNN
    doing a 5-fold stratified CV with the provided Training data and labels.
    The hyperparameter is obtained by considering the odd sequence up to the
    value provided as input.
    We start with 1 as the first value. The model is returned.
    We set a seed, given as input."""
    CV_data= []
    for k_neigh in range(1, maximum_k, 2):
        neigh = KNeighborsClassifier(n_neighbors= k_neigh)
        neigh.fit(X_train, y_train.values.ravel())
        cv_eval= model_selection.cross_val_score(neigh, X_train,
                                                 y_train.values.ravel(),
                                                 cv=StratifiedKFold(n_splits=5,random_state= cv_seed,shuffle=True))
        CV_data.append([k_neigh, np.mean(cv_eval), np.std(cv_eval), neigh.score(X_train, y_train)])
    CV_data= pd.DataFrame(CV_data)
    CV_data.columns= ["k", "CVAccuracy", "CVStd", "TrainAccuracy"]
    x= CV_data[["k"]]
    y_cv= CV_data[["CVAccuracy"]]
    y_train= CV_data[["TrainAccuracy"]]
    
    plt.figure(3)
    plt.plot(x, 1 - y_cv, label= "CV Error")
    plt.plot(x, 1 - y_train, label= "Train Error")
    plt.xlabel("Number k of Neighbours")
    plt.ylabel("Error")
    plt.title("k Nearest Neighbours")
    plt.legend()
    plt.show()
    plt.close()
    return CV_data

def knn_fitting(X_train, y_train, CV_data, cv_seed):
    """This function fits the Knn considering
    the best value obtained in tuning. It returns the model.
    We set a seed, given as input."""
    index= CV_data['CVAccuracy'].idxmax()
    k_neigh= CV_data['k'][index]
    neigh = KNeighborsClassifier(n_neighbors= k_neigh)
    neigh.fit(X_train, y_train.values.ravel())
    cv_eval= model_selection.cross_val_score(neigh, X_train,
                                                 y_train.values.ravel(),
                                                 cv=StratifiedKFold(n_splits=5,random_state= cv_seed,shuffle=True))
    print("\nkNN Tuning:\n\tCV ErrorMean: %3.4f\t Std: %3.4f" % (1 - np.average(cv_eval), np.std(cv_eval))) 
    return neigh

def knn_training(X_train, y_train, cv_seed, maximum_k):
    """This function wraps the tuning and fitting of the kNN.
    It return the model. We set a seed, given as input."""
    CV_data= knn_tuning(X_train, y_train, cv_seed, maximum_k)
    
    
    neigh= knn_fitting(X_train, y_train, CV_data, cv_seed)
    return neigh

def knn_testing(X_test, y_test, neigh):
    """This function tests the logistic regression provided
    on a given test set. It return the confusion matrix."""
    y_pred= neigh.predict(X_test)
    knn_confusion_matrix= confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = knn_confusion_matrix.ravel()
    print("\nkNN Testing:")
    print("\tTrue Negative: %d" % tn)
    print("\tFalse Positive: %d" % fp)
    print("\tFalse Negative: %d" % fn)
    print("\tTrue Positive: %d" % tp)
    print("Error: \t%3.4f" % (1 - accuracy_score(y_test, y_pred)))
    return knn_confusion_matrix


####### FUNCTIONS TO IMPLEMENT NAIVE BAYES ####################################
def compute_log_priors(y_train):
    first_class= np.sum(y_train) / y_train.shape[0] 
    zero_class= 1 - first_class
    return np.array(np.log(first_class)), np.array(np.log(zero_class))
    
def separate_classes(X, y):
    class_index= (y == 1).values.ravel()
    X_first_class= X.iloc[class_index,:]
    X_zero_class= X.iloc[~class_index,:]
    return X_first_class, X_zero_class

def log_likelihood(x, X_train, bandwidth, kernel):
    num_dim= X_train.shape[1]
    x= np.array(x)[:, np.newaxis]
    log_dens= np.zeros(1)
    for i in range(0, num_dim):
        kde= KernelDensity(bandwidth= bandwidth, kernel= kernel)
        feature= np.array(X_train.iloc[:,i])
        feature= np.array(feature)[:, np.newaxis]
        kde.fit(feature)
        log_dens+= kde.score(x)
    return log_dens

def classify(prior_one, prior_zero, likelihood_one, likelihood_zero):
    if (prior_one + likelihood_one) > (prior_zero + likelihood_zero):
        return 1
    else:
        return 0
    
def prediction_error(X_train, y_train, X_test, y_test, bandwidth, kernel= 'gaussian'):
    prior_one, prior_zero= compute_log_priors(y_train)
    X_one, X_zero= separate_classes(X_train, y_train)
    dim_test= X_test.shape[0]
    y_predict= []
    for i in range(0, dim_test):
        current_X= np.array(X_test.iloc[i,:])
        likelihood_one= log_likelihood(current_X, X_one, bandwidth, kernel)
        likelihood_zero= log_likelihood(current_X, X_zero, bandwidth, kernel)
        y_predict.append(classify(prior_one, prior_zero, likelihood_one, likelihood_zero))
    y_predict= np.array(y_predict)
    error= 1 - accuracy_score(y_test, y_predict)
    #print("Misclassification error: %3.2f" % error)
    return error

def bayes_cv(X_train, y_train, cv_seed, bandwidth):
    skf = StratifiedKFold(n_splits=5, random_state= cv_seed, shuffle= True)
    error= []
    for train, test in skf.split(X_train, y_train.values.ravel()):
        curr_error= prediction_error(X_train.iloc[train,:], y_train.iloc[train,:],
                                 X_train.iloc[test,:], y_train.iloc[test,:],
                                 bandwidth= bandwidth)
        error.append(curr_error)
    error= np.array(error)
    return(np.mean(error), np.std(error))

def bayes_tuning(X_train, y_train, cv_seed, bandwidth_max):
    cv_error= []
    for i in np.arange(0.01, bandwidth_max, 0.02):
        print("Current Bandwidth %3.2f" % i)
        curr_cv_error= bayes_cv(X_train, y_train, cv_seed, i)
        cv_error.append(curr_cv_error)
    return np.array(cv_error)

def bayes_testing(X_train, y_train, X_test, y_test, cv_bayes, kernel= 'gaussian'):
    bandwidth= cv_bayes[:,0].min()
    test_error= prediction_error(X_train, y_train, X_test, y_test, bandwidth)
    print("The testing error is: %3.2f" % test_error)
    return test_error

filename= 'TP1-data.csv'
Train_Size= 0.66
Seed= 10182017
Column_Names= ['Variance', 'Skewness', 'Curtosis', 'Entropy']
X_train, X_test, y_train, y_test= preprocess_data(filename, Train_Size, Seed, Column_Names)

cv_seed= 52222
logistic= logistic_regression_training(X_train, y_train, cv_seed, 20)
logistic_confusion_matrix= logistic_testing(X_test, y_test, logistic)

neigh= knn_training(X_train, y_train, cv_seed, 40)
knn_confusion_matrix= knn_testing(X_test, y_test, neigh)


cv_bayes= bayes_tuning(X_train, y_train, cv_seed, 1)
bayes_error= bayes_testing(X_train, y_train, X_test, y_test, cv_bayes)

