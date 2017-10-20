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


############### FUNCTIONS TO COMPUTE THE LOGISTIC REGRESSION ##################
def logistic_regression_tuning(X_train, y_train, kfolds, cv_seed, logistic_iterations):
    """This function trains the hyperparamater of the linear regression
    doing a 5-fold stratified CV with the provided Training data and labels.
    The hyperparameter is obtained by doubling at every iteration the previous one.
    We start with 1 as the first value. The model is returned.
    We set a seed, given as input."""
    cv_data= []
    for i in range(0, logistic_iterations):
        inverse_penalisation= pow(2, i)
        logistic= LogisticRegression(C= inverse_penalisation)
        logistic.fit(X_train, y_train.values.ravel())
        cv_eval= cross_val_score(logistic, X_train,
                                 y_train.values.ravel(),
                                 cv= StratifiedKFold(n_splits= kfolds, random_state= cv_seed, shuffle= True))
        cv_data.append([inverse_penalisation, np.mean(cv_eval), logistic.score(X_train, y_train)])
        
    cv_data= pd.DataFrame(cv_data)
    cv_data.columns= ["InvPenalisation", "CVAccuracy", "TrainAccuracy"]

    logistic_regression_plotting(np.log(cv_data[["InvPenalisation"]]), 
                                 cv_data[["CVAccuracy"]],
                                 cv_data[["TrainAccuracy"]]) 
    return cv_data


def logistic_regression_plotting(inv_penalisations, cv_accuracy, train_accuracy):
    ax = plt.figure().add_subplot(111)
    plt.plot(inv_penalisations, 1 - cv_accuracy['CVAccuracy'], label= "CV Error")
    plt.plot(inv_penalisations, 1 - train_accuracy['TrainAccuracy'], label= "Train Error")
    plt.xlabel("Logarithm of Inverse Penalisation")
    plt.ylabel("Error")
    plt.title("L2 Logistic Regression Regularization")
    plt.legend()
    # annotate selected k value
    idxmin= (1 - cv_accuracy['CVAccuracy']).idxmin()
    xmin= inv_penalisations['InvPenalisation'][idxmin]
    ymin= (1 - cv_accuracy['CVAccuracy'])[idxmin]
    plt.plot([xmin], [ymin], marker='o', markersize=3, color="red")
    ax.annotate('(%0.3f, %0.3f)' % (xmin, ymin),
                xy= (xmin, ymin), 
                xytext= (xmin, ymin-0.0006))
    plt.savefig('Best C value - L2 Logistic Regression')
    plt.show()
    plt.close()
    
    
def logistic_regresion_fitting(X_train, y_train, cv_data, kfolds, cv_seed):
    """This function fits the logistic regression considering
    the best value obtained in tuning. It returns the model.
    We set a seed, given as input."""
    index= cv_data['CVAccuracy'].idxmax()
    inverse_penalisation= cv_data['InvPenalisation'][index]
    logistic_regression= LogisticRegression(C= inverse_penalisation)
    logistic_regression.fit(X_train, y_train.values.ravel())
    cv_eval= cross_val_score(logistic_regression, 
                             X_train,
                             y_train.values.ravel(),
                             cv= StratifiedKFold(n_splits= kfolds, random_state= cv_seed, shuffle= True))
    print("\nLogistic Regression Tuning:\n\t" 
          "Inverse Penalisation: %d\n\tCV ErrorMean: %3.4f\n\tCV Std: %3.4f" % 
          (inverse_penalisation, 1 - np.average(cv_eval), np.std(cv_eval))) 
    return logistic_regression
  
    
def logistic_regression_training(X_train, y_train, kfolds, cv_seed, logistic_iterations):
    """This function wraps the tuning and fitting of the logistic regression.
    It return the model. We set a seed, given as input."""
    cv_data= logistic_regression_tuning(X_train, y_train, kfolds, cv_seed, logistic_iterations)
    logistic= logistic_regresion_fitting(X_train, y_train, cv_data, kfolds, cv_seed)
    return logistic


def logistic_regression_testing(X_test, y_test, logistic_regression):
    """This function tests the logistic regression provided
    on a given test set. It return the confusion matrix."""
    y_pred= logistic_regression.predict(X_test)
    logistic_confusion_matrix= confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = logistic_confusion_matrix.ravel()
    print("\nL2 Logistic Regression Testing:")
    print("\tTrue Negative: %d" % tn)
    print("\tFalse Positive: %d" % fp)
    print("\tFalse Negative: %d" % fn)
    print("\tTrue Positive: %d" % tp)
    print("Test Error: \n\t%3.4f" % (1 - accuracy_score(y_test, y_pred)))
    return logistic_confusion_matrix


########### FUNCTIONS TO COMPUTE THE kNN ######################################
def knn_tuning(X_train, y_train, kfolds, cv_seed, knn_max):
    """This function trains the k-parameter of the kNN
    doing a 5-fold stratified CV with the provided Training data and labels.
    The hyperparameter is obtained by considering the odd sequence up to the
    value provided as input.
    We start with 1 as the first value. The model is returned.
    We set a seed, given as input."""
    cv_data= []
    for k_neigh in range(1, knn_max, 2):
        neigh = KNeighborsClassifier(n_neighbors= k_neigh)
        neigh.fit(X_train, y_train.values.ravel())
        cv_eval= cross_val_score(neigh, X_train,
                                 y_train.values.ravel(),
                                 cv=StratifiedKFold(n_splits= kfolds, random_state= cv_seed, shuffle= True))
        cv_data.append([k_neigh, np.mean(cv_eval), neigh.score(X_train, y_train)])
        
    cv_data= pd.DataFrame(cv_data)
    cv_data.columns= ["k", "CVAccuracy", "TrainAccuracy"]
    
    knn_plotting(cv_data[["k"]],
                 cv_data[["CVAccuracy"]],
                 cv_data[["TrainAccuracy"]])
   
    return cv_data


def knn_plotting(x, y_cv, y_train):
    ax = plt.figure().add_subplot(111)
    plt.plot(x, 1 - y_cv, label= "CV Error")
    plt.plot(x, 1 - y_train, label= "Train Error")
    plt.xlabel("Number k of Neighbours")
    plt.ylabel("Error")
    plt.title("k Nearest Neighbours")
    plt.legend()
    # annotate selected inverted penalisation constant
    idxmin= (1 - y_cv['CVAccuracy']).idxmin()
    xmin= x['k'][idxmin]
    ymin= (1 - y_cv['CVAccuracy'])[idxmin]
    plt.plot([xmin], [ymin], marker='o', markersize=3, color="red")
    ax.annotate('(%d, %0.3f)' % (xmin, ymin),
                xy= (xmin, ymin), 
                xytext= (xmin, ymin-0.0008))
    plt.savefig('Best K value - K-nearest Neighbours')
    plt.show()
    plt.close()


def knn_fitting(X_train, y_train, cv_data, kfolds, cv_seed):
    """This function fits the Knn considering
    the best value obtained in tuning. It returns the model.
    We set a seed, given as input."""
    index= cv_data['CVAccuracy'].idxmax()
    k_neigh= cv_data['k'][index]
    neigh = KNeighborsClassifier(n_neighbors= k_neigh)
    neigh.fit(X_train, y_train.values.ravel())
    cv_eval= cross_val_score(neigh, 
                             X_train,
                             y_train.values.ravel(),
                             cv=StratifiedKFold(n_splits= kfolds, random_state= cv_seed, shuffle= True))
    print("\nkNN Tuning:\n\tCV ErrorMean: %3.4f\t Std: %3.4f" % (1 - np.average(cv_eval), np.std(cv_eval))) 
    return neigh


def knn_training(X_train, y_train, kfolds, cv_seed, knn_max):
    """This function wraps the tuning and fitting of the kNN.
    It return the model. We set a seed, given as input."""
    cv_data= knn_tuning(X_train, y_train, kfolds, cv_seed, knn_max)
    neigh= knn_fitting(X_train, y_train, cv_data, kfolds, cv_seed)
    return neigh


def knn_testing(X_test, y_test, knn):
    """This function tests the logistic regression provided
    on a given test set. It return the confusion matrix."""
    y_pred= knn.predict(X_test)
    knn_confusion_matrix= confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = knn_confusion_matrix.ravel()
    print("\nkNN Testing:")
    print("\tTrue Negative: %d" % tn)
    print("\tFalse Positive: %d" % fp)
    print("\tFalse Negative: %d" % fn)
    print("\tTrue Positive: %d" % tp)
    print("Error: \t%3.4f" % (1 - accuracy_score(y_test, y_pred)))
    print("\n")
    return knn_confusion_matrix


####### FUNCTIONS TO IMPLEMENT NAIVE BAYES ####################################
def compute_log_priors(y_train):
    """This function compute the logarithm of the priors by taking the ratio
    of classes in the training set"""
    first_class= np.sum(y_train) / y_train.shape[0]
    zero_class= 1 - first_class
    return np.array(np.log(first_class)), np.array(np.log(zero_class))

def separate_classes(X, y):
    """This function separates a data matrix according to the class label.
    It returns the separated databases"""
    class_index= (y == 1).values.ravel()
    X_first_class= X.iloc[class_index,:]
    X_zero_class= X.iloc[~class_index,:]
    return X_first_class, X_zero_class


def log_likelihood(x, X_train, bandwidth, kernel):
    """This function computes the likelihood of a new point considering a kernel"""
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
    """This function classifies a point considering the priors
    and the likelihoods"""
    if (prior_one + likelihood_one) > (prior_zero + likelihood_zero):
        return 1
    else:
        return 0


def prediction_error(X_train, y_train, X_test, y_test, bandwidth, kernel= 'gaussian'):
    """This function returns the prediction error of a testing set"""
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
    return error, y_predict


def bayes_cv(X_train, y_train, kfolds, cv_seed, bandwidth):
    """This function computes the cv error for Naive Bayes."""
    skf = StratifiedKFold(n_splits= kfolds, random_state= cv_seed, shuffle= True)
    errors= []
    for train, test in skf.split(X_train, y_train.values.ravel()):
        err= prediction_error(X_train.iloc[train,:], y_train.iloc[train,:],
                                 X_train.iloc[test,:], y_train.iloc[test,:],
                                 bandwidth= bandwidth)
        errors.append(err)
    errors= np.array(errors)
    return np.mean(errors), np.std(errors)


def bayes_tuning(X_train, y_train, kfolds, cv_seed, bandwidth_max):
    """This function is to tune the bandwidth parameter for Naive Bayes
    Classifier"""
    cv_errors= []
    bandwidth = np.arange(0.01, bandwidth_max, 0.02)
    for i in bandwidth:
        print("Current Bandwidth %3.2f" % i)
        err= bayes_cv(X_train, y_train, kfolds, cv_seed, i)
        cv_errors.append(err)
    cv_errors= np.array(cv_errors)
   
    bayes_plotting(bandwidth, cv_errors)
    
    return cv_errors

def bayes_plotting(bandwidth, cv_error):
    plt.plot(bandwidth, cv_error[:,0], label= "CV Error")
    plt.xlabel("Bandwidth")
    plt.ylabel("Error")
    plt.title("Nonparamentric Naive Bayes")
    plt.legend()
    plt.show()
    plt.close()

def bayes_testing(X_train, y_train, X_test, y_test, cv_bayes, kernel= 'gaussian'):
    """This function returns the prediction for the testing set"""
    bandwidth= cv_bayes[:,0].min()
    test_error, y_predicted= prediction_error(X_train, y_train, X_test, y_test, bandwidth)
    print("The testing error is: %3.2f" % test_error)
    bayes_confusion_matrix= confusion_matrix(y_test, y_predicted)
    return test_error, bayes_confusion_matrix

def mc_nemar_test(e01, e10):
    return pow((np.abs(e01 - e10) - 1), 2) / (e01 + e10)

def compare_classifiers(logistic_regression_confusion_matrix,
                        knn_confusion_matrix,
                        bayes_confusion_matrix):
    # Logistic Regression results
    lr_tn, lr_fp, lr_fn, lr_tp = logistic_regression_confusion_matrix.ravel()
    # K-nearest Neighbours results
    knn_tn, knn_fp, knn_fn, knn_tp = knn_confusion_matrix.ravel()
    # Naive Bayes results
    nb_tn, nb_fp, nb_fn, nb_tp = bayes_confusion_matrix.ravel()
    # Compare classifiers
    print("\nMcNemar tests:")
    print("\tLogistic Regression VS K-nearest neighbours: %0.3f" % (mc_nemar_test(0, 0)))
    print("\tLogistic Regression VS Naive Bayes: %0.3f\n" % mc_nemar_test(0, 0))
    print("\tK-nearest neighbours VS Logistic Regression: %0.3f" % (mc_nemar_test(0, 0)))
    print("\tK-nearest neighbours VS Naive Bayes: %0.3f\n" % mc_nemar_test(0, 0))
    print("\tNaive Bayes VS Logistic Regression: %0.3f\n" % mc_nemar_test(0, 0))
    print("\tNaive Bayes VS K-nearest neighbours: %0.3f\n" % mc_nemar_test(0, 0))


def main():
    filename= 'TP1-data.csv'
    feature_names= ['Variance', 'Skewness', 'Curtosis', 'Entropy']
    train_size= 2./3.
    split_seed= 10182017
    kfolds= 5
    cv_seed= 52222
    logistic_iterations= 20
    knn_max= 40
    bandwidth_max= 1
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test= preprocess_data(filename, train_size, split_seed, feature_names)
    
    # Logistic Regression Classifier
    logistic_regression= logistic_regression_training(X_train, y_train, kfolds, cv_seed, logistic_iterations)
    logistic_regression_confusion_matrix= logistic_regression_testing(X_test, y_test, logistic_regression)
    
    # K-nearest Neighbours Classifier
    knn= knn_training(X_train, y_train, kfolds, cv_seed, knn_max)
    knn_confusion_matrix= knn_testing(X_test, y_test, knn)

    # Naive Bayes Classifier
    cv_bayes= bayes_tuning(X_train, y_train, kfolds, cv_seed, bandwidth_max)
    bayes_error, bayes_confusion_matrix= bayes_testing(X_train, y_train, X_test, y_test, cv_bayes)
    
    # Compare classifiers with Mc Nemar's test
    compare_classifiers(logistic_regression_confusion_matrix,
                        knn_confusion_matrix)
                        bayes_confusion_matrix)
    
main()