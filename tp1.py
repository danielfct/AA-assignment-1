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
        cv_data.append([np.log10(inverse_penalisation), 1-np.mean(cv_eval), 1-logistic.score(X_train, y_train)])
        
    cv_data= pd.DataFrame(cv_data)
    cv_data.columns= ["InvPenalisation", "CvError", "TrainError"]

    idxmin= cv_data['CvError'].idxmin()
    optimal_c= cv_data['InvPenalisation'].iloc[idxmin]
    min_cv_error= cv_data['CvError'].iloc[idxmin]
    logistic_regression_plotting(optimal_c, min_cv_error, 
                                 cv_data[["InvPenalisation"]], 
                                 cv_data[["CvError"]],
                                 cv_data[["TrainError"]]) 
    return 10**optimal_c, cv_data


def logistic_regression_plotting(optimal_c, min_cv_error, inv_penalisations, cv_error, train_error):
    
    ax = plt.figure().add_subplot(111)
    plt.plot(inv_penalisations, cv_error['CvError'], label= "CV Error")
    plt.plot(inv_penalisations, train_error['TrainError'], label= "Train Error")
    plt.xlabel("Logarithm of Inverse Penalisation")
    plt.ylabel("Error")
    plt.title("L2 Logistic Regression Regularization")
    plt.legend()
    # annotate selected k value
    plt.plot([optimal_c], [min_cv_error], marker='o', markersize=3, color="red")
    ax.annotate('(%0.3f, %0.3f)' % (optimal_c, min_cv_error),
                xy= (optimal_c, min_cv_error), 
                xytext= (optimal_c, min_cv_error-0.0006))
    plt.savefig('Best C value - L2 Logistic Regression')
    plt.show()
    plt.close()
    
    
def logistic_regresion_fitting(X_train, y_train, cv_data, kfolds, cv_seed, optimal_c):
    """This function fits the logistic regression considering
    the best value obtained in tuning. It returns the model.
    We set a seed, given as input."""
    logistic_regression= LogisticRegression(C= optimal_c)
    logistic_regression.fit(X_train, y_train.values.ravel())
    cv_eval= cross_val_score(logistic_regression, 
                             X_train,
                             y_train.values.ravel(),
                             cv= StratifiedKFold(n_splits= kfolds, random_state= cv_seed, shuffle= True))
    print("\nL2 Logistic Regression Tuning:\n" 
          "\tOptimal Inverse C: %d\n"
          "\tCV Error Mean: %3.4f\n"
          "\tCV Error Std: %3.4f" % 
          (optimal_c, 1 - np.average(cv_eval), np.std(cv_eval))) 
    return logistic_regression
  
    
def logistic_regression_training(X_train, y_train, kfolds, cv_seed, logistic_iterations):
    """This function wraps the tuning and fitting of the logistic regression.
    It return the model. We set a seed, given as input."""
    optimal_c, cv_data= logistic_regression_tuning(X_train, y_train, kfolds, cv_seed, logistic_iterations)
    logistic= logistic_regresion_fitting(X_train, y_train, cv_data, kfolds, cv_seed, optimal_c)
    return logistic


def logistic_regression_testing(X_test, y_test, logistic_regression):
    """This function tests the logistic regression provided
    on a given test set. It return the confusion matrix."""
    y_pred= logistic_regression.predict(X_test)
    logistic_confusion_matrix= confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = logistic_confusion_matrix.ravel()
    print("\nL2 Logistic Regression Testing:\n"
          "\tTrue Negative: %d\n"
          "\tFalse Positive: %d\n"
          "\tFalse Negative: %d\n"
          "\tTrue Positive: %d\n"
          "\tTest Error: %3.4f" % (tn, fp, fn, tp, 1-accuracy_score(y_test, y_pred)))
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
        knn = KNeighborsClassifier(n_neighbors= k_neigh)
        knn.fit(X_train, y_train.values.ravel())
        cv_eval= cross_val_score(knn, 
                                 X_train,
                                 y_train.values.ravel(),
                                 cv=StratifiedKFold(n_splits= kfolds, random_state= cv_seed, shuffle= True))
        cv_data.append([k_neigh, 1-np.mean(cv_eval), 1-knn.score(X_train, y_train)])
        
    cv_data= pd.DataFrame(cv_data)
    cv_data.columns= ["k", "CvError", "TrainError"]
    
    minIdx= cv_data['CvError'].idxmin()
    optimal_k= cv_data['k'].iloc[minIdx]
    min_cv_error= cv_data['CvError'].iloc[minIdx]
    knn_plotting(optimal_k, min_cv_error,
                 cv_data[["k"]],
                 cv_data[["CvError"]],
                 cv_data[["TrainError"]])
   
    return optimal_k, cv_data


def knn_plotting(optimal_k, min_cv_error, neighbours, cv_error, train_error):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(neighbours, cv_error, label= "CV Error")
    plt.plot(neighbours, train_error, label= "Train Error")
    plt.xlabel("Number k of Neighbours")
    plt.ylabel("Error")
    plt.title("k Nearest Neighbours")
    plt.legend()
    # annotate selected k value
    plt.plot([optimal_k], [min_cv_error], marker='o', markersize=3, color="red")
    ax.annotate('(%d, %0.3f)' % (optimal_k, min_cv_error),
                xy= (optimal_k, min_cv_error), 
                xytext= (optimal_k, min_cv_error-0.0008))
#    plt.savefig('Best K value - K-nearest Neighbours')
    plt.show()
    plt.close()


def knn_fitting(X_train, y_train, cv_data, kfolds, cv_seed, optimal_k):
    """This function fits the Knn considering
    the best value obtained in tuning. It returns the model.
    We set a seed, given as input."""
    knn = KNeighborsClassifier(n_neighbors= optimal_k)
    knn.fit(X_train, y_train.values.ravel())
    cv_eval= cross_val_score(knn, 
                             X_train,
                             y_train.values.ravel(),
                             cv=StratifiedKFold(n_splits= kfolds, random_state= cv_seed, shuffle= True))
    print("\nkNN Tuning:\n"
          "\tOptimal K: %d\n"
          "\tCV Error Mean: %3.4f\n"
          "\tCV Error Std: %3.4f" % (optimal_k, 1-np.average(cv_eval), np.std(cv_eval))) 
    return knn


def knn_training(X_train, y_train, kfolds, cv_seed, knn_max):
    """This function wraps the tuning and fitting of the kNN.
    It return the model. We set a seed, given as input."""
    optimal_k, cv_data= knn_tuning(X_train, y_train, kfolds, cv_seed, knn_max)
    neigh= knn_fitting(X_train, y_train, cv_data, kfolds, cv_seed, optimal_k)
    return neigh


def knn_testing(X_test, y_test, knn):
    """This function tests the logistic regression provided
    on a given test set. It return the confusion matrix."""
    y_pred= knn.predict(X_test)
    knn_confusion_matrix= confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = knn_confusion_matrix.ravel()
    print("\nkNN Testing:\n"
          "\tTrue Negative: %d\n"
          "\tFalse Positive: %d\n"
          "\tFalse Negative: %d\n"
          "\tTrue Positive: %d\n"
          "\tTest Error: %3.4f" % (tn, fp, fn, tp, 1-accuracy_score(y_test, y_pred)))
    return knn_confusion_matrix


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
        #print(accuracy_score(y_testcv, y_predictedcv))
    return np.mean(errors), np.std(errors)
    
def bayes_cv(K, max_h, X_train, y_train, kfolds, cv_seed):
    print("\n\nTraining BAYES")
    cv_error= []
    for curr_bandwidth in np.arange(0.01, max_h, 0.01):
        print("Current Bandwidth %3.2f" % curr_bandwidth)
        curr_err, curr_std= bayes_cv_with_bandwidth(K, curr_bandwidth, X_train, y_train, kfolds, cv_seed)
        cv_error.append([curr_bandwidth, curr_err, curr_std])
    return np.array(cv_error)

def bayes_tuning(cv_error):
    index_best= cv_error[:,1].argmin()
    optimal_bandwidth= cv_error[index_best, 0]
    min_cv_error= cv_error[index_best, 1]
    bandwidths= cv_error[:,0]
    cv_errors= cv_error[:,1]
    bayes_plotting(optimal_bandwidth, min_cv_error, bandwidths, cv_errors)
    return cv_error[index_best, 0]

def bayes_test(K, h, X_train, Y_train, X_test, Y_test):
    Y_predict= bayes_classify(K, h, X_train, Y_train.values.ravel(), X_test)
    bayes_confusion_matrix= confusion_matrix(Y_test, Y_predict)
    return bayes_confusion_matrix

def bayes_predict(K, h, X_train, Y_train, X_test):
    Y_predict= bayes_classify(K, h, X_train, Y_train.values.ravel(), X_test)
    return Y_predict

def bayes_plotting(optimal_bandwidth, min_cv_error, bandwidths, cv_error):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(bandwidths, cv_error, label= "CV Error")
    #plt.plot(x, 1 - y_train, label= "Train Error")
    plt.xlabel("Bandwidth")
    plt.ylabel("Error")
    plt.title("Nonparamentric Naive Bayes")
    plt.legend()
    plt.plot([optimal_bandwidth], [min_cv_error], marker='o', markersize=3, color="red")
    ax.annotate('(%0.3f, %0.2f)' % (optimal_bandwidth, min_cv_error),
                xy= (optimal_bandwidth, min_cv_error), 
                xytext= (optimal_bandwidth, min_cv_error-0.0008))
    plt.savefig('Best Bandwidth - Naive Bayes')
    plt.show()
    plt.close()



########################MCNEMAR ###############################################
def mc_nemar_test(e01, e10):
    return pow((abs(e01 - e10) - 1), 2) / (e01 + e10)

def compare_classifiers(first_clf_pred, second_clf_pred, y_test):
    e01 = e10 = 0
    for i in range (len(y_test)):
        curr_first_pred = first_clf_pred[i]
        curr_second_pred = second_clf_pred[i]
        curr_correct_class = y_test['Class'].iloc[i]
        
        if (abs(curr_first_pred-curr_correct_class) == 1 and curr_second_pred-curr_correct_class == 0):
            e01 += 1
        if (curr_first_pred-curr_correct_class == 0 and abs(curr_second_pred-curr_correct_class) == 1):
            e10 += 1
   
  #  print("\ne01:", e01, "e10:", e10)
    return mc_nemar_test(e01, e10);

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
    cv_bayes= bayes_cv('gaussian', 1, X_train, y_train, 5, cv_seed)
    optimal_bandwidth= bayes_tuning(cv_bayes)
    bayes__confusion_matrix= bayes_test('gaussian', optimal_bandwidth, X_train, y_train, X_test, y_test)
    
    # Compare classifiers with Mc Nemar's test
    lr_vs_knn = compare_classifiers(logistic_regression.predict(X_test), 
                                    knn.predict(X_test), 
                                    y_test)
    
    lr_vs_bayes = compare_classifiers(logistic_regression.predict(X_test), 
                                   bayes_predict('gaussian', optimal_bandwidth, X_train, y_train, X_test), 
                                   y_test)
    
    knn_vs_bayes = compare_classifiers(knn.predict(X_test), 
                                       bayes_predict('gaussian', optimal_bandwidth, X_train, y_train, X_test), 
                                       y_test)
    
    print("\nMcNemar tests:")
    print("\tLogistic Regression VS K-nearest neighbours: %0.3f\n" % lr_vs_knn)
    print("\tLogistic Regression VS Naive Bayes: %0.3f\n" % lr_vs_bayes)
    print("\tK-nearest neighbours VS Naive Bayes: %0.3f\n" % knn_vs_bayes)

    
main()


#class NaiveBayes:
   # score: accuracy_score(y_train.iloc[valid,:], y_predict)
 #    knn.predict(X_test)
  #  knn.fit(X_train, y_train.values.ravel())
   #  def predict(self):
    #    return 'hello world'
