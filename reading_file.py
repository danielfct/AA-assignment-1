# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 21:19:26 2017

@author: Andrea
"""

#Importing the libraries required
import numpy as np
from sklearn import preprocessing

#Defining a function to read the data (code taken from the 2nd tutorial)
def read_data(file_name):
    """Returns a matrix with the data
        and print the dimension and first ten
        lines of the data matrix"""
    rows= []
    lines= open(file_name).readlines()
    for line in lines:
        parts= line.split(',')
        rows.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])))
    data= np.array(rows) #I am pretty sure there is a smarter way to do this
    print("\nPrinting the first ten rows...")
    print(data[0:9,:])
    print('\n')
    return data

def shuffle(data):
    """This function randomly shuffles
    the data to ensure that no patterns
    due to the way data were given 
    compromises the analysis"""
    np.random.shuffle(data)
    return data
    
def output_input_separation(data):
    """This function outputs the data matrix
    and the vector of responses"""
    return data[:,0:3], data[:,4]

def normalize(data_matrix):
    """This function normalises the data
    first by subtracting the mean and then
    by dividing by the standard deviation"""
    data_matrix_scaled= preprocessing.scale(data_matrix)
    print("Printing the first ten NORMALISED rows...")
    print(data_matrix_scaled[0:9,:])
    return data_matrix_scaled

def input_data(file_name):
    data_matrix, response= output_input_separation(shuffle(read_data(file_name)))
    data_matrix= normalize(data_matrix)
    return data_matrix, response

data_matrix, response= input_data("TP1-data.csv")

    
    

