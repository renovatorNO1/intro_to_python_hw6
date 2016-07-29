# *******************************************************
# Name: Lucas Liu
# UNI: yl3433
# hw5b module
# Assignment 6 Part 1
# ENGI E1006
# *******************************************************

import numpy as np
from scipy.spatial import distance
import statistics as stats

def KNNclassifier(training, test, k, distance_matrix):
    '''classify the test data with labels provided by training data by applying nearest
        neighbor algorithm. The out put is a numpy array of labels for each data point'''
    #Store #of rows of test into variable row_num
    row_num = test.shape[0]
    
    #Specify the dimension of the ourput array
    order = np.zeros(row_num)
    
    #Create a numpy array whose elements are the values of distance from every pair of points    
    d = distance.cdist(test, training[:, 1:], distance_matrix)
    #d = distance.cdist( test, training[:, 1:], distance_matrix)
    
    #Rearrange elements of each row of d in ascending order, represented by their indice
    d_order = d.argsort()
    

    
    #Modify each element of order. Turn it into a vector with all the labels for test data set
    for x in range(test.shape[0]):
        
        k_min_d_indice = d_order[x][:k]
        k_labels = []
        
        for i in range(k_min_d_indice.shape[0]):
            
            #Append each label into k_labels
            k_labels.append(training[k_min_d_indice[i]][0])
            
        #Modify the corresponding entry of order            
        order[x] = stats.mode(k_labels)
     
        

    
    #Specify the shape of order to meet the output qualification
    order = order.reshape(row_num,1)
    
    return order
    
    #Modify each element of order. Turn it into a vector with all the labels for test data set
    '''for x in range(test.shape[0]):
        k_min_d_indice = d_order[x][:k]
        print(k_min_d_indice.shape)
        k_labels = []
        for i in range(k_min_d_indice.shape[0]):
            k_labels.append(training[k_min_d_indice[i]])
        order[x] = stats.mode(k_labels)   
    
    #Specify the shape of order to meet the output qualification
    order = order.reshape(row_num,1)
    
    return order'''
    
def n_validator(data, p, classifier, k, distance_matrix):
    '''Estimate the performance of the NNclassifier function with a real data set. The output 
       is a float between 0 and 1 that measures the performance of the classifier. Higher score
       indicates higher accuracy of the classifer function.'''
       
    #Store #of rows of data into the variable n    
    n = data.shape[0]
    
    #The following codes are intended to divide the data into p parts as evenly as possible
    #so that no single section has more than one observation than any other section.
    remainder = n % p
    sections = np.array([n//p]*p)
    sections[:remainder] += 1
    
    #Initiate counter. Use k to keep track of sections. 
    #Use score to keep track of the accuracy of labels
    t = 0
    score = 0

    #Loop p times to make sure every section gets treated as test data set for one time    
    for i in range(p):
        
        #Specify the test data set and training data set
        test = data[t:t+sections[i]]
        test_no_labels = np.delete(data[t:t+sections[i]], 0, 1)
        
        training = np.delete(data, range(t, t+sections[i]), 0)
        
        #Increment k so next loop will start on a new section
        t = t + sections[i]
        
        #Call the calssifier function 
        labels = classifier(training, test_no_labels, k, distance_matrix)
        
        #Increase score by 1 for each correct label        
        for x in range(labels.shape[0]):
            if labels[x] == test[x][0]:
                score += 1
    
    #Return the percentage of correct labels
    return score/n



    

    
            
            
            
            

