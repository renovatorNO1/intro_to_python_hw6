# *******************************************************
# Name: Lucas Liu
# UNI: yl3433
# hw5b module
# Assignment 6 Part 1
# ENGI E1006
# *******************************************************

import matplotlib.pyplot as plt
import numpy as np
import nn
import time


def synthetic_data():
    '''Generate two classes of data set using multivariate normal distribution.
        Combine the two to create a synthetic data set'''
    m1 = [2.5, 3.5]
    m2 = [0.5, 1]
    cov1 = [[1,1],[1,4.5]]
    cov2 = [[2,0],[0,1]]
    x1 = np.random.multivariate_normal(m1, cov1, 200)
    x2 = np.random.multivariate_normal(m2, cov2, 300)

    #plt.plot(x1[:,0],x1[:,1],'ro',x2[:,0],x2[:,1],'bo')
    #plt.show()
    
    labels1 = np.ones(200).reshape(200,1)
    x1_labels = np.hstack((x1, labels1))
    x1_labels.shape
    
    labels2 = 2 * np.ones(300).reshape(300,1)
    x2_labels = np.hstack((x2, labels2))
    x2_labels.shape
    
    data = np.vstack((x1_labels, x2_labels))
    data[:,[0, 2]] = data[:,[2, 0]]
    
    np.random.shuffle(data)
    
    return data

def real_data():
    '''Import a real data set from wdbc.txt'''
    with open('wdbc.txt', 'r') as infile:        
        temp_data = []
        for line in infile:
            line = line.split()
            temp_data.append(line)
            
        temp_data = np.array(temp_data)
        temp_data = temp_data.astype(float)
        data_set = np.delete(temp_data, 0, 1)
    return data_set
    
def return_best_k_score1(data, distance_matrix):
    '''This is for real data set only.
       Return the best k with the corresponding score
       for a designated distance matrix, given a data set'''
    #Create a list, score_chart, that records all the socres
    score_chart = [] 
    
    
    #Calcuate score for each odd k, and register each score into score_chart
    for k in range(16):
        if k % 2 == 1:
                     
            score = return_score(data, k, distance_matrix)
            score_chart.append(score)               
            
    #Find the best_k that yields to the highest score
    score_chart = np.array(score_chart)
   
    score_order = np.argsort(score_chart)
    
    best_k = score_order[-1] * 2 + 1
    best_score = score_chart[score_order[-1]]
    
    
    return (best_k, best_score)
    
def return_best_k_score2(data, distance_matrix):
    '''this is for synthetic data only.
       Return the best k with its corresponding value
       for a designated distance matrix, given a data set'''
    #Create a list, score_chart, that records all the socres
    score_chart = [] 
    trials = 50
    
    #Calcuate score for each odd k, and register each score into score_chart
    for k in range(16):
        if k % 2 == 1:
            #create a list to keep track of all the scores in trials
            scores_in_trials = []
            
            for times in range(trials):
                #Append each score into the trial          
                score = return_score(data, k, distance_matrix)
                scores_in_trials.append(score)
                
            ave_score = sum(scores_in_trials) / trials
    
            score_chart.append(ave_score)   
            
    #Find the best_k that yields to the highest score
    score_chart = np.array(score_chart)
   
    score_order = np.argsort(score_chart)
    
    best_k = score_order[-1] * 2 + 1
    best_score = score_chart[score_order[-1]]
    
    
    return (best_k, best_score)

def return_score(data, k, distance_matrix):
    '''Return the score for a designated combination of distance_matrix and k'''
    score = nn.n_validator(data, 5, nn.KNNclassifier, k, distance_matrix)
    return score

def print_out1(distance_methods,data):
    '''This is for real data only. 
       Print to console the results of utilizing different matrices'''
    #Initiate two variables to keep track of the values of k and score
    k_chart = []
    scores_chart = []

    print('distance methods \t best k \t score')
    
    #Print out a chart with the first column as method, the second column
    #as the value of best k
    for name, method in distance_methods:
        #From this index find the best combination of matrix-k

        k = return_best_k_score1(data, method)[0]
        score = return_best_k_score1(data, method)[1]
        
        k_chart.append(k)
        scores_chart.append(score)
            
        #From this index find the best combination of matrix-k
        print('{:s} \t {:d} \t \t {:f}'.format(name, k, score))
    
    #Find the index that indicates the best performances    
    best_score = max(scores_chart)
    best_score_index = scores_chart.index(best_score)
    
    #From this index find the best combination of matrix-k
    best_k = k_chart[best_score_index]
    best_method = distance_methods[best_score_index][0]
    
    print()
    print("The best matrix-k combination is")
    print(str(best_method) + "-" + str(best_k))
def print_out2(distance_methods, data):
    '''This is for synthetic data set only.
       Print to console the results of utilizing different matrices'''
    #Initiate two variables to keep track of the values of k and score
    k_chart = []
    scores_chart = []

    print('distance methods \t best k \t score')
    
    #Print out a chart with the first column as method, the second column
    #as the value of best k
    for name, method in distance_methods:
        #From this index find the best combination of matrix-k

        k = return_best_k_score2(data, method)[0]
        score = return_best_k_score2(data, method)[1]
        
        k_chart.append(k)
        scores_chart.append(score)
            
        #From this index find the best combination of matrix-k
        print('{:s} \t {:d} \t \t {:f}'.format(name, k, score))
    
    #Find the index that indicates the best performances    
    best_score = max(scores_chart)
    best_score_index = scores_chart.index(best_score)
    
    #From this index find the best combination of matrix-k
    best_k = k_chart[best_score_index]
    best_method = distance_methods[best_score_index][0]
    
    print()
    print("The best matrix-k combination is")
    print(str(best_method) + "-" + str(best_k))

   
def main():
    '''A control function that musters all the functions and produce
        desired results.'''
        
    data_set1 = real_data()    
    data_set2 = synthetic_data() 
    
    #Create a list that contains all the distance methods to be used
    distance_methods = [('Euclidean Distance','euclidean'),\
                        ('Manhattan Distance','cityblock'),\
                        ('Correlation Distance','correlation')]
    
    print("The following is the test result of real data set")  
    print()
    print_out1(distance_methods, data_set1)


    print()
    print("The following is the test result of a synthetic data set")
    print_out2(distance_methods, data_set2)


start_time = time.time()
main()
print("--- %s minutes ---" % ((time.time() - start_time)/60))