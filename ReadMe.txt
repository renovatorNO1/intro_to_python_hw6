# *******************************************************
# Name: Lucas Liu
# UNI: yl3433
# hw5b module
# Assignment 6 Part 1
# ENGI E1006
# *******************************************************

For nn.py
for KNNclassifier function:
    I add an additional parameter for the distance name. The output should be n-1 dimensional vector whose elements are the labels.

For n_valudator.py

I add an additional parameter for the distance matrix. The out put should be a score that indicates the performances

For nn_testers.py

I have several helper functions in place. 

function synthetic_data() is intended to generate a synthetic data set

function real_data() is intended to import a real data set

return_best_k_score1() is intended for real data set only, and it is supposed to return a tuple that contains the
best k and the best score, given data set and a specific distance matrix

return_best_k_score2() is intended for synthetic data set only, and it is supposed to return a tuple that contains the best k and the best score, given data set and a specific distance matrix

print_out1() is intended for real data set only, and it is supposed to print out the result onto the python console.

print_out2() is intended for synthetic data set only, and it is supposed to print out the result onto the python console. 

The main() function is to control all the calling of functions 

the last a few lines of codes are intended to measure the running time.

