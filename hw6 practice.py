# *******************************************************
# Name: Lucas Liu
# UNI: yl3433
# hw5b module
# Assignment 6 Part 1
# ENGI E1006
# *******************************************************

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

'''mean1 = [2.5,3.5] 
covariance1 = [[1, 1], [1, 4.5] ]
sample1 = 300
data_set1 = np.random.multivariate_normal(mean1, covariance1, sample1)
print(data_set1.shape)

labels = np.arange(0,300)
labels.resize(300,1)
data_set1 = np.concatenate((labels, data_set1), 1)
print(data_set1)

mean2 = [.5, 1]
covariance2 = [[2, 0], [0, 1]]
data_set2 = np.random.multivariate_normal(mean2, covariance2, sample1)
print(data_set2.shape)'''

m1 = [2.5, 3.5]
m2 = [0.5, 1]
cov1 = [[1,1],[1,4.5]]
cov2 = [[2,0],[0,1]]
x1 = np.random.multivariate_normal(m1, cov1, 200)
x2 = np.random.multivariate_normal(m2, cov2, 300)

plt.plot(x1[:,0],x1[:,1], 'ro')
plt.show()

plt.plot(x2[:,0], x2[:,1], 'bo')
plt.show()

plt.plot(x1[:,0],x1[:,1],'ro',x2[:,0],x2[:,1],'bo')
plt.show()

labels1 = np.ones(200).reshape(200,1)
x1_labels = np.hstack((x1, labels1))
x1_labels.shape

labels2 = 2 * np.ones(300).reshape(300,1)
x2_labels = np.hstack((x2, labels2))
x2_labels.shape

data = np.vstack((x1_labels, x2_labels))

np.random.shuffle(data)

data[:,[0, 2]] = data[:,[2, 0]]

def NNclassifier(training, test):
    
    row_num = test.shape[0]
    order = np.zeros(row_num)
    
    
    d = distance.cdist(test[:, 1:], training[:, 1:],'euclidean')
    
    distance_order = d.argsort()
    distance_rank = distance_order[:, 0]
    
    
    for x in range(test.shape[0]):
        min_distance = distance_rank[x]
        order[x] = training[min_distance][0]
    
    order = order.reshape(row_num,1)
    
    
    
    '''for x in range(row_num):
        distance_list = np.zeros(row_num)
        
        for i in range(row_num):
            XA = test[x][1:]
            XA = XA.reshape(1,2)
            
            XB = training[x][1:]
            XB = XB.reshape(1,2)
            
           
            d = distance.cdist(XA, XB, 'euclidean')
            
            distance_list[i] = d

            distance_order = distance_list.argsort()
            min_d_index = distance_order[0] 
            order[x] = training[min_d_index][0]'''
            
    
    return order
    
def n_validator(data, p, classifier, *arg):
    n = data.shape[0]
    remainder = n % p
    sections = np.array([n//p]*p)
    sections[:remainder] = sections[:remainder] + 1
    
    k = 0
    score = 0
    for i in range(p):
        test = data[k:k+sections[i]]
        training = np.delete(data, range(k, k+sections[i]), 0)
        
        
        k = k + sections[i]
        
        labels = classifier(training, test)
        
        for x in range(labels.shape[0]):
            if labels[x] == test[x][0]:
                score += 1
    
    return score/n
#print(NNclassifier(data_set1, data_set2).shape)
with open('wdbc.txt', 'r') as infile:
    temp_data = []
    for line in infile:
        line = line.split()
        temp_data.append(line)
        
    temp_data = np.array(temp_data)
    temp_data = temp_data.astype(float)
    data_set = np.delete(temp_data, 0, 1)
    np.random.shuffle(data_set)


print(n_validator(data_set, 5, NNclassifier))

'''coords = np.array([(35.0456, -85.2672),
          (35.1174, -89.9711),
          (35.9728, -83.9422),
          (36.1667, -86.7833)])
          
a = distance.cdist(coords, coords, 'euclidean')
print(a)
print(a.argsort())   
            '''
def f(t):
    return t**2*np.exp(-t**2)
    
t=np.linspace(0,3,51)
y= f(t) # makes a numpy array
    
#plt.plot(t,y)
#plt.show()   

a = np.array([1,2,3])
b = a[:1]
print (b)            
            
            
