#from audioop import mul
#from multiprocessing.context import assert_spawning
import os
#from statistics import mode
import timeit
#from bitarray import test

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
#from sympy import N

#from dataExploration import Data_Exploration
from read_transform_data import read_and_transform
from simulate_clients import simultedClients

def plot_f1(local_f1, glob_f1):
    local_T = list(local_f1.T)
    glob_T = list(glob_f1.T)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17,5))
    fig.suptitle('f1-score')
    X = np.arange(6)
    color_l = ['b','g', 'y','r','c', 'm', 'k', 'black','purple', 'pink', 'olive', 'gray', 'orange', 'lime']
    for i in range(np.shape(local_T)[1]):
        ax1.bar(X[0] + i, local_T[0][i], color = 'red', width = 0.2)
        ax1.bar(X[0] + 0.2+i, glob_T[0][i], color = 'green', width = 0.2)
        ax2.bar(X[0] + i, local_T[1][i], color = 'red', width = 0.2)
        ax2.bar(X[0] + 0.2+i, glob_T[1][i], color = 'green', width = 0.2)
        ax3.bar(X[0] + i, local_T[2][i], color = 'red', width = 0.2)
        ax3.bar(X[0] + 0.2+i, glob_T[2][i], color = 'green', width = 0.2)
    ax1.set_title('High')
    ax1.set_xlabel('')
    ax1.set_xticks([0,1,2,3,4,5])
    #ax1.set_xticks(['High','Low','Medium'])
    ax1.set_xticklabels(['Client 0','Client 1', 'Client 2', 'Client 3', 'Client 4', 'Client 5'])
    #ax1.axhline(y=0.5, color='r', linestyle='-')
    ax1.set_ylim([0.4, 0.8])

    ax2.set_title('Low')
    ax2.set_xticks([0,1,2,3,4,5])
    ax2.set_xticklabels(['Client 0','Client 1', 'Client 2', 'Client 3', 'Client 4', 'Client 5'])
    #ax2.axhline(y=0.5, color='r', linestyle='-')
    #ax2.set_xticks([0.2,1.2])
    ax2.set_ylim([0.4, 0.8])
     
    ax3.set_title('Medium')
    ax3.set_xticks([0,1,2,3,4,5])
    ax3.set_xticklabels(['Client 0','Client 1', 'Client 2', 'Client 3', 'Client 4', 'Client 5'])
    #ax2.axhline(y=0.5, color='r', linestyle='-')
    #ax3.set_xticks([0.2,1.2])
    
    ax3.set_ylim([0.4, 0.8])
    plt.savefig('f1_score.png')

def plot_f12c(local_f1, glob_f1):
    local_T = list(local_f1.T)
    glob_T = list(glob_f1.T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,5))
    fig.suptitle('f1-score')
    X = np.arange(6)
    color_l = ['b','g', 'y','r','c', 'm', 'k', 'black','purple', 'pink', 'olive', 'gray', 'orange', 'lime']
    for i in range(np.shape(local_T)[1]):
        ax1.bar(X[0] + i, local_T[0][i], color = 'red', width = 0.2)
        ax1.bar(X[0] + 0.2+i, glob_T[0][i], color = 'green', width = 0.2)
        ax2.bar(X[0] + i, local_T[1][i], color = 'red', width = 0.2)
        ax2.bar(X[0] + 0.2+i, glob_T[1][i], color = 'green', width = 0.2)
    ax1.set_title('High')
    ax1.set_xlabel('')
    ax1.set_xticks([0,1,2,3,4,5])
    #ax1.set_xticks(['High','Low','Medium'])
    ax1.set_xticklabels(['Client 0','Client 1', 'Client 2', 'Client 3', 'Client 4', 'Client 5'])
    #ax1.axhline(y=0.5, color='r', linestyle='-')
    ax1.set_ylim([0.8, 0.86])

    ax2.set_title('Low')
    ax2.set_xticks([0,1,2,3,4,5])
    ax2.set_xticklabels(['Client 0','Client 1', 'Client 2', 'Client 3', 'Client 4', 'Client 5'])
    #ax2.axhline(y=0.5, color='r', linestyle='-')
    #ax2.set_xticks([0.2,1.2])
    ax2.set_ylim([0.8, 0.86])

    plt.savefig('f1_score.png')


print("---Reading input file to pandas Dataframe---")
data = read_and_transform()

print('---Simulating clients based on geographical locations of the banks---') 
test_client = simultedClients(data=data, n_clients=30)
X_train, X_test, y_train, y_test = test_client.createBalancedClients(algo='downsampling')



class LR_ScikitModel():
    def __init__(self):
        self.name = 'LR'

    def fit(self, X_train, X_test, y_train, y_test):

        clf = LogisticRegression(multi_class='multinomial', max_iter=5000, n_jobs=-1)
        #clf = LogisticRegression(multi_class='auto', max_iter=500, n_jobs=-1)
        
        starttime = timeit.default_timer()
        #Train the model using the training sets y_pred=clf.predict(X_test)
        
        clf.fit(X_train, y_train)
        model_params = clf.get_params() 
        training_time = timeit.default_timer() - starttime
        #starttime = timeit.default_timer()
        y_pred=clf.predict(X_test)
        #precison = metrics.precision_score(y_test, y_pred, average='weighted')
        #print('Precison: ', precison)
        #recall = metrics.recall_score(y_test, y_pred, average='weighted')
        #print('Recall: ', recall)
        f1 = f1_score(y_test, y_pred, average=None)
        f1_ave = round(f1_score(y_test, y_pred, average='weighted'), 4)
                

        #print('F1: ', f1)
        accuracy = round(accuracy_score(y_test, y_pred)*100,2)

        print(classification_report(y_test, y_pred, zero_division=0))
        report = classification_report(y_test, y_pred, zero_division=0)
        print("The training time is :", training_time)
        print('Accuracy: ', accuracy)

        #print('Intercept')
        #print(clf.intercept_)
        #print('Coefficients')
        #print(clf.coef_)
        
        #testing_time = timeit.default_timer() - starttime
        #print("The testing time is :", testing_time)
        return clf.intercept_, clf.coef_, accuracy, f1, f1_ave, report


print('---Training a big model for all---')
#Training Local model
intercept_l = []
coef_l = []
row_l = []
accuracy_l = []
report_l = []
model = LR_ScikitModel()
X = selected_df.drop(columns=['Churn_risk'])
y = selected_df['Churn_risk']

# Data normalization
#X = normalize(X, norm='l2', axis=1, copy=True, return_norm=False)
#print(X)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X, y, test_size=0.3, random_state=42) 


# balance data
X_train_b, y_train_b = data_balance(X_train_b, y_train_b, algo='downsampling')
X_test_b, y_test_b = data_balance(X_test_b, y_test_b, algo='downsampling')
print('Training Labels count: \n', y_train_b.value_counts())
print('Testing Labels count: \n', y_test_b.value_counts())
intercept_b, coef_b, accuracy_b, f1_b, f1_ave_b, report_b =  model.fit(X_train_b, X_test_b, y_train_b, y_test_b)


#print(intercept_l)
#print(coef_l)
#print(accuracy_l)

print('---Training local models at local clients---')
#Training Local model
intercept_l = []
coef_l = []
row_l = []
accuracy_l = []
report_l = []
f1_l = []
f1_ave_l = []
for i in range(np.size(clients_data)):
    print(f'client No {i} - local model')
    model = LR_ScikitModel()
    #print(y_train[i].unique(), y_test[i].unique())
    print('Training Labels count: \n', y_train[i].value_counts())
    print('Testing Labels count: \n', y_test[i].value_counts())
    intercept, coef, accuracy, f1, f1_ave, report =  model.fit(X_train[i], X_test[i], y_train[i], y_test[i])
    intercept_l.append(intercept)
    coef_l.append(coef)
    accuracy_l.append(accuracy)
    row_l.append(np.size(accuracy))
    report_l.append(report)
    f1_l.append(f1)
    f1_ave_l.append(f1_ave)
#print(intercept_l)
#print(coef_l)
#print(accuracy_l)


print('---Aggregating at the aggregation server---')
#averaged the local weights
#print(np.sum(intercept_l,axis=0))
#print(np.sum(coef_l,axis=0))   # axis1=3 becasue there is 3 classes


print('---Constructing model---')
#averaged the local weights
global_intercept = np.sum(intercept_l,axis=0)/6
global_coef = np.sum(coef_l,axis=0)/6 
#print(np.shape(global_intercept))
#print(np.shape(global_coef))   # axis1=3 becasue there is 3 classes
#global_intercept = np.sum(intercept_l*row_l)/np.sum(row_l)
#global_coef = np.sum(np.multiply(coef_l, row_l))/np.sum(row_l)
#print(np.shape(global_intercept))
#print(np.shape(global_coef))


print('---Testing on local clients---')
#
#print(np.sum(intercept_l,axis=0))
#print(np.sum(coef_l,axis=0))   # axis1=3 becasue there is 3 classes



def multiclass_LogisticFunction(X, W, b):
    '''
    Logistics Regression function
    Input: 
        X: input data in form of a matrix with size (n_samples, n_features)
        W: Weight or logistics coefficient matrix with size (n_classes, n_features)
        b: bias or intercept vector with size (n_classes)  
        ref: https://github.com/bamtak/machine-learning-implemetation-python/blob/master/Multi%20Class%20Logistic%20Regression.ipynb
    '''

    def softmax(z):
        prob = np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)
        return prob
    
    def _sigmoid_function(x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)
    def _sigmoid(x):
        return np.array([_sigmoid_function(value) for value in x])



    def predict_(X, W, b):

        assert np.shape(X)[1] == np.shape(W)[1]   
        assert np.shape(W)[0] == np.shape(b)[0]   

        pre_vals = np.dot(X, W.T) + b
        if np.size(b) > 2:            
            return softmax(pre_vals)
        else:
            return _sigmoid(pre_vals)


    probability = predict_(X, W, b)

    if np.size(b) > 2:
        max_prob = np.amax(probability, axis=1, keepdims=True)
        #print(np.shape(max_prob))
        label = np.argmax(probability, axis=1)
    else:
        label = [1 if p > 0.5 else 0 for p in probability]

    return label



print('---Testing global model on local testing data---')
gl_m_accuracy_l = []
global_report_l = []
gl_m_f1_l = []
gl_m_f1_ave_l = []
for i in range(np.size(clients_data)):
    print(f'client No {i} - global model')
    model = LR_ScikitModel()

    label =  multiclass_LogisticFunction(X_test[i], np.array(global_coef), np.array(global_intercept))
    gl_m_accuracy = round(accuracy_score(y_test[i], label)*100,2)
    gl_m_accuracy_l.append(gl_m_accuracy)
    f1 = f1_score(y_test[i], label, average=None)
    gl_m_f1_l.append(f1)
    f1_ave = round(f1_score(y_test[i], label, average='weighted'),4)
    gl_m_f1_ave_l.append(f1_ave)
    global_report = classification_report(y_test[i], label, zero_division=0)
    print(classification_report(y_test[i], label, zero_division=0))
    global_report_l.append(global_report)
    print('Global model Accuracy: ', gl_m_accuracy)
    
#print(accuracy_l, np.mean(accuracy_l))
#print(gl_m_accuracy_l, np.mean(gl_m_accuracy_l))

print(f1_l, np.mean(f1_l))
print(gl_m_f1_l)
print('\n')
print(np.array(f1_ave_l))
print(np.array(gl_m_f1_ave_l))
print('\n')
print(np.array(f1_l))
print('\n')
print(np.array(gl_m_f1_l))

print(np.shape(f1_ave_l))
print(np.shape(gl_m_f1_ave_l))
print(np.shape(f1_l))
print(np.shape(gl_m_f1_l))

#plot_f1(np.array(f1_l), np.array(gl_m_f1_l))
plot_f12c(np.array(f1_l), np.array(gl_m_f1_l))

print('Global model')
print(global_coef)
print(global_intercept)

for i, c in enumerate(coef_l):
    print(f'Local model {i}')
    print(coef_l[i])
    print('\n')
    print(intercept_l[i])
    print('\n')