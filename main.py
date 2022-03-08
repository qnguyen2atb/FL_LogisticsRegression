#from audioop import mul
#from multiprocessing.context import assert_spawning
import os
#from statistics import mode
import timeit
#from bitarray import test

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
#from sympy import N


print("---Reading input file to pandas Dataframe---")
# dataset path
path = 'data'
file_name = 'churnsimulateddata.csv'
file = os.path.join(path, file_name)
print(file)
# read data
df = pd.read_csv(file)
print(f'Shape of original data: {df.shape}')

print("---Select features---")
feature_names = ['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products', 'Churn_risk']
#feature_names = ['Age','PSYTE_Segment','Total_score','Churn_risk']
#feature_names = ['PSYTE_Segment','Total_score','Churn_risk']

selected_df = df[feature_names].dropna()

#selected_df['Churn_risk'][selected_df['Churn_risk'] == 'Medium'] = 'High'  

selected_df['Churn_risk'] = selected_df.Churn_risk.astype("category").cat.codes

print(selected_df['Churn_risk'].unique())

selected_df = selected_df.dropna()
#selected_df = selected_df.drop(selected_df[(selected_df.Churn_risk != 0) or (selected_df.Churn_risk != 1) (selected_df.Churn_risk != 2)].index)
#print(selected_df['Churn_risk'].unique())


print('---Simulating clients based on geographical locations of the banks---') 
geo_split = 'T'
if geo_split:
    selected_df_v2 = selected_df.sample(frac=1)
    clients_data = []
    for i in range(0, 60, 4):
        clients_data.append(selected_df_v2[(selected_df_v2.PSYTE_Segment >= i) & (selected_df_v2.PSYTE_Segment < i+10)]) 
    
else:
    n_clients = 10
    clients_data = np.array_split(selected_df.sample(frac=1), n_clients)
    
print(f'Number of {np.size(clients_data)} clients. ')
X_train = []
X_test = []
y_train = []
y_test = []
for i, client_data in enumerate(clients_data):
    X = client_data.drop(columns=['Churn_risk','PSYTE_Segment'])
    print(X.columns)
    y = client_data['Churn_risk']
    
    _X_train, _X_test, _y_train, _y_test = train_test_split(
    X, y, test_size=0.2, random_state=42) 
    print(np.shape(_X_train))
    
    from imblearn.over_sampling import SMOTE, ADASYN
    #_X_train, _y_train = SMOTE().fit_resample(_X_train, _y_train)
    #_X_train, _y_train = ADASYN().fit_resample(_X_train, _y_train)

    
    print(np.shape(_X_train))

    X_train.append(_X_train)
    X_test.append(_X_test)
    y_train.append(_y_train)
    y_test.append(_y_test)


class LR_ScikitModel():
    def __init__(self):
        self.name = 'LR'

    def fit(self, X_train, X_test, y_train, y_test):

        clf = LogisticRegression(multi_class='multinomial', max_iter=10000, n_jobs=-1)
        starttime = timeit.default_timer()
        #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)
        model_params = clf.get_params() 
        training_time = timeit.default_timer() - starttime
        print("The training time is :", training_time)
        #starttime = timeit.default_timer()
        y_pred=clf.predict(X_test)
        #precison = metrics.precision_score(y_test, y_pred, average='weighted')
        #print('Precison: ', precison)
        #recall = metrics.recall_score(y_test, y_pred, average='weighted')
        #print('Recall: ', recall)
        #f1 = metrics.f1_score(self.y_test, self.y_pred, average='weighted')
        #print('F1: ', f1)
        accuracy = round(accuracy_score(y_test, y_pred)*100,2)
        print('Accuracy: ', accuracy)
        print(classification_report(y_test, y_pred, zero_division=0))
        #print('Intercept')
        #print(clf.intercept_)
        #print('Coefficients')
        #print(clf.coef_)
        
        #testing_time = timeit.default_timer() - starttime
        #print("The testing time is :", testing_time)
        return clf.intercept_, clf.coef_, accuracy


print('---Training local models at local clients---')
#Training Local model
intercept_l = []
coef_l = []
accuracy_l = []
for i in range(np.size(clients_data)):
    print(f'client No {i}')
    model = LR_ScikitModel()
    intercept, coef, accuracy =  model.fit(X_train[i], X_test[i], y_train[i], y_test[i])
    intercept_l.append(intercept)
    coef_l.append(coef)
    accuracy_l.append(accuracy)
#print(intercept_l)
#print(coef_l)
#print(accuracy_l)


print('---Aggregating at the aggregation server---')
#averaged the local weights
#print(np.sum(intercept_l,axis=0))
#print(np.sum(coef_l,axis=0))   # axis1=3 becasue there is 3 classes


print('---Constructing model---')
#averaged the local weights
global_intercept = np.sum(intercept_l,axis=0)
global_coef = np.sum(coef_l,axis=0) 
print(global_intercept)
print(global_coef)   # axis1=3 becasue there is 3 classes


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
            
    def predict_(X, W, b):

        assert np.shape(X)[1] == np.shape(W)[1]   
        assert np.shape(W)[0] == np.shape(b)[0]   

        pre_vals = np.dot(X, W.T) + b
        return softmax(pre_vals)
    
    probability = predict_(X, W, b)
    max_prob = np.amax(probability, axis=1, keepdims=True)
    #print(np.shape(max_prob))
    label = np.argmax(probability, axis=1)

    return label



print('---Testing global model on local testing data---')
gl_m_accuracy_l = []
for i in range(np.size(clients_data)):
    print(f'client No {i}')
    model = LR_ScikitModel()

    label =  multiclass_LogisticFunction(X_test[i], np.array(global_coef), np.array(global_intercept))
    gl_m_accuracy = round(accuracy_score(y_test[i], label)*100,2)
    print('Global model Accuracy: ', gl_m_accuracy)
    gl_m_accuracy_l.append(gl_m_accuracy)
    print(classification_report(y_test[i], label, zero_division=0))
    
    
print(accuracy_l, np.mean(accuracy_l))
print(gl_m_accuracy_l, np.mean(gl_m_accuracy_l))


