import os
from statistics import mode
import timeit

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sympy import N


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
feature_names = ['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products','mean_trnx_amt', 'Churn_risk']
selected_df = df[feature_names]
selected_df = selected_df.dropna()

print('---Simulating clients based on geographical locations of the banks---') 
geo_split = 'T'
if geo_split:
    selected_df_v2 = selected_df.sample(frac=1)
    clients_data = []
    for i in range(0, 60, 10):
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
    X = client_data.drop(columns=['Churn_risk'])
    y = client_data['Churn_risk']
    _X_train, _X_test, _y_train, _y_test = train_test_split(
    X, y, test_size=0.2, random_state=42) 
    X_train.append(_X_train)
    X_test.append(_X_test)
    y_train.append(_y_train)
    y_test.append(_y_test)


class LR_ScikitModel():
    def __init__(self):
        self.name = 'LR'

    def fit(self, X_train, X_test, y_train, y_test):

        clf = LogisticRegression(multi_class='ovr', max_iter=1000)
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
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy: ', accuracy)
        print(classification_report(y_test, y_pred, zero_division=0))
        #print('Intercept')
        #print(clf.intercept_)
        #print('Coefficients')
        print(clf.coef_)
        
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
print(intercept_l)
print(coef_l)
print(accuracy_l)

print('---Aggregating at the aggregation server---')
#averaged the local weights
print(np.sum(intercept_l,axis=0))
print(np.sum(coef_l,axis=0))   # axis1=3 becasue there is 3 classes


print('---Constructing model---')
#averaged the local weights
print(np.sum(intercept_l,axis=0))
print(np.sum(coef_l,axis=0))   # axis1=3 becasue there is 3 classes


print('---Testing on local clients---')
#averaged the local weights
print(np.sum(intercept_l,axis=0))
print(np.sum(coef_l,axis=0))   # axis1=3 becasue there is 3 classes

