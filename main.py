import os
from statistics import mode
import timeit

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


print("---Reading input file to pandas Dataframe---")
# dataset path
path = '../data'
file_name = 'churnsimulateddata.csv'
file = os.path.join(path, file_name)
print(file)
# read data
df = pd.read_csv(file)
print('Shape of original data: {df.shape()}')

print("---Select features---")
feature_names = ['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products','mean_trnx_amt', 'Churn_risk']
selected_df = df[feature_names]
selected_df = selected_df.dropna()

print('---Split data to create simulated clients---') 
n_clients = 10
clients_data = np.array_split(selected_df.sample(frac=1), n_clients)
print('Number of {np.size(local_clients)} clients. ')

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

class LR_Model():
    def __init__(self):
        self.name = 'LR'


    def fit(self, X_train, X_test, y_train, y_test):
        clf = LogisticRegression(multi_class='ovr', max_iter=500)

        starttime = timeit.default_timer()
        #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)
        model_params = clf.get_params() 
        print(model_params)

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
        print('Coefficients')
        print(clf.coef_)
        
        #testing_time = timeit.default_timer() - starttime
        #print("The testing time is :", testing_time)
        return clf.intercept_, clf.coef_

'''
# Test on all data
X = selected_df[feature_names].drop(columns=['Churn_risk'])
y = selected_df['Churn_risk']
print(np.unique(y, return_counts=True))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LR_Model()
model.fit(X_train, X_test, y_train, y_test)

'''

#Training Local model
for i in range(10):
    print(f'client No {i}')
    model = LR_Model()
    intercept, coef =  model.fit(X_train[i], X_test[i], y_train[i], y_test[i])
    print(model.intercepts_)


#Aggregation


