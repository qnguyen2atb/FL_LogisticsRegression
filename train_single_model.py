import os
import pandas as pd
import numpy as np
import timeit

import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import seaborn as sns


def train_single_model(file_name):
    path = 'data'
    file_name = file_name
    file = os.path.join(path, file_name)
    print(file)
    # read data
    df = pd.read_csv(file)
    feature_names = ['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products', 'Churn_risk']
    #feature_names = ['Age','Tenure','PSYTE_Segment','Trnx_count','num_products', 'Churn_risk']

    selected_df = df[feature_names].dropna()

    # binarize
    selected_df['Churn_risk'][selected_df['Churn_risk'] == 'Medium'] = 'High'  
    #selected_df = selected_df.drop(selected_df[selected_df['Churn_risk'] == 'Medium'].index) 
    

    selected_df['Churn_risk'] = selected_df.Churn_risk.astype("category").cat.codes
    client_data = selected_df.dropna()

    #for rat in range(100,1,-1):
    #data_s = data.sample(frac=0.01)
    #data_r = data_s.sample(frac=rat/100.)
    client_data = client_data.sample(frac=1)
    X = client_data.drop(columns=['Churn_risk','PSYTE_Segment'])
    y = client_data['Churn_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42) 
    
    X = pd.concat([X_train, y_train], axis=1)
    low = X[X.Churn_risk==1]
    high = X[X.Churn_risk==0]
    #medium = X[X.Churn_risk==2]
    
    X.Churn_risk.value_counts()

    low_downsampled = resample(low,
                                replace = True, # sample without replacement
                                n_samples = len(high), # match minority n
                                random_state = 27) # reproducible results

    # combine minority and downsampled majority
    downsampled = pd.concat([low_downsampled, high])

    # checking counts
    print('Labels counts after balancing.')
    print(downsampled.Churn_risk.value_counts())

    X_train = downsampled.drop('Churn_risk', axis=1)
    y_train = downsampled.Churn_risk
    
    print(f'Traing size {np.shape(y_train)}')
    print(f'Testing size {np.shape(y_test)}')

    LR = LogisticRegression()
    LRparam_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'tol': [1e-6, 1e-5, 1e-4, 1e-2, 1e-1, 1],
    'penalty': ['l1', 'l2'],
    'max_iter': list(range(100,800,100)),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }

    clf = GridSearchCV(LR, param_grid=LRparam_grid, refit = True, verbose = 3, cv=10, scoring='f1')

    # fitting the model for grid search 
    clf.fit(X_train , y_train)
    print('best parameters', clf.best_params_)
    # summarize
    print('Mean f1: %.3f' % clf.best_score_)
    print('Best parameters Config: %s' % clf.best_params_)

    #clf = LogisticRegressionCV( cv=10, max_iter=100, tol=1e-2, n_jobs=-1, solver='liblinear', scoring='f1')
    #clf = LogisticRegressionCV( cv=10, max_iter=100000, tol=1e-6, n_jobs=-1, solver='lbfgs')

    #clf.fit(X_train, y_train)

    #spit the test data into 2 set
    X_combined = pd.concat([X_test, y_test], axis=1)
    X_test_b = np.array_split(X_combined.sample(frac=1), 10)
    #a = pd.DataFrame(X_test_b[0])

    # train    
    y_pred3=clf.predict(X_train)
    f12 = round(np.max(f1_score(y_train, y_pred3, average=None))*100, 2)
    print(f'The performance of the model: {np.round(clf.score(X_test, y_test)*100,2)}')

    # folding
    #spit the test data into 2 set
    X_combined = pd.concat([X_test, y_test], axis=1)
    X_test_b = np.array_split(X_combined.sample(frac=1), 10)

    for k, test in enumerate(X_test_b):
        y_pred_1=clf.predict(test.drop(columns='Churn_risk'))
        f1 = round(np.max(f1_score(test['Churn_risk'], y_pred_1, average=None))*100, 2)
        print(f'f1 for {k}th-fold test data', f1)

    '''
    #f1_ave = round(f1_score(y_test, y_pred, average='weighted')*100, 2)
    #accuracy = round(accuracy_score(y_test, y_pred)*100,2)
    report1 = classification_report(X_test_b[0]['Churn_risk'], y_pred_1, zero_division=0)
    print(report1)
    report2 = classification_report(X_test_b[1]['Churn_risk'], y_pred_2, zero_division=0)
    print(report2)
    
    print(classification_report(y_train, y_pred3, zero_division=0))
    print(confusion_matrix(X_test_b[0]['Churn_risk'], y_pred_1))
    print(confusion_matrix(X_test_b[1]['Churn_risk'], y_pred_2))
    print('f1 for test 1 data', f1)
    print('f1 for test 2 data', f2)
    print('f1 for train data', f12)

    '''
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 14))
    idx_feature = 0  

    _X = pd.concat([X_train, y_train], axis=1).reset_index().drop(columns='index')
    print(np.shape(_X))
    for i in range(ax.shape[0]):
        for j in range(0, ax.shape[1]):
            try:  
                feature = _X.columns[idx_feature]    
                plot = sns.histplot(_X, 
                                hue = 'Churn_risk', 
                                x = feature, 
                                multiple = 'stack',
                                binwidth = 1,
                                bins=5,
                                stat = 'count',
                                ax=ax[i][j])
                #plt.show()
                idx_feature +=1
            except IndexError:
                pass
            plt.savefig(f'feature_distribution_train.png')

    _Y = pd.concat([X_test, y_test], axis=1).reset_index().drop(columns='index')

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 14))
    idx_feature = 0  
    for i in range(ax.shape[0]):
        for j in range(0, ax.shape[1]):
            try:  
                feature = _Y.columns[idx_feature]    
                plot = sns.histplot(_Y, 
                                hue = 'Churn_risk', 
                                x = feature, 
                                multiple = 'stack',
                                binwidth = 1,
                                bins=5,
                                stat = 'count',
                                ax=ax[i][j])
                #plt.show()
                idx_feature +=1
            except IndexError:
                pass
            plt.savefig(f'feature_distribution_test.png')
            
train_single_model('churnsimulateddata.csv')