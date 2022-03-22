
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import timeit

import numpy as np

class LR_ScikitModel():
    def __init__(self):
        self.name = 'LR'

    def fit(self, X_train, X_test, y_train, y_test):
        print(f'fiting LR model with theses classes: {y_train.value_counts()}')
        clf = LogisticRegression(multi_class='auto', max_iter=10000, n_jobs=-1, warm_start=True)
        starttime = timeit.default_timer()
        #Train the model using the training sets y_pred=clf.predict(X_test)        
        clf.fit(X_train, y_train)
        model_params = clf.get_params() 
        training_time = timeit.default_timer() - starttime
        #starttime = timeit.default_timer()
        y_pred=clf.predict(X_test)
        f1 = round(np.max(f1_score(y_test, y_pred, average=None))*100, 2)
        f1_ave = round(f1_score(y_test, y_pred, average='weighted')*100, 2)
        accuracy = round(accuracy_score(y_test, y_pred)*100,2)
        #print(classification_report(y_test, y_pred, zero_division=0))
        report = classification_report(y_test, y_pred, zero_division=0)
        print("The training time is :", training_time)
        print('Accuracy: ', accuracy)
        print('f1', f1)
        #testing_time = timeit.default_timer() - starttime
        #print("The testing time is :", testing_time)
        return clf.intercept_, clf.coef_, accuracy, f1, f1_ave, report



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
        #assert np.shape(W)[0] == np.shape(b)[0]   

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


