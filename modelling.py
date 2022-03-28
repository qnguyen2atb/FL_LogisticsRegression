from lib import *

class LR_ScikitModel():
    def __init__(self):
        self.name = 'LR'

    def fit(self, X_train, X_test, y_train, y_test):
        print(f'fiting LR model with theses classes: {y_train.value_counts()}')
        #best_param = {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear', 'tol': 1e-06}
        #clf = LogisticRegression(**best_param)
        starttime = timeit.default_timer()
        LR = LogisticRegression()
        LRparam_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10], #, 1, 10, 100
        'tol': [1e-7, 1e-6, 1e-5, 1e-4,1e-3, 1e-2],
        'penalty': ['l1', 'l2'],
        'max_iter': list(range(200,1200,200)),
        'solver': ['newton-cg', 'lbfgs',  'liblinear'], #'sag', 'saga', 'lbfgs',
        }

        LR_search = GridSearchCV(LR, param_grid=LRparam_grid, refit = True, verbose = 3, cv=10, scoring='f1', n_jobs=-1)

        # fitting the model for grid search 
        LR_search.fit(X_train , y_train)
        print('best parameters', LR_search.best_params_)
        # summarize
        print('Mean f1: %.3f' % LR_search.best_score_)
        print('Best parameters Config: %s' % LR_search.best_params_)

        #Train the model using the training sets y_pred=clf.predict(X_test)        
        best_param = LR_search.best_params_
        
        clf= LogisticRegression(**best_param)
        clf.fit(X_train, y_train)

        self.predict_proba = clf.predict_proba(X_test) 
        model_params = clf.get_params() 
        training_time = timeit.default_timer() - starttime
        #starttime = timeit.default_timer()
        y_pred=clf.predict(X_test)
        f1 = round(np.max(f1_score(y_test, y_pred, average=None))*100, 2)
        f1 = round(np.mean(f1_score(y_test, y_pred, labels=0, average='binary'))*100, 2)
        f1_ave = round(f1_score(y_test, y_pred, average='weighted')*100, 2)
        accuracy = round(accuracy_score(y_test, y_pred)*100,2)
        report = classification_report(y_test, y_pred, zero_division=0)
        print(confusion_matrix(y_test, y_pred))
        print("The training time is :", training_time)
        print('Accuracy: ', accuracy)
        print('f1', f1)
        #testing_time = timeit.default_timer() - starttime
        #print("The testing time is :", testing_time)   
        self.coeff = clf.coef_
        self.X_test = X_test

        return clf.intercept_, clf.coef_, accuracy, f1, f1_ave, report

    def coefficient_error(self): 

        '''https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture26.pdf'''    
        #c = np.exp(np.tensordot(self.X_test,self.coeff.T, axes=(1))) / (1+np.exp(np.tensordot(self.X_test,self.coeff.T, axes=(1)))**2)
        d = np.exp(np.tensordot(self.X_test,self.coeff.T, axes=(1))) / (1+np.exp(np.tensordot(self.X_test,self.coeff.T, axes=(1)))**2)
        e = np.diag(d.flatten())
        f = np.matmul(self.X_test.T, e)
        g = np.matmul(f, self.X_test)
        h = inv(g)
        i = np.diag(h)
        self.coeff_error = np.sqrt(i)
        print(f'standard error of coefficients: {self.coeff_error}')
        return self.coeff_error

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
        label = np.argmax(probability, axis=1)
    else:
        label = [1 if p > 0.5 else 0 for p in probability]

    return label


