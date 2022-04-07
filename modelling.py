from lib import *

class LR_ScikitModel():
    
    
    def __init__(self, X_train, X_test, y_train, y_test, figname_prefix='default'):
        self.name = 'LR'
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.figname_prefix = figname_prefix
    
    def fit(self):
        print(f'fiting LR model')
        starttime = timeit.default_timer()
        LR = LogisticRegression()
        LRparam_grid = {
        #'C': [0.0001, 0.001, 0.01, 0.1, 1, 10], #, 1, 10, 100
        #'tol': [1e-7, 1e-6, 1e-5, 1e-4,1e-3, 1e-2],
        #'penalty': ['l1', 'l2'],
        #'max_iter': list(range(200,1200,200)),
        'solver': ['newton-cg']#, 'lbfgs',  'liblinear'],#, 'sag', 'saga',], #'sag', 'saga', 'lbfgs',
        }

        LR_search = GridSearchCV(LR, param_grid=LRparam_grid, refit = True, verbose = 3, cv=10, scoring='f1', n_jobs=-1)

        # fitting the model for grid search 
        LR_search.fit(self.X_train , self.y_train)
        print('Mean f1: %.3f' % LR_search.best_score_)
        print('Best parameters Config: %s' % LR_search.best_params_)

        #Train the model using the training sets y_pred=clf.predict(X_test)        
        best_param = LR_search.best_params_
        
        clf= LogisticRegression(**best_param)
        clf.fit(self.X_train, self.y_train)

        self.predict_proba = clf.predict_proba(self.X_test) 
        model_params = clf.get_params() 
        #best_param = {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear', 'tol': 1e-06}
        #clf = LogisticRegression(**best_param)
        
        training_time = timeit.default_timer() - starttime
        #starttime = timeit.default_timer()
        y_pred=clf.predict(self.X_test)
        f1 = round(np.mean(f1_score(self.y_test, y_pred, labels=0, average='binary'))*100, 2)
        f1_ave = round(f1_score(self.y_test, y_pred, average='weighted')*100, 2)
        accuracy = round(accuracy_score(self.y_test, y_pred)*100,2)
        report = classification_report(self.y_test, y_pred, zero_division=0)
        confusion = confusion_matrix(self.y_test, y_pred)
        #print("The training time is :", training_time)
        #testing_time = timeit.default_timer() - starttime
        #print("The testing time is :", testing_time)   
        self.coeff = clf.coef_
        self.intercept = clf.intercept_

        # errors of coefficients
        self.coeff_error = self.coefficient_error()

        output = {}
        output.update({'intercept': self.intercept, \
                       'coefficients': self.coeff,\
                       'err_coefficients': self.coeff_error,\
                        'f1-score': f1,\
                       'ave_f1-score': f1_ave,\
                       'accuracy': accuracy,\
                        'tunning_parmeter': model_params,\
                       'confusion_matrix': confusion,\
                       'classification_report': report,\
                        'predicted_prob': self.predict_proba,
                       
                        })
            
        self.plot_dist(self.X_train, self.y_train, figname=self.figname_prefix+'_train')
        self.plot_dist(self.X_test, self.y_test, figname=self.figname_prefix+'_test')
        
        return output

    def coefficient_error(self): 

        '''https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture26.pdf'''    
        #c = np.exp(np.tensordot(self.X_test,self.coeff.T, axes=(1))) / (1+np.exp(np.tensordot(self.X_test,self.coeff.T, axes=(1)))**2)
        d = np.exp(np.tensordot(self.X_test,self.coeff.T, axes=(1))) / (1+np.exp(np.tensordot(self.X_test,self.coeff.T, axes=(1)))**2)
        e = np.diag(d.flatten())
        f = np.matmul(self.X_test.T, e)
        g = np.matmul(f, self.X_test)
        h = np.linalg.inv(g)
        i = np.diag(h)
        self.coeff_error = np.sqrt(i)
        print(f'standard error of coefficients: {self.coeff_error}')
        return self.coeff_error

    def plot_dist(self, X, y, figname='default'):
        plots_path = './plots/'
        #sns.set_palette("hls")
        sns.set(rc={'figure.figsize':(16,9)})
        
        _data = pd.concat([X, y], axis=1).reset_index().drop(columns='index')#.drop_duplicates()
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 14))
        
        idx_feature = 0  
        for i in range(ax.shape[0]):
            for j in range(0, ax.shape[1]):
                feature = _data.columns[idx_feature]
                plot = sns.histplot(_data, 
                            hue = 'Churn_risk', 
                            x = feature, 
                            multiple = 'dodge',
                            #binwidth = 1,
                            bins=20,
                            stat = 'count',
                            kde='True',
                            ax=ax[i][j])
                #plot.set(xlim=(x_min,x_max))
                idx_feature += 1 

        fig.suptitle(f'Feature Distribution of Data')
        if figname == 'default':
            plt.savefig('plots/feature_distribution_defaultplot.png')
        else:
            plt.savefig('plots/feature_dist_'+figname+'.png')


        
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

