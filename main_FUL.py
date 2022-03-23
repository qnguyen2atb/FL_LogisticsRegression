from lib import *

#from dataExploration import Data_Exploration
from read_transform_data import read_and_transform
from simulate_clients import simultedClients, data_balance
from modelling import LR_ScikitModel, multiclass_LogisticFunction
from plotting import plot_f1

def test_global_model(X_test, y_test, unlearn_global_coef, unlearn_global_intercept, f1_l, f1_l_final_global, unlearn):
    '''
    Inputs: 
        X_test: a list of multidimensional testing data at client sites
        X_test: a list of labels of testing data at client sites  
        global_coef, global_intercept: aggragated Logistic Regrestion parameters
    '''  
    print('---Testing global model on local testing data---')

    f1_unlearn_final_model = []

    # UNLEARN MODELS
    for i in range(np.shape(X_test)[0]):
        print(f'\n Client No {i} - global model')
        model = LR_ScikitModel()
        gl_labels =  multiclass_LogisticFunction(X_test[i], np.array(unlearn_global_coef), np.array(unlearn_global_intercept))
        f1 = round(np.max(f1_score(y_test[i], gl_labels, average=None))*100,2)
        f1_unlearn_final_model.append(f1)
        print(classification_report(y_test[i], gl_labels, zero_division=0))
        print('Confusion matrix \n', confusion_matrix(y_test[i], gl_labels))    

        #plot_f12c(np.array(f1_l), np.array(gl_m_f1_l))
        #plot_f12c(np.array(f1_l), np.array(final_gl_model),figname='f1_scoreNEW.png')
        #plot_f1(np.array(f1_l), np.array(f1_l_final_global), \
        #    figname='f1_score_unlearn_'+str(unlearn)+'_clients.png')
        #plot_f12c(np.array(f1_l), np.array(final_gl_model),figname='f1_scoreNEW.png')
    plot_f1(np.array(f1_l), np.array(f1_l_final_global),np.array(f1_unlearn_final_model), \
             figname='f1_score_unlearn_'+str(unlearn)+'_clients.png')
    print(np.mean(np.abs(np.array(f1_unlearn_final_model) - np.array(f1_l_final_global))))
    print(np.max(np.abs(np.array(f1_unlearn_final_model) - np.array(f1_l_final_global))))
    
    print((np.array(f1_unlearn_final_model) - np.array(f1_l_final_global)))
    print((np.array(f1_unlearn_final_model),np.array(f1_l_final_global)))
    


print("---EXECUTING THE MAIN PIPELINE---")
print("---Reading input file to pandas Dataframe---")
data = read_and_transform(binary_or_multiclass='binary')

print('---Training a big model for all---')
model = LR_ScikitModel()
X = data.drop(columns=['Churn_risk'])
y = data['Churn_risk']
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X, y, test_size=0.3, random_state=42) 

# balance data
X_train_b, y_train_b = data_balance(X_train_b, y_train_b, algo='downsampling')
print('Training Labels count: \n', y_train_b.value_counts())
print('Testing Labels count: \n', y_test_b.value_counts())
intercept_b, coef_b, accuracy_b, f1_b, f1_ave_b, report_b =  model.fit(X_train_b, X_test_b, y_train_b, y_test_b)
print('F1 score of the big fat model: ', f1_b)


print('---Simulating clients based on geographical locations of the banks---') 
test_client = simultedClients(data=data, split_feature= 'geo', n_clients=60)
X_train, X_test, y_train, y_test = test_client.createBalancedClients(algo='downsampling', balance_test_data=False)


print('---Training local models at local clients---')
intercept_l = []
coef_l = []
f1_l = []

for i in range(np.shape(X_train)[0]):
    print(f'client No {i} - local model')
    model = LR_ScikitModel()
    print('Training Labels count: \n', y_train[i].value_counts())
    print('Testing Labels count: \n', y_test[i].value_counts())
    intercept, coef, accuracy, f1, f1_ave, report =  model.fit(X_train[i], X_test[i], y_train[i], y_test[i])
    intercept_l.append(intercept)
    coef_l.append(coef)
    f1_l.append(f1)


print('---Aggregating at the aggregation server---')
#averaged the local weights & biases
global_intercept = np.mean(intercept_l,axis=0)
global_coef = np.mean(coef_l,axis=0)


print('---Testing global model on local testing data---')

f1_l_global = []
for i in range(np.shape(X_test)[0]):
    print(f'\n Client No {i} - global model')
    model = LR_ScikitModel()
    global_labels =  multiclass_LogisticFunction(X_test[i], np.array(global_coef), np.array(global_intercept))
    f1 = round(np.max(f1_score(y_test[i], global_labels, average=None))*100,2)
    f1_l_global.append(f1)

f1_l_final_global = []

for i, _f1  in enumerate(f1_l_global):
    if f1_l[i] < f1_l_global[i]:
        f1_l_final_global.append(f1_l_global[i])  
    else:
        f1_l_final_global.append(f1_l[i]) 
        

plot_f1(np.array(f1_l), np.array(f1_l_final_global), \
         figname='f1_score_pre-unlearn_global_model.png')


print('---Unlearn the entire client---')
#averaged the local weights

print(intercept_l)
print(np.size(intercept_l))
print(np.size(intercept_l[0:10]))
print(np.size(np.delete(intercept_l, [1,2,3]) ))
print(intercept_l, np.delete(intercept_l, [1,2,3]) )

'''
for i in range(0, np.size(intercept_l)):
    unlearn_clients = np.arange(0, i )
    intercept_l_u = np.delete(intercept_l, unlearn_clients, axis=0)
    coef_l_u = np.delete(coef_l, unlearn_clients, axis=0)
    unlearn_global_intercept = np.mean(intercept_l_u,axis=0)
    unlearn_global_coef = np.mean(coef_l_u,axis=0)
    print('COEFF ', global_coef, unlearn_global_coef, global_intercept, unlearn_global_intercept)

    test_global_model(X_test, y_test, unlearn_global_coef, unlearn_global_intercept, f1_l, f1_l_final_global, i)

'''



for rat in range(100,1,-1):
    #print('---Training a big model for all---')
    model = LR_ScikitModel()
    data_s = data.sample(frac=0.01)
    data_r = data_s.sample(frac=rat/100.)
    X = data_r.drop(columns=['Churn_risk'])
    y = data_r['Churn_risk']
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y, test_size=0.3, random_state=42) 

    # balance data
    X_train_b, y_train_b = data_balance(X_train_b, y_train_b, algo='downsampling')
    #print('Training Labels count: \n', y_train_b.value_counts())
    #print('Testing Labels count: \n', y_test_b.value_counts())
    intercept_b, coef_b, accuracy_b, f1_b, f1_ave_b, report_b =  model.fit(X_train_b, X_test_b, y_train_b, y_test_b)
    print(f'F1 score of the big fat model: {f1_b}, for data is sampled at {rat/100.}')


