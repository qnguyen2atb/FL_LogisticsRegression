from lib import *

from read_transform_data import read_and_transform
from simulate_clients import simultedClients, data_balance
from modelling import LR_ScikitModel, multiclass_LogisticFunction
from plotting import plot_f1, plot_hist, plot_comparision, plot_coef_dist


def train_big_model(train_big=False):
    '''
    This function train one churn risk model for all data 
    using the Logistic Regression from Scikitlearn
    '''

    if train_big:
        print('---Training a big model---') 
        #split data
        data = data.sample(frac=1)
        X = data.drop(columns=['Churn_risk'])
        y = data['Churn_risk']
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X, y, test_size=0.3, random_state=42) 

        # balance data
        X_train_b, y_train_b = data_balance(X_train_b, y_train_b, algo='downsampling')
        print('Training Labels count: \n', y_train_b.value_counts())
        print('Testing Labels count: \n', y_test_b.value_counts())
        
        model = LR_ScikitModel(X_train_b, X_test_b, y_train_b, y_test_b, 'bigmodel')
        output =  model.fit()
        pprint.pprint(output)
        print('F1 score of the big fat model: ', output['f1-score'])
        return output


def train_local(X_train, X_test, y_train, y_test, retrain=False):
    '''
    This function train all local models 
    using the Logistic Regression from Scikitlearn
    Input: list of X_train, X_test, y_train, y_test of local models
    Output: f1, error, coef, and intercept of all local models 
    '''

    print('---Retrain the local models or load the pre-trained models---')  
    if retrain==True:
        print('---Training local models at local clients---')
        intercept_l = []
        coef_l = []
        f1_l = []

        error_l = []

        for i in range(np.shape(X_train)[0]):
            print(f'client No {i} - local model')
            
            model = LR_ScikitModel(X_train[i], X_test[i], y_train[i], y_test[i], f'local_model_{i}')
            print('Training Labels count: \n', y_train[i].value_counts())
            print('Testing Labels count: \n', y_test[i].value_counts())
            output =  model.fit()
            pprint.pprint(output)
            print(f"F1 score of the  model for client {i}: {output['f1-score']}")
            intercept_l.append(output['intercept'])
            coef_l.append(output['coefficients'])
            f1_l.append(output['f1-score'])
            # predicted probability of class 1
            prob = output['predicted_prob'][:,1]
            print('predicted probability of class 1', prob[prob > 0.5])
            plot_hist(prob[prob > 0.5], f'prob_dist_local_model_{i}')
            error = output['err_coefficients']
            error_l.append(error)
        
        with open('output_models/f1_parameters.npy', 'wb') as f:
            np.save(f, f1_l)
        with open('output_models/fitting_coef_err.npy', 'wb') as f:
            np.save(f, error_l)
        with open('output_models/fitting_coef.npy', 'wb') as f:
            np.save(f, coef_l)
        with open('output_models/fitting_intercept.npy', 'wb') as f:
            np.save(f, intercept_l)

    else:
        with open('output_models/f1_parameters.npy', 'rb') as f:
            f1_l = np.load(f)
        with open('output_models/fitting_coef_err.npy', 'rb') as f:
            error_l = np.load(f)
        with open('output_models/fitting_coef.npy', 'rb') as f:
            coef_l = np.load(f)
        with open('output_models/fitting_intercept.npy', 'rb') as f:
            intercept_l = np.load(f)
    return f1_l, error_l, coef_l, intercept_l

def test_global_model(X_test, y_test, global_coef, global_intercept, f1_local):
    '''
    This function test all global models on local clients' data
    using the Customized Logistic Regression
    Output: f1 of global models on local data
            comparision plot between f1_l_final_global f1_local
    '''

    print('---Testing global model on local testing data---')
    f1_l_global = []
    for i in range(np.shape(X_test)[0]):
        print(f'\n Client No {i} - global model')
        global_labels =  multiclass_LogisticFunction(X_test[i], np.array(global_coef), np.array(global_intercept))
        f1 = round(np.mean(f1_score(y_test[i], global_labels, average='weighted'))*100,2)
        f1_l_global.append(f1)
        
    f1_l_final_global = []
    # Replace local model with global model if global model is better
    for i, _f1  in enumerate(f1_l_global):
        if f1_local[i] < f1_l_global[i]:
            f1_l_final_global.append(f1_l_global[i])  
        else:
            f1_l_final_global.append(f1_local[i]) 

    plot_comparision(f1_l_final_global, f1_local)


def main():
    
    '''
    This main class will execute the federated learning pipeline, this main class will:  
    - Read and transform the the simulated churn risk data from ATB
    - Optional: Train a big model for all data 
    - Simulating clients based on geographical locations of the banks
    - Optional: Retrain the local models OR load the pre-trained models
    - Aggregate the local models at the simulated aggregation server
    - Test the global model on clients' local testing data
    Input: churnsimulateddata.csv
    '''

    print("---Read & Transform Data---")    
    data = read_and_transform(binary_or_multiclass='binary')
    output_bigmodel = train_big_model(train_big=False)

    print('---Simulate clients based on geographical locations of the banks---') 
    test_client = simultedClients(data=data, split_feature= 'geo', n_clients=60)
    X_train, X_test, y_train, y_test = test_client.createBalancedClients(algo='downsampling', balance_test_data=False)

    print('---Train local models---')
    f1_local, error_l, coef_l, intercept_l = train_local(X_train, X_test, y_train, y_test, retrain=True)

    print('---Aggregating at the aggregation server---')
    #averaged the local weights & biases
    global_intercept = np.mean(intercept_l,axis=0)
    global_coef = np.mean(coef_l,axis=0)
    print('GLOBAL intercept and coefficient: \n', global_intercept, global_coef)

    test_global_model(X_test, y_test, global_coef, global_intercept, f1_local)    

if __name__ == "__main__":
    main()
