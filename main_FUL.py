from lib import *
import numpy as np 


#from dataExploration import Data_Exploration
from read_transform_data import read_and_transform
from simulate_clients import simultedClients, data_balance
from modelling import LR_ScikitModel, multiclass_LogisticFunction
from plotting import plot_f1, plot_hist, plot_coef_dist

                                                            
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
    for i in range(np.shape(X_test)[0]-5):
        print(f'\n Client No {i} - global model')
        #model = LR_ScikitModel()
        gl_labels =  multiclass_LogisticFunction(X_test[i], np.array(unlearn_global_coef), np.array(unlearn_global_intercept))
        f1 = round(np.max(f1_score(y_test[i], gl_labels, average='binary'))*100,2)

        # folding
        #spit the test data into 2 set
        X_combined = pd.concat([X_test[i], y_test[i]], axis=1)
        X_test_b = np.array_split(X_combined.sample(frac=1), 10)

        for k, test in enumerate(X_test_b):
            y_pred_1= multiclass_LogisticFunction(test.drop(columns='Churn_risk'), np.array(unlearn_global_coef), np.array(unlearn_global_intercept))
            f1 = round(np.mean(f1_score(test['Churn_risk'], y_pred_1, average='binary'))*100, 2)
            print(f'f1 for {k}th-fold test data', f1)

        f1_unlearn_final_model.append(f1)
        #print(classification_report(y_test[i], gl_labels, zero_division=0))
        #print('Confusion matrix \n', confusion_matrix(y_test[i], gl_labels))    
    print('TESTF1', np.shape(f1_l), np.shape(f1_l_final_global), np.shape(f1_unlearn_final_model))
    f1_l = f1_l[0:51]
    f1_l_final_global = f1_l_final_global[0:51]

    plot_f1(np.array(f1_l), np.array(f1_l_final_global), np.array(f1_unlearn_final_model), \
             plot_l = ['local','global'], figname='f1_score_local_vs_unlearn_model:_'+str(unlearn)+'_first_clients')
               
    #plot_improve(np.array(f1_l), np.array(f1_l_final_global), np.array(f1_unlearn_final_model), \
    #         plot_l = ['local','global'], figname='f1_score_local_vs_unlearn_model:_'+str(unlearn)+'_first_clients')
    #print('mean', np.mean(np.abs(np.array(f1_unlearn_final_model) - np.array(f1_l_final_global))))
    #print('max', np.max(np.abs(np.array(f1_unlearn_final_model) - np.array(f1_l_final_global))))
    #f1_local, f1_preunlearn_final, f1_global=None, plot_l = ['local','preunlearn','global'], figname='default'
    improve = np.array(f1_unlearn_final_model) - np.array(f1_l)
    print(improve < 0)
    nr_worse_clients = np.count_nonzero(improve <= 0)
    nr_better_clients = np.count_nonzero(improve > 0)
    print('improvement', nr_worse_clients, nr_better_clients)
    f1_improvement = np.array(f1_unlearn_final_model) - np.array(f1_l)
    print('f1 improvement', (f1_improvement))
    print((np.array(f1_unlearn_final_model),np.array(f1_l_final_global)))

    nr_worse_clientsb = np.count_nonzero(improve <= 0.25)
    nr_better_clientsb = np.count_nonzero(improve > 0.25)

    return nr_worse_clients, nr_better_clients, nr_worse_clientsb, nr_better_clientsb, f1_improvement, f1_unlearn_final_model

def main():
    print("---EXECUTING THE MAIN PIPELINE---")
    print("---Read & Transform Data---")
    data = read_and_transform(binary_or_multiclass='binary')

    trainbig=False
    if trainbig:
        print('---Training a big model for all---') 
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

    print('---Simulating clients based on geographical locations of the banks---') 
    test_client = simultedClients(data=data, split_feature= 'geo', n_clients=60)
    X_train, X_test, y_train, y_test = test_client.createBalancedClients(algo='downsampling', balance_test_data=False)
    print('TRAINTEST', np.shape(X_train), np.shape(X_test))
    
    
    retrain=False
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
        
        with open('output/f1_parameters.npy', 'wb') as f:
            np.save(f, f1_l)
        with open('output/fitting_coef_err.npy', 'wb') as f:
            np.save(f, error_l)
        with open('output/fitting_coef.npy', 'wb') as f:
            np.save(f, coef_l)
        with open('output/fitting_intercept.npy', 'wb') as f:
            np.save(f, intercept_l)

    else:
        with open('output/f1_parameters.npy', 'rb') as f:
            f1_l = np.load(f)
        with open('output/fitting_coef_err.npy', 'rb') as f:
            error = np.load(f)
        with open('output/fitting_coef.npy', 'rb') as f:
            coef_l = np.load(f)
        with open('output/fitting_intercept.npy', 'rb') as f:
            intercept_l = np.load(f)


    print('---Aggregating at the aggregation server---')
    #averaged the local weights & biases
    global_intercept = np.mean(intercept_l,axis=0)
    global_coef = np.mean(coef_l,axis=0)
    print('GLOBAL intercept and coefficient: \n', global_intercept, global_coef)

 
    print('---Testing global model on local testing data---')
    f1_l_global = []
    for i in range(np.shape(X_test)[0]):
        print(f'\n Client No {i} - global model')
        #model = LR_ScikitModel()
        global_labels =  multiclass_LogisticFunction(X_test[i], np.array(global_coef), np.array(global_intercept))
        #f1 = round(np.max(f1_score(y_test[i], global_labels, average=None))*100,2)
        f1 = round(np.mean(f1_score(y_test[i], global_labels, average='weighted'))*100,2)
        f1_l_global.append(f1)
        
    f1_l_final_global = []
    # Replace local model with global model if global model is better
    for i, _f1  in enumerate(f1_l_global):
        if f1_l[i] < f1_l_global[i]:
            f1_l_final_global.append(f1_l_global[i])  
        else:
            f1_l_final_global.append(f1_l[i]) 

    #comparision between global model and local model
    figname = 'comparision_global_local'
    print('comparep', f1_l_final_global - f1_l)
    n_bins = 10
    fig, ((ax0)) = plt.subplots(nrows=1, ncols=1, figsize=(20,15))
    colors = ['red', 'tan', 'lime']
    ax0.hist(f1_l_final_global - f1_l, n_bins, density=False, range=(0,10), histtype='bar')#, color=colors, label=colors)
    ax0.legend(prop={'size': 10})
    ax0.set_title('Intercept', fontsize=18)
    fig.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if figname == 'default':
        plt.savefig('plots/comparision_global_local.png')
    else:
        plt.savefig('plots/compa_'+figname)    



    unlearn=False 
    if unlearn:
        print('---Unlearn the entire client---')
        #averaged the local weights
        print(intercept_l)
        print(np.size(intercept_l))
        print(np.size(intercept_l[0:10]))
        print(np.size(np.delete(intercept_l, [1,2,3]) ))
        print(intercept_l, np.delete(intercept_l, [1,2,3]) )


        worse_clients = [] 
        better_clients = []
        worse_clientsb = [] 
        better_clientsb = []
        f1_improvement_l = []
        f1_positive_improvement_l = []
        count_f1_positive_improvement_l = []
        sum_f1_positive_improvement_l = []
        
        
        for i in range(0, np.size(intercept_l)):
            print('Shape of preunlearn local intercepts', np.shape(intercept_l))
            print('Shape of preunlearn  local coeffs', np.shape(coef_l))
            print('Shape of preunlearn  global coeffs', (np.mean(coef_l,axis=0)))

            unlearn_clients = np.arange(0, i )
            intercept_l_u = np.delete(intercept_l, unlearn_clients, axis=0)
            coef_l_u = np.delete(coef_l, unlearn_clients, axis=0)
            print('Shape of unlearn local intercepts', np.shape(intercept_l_u))
            print('Shape of unlearn local coeffs', np.shape(coef_l_u))
            print('Shape of global coeffs', (np.mean(coef_l_u,axis=0)))

            unlearn_global_intercept = np.mean(intercept_l_u,axis=0)
            print('UNLEARN CLIENTS', unlearn_clients)
            print('PREUNLEARN GLOBAL INTERCEPT', np.mean(intercept_l,axis=0) )
            unlearn_global_coef = np.mean(coef_l_u,axis=0)
            print('UNLEARN GLOBAL INTERCEPT', unlearn_global_intercept)
            print('PREUNLEARN GLOBAL COEFF', np.mean(coef_l,axis=0))
            print('UNLEARN GLOBAL COEFF', unlearn_global_coef)


            #print('COEFF ', global_coef, unlearn_global_coef, global_intercept, unlearn_global_intercept)

            nr_worse_clients, nr_better_clients, nr_worse_clientsb, nr_better_clientsb, f1_improvement = test_global_model(X_test, y_test, unlearn_global_coef, unlearn_global_intercept, f1_l, f1_l_final_global, i)
            worse_clients.append(nr_worse_clients) 
            better_clients.append(nr_better_clients)
            worse_clientsb.append(nr_worse_clientsb) 
            better_clientsb.append(nr_better_clientsb)
            a = np.array(f1_improvement)#[(f1_improvement) >0]
            f1_positive_improvement_l.append(a)
            count_f1_positive_improvement_l.append(np.size(a))
            sum_f1_positive_improvement_l.append(np.sum(a))
            f1_improvement_l.append(f1_improvement)

            #comparision between global model and local model
        figname = 'comparision_f1_unlearned_vs_local'
        print('comparep', f1_l_final_global - f1_l)
        print('f1_improvement', np.shape(f1_improvement_l))
        print('mean f1_improvement', np.mean(f1_improvement_l, axis=0))
        print('std f1_improvement', np.std(f1_improvement_l, axis=0))
        print('max f1_improvement', np.max(f1_improvement_l, axis=0))
        print('min f1_improvement', np.min(f1_improvement_l, axis=0))
        print('sum f1_improvement', sum_f1_positive_improvement_l)
        print('count f1_improvement', count_f1_positive_improvement_l)
        print('average f1_improvement', np.array(sum_f1_positive_improvement_l)/np.array(count_f1_positive_improvement_l))


        print('f1_improvement averaged over all different rounds of unlearn', np.shape(np.mean(f1_improvement_l, axis=0)))
        print(np.shape(f1_improvement_l[0]))
        print(f1_improvement_l[55])
        print(f1_improvement_l[:][55])
        for i in range(0, 51):
            n_bins = 10
            print(i)
            fig, ((ax0)) = plt.subplots(nrows=1, ncols=1, figsize=(20,15))
            colors = ['red', 'tan', 'lime']
            ax0.hist(np.array(f1_improvement_l)[:,i], n_bins, density=False, range=(np.min(np.array(f1_improvement_l)[:,i]),np.max(np.array(f1_improvement_l)[:,i])), histtype='bar')#, color=colors, label=colors)
            ax0.legend(prop={'size': 10})
            ax0.set_title('Intercept', fontsize=18)
            ax0.set_xlabel('f1-score improvement (%)',fontsize=16)
            ax0.set_ylabel('Number of improved clients',fontsize=16)
            fig.tight_layout()
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.savefig(f'plots/comparision_f1_unlearn_local_{i}.png')
            
        for i in range(0, 51):
            n_bins = 10
            print(i)
            fig, ((ax0)) = plt.subplots(nrows=1, ncols=1, figsize=(20,15))
            colors = ['red', 'tan', 'lime']
            ax0.hist(f1_positive_improvement_l[i], n_bins, density=False, histtype='bar')#, color=colors, label=colors)
            ax0.legend(prop={'size': 10})
            ax0.set_title('Intercept', fontsize=18)
            ax0.set_xlabel('f1-score improvement (%)',fontsize=16)
            ax0.set_ylabel('Number of improved clients',fontsize=16)
            fig.tight_layout()
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.savefig(f'plots/comparision_f1_unlearn_local_{i}.png')

        labels = np.arange(np.size(better_clients) )
        print(labels)
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=[15,10])
        rects1 = ax.bar(x - width/2, worse_clients, width, label='Worse')
        rects2 = ax.bar(x + width/2, better_clients, width, label='Better')

        trend_line = plt.plot(x - width/2, worse_clients,marker='o', color='#5b74a8', label='Worse')
        trend_line = plt.plot(x + width/2, better_clients,marker='o', color='black', label='Better')



        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Number of clients')
        ax.set_ylim([10, 50])
        ax.set_title('Comparision between unlearn models and local models ')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()


        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')


        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        plt.savefig('plots/improvement.png')




        labels = np.arange(np.size(better_clientsb) )
        print(labels)
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=[15,10])
        rects1 = ax.bar(x - width/2, worse_clientsb, width, label='Worse')
        rects2 = ax.bar(x + width/2, better_clientsb, width, label='Better')

        trend_line = plt.plot(x - width/2, worse_clientsb,marker='o', color='#5b74a8', label='Worse')
        trend_line = plt.plot(x + width/2, better_clientsb,marker='o', color='black', label='Better')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Number of clients')
        ax.set_ylim([10, 50])
        ax.set_title('Comparision between unlearn models and local models ')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()


        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')


        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        plt.savefig('plots/improvementb.png')


        labels = np.arange(np.size(better_clientsb) )
        print(labels)
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=[15,10])
        rects1 = ax.bar(x - width/2, worse_clientsb, width, label='Worse')
        rects2 = ax.bar(x + width/2, better_clientsb, width, label='Better')

        trend_line = plt.plot(x - width/2, worse_clientsb,marker='o', color='#5b74a8', label='Worse')
        trend_line = plt.plot(x + width/2, better_clientsb,marker='o', color='black', label='Better')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Number of clients')
        ax.set_title('Comparision between unlearn models and local models ')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()


        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')


        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        plt.savefig('plots/improvementb.png')
            
        intercept_s = str(intercept_l)
        coef_s = str(coef_l)
        error_s = str(error_l)

        f=open('model_parameters.txt','w')
        f.write('intercept'+'\n')
        f.write(intercept_s)
        f.write('\n'+'coef'+'\n')
        f.write(coef_s)
        f.write('\n'+'error'+'\n')
        f.write(error_s)
        f.close()


        print(worse_clients)
        print(better_clients)
        print('f1 improvment ', f1_improvement_l)
        print('intercept ', intercept_l)
        print('coefficient ', coef_l)
        print('error ', error_l)
        
        a = np.array(sum_f1_positive_improvement_l)/np.array(count_f1_positive_improvement_l)
        print(a)

        n_bins = 10

        fig, ((ax0)) = plt.subplots(nrows=1, ncols=1, figsize=(15,10))

        colors = ['red', 'tan', 'lime']
        ax0.hist(a, n_bins, density=False, histtype='bar')#, color=colors, label=colors)
        ax0.legend(prop={'size': 10})
        ax0.set_title('Average f1-score improvement', fontsize=18)
        ax0.set_xlabel('f1-score improvement (%)',fontsize=16)
        ax0.set_ylabel('Number of improved clients',fontsize=16)
        fig.tight_layout()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
                
        fig.tight_layout()
        plt.show()


        print(np.shape(a))

        plot_coef_dist(f1_l, error, coef_l, intercept_l, figname='Coef_dist_local_model')



    #########UNLEARN ONLY BIG GUY##########
    print('---Unlearn the entire client---')
    #averaged the local weights

    worse_clients = [] 
    better_clients = []
    worse_clientsb = [] 
    better_clientsb = []
    f1_improvement_l = []
    f1_positive_improvement_l = []
    f1_unlearn_final_model_l = []
    count_f1_positive_improvement_l = []
    sum_f1_positive_improvement_l = []
    
    unlearn_clients = [10, 6, 23, 55 ] #12, 8, 25, 57
    intercept_l_u = np.delete(intercept_l, unlearn_clients, axis=0)
    coef_l_u = np.delete(coef_l, unlearn_clients, axis=0)
    print('BShape of unlearn local intercepts', np.shape(intercept_l_u))
    print('BShape of unlearn local coeffs', np.shape(coef_l_u))
    print('BShape of global coeffs', (np.mean(coef_l_u,axis=0)))

    unlearn_global_intercept = np.mean(intercept_l_u,axis=0)
    print('UNLEARN CLIENTS', unlearn_clients)
    print('PREUNLEARN GLOBAL INTERCEPT', np.mean(intercept_l,axis=0) )
    unlearn_global_coef = np.mean(coef_l_u,axis=0)
    print('UNLEARN GLOBAL INTERCEPT', unlearn_global_intercept)
    print('PREUNLEARN GLOBAL COEFF', np.mean(coef_l,axis=0))
    print('UNLEARN GLOBAL COEFF', unlearn_global_coef)


    #print('COEFF ', global_coef, unlearn_global_coef, global_intercept, unlearn_global_intercept)

    for i in range(0, 51):
        nr_worse_clients, nr_better_clients, nr_worse_clientsb, nr_better_clientsb, f1_improvement, f1_unlearn_final_model = test_global_model(X_test, y_test, unlearn_global_coef, unlearn_global_intercept, f1_l, f1_l_final_global, i)
        worse_clients.append(nr_worse_clients) 
        better_clients.append(nr_better_clients)
        worse_clientsb.append(nr_worse_clientsb) 
        better_clientsb.append(nr_better_clientsb)
        a = np.array(f1_improvement)#[(f1_improvement) >0]
        f1_positive_improvement_l.append(a)
        count_f1_positive_improvement_l.append(np.size(a))
        sum_f1_positive_improvement_l.append(np.sum(a))
        f1_improvement_l.append(f1_improvement)
        f1_unlearn_final_model_l.append(f1_unlearn_final_model)

    with open('output/f1_parameters_unlearn_big.npy', 'wb') as f:
        np.save( f, f1_unlearn_final_model_l)
    with open('output/f1_parameters_preunlearn_global.npy', 'wb') as f:
        np.save( f, f1_l_final_global)
    with open('output/f1_parameters_local.npy', 'wb') as f:
        np.save( f, f1_l)



    #########UNLEARN ONLY SMALL GUYS ##########
    print('---Unlearn the entire client---')
    #averaged the local weights

    worse_clients = [] 
    better_clients = []
    worse_clientsb = [] 
    better_clientsb = []
    f1_improvement_l = []
    f1_positive_improvement_l = []
    f1_unlearn_final_model_l = []
    count_f1_positive_improvement_l = []
    sum_f1_positive_improvement_l = []
    
    unlearn_clients = [44, 49, 32, 47, 37, 26, 18, 7, 54, 21, 43, 35, 46, 42, 17 ] #12, 8, 25, 57
    intercept_l_u = np.delete(intercept_l, unlearn_clients, axis=0)
    coef_l_u = np.delete(coef_l, unlearn_clients, axis=0)
    print('BShape of unlearn local intercepts', np.shape(intercept_l_u))
    print('BShape of unlearn local coeffs', np.shape(coef_l_u))
    print('BShape of global coeffs', (np.mean(coef_l_u,axis=0)))

    unlearn_global_intercept = np.mean(intercept_l_u,axis=0)
    print('UNLEARN CLIENTS', unlearn_clients)
    print('PREUNLEARN GLOBAL INTERCEPT', np.mean(intercept_l,axis=0) )
    unlearn_global_coef = np.mean(coef_l_u,axis=0)
    print('UNLEARN GLOBAL INTERCEPT', unlearn_global_intercept)
    print('PREUNLEARN GLOBAL COEFF', np.mean(coef_l,axis=0))
    print('UNLEARN GLOBAL COEFF', unlearn_global_coef)


    #print('COEFF ', global_coef, unlearn_global_coef, global_intercept, unlearn_global_intercept)

    for i in range(0, 51):
        nr_worse_clients, nr_better_clients, nr_worse_clientsb, nr_better_clientsb, f1_improvement, f1_unlearn_final_model = test_global_model(X_test, y_test, unlearn_global_coef, unlearn_global_intercept, f1_l, f1_l_final_global, i)
        worse_clients.append(nr_worse_clients) 
        better_clients.append(nr_better_clients)
        worse_clientsb.append(nr_worse_clientsb) 
        better_clientsb.append(nr_better_clientsb)
        a = np.array(f1_improvement)#[(f1_improvement) >0]
        f1_positive_improvement_l.append(a)
        count_f1_positive_improvement_l.append(np.size(a))
        sum_f1_positive_improvement_l.append(np.sum(a))
        f1_improvement_l.append(f1_improvement)
        f1_unlearn_final_model_l.append(f1_unlearn_final_model)

    with open('output/f1_parameters_unlearn_small.npy', 'wb') as f:
        np.save( f, f1_unlearn_final_model_l)


if __name__ == "__main__":
    main()
