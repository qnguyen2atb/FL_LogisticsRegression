from lib import *

#from dataExploration import Data_Exploration
from read_transform_data import read_and_transform
from simulate_clients import simultedClients, data_balance
from modelling import LR_ScikitModel, multiclass_LogisticFunction
from plotting import plot_f1

def plot_hist(prob, figname='default'):
    #worse_clients = [26, 25, 22, 31, 25, 26, 31, 20, 26, 27, 21, 26, 19, 24, 25, 17, 27, 26, 30, 25, 31, 22, 29, 32, 26, 24, 26, 31, 27, 25, 25, 25, 28, 21, 17, 26, 32, 22, 25, 28, 24, 23, 27, 23, 23, 22, 23, 23, 21, 21, 22, 19, 23, 22, 18, 28]
    #better_clients = [30, 31, 34, 25, 31, 30, 25, 36, 30, 29, 35, 30, 37, 32, 31, 39, 29, 30, 26, 31, 25, 34, 27, 24, 30, 32, 30, 25, 29, 31, 31, 31, 28, 35, 39, 30, 24, 34, 31, 28, 32, 33, 29, 33, 33, 34, 33, 33, 35, 35, 34, 37, 33, 34, 38, 28]


    #labels = 3*np.arange(np.size(better_clients) )
    #print(labels)
    #x = np.arange(len(labels))  # the label locations
    #width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=[15,10])
    x = prob
    plt.hist(x, density=True, bins=20)  # density=False would make counts
    #plt.ylabel('Probability')
    plt.xlabel('Predicted Probability')
    #rects1 = ax.bar(x - width/2, worse_clients, width, label='Worse')
    #rects2 = ax.bar(x + width/2, better_clients, width, label='Better')

    #trend_line = plt.plot(x - width/2, worse_clients,marker='o', color='#5b74a8', label='Worse')
    #trend_line = plt.plot(x + width/2, better_clients,marker='o', color='black', label='Better')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of clients')
    #ax.set_title('Comparision between unlearn models and local models ')
    #ax.set_xticks(x)
    #ax.set_xticklabels(labels)
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


    #autolabel(rects1)
    #autolabel(rects2)

    fig.tight_layout()
    if figname=='default':
        plt.savefig('plots/prob_hist.png')
    else:
        plt.savefig('plots/'+figname+'.png')

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

        # folding
        #spit the test data into 2 set
        X_combined = pd.concat([X_test[i], y_test[i]], axis=1)
        X_test_b = np.array_split(X_combined.sample(frac=1), 10)

        for k, test in enumerate(X_test_b):
            y_pred_1= multiclass_LogisticFunction(test.drop(columns='Churn_risk'), np.array(unlearn_global_coef), np.array(unlearn_global_intercept))
            f1 = round(np.max(f1_score(test['Churn_risk'], y_pred_1, average=None))*100, 2)
            print(f'f1 for {k}th-fold test data', f1)

        f1_unlearn_final_model.append(f1)
        #print(classification_report(y_test[i], gl_labels, zero_division=0))
        #print('Confusion matrix \n', confusion_matrix(y_test[i], gl_labels))    

    plot_f1(np.array(f1_l), np.array(f1_l_final_global), np.array(f1_unlearn_final_model), \
             plot_l = ['local','global'], figname='f1_score_local_vs_unlearn_model:_'+str(unlearn)+'_first_clients')
    #plot_improve(np.array(f1_l), np.array(f1_l_final_global), np.array(f1_unlearn_final_model), \
    #         plot_l = ['local','global'], figname='f1_score_local_vs_unlearn_model:_'+str(unlearn)+'_first_clients')
    print('mean', np.mean(np.abs(np.array(f1_unlearn_final_model) - np.array(f1_l_final_global))))
    print('max', np.max(np.abs(np.array(f1_unlearn_final_model) - np.array(f1_l_final_global))))
    #f1_local, f1_preunlearn_final, f1_global=None, plot_l = ['local','preunlearn','global'], figname='default'
    improve = np.array(f1_unlearn_final_model) - np.array(f1_l)
    print(improve < 0)
    nr_worse_clients = np.count_nonzero(improve <= 0)
    nr_better_clients = np.count_nonzero(improve > 0)
    print('improvement', nr_worse_clients, nr_better_clients)
    f1_improvement = np.array(f1_unlearn_final_model) - np.array(f1_l)
    print('f1 improvement', (f1_improvement))
    print((np.array(f1_unlearn_final_model),np.array(f1_l_final_global)))

    nr_worse_clientsb = np.count_nonzero(improve <= 0.1)
    nr_better_clientsb = np.count_nonzero(improve > 0.1)
    print('improvement', nr_worse_clients, nr_better_clients)
    f1_improvement = np.array(f1_unlearn_final_model) - np.array(f1_l)
    print('f1 improvement', (f1_improvement))

    print((np.array(f1_unlearn_final_model),np.array(f1_l_final_global)))



    return nr_worse_clients, nr_better_clients, nr_worse_clientsb, nr_better_clientsb, f1_improvement


print("---EXECUTING THE MAIN PIPELINE---")
print("---Reading input file to pandas Dataframe---")
data = read_and_transform(binary_or_multiclass='binary')

print('---Training a big model for all---')
trainbig=False
if trainbig:
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

retrain=True
if retrain==True:
    print('---Training local models at local clients---')
    intercept_l = []
    coef_l = []
    f1_l = []

    error_l = []
    for i in range(np.shape(X_train)[0]):
        print(f'client No {i} - local model')
        
        model = LR_ScikitModel()
        print('Training Labels count: \n', y_train[i].value_counts())
        print('Testing Labels count: \n', y_test[i].value_counts())
        intercept, coef, accuracy, f1, f1_ave, report =  model.fit(X_train[i], X_test[i], y_train[i], y_test[i])
        intercept_l.append(intercept)
        coef_l.append(coef)
        f1_l.append(f1)
        # predicted probability of class 1
        prob = model.predict_proba[:,1]
        print('predicted probability of class 1', prob[prob > 0.5])
        plot_hist(prob[prob > 0.5], figname=f'prob_dist_local_model_{i}')
        error = model.coefficient_error()
        error_l.append(error)
else:
    from numpy import array
    intercept_l = np.array([array([-2.95291883]), array([-2.89353374]), array([-3.55656052]), array([-3.38050763]), array([-3.21683201]), array([-3.54494914]), array([-2.14538487]), array([-3.33804037]), array([-1.06269912]), array([-3.21602606]), array([-2.5269128]), array([-3.4356709]), array([-3.4482794]), array([-2.41308494]), array([-3.30956452]), array([-2.93866427]), array([-3.29240062]), array([-1.88534872]), array([-2.10096403]), array([-1.91338313]), array([-2.30963623]), array([-2.96646752]), array([-3.49770571]), array([-2.89553615]), array([-3.11775328]), array([-3.42342434]), array([-3.06958515]), array([-0.00167468]), array([-3.04044733]), array([-3.30070063]), array([-2.99296621]), array([-2.70900735]), array([-2.81648284]), array([-0.00746212]), array([-3.15353145]), array([-3.26113029]), array([-0.33299378]), array([-3.09137911]), array([-2.15984769]), array([-2.45649579]), array([-3.156661]), array([-2.22107374]), array([-2.90168033]), array([-2.36432563]), array([-2.10685214]), array([-2.73649335]), array([-2.00169189]), array([-3.24813136]), array([0.]), array([-0.62973549]), array([-2.19272681]), array([-3.91298047]), array([-2.5053185]), array([-2.22922473]), array([-1.77510824]), array([-3.07327028])])
    coef_l = [array([[-1.41367347e-02,  3.80874191e-02,  1.47716712e-01,
         2.24292617e-04,  4.88092809e-01]]), array([[-1.62803092e-02,  2.82168212e-02,  1.50496185e-01,
         4.45310901e-05,  5.29501642e-01]]), array([[-1.04250206e-02,  3.14707349e-02,  1.82972105e-01,
         2.66437464e-04,  3.93183955e-01]]), array([[-0.01204086,  0.03784418,  0.18399849,  0.00033031,  0.31274546]]), array([[-1.88393184e-02,  4.69083218e-02,  1.78579740e-01,
        -2.22221938e-05,  4.36483424e-01]]), array([[-0.01550586,  0.04901334,  0.17556384,  0.00043578,  0.40298164]]), array([[-0.02688634,  0.04590823,  0.15132083,  0.00031172,  0.28725889]]), array([[-1.45408023e-02,  5.06318567e-02,  1.58099799e-01,
         3.00784283e-04,  5.17719484e-01]]), array([[-0.03973736,  0.02396298,  0.18576577,  0.00026139,  0.11182715]]), array([[-1.86187303e-02,  5.22029106e-02,  1.69218891e-01,
         3.01062313e-04,  3.55022492e-01]]), array([[-0.01612691,  0.0378383 ,  0.17246729,  0.00033231,  0.09581076]]), array([[-1.68239851e-02,  5.15137095e-02,  1.73253137e-01,
         3.44798000e-04,  4.58645185e-01]]), array([[-1.37669351e-02,  3.50041348e-02,  1.81470863e-01,
         2.76062753e-04,  3.93519997e-01]]), array([[-0.02373095,  0.0435327 ,  0.15121564,  0.00057187,  0.31176868]]), array([[-1.43304756e-02,  4.25752812e-02,  1.71218994e-01,
         2.83280980e-04,  4.84852763e-01]]), array([[-2.17115229e-02,  4.24208290e-02,  1.73822752e-01,
         3.19823675e-04,  4.62837336e-01]]), array([[-1.12492812e-02,  4.24992021e-02,  1.45888790e-01,
         1.79209378e-04,  5.27882982e-01]]), array([[-2.28030226e-02,  3.61108666e-02,  1.33702575e-01,
        -3.60405357e-06,  3.70035889e-01]]), array([[-3.23219972e-02,  5.51545777e-02,  1.67054843e-01,
         1.15556734e-04,  2.59483825e-01]]), array([[-4.31524688e-02,  8.41344946e-02,  1.63416763e-01,
         7.51418477e-05,  2.78320189e-01]]), array([[-2.56886613e-02,  4.64895652e-02,  1.51960071e-01,
         5.23495713e-05,  4.25427540e-01]]), array([[-1.99528206e-02,  4.51812899e-02,  1.60090402e-01,
         1.66669736e-04,  5.12817589e-01]]), array([[-5.45832319e-03,  4.28884451e-02,  1.76480435e-01,
        -6.52696608e-05,  2.32116193e-01]]), array([[-1.49189352e-02,  2.50263356e-02,  1.51451425e-01,
         2.09988723e-04,  4.91941026e-01]]), array([[-0.01502059,  0.04242774,  0.15701705,  0.00049235,  0.36810996]]), array([[-0.01233242,  0.0393154 ,  0.16682888,  0.0006454 ,  0.40980726]]), array([[-2.09751946e-02,  5.30045368e-02,  1.68755476e-01,
         3.54579206e-04,  4.46223401e-01]]), array([[-0.01626417,  0.01839327,  0.04935328, -0.00070757,  0.0025515 ]]), array([[-0.01885608,  0.05750488,  0.17879662,  0.00052984,  0.15339196]]), array([[-1.30614238e-02,  5.04070723e-02,  1.55391881e-01,
         3.40955254e-04,  3.64981989e-01]]), array([[-0.0197101 ,  0.04778308,  0.16142618,  0.00054439,  0.36444763]]), array([[-0.01781733,  0.04912245,  0.16163525,  0.00045714,  0.1808141 ]]), array([[-2.22363246e-02,  5.65186018e-02,  1.45875899e-01,
         1.97097560e-04,  3.52364712e-01]]), array([[-0.05285021,  0.04798841,  0.12308677, -0.00075326,  0.0114244 ]]), array([[-0.01616902,  0.04780529,  0.16282311,  0.00087745,  0.31363226]]), array([[-9.36402411e-03,  2.96751070e-02,  1.64201367e-01,
        -7.94248283e-05,  4.08255965e-01]]), array([[-0.04984827,  0.04895229,  0.11960255, -0.0004446 ,  0.28239835]]), array([[-1.35250669e-02,  4.78007702e-02,  1.54975344e-01,
         3.27702670e-04,  4.40055217e-01]]), array([[-0.02340094,  0.01773919,  0.13547141, -0.00099006,  0.65034889]]), array([[-0.01851409,  0.05182574,  0.13580391,  0.00040392,  0.27677855]]), array([[-0.01571097,  0.05926837,  0.15664096,  0.00079383,  0.23242466]]), array([[-2.56303440e-02,  5.61524112e-02,  1.29933887e-01,
        -3.94992057e-04,  4.62230332e-01]]), array([[-1.67823827e-02,  3.99217902e-02,  1.59521430e-01,
         2.97986958e-04,  4.44452594e-01]]), array([[-0.02915151,  0.05056271,  0.17033991,  0.0003071 ,  0.27992992]]), array([[-0.03059831,  0.07060066,  0.15042849, -0.00115839,  0.23783674]]), array([[-0.00980248,  0.05175754,  0.15366447,  0.00045604,  0.1851195 ]]), array([[-2.79436068e-02,  6.24146663e-02,  1.32308957e-01,
        -2.54064831e-04,  4.18677743e-01]]), array([[-0.00696753,  0.04921902,  0.16601031,  0.00060281,  0.19266306]]), array([[-0.00402843,  0.        ,  0.02388564,  0.00025645,  0.        ]]), array([[-0.08434883,  0.04992097,  0.09566007,  0.00305911,  1.1571078 ]]), array([[-2.55873897e-02,  4.89243635e-02,  1.26009539e-01,
         1.91584008e-04,  4.61405687e-01]]), array([[-1.04386135e-02,  4.54822605e-02,  1.88816897e-01,
         2.06887123e-04,  4.28588762e-01]]), array([[-2.45084100e-02,  5.22678797e-02,  1.41112191e-01,
         2.99182996e-04,  4.96171709e-01]]), array([[-2.22190572e-02,  5.44442198e-02,  1.37977085e-01,
         3.78445162e-05,  4.44679760e-01]]), array([[-2.70797221e-02,  3.76781463e-02,  1.28263628e-01,
        -1.63184509e-04,  4.43036953e-01]]), array([[-0.01492703,  0.03600221,  0.15149321,  0.00058832,  0.52205352]])]


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
    #f1 = round(np.max(f1_score(y_test[i], global_labels, average=None))*100,2)
    f1 = round(np.mean(f1_score(y_test[i], global_labels, average='weighted'))*100,2)
    
    f1_l_global.append(f1)
    
f1_l_final_global = []

for i, _f1  in enumerate(f1_l_global):
    if f1_l[i] < f1_l_global[i]:
        f1_l_final_global.append(f1_l_global[i])  
    else:
        f1_l_final_global.append(f1_l[i]) 
        

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

    f1_improvement_l.append(f1_improvement)

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



import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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


'''
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



