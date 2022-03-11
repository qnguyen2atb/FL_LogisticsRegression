from pydoc import cli
from re import X
from unicodedata import east_asian_width
from lib import *
from read_transform_data import read_and_transform
from simulate_clients import simultedClients
import seaborn as sns

binary_or_multiclass = 'multiclass'
data = read_and_transform(binary_or_multiclass='multiclass')
test_client = simultedClients(data=data, n_clients=3)
print(np.shape(test_client))
X_train_l, X_test_l, y_train_l, y_test_l = test_client.createBalancedClients(algo='downsampling')


plots_path = './plots/'

for client, X_train in enumerate(X_train_l):
    print(client)
    data = X_train_l[client].copy()
    data['Churn_risk'] = y_train_l[client]
    data = data.drop_duplicates()
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 14))
    
    idx_feature = 0  
    for i in range(ax.shape[0]):
        for j in range(0, ax.shape[1]):
            feature = data.columns[idx_feature]
            print(feature)
            x_min=0
            if feature == 'Trnx_count':
                x_max = 600
            elif feature == 'num_products':
                x_max = 12
            elif feature == 'Tenure':
                x_max = 50
            elif feature == 'Total_score':
                x_max = 60
            else:
                x_max = data[feature].values.max()
                
            plot = sns.histplot(data, 
                        hue = 'Churn_risk', 
                        x = feature, 
                        multiple = 'stack',
                        binwidth = 0.50,
                        bins=10,
                        stat = 'count',
                        ax=ax[i][j], color='green')
            plot.set(xlim=(x_min,x_max))
            idx_feature += 1 
    fig.suptitle(f'Feature Distribution of client {int(client)}')# {binary_or_multiclass}')
    #plt.show()
    plt.savefig(plots_path+'client_'+str(int(client))+'_balanced_'+str(binary_or_multiclass))






