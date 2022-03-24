from lib import *
from read_transform_data import read_and_transform
from simulate_clients import simultedClients
import seaborn as sns


binary_or_multiclass = 'multiclass'
n_clients = 6
data = read_and_transform(binary_or_multiclass=binary_or_multiclass)
test_client = simultedClients(data=data, n_clients=30, split_feature= 'Age')
print(np.shape(test_client))
X_train_l, X_test_l, y_train_l, y_test_l = test_client.createBalancedClients(algo='downsampling')


plots_path = './plots/'
#sns.set_palette("hls")
sns.set(rc={'figure.figsize':(16,9)})

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
            if feature != 'Churn_risk':
                print(feature)
                x_min=0
                if feature == 'Trnx_count':
                    x_max = 600
                    y_max = data[feature].count()/2
                elif feature == 'num_products':
                    x_max = 12
                elif feature == 'Tenure':
                    x_max = 50
                elif feature == 'Total_score':
                    x_max = 60
                else:
                    x_max = data[feature].values.max()
                    y_max= data[feature].count()

                plot = sns.histplot(data, 
                            hue = 'Churn_risk', 
                            x = feature, 
                            multiple = 'stack',
                            binwidth = 1,
                            bins=5,
                            stat = 'count',
                            ax=ax[i][j])
                plot.set(xlim=(x_min,x_max))
                idx_feature += 1 
    fig.suptitle(f'Feature Distribution of client {int(client)} out of {int(n_clients)}')# {binary_or_multiclass}')
    plt.savefig(plots_path+'client_'+str(int(client))+'_balanced_Age'+str(int(n_clients))+str(binary_or_multiclass))






