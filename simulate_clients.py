from pydoc import cli
from unicodedata import east_asian_width
from lib import *
from read_transform_data import read_and_transform


def data_balance(_X_train, _y_train, algo='downsampling'):
    print(np.size(_y_train.unique()))
    if np.size(_y_train.unique()) == 3:
        if algo=='downsampling':
            # downsample majority
            # concatenate our training data back together
            X = pd.concat([_X_train, _y_train], axis=1)
            low = X[X.Churn_risk==1]
            high = X[X.Churn_risk==0]
            medium = X[X.Churn_risk==2]
            
            X.Churn_risk.value_counts()

            low_downsampled = resample(low,
                                        replace = True, # sample without replacement
                                        n_samples = len(high), # match minority n
                                        random_state = 27) # reproducible results
            medium_downsampled = resample(medium,
                                        replace = True, # sample without replacement
                                        n_samples = len(high), # match minority n
                                        random_state = 27) # reproducible results

            
            # combine minority and downsampled majority
            downsampled = pd.concat([low_downsampled, high, medium_downsampled])

            # checking counts
            print('Labels counts after balancing.')
            print(downsampled.Churn_risk.value_counts())

            _X_train = downsampled.drop('Churn_risk', axis=1)
            _y_train = downsampled.Churn_risk
        elif algo == 'SMOTE':
            #oversamppling
            from imblearn.over_sampling import SMOTE, ADASYN
            _X_train, _y_train = SMOTE().fit_resample(_X_train, _y_train)
        elif algo == 'ADASYN':
            from imblearn.over_sampling import SMOTE, ADASYN
            _X_train, _y_train = ADASYN().fit_resample(_X_train, _y_train)
        else:
            ValueError('No sampling algorithm specified.')
    else:
        X = pd.concat([_X_train, _y_train], axis=1)
        low = X[X.Churn_risk==1]
        high = X[X.Churn_risk==0]


        X.Churn_risk.value_counts()
        
        try:
            low_downsampled = resample(low,
                                    replace = True, # sample without replacement
                                    n_samples = len(high), # match minority n
                                    random_state = 27) # reproducible results
        except ValueError:
            low_downsampled = low.copy()

        # combine minority and downsampled majority
        downsampled = pd.concat([low_downsampled, high])

        # checking counts
        print('Labels counts after balancing.')
        print(downsampled.Churn_risk.value_counts())

        _X_train = downsampled.drop('Churn_risk', axis=1)
        _y_train = downsampled.Churn_risk
        
    return _X_train, _y_train



class simultedClients():
    def __init__(self, data, n_clients, split_feature= 'geo', split_type='geo') -> None:
        '''
        split_type: geo - split based on geoloction.
                     uniform - split randomly
        '''
        self.data = data
        self.n_clients = n_clients
        self.split_feature = split_feature
        self.split_type = split_type
        print(f'---Simulate {self.n_clients} clients.')

    def createClients(self):
        if (self.split_type == 'geo') & (self.split_feature == 'geo'):
            print(f'Perform geo spliting to {self.n_clients} clients.')
            
            _data_v2 = self.data.sample(frac=1)
            self.clients_data = []
            
            _step_size = int(60/self.n_clients)
            for i in range(0, 60, _step_size):
                print(f'Client with PSYTE from {i} to {i+_step_size}')
                y = _data_v2[(_data_v2.PSYTE_Segment >= i) & (_data_v2.PSYTE_Segment < i+_step_size)]
                self.clients_data.append(y)     
        elif (self.split_type == 'uniform') & (self.split_feature == 'geo'):
            self.clients_data = np.array_split(self.data.sample(frac=1), self.n_clients)
        elif self.split_feature == 'Age':
            print(f'Perform geo spliting to {self.n_clients} clients based on Age.')
            
            _data_v2 = self.data.sample(frac=1)
            self.clients_data = []
            
            _step_size = int(_data_v2.Age.max()/self.n_clients)
            for i in range(0, _data_v2.Age.max(), _step_size):
                print(f'Client with AGE from {i} to {i+_step_size}')
                y = _data_v2[(_data_v2.Age >= i) & (_data_v2.Age < i+_step_size)]
                self.clients_data.append(y)     
        else:
            raise NameError(' split type is not correctly defined')   
        print(f'Number of {np.size(self.clients_data)} clients. ')
        return self.clients_data

    def createBalancedClients(self, algo='downsampling'):
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        self.clients_data = self.createClients()
        print(f'Shape of clients data {np.shape(self.clients_data)}. ')
        #print(self.clients_data[0])
        for i, client_data in enumerate(self.clients_data):
            print(f'client {i} with shape {np.shape(client_data)}')
            if np.shape(client_data)[0] > 100:
                if self.split_feature == 'geo':
                    X = client_data.drop(columns=['Churn_risk','PSYTE_Segment'])
                elif self.split_feature == 'Age':
                    X = client_data.drop(columns=['Churn_risk','Age'])
                y = client_data['Churn_risk']
                _X_train, _X_test, _y_train, _y_test = train_test_split(
                X, y, test_size=0.2, random_state=42) 

                # balance data
                _X_train, _y_train = data_balance(_X_train, _y_train, algo='downsampling')
                _X_test, _y_test = data_balance(_X_test, _y_test, algo='downsampling')
                
                X_train.append(_X_train)
                X_test.append(_X_test)
                y_train.append(_y_train)
                y_test.append(_y_test)
        return  X_train, X_test, y_train, y_test


#data = read_and_transform()
#test_client = simultedClients(data=data, n_clients=30)
#X_train_l, X_test_l, y_train_l, y_test_l = test_client.createBalancedClients(algo='downsampling')
