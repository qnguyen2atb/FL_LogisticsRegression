from lib import *

def read_and_transform(binary_or_multiclass='multiclass'):
    '''
    Read and Transform data'''


    print("---Reading input file to pandas Dataframe---")
    # dataset path
    path = 'data'
    file_name = 'churnsimulateddata.csv'
    file = os.path.join(path, file_name)
    print(file)
    # read data
    df = pd.read_csv(file)
    print(f'Shape of original data: {df.shape}')

    print("---Select features---")
    feature_names = ['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products', 'Churn_risk']

    selected_df = df[feature_names].dropna()

    # binary
    if binary_or_multiclass =='binary':
        selected_df['Churn_risk'][selected_df['Churn_risk'] == 'Medium'] = 'High'  

    selected_df['Churn_risk'] = selected_df.Churn_risk.astype("category").cat.codes

    # Data normalization
    X_data = selected_df.drop(columns=['PSYTE_Segment','Churn_risk'])
    X_target = selected_df[['PSYTE_Segment','Churn_risk']]
    print(X_target)

    X_data_b = pd.DataFrame(normalize(X_data, norm='l2', axis=1, copy=True, return_norm=False),columns=X_data.columns)
    print(X_data_b)
    X = pd.concat([X_data_b, X_target], axis=1)
    X = X.dropna()

    return X