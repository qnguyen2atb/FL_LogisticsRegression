from lib import *


def read_and_transform(binary_or_multiclass='multiclass'):
    print("---Reading input file to pandas Dataframe---")
    # dataset path
    #read_transform_data.py
    path = 'data'
    file_name = 'churnsimulateddata.csv'
    file = os.path.join(path, file_name)
    print(file)
    # read data
    df = pd.read_csv(file)
    print(f'Shape of original data: {df.shape}')

    print("---Select features---")
    feature_names = ['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products', 'Churn_risk']
    #feature_names = ['Age','PSYTE_Segment','Total_score','Churn_risk']
    #feature_names = ['PSYTE_Segment','Total_score','Churn_risk']

    selected_df = df[feature_names].dropna()

    # binary
    if binary_or_multiclass =='binary':
        selected_df['Churn_risk'][selected_df['Churn_risk'] == 'Medium'] = 'High'  

    selected_df['Churn_risk'] = selected_df.Churn_risk.astype("category").cat.codes
    #selected_df['A'] = selected_df.Age * selected_df.Total_score
    #selected_df['B'] = selected_df.Churn_risk * selected_df.Total_score
    #print(selected_df[['Churn_risk_number', 'Churn_risk']])

    #print(selected_df['Churn_risk'][selected_df['Churn_risk_code'] == 0]) #HIGH
    #print(selected_df['Churn_risk'][selected_df['Churn_risk_code'] == 1]) #LOW
    #print(selected_df['Churn_risk'][selected_df['Churn_risk_code'] == 2]) #MED

    #selected_df['Churn_risk'][selected_df['Churn_risk'] == 'Medium'] = '1'
    #selected_df['Churn_risk'][selected_df['Churn_risk'] == 'Low'] = '0'
    #selected_df['Churn_risk'][selected_df['Churn_risk'] == 'High'] = '2'


    print(selected_df['Churn_risk'].value_counts())

    selected_df = selected_df.dropna()
    #selected_df = selected_df.drop(selected_df[(selected_df.Churn_risk != 0) or (selected_df.Churn_risk != 1) (selected_df.Churn_risk != 2)].index)
    #print(selected_df['Churn_risk'].unique())
    return selected_df