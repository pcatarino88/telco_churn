import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler


def cleaning (df):
    """
    Function that:
    1) fills null values: i) nulls of 'Churn Reason', 'Churn Category' with 'No Churn'; ii) nulls of 'Offer' with 'No Offer'; iii) null of 'Internet Type' with 'No Internet'
    2) normalizes total dollar columns per 'Tenure in Months'
    3) creates bins for 'Number of Referrals' column
    4) drops unnecessary columns
    
    """

    # 1. fill null values
    df['Churn Reason'] = df['Churn Reason'].fillna('No Churn')
    df['Churn Category'] = df['Churn Category'].fillna('No Churn')
    df['Offer'] = df['Offer'].fillna('No Offer')
    df['Internet Type'] = df['Internet Type'].fillna('No Internet')

    # 2. Divide total dollar columns per 'Tenure in Months' and rename with 'per Month' suffix
    dollar_cols = ['Total Charges', 'Total Refunds', 'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']
    for col in dollar_cols:
        df[col + ' per Month'] = df[col] / df['Tenure in Months']
    
    # 3. create referrals_bins with 3 possible values: 0, 1, 2+
    df['Number of Referrals_bins'] = pd.cut(df['Number of Referrals'], bins=[-1, 0, 1, np.inf], labels=['0', '1', '2+'])

    # 4. drop unnecessary columns
    df = df.drop(
        columns=[
            'Under 30', 'Age', 'Number of Dependents', 'Country', 'State','Zip Code','Lat Long','Referred a Friend','Number of Referrals', 'Churn Label','Churn Score',
            'Quarter','ID','Total Charges', 'Total Refunds', 'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue','Avg Monthly Long Distance Charges','CLTV','Population'
            ]
        )
    
    return df


def create_geoclusters(df, random_state=42):
    """
    Function to create geoclusters based on latitude and longitude.
    Adds a new column 'GeoCluster' to the dataframe indicating the cluster each record belongs to.
    """
    # Extract latitude and longitude
    coords = df[['Latitude', 'Longitude']].dropna()

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=random_state)
    df.loc[coords.index, 'GeoCluster'] = kmeans.fit_predict(coords)

    # drop 'City', 'Latitude' and 'Longitude' columns
    df = df.drop(columns=['City', 'Latitude', 'Longitude'])

    return df


def feature_engineering(df):
    """
    Function that performs the followting transformations:
    1) Create a new feature 'Streaming Services' that indicates if a customer has any streaming services
    2) Transfors 'Internet Type' into an ordereed categorical variable from lower to higher speed: 0 Cable, 1 DSL, 2 Fiber Optic
    3) Creates 'Value Added Services' feature that indicates if a customer has any of the following services: 'Device Protection Plan', 'Premium Tech Support', 'Online Security', 'Online Backup'
    4) Creates ordered "Phone Service" categorical variable: 0 'No Phone Service', 1 'Yes' and 2 'Yes_Multiple Lines' (when 'Phone Service' is 'Yes' and 'Multiple Lines' is 'Yes')
    5) Drop leakage columns for modelling
    """
    # 1. Create a new feature 'Streaming Services' that indicates if a customer has any streaming services and drops the original columns
    df['Streaming Services'] = np.where(
        (df['Streaming Music'] == 'Yes') | (df['Streaming TV'] == 'Yes') | (df['Streaming Movies'] == 'Yes'),
        'Yes',
        'No'
    )
    df = df.drop(columns=['Streaming Music', 'Streaming TV', 'Streaming Movies'])

    # 2. Transfors 'Internet Type' into an ordereed categorical variable from lower to higher speed: 0 Cable, 1 DSL, 2 Fiber Optic. Drop 'Internet Service' column.
    internet_type_mapping = {
        'No Internet': -1,
        'Cable': 0,
        'DSL': 1,
        'Fiber Optic': 2
    }
    df['Internet Type'] = df['Internet Type'].map(internet_type_mapping)
    df = df.drop(columns=['Internet Service'])

    # 3. Creates 'Value Added Services' feature that indicates if a customer has any of the following services: 'Device Protection Plan', 'Premium Tech Support', 'Online Security', 'Online Backup'. drops original columns.
    df['Value Added Services'] = np.where(
        (df['Device Protection Plan'] == 'Yes') | (df['Premium Tech Support'] == 'Yes') | (df['Online Security'] == 'Yes') | (df['Online Backup'] == 'Yes'),
        'Yes',
        'No'
    )
    df = df.drop(columns=['Device Protection Plan', 'Premium Tech Support', 'Online Security', 'Online Backup'])

    # 4. Creates ordered "Phone Service" categorical variable: 0 'No Phone Service', 1 'Yes' and 2 'Yes_Multiple Lines' (when 'Phone Service' is 'Yes' and 'Multiple Lines' is 'Yes'). Drops 'Multiple Lines' column.
    df['Phone Service'] = np.where(
        df['Phone Service'] == 'No',
        0,
        np.where(
            (df['Phone Service'] == 'Yes') & (df['Multiple Lines'] == 'Yes'),
            2,
            1
        )
    )
    df = df.drop(columns=['Multiple Lines'])

    # 5. drop leakage columns for modelling
    df = df.drop(columns=['Offer','Satisfaction Score','Customer Status', 'Churn Reason', 'Churn Category'])
      
    return df


def encoding(df):
    """
    Function that encodes categorical variables according to fhe following rules:
    1. 'Yes'/'No' variables are encoded as 1/0
    2. 'Payment Method' variable is one-hot encoded
    3. 'Contract Type' and 'Number of Referrals_bins' variable are ordinally encoded: i) 'Month-to-Month' as 0, 'One Year' as 1, and 'Two Year' as 2; ii) '0' as 0, '1' as 1, and '2+' as 2
    4. 'Gender' is encoded as 1 for 'Male' and 0 for 'Female' and column name is changed to 'Male'
    
    """
    # 1. 'Yes'/'No' variables are encoded as 1/0
    yes_no_cols = df.select_dtypes(include=['object']).columns
    for col in yes_no_cols:
        if set(df[col].unique()) == {'Yes', 'No'}:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # 2. 'Payment Method' variable is one-hot encoded and drop first to avoid multicollinearity
    df = pd.get_dummies(df, columns=['Payment Method'], drop_first=True)

    dummy_cols = [col for col in df.columns if col.startswith('Payment Method_')]
    df[dummy_cols] = df[dummy_cols].astype(int)


    # 3. 'Contract' and 'Number of Referrals_bins' variable are ordinally encoded: i) 'Month-to-Month' as 0, 'One Year' as 1, and 'Two Year' as 2; ii) '0' as 0, '1' as 1, and '2+' as 2
    contract_type_mapping = {
        'Month-to-Month': 0,
        'One Year': 1,
        'Two Year': 2
    }
    df['Contract Duration'] = df['Contract'].map(contract_type_mapping)
    df = df.drop(columns=['Contract'])
    
    referrals_bins_mapping = {
        '0': 0,
        '1': 1,
        '2+': 2
    }
    df['Number of Referrals_bins'] = df['Number of Referrals_bins'].map(referrals_bins_mapping).astype(int)
    
    # 4. 'Gender' is encoded as 1 for 'Male' and 0 for 'Female' and column name is changed to 'Male'
    df['Male'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df = df.drop(columns=['Gender'])

    return df


def scaling(df):
    """
    Function that scales the df using RobustScaler. If a column is non-numerical an error message is raised.
    """
    
    # print error message if any column is non-numerical
    non_numerical_cols = df.select_dtypes(exclude=['int64', 'int32', 'float64']).columns
    if len(non_numerical_cols) > 0:
        raise ValueError(f"The following columns are non-numerical: {non_numerical_cols.tolist()}. Please encode them before scaling.")
    
    # apply RobustScaler
    scaler = RobustScaler()

    df[df.columns] = scaler.fit_transform(df[df.columns])
    
    return df


def select_important_variables(df):
    """
    Selects only important variables from the dataframe that were identified during feature selection and adds 'Churn Value' column.
    """
    important_variables = [
        'Contract Duration', 'Dependents', 'Internet Type', 'Monthly Charge', 'Number of Referrals_bins', 'Paperless Billing', 
        'Payment Method_Credit Card', 'Senior Citizen', 'Tenure in Months', 'Total Extra Data Charges per Month', 
        'Total Long Distance Charges per Month', 'Unlimited Data'] + ['Churn Value']

    return df[important_variables]