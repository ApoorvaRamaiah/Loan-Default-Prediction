# import pandas as pd
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# import joblib

# def preprocess_data(df, is_training=True):
#     """
#     Preprocess the data for training or inference.
    
#     Args:
#         df (pd.DataFrame): Input data as a Pandas DataFrame.
#         is_training (bool): Whether preprocessing is for training (True) or inference (False).
    
#     Returns:
#         pd.DataFrame: Preprocessed features ready for the model.
#     """
#     # Define numerical and categorical columns
#     numerical_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
#                       'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
#     categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 
#                         'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
    
#     # Exclude LoanID as it's not predictive
#     df = df.drop(columns=['LoanID'])
    
#     # Handle missing values (fill with mean for numerical and mode for categorical)
#     df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
#     df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
#     if is_training:
#         # Fit and transform encoders and scalers during training
#         encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#         # encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
#         scaler = StandardScaler()
        
#         categorical_data = encoder.fit_transform(df[categorical_cols])
#         numerical_data = scaler.fit_transform(df[numerical_cols])
        
#         # Save the fitted encoders and scalers for future use
#         joblib.dump(encoder, 'model/encoder.pkl')
#         joblib.dump(scaler, 'model/scaler.pkl')
    
#     else:
#         # Load previously fitted encoders and scalers during inference
#         encoder = joblib.load('model/encoder.pkl')
#         scaler = joblib.load('model/scaler.pkl')
        
#         categorical_data = encoder.transform(df[categorical_cols])
#         numerical_data = scaler.transform(df[numerical_cols])
    
#     # Combine preprocessed numerical and categorical features
#     categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out())
#     numerical_df = pd.DataFrame(numerical_data, columns=numerical_cols)
#     preprocessed_data = pd.concat([numerical_df, categorical_df], axis=1)
    
#     return preprocessed_data

from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import joblib

def preprocess_data(df, is_training=True):
    """
    Preprocess the data for training or inference.
    
    Args:
        df (pd.DataFrame): Input data as a Pandas DataFrame.
        is_training (bool): Whether preprocessing is for training (True) or inference (False).
    
    Returns:
        pd.DataFrame: Preprocessed features ready for the model.
    """
    numerical_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                      'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
    categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 
                        'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
    
    # Handle missing values
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    if is_training:
        # Updated: Use sparse_output instead of sparse
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        scaler = StandardScaler()
        
        categorical_data = encoder.fit_transform(df[categorical_cols])
        numerical_data = scaler.fit_transform(df[numerical_cols])
        
        joblib.dump(encoder, 'model/encoder.pkl')
        joblib.dump(scaler, 'model/scaler.pkl')
    else:
        encoder = joblib.load('model/encoder.pkl')
        scaler = joblib.load('model/scaler.pkl')
        
        categorical_data = encoder.transform(df[categorical_cols])
        numerical_data = scaler.transform(df[numerical_cols])
    
    categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out())
    numerical_df = pd.DataFrame(numerical_data, columns=numerical_cols)
    return pd.concat([numerical_df, categorical_df], axis=1)
