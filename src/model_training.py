# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import joblib

# def train_model(df):
#     """Train a model and save it."""
#     X = df.drop(columns=['Loan_Status'])
#     y = df['Loan_Status']
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train a model
#     model = LogisticRegression()
#     model.fit(X_train, y_train)
    
#     # Save the model
#     joblib.dump(model, 'model/loan_default_model.pkl')
    
#     # Evaluate
#     y_pred = model.predict(X_test)
#     print(classification_report(y_test, y_pred))
import os
print("Current Working Directory:", os.getcwd())
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import preprocess_data
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_model():
    # Load the data
    df = pd.read_csv('data/loan_data.csv')
    print(df.columns)
    # Preprocess the data
    TARGET_COLUMN = 'Default'
    X = preprocess_data(df.drop(columns=[TARGET_COLUMN]), is_training=True)
    y = df[TARGET_COLUMN]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, 'model/loan_default_model.pkl')
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model()
