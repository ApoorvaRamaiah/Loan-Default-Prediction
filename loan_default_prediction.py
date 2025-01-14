import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load Data
data_path = "data/loan_data.csv"  # Update path to your dataset
df = pd.read_csv(data_path)

# Step 2: Exploratory Data Analysis (EDA)
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values (Example: fill numerical with mean)
df.fillna(df.mean(), inplace=True)

# Visualize relationships
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Step 3: Data Preprocessing
# Encode categorical variables
categorical_cols = ['Gender', 'Married', 'Education']
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[categorical_cols]).toarray()

# Normalize numerical features
numerical_cols = ['ApplicantIncome', 'LoanAmount']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numerical_cols])

# Combine processed features
processed_data = np.hstack((scaled_data, encoded_data))
target = df['Loan_Status']  # Replace 'Loan_Status' with your target column

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    processed_data, target, test_size=0.2, random_state=42
)

# Step 4: Train a Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
