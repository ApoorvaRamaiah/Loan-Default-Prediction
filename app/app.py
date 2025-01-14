# from flask import Flask, request, jsonify
# import joblib

# app = Flask(__name__)

# # Load model
# model = joblib.load('model/loan_default_model.pkl')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     prediction = model.predict([data['features']])
#     return jsonify({'prediction': prediction[0]})

# if __name__ == '__main__':
#     app.run(debug=True)

import os
print("Current Working Directory:", os.getcwd())
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
import pandas as pd
import joblib
from src.data_preprocessing import preprocess_data

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/loan_default_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.json
        df = pd.DataFrame([data])  # Convert to DataFrame
        
        # Preprocess the data
        features = preprocess_data(df, is_training=False)
        
        # Predict the output
        prediction = model.predict(features)
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, jsonify
# import pandas as pd
# import joblib
# from src.data_preprocessing import preprocess_data  # Ensure this is the correct path

# app = Flask(__name__)

# # Load the trained model
# model = joblib.load('model/loan_default_model.pkl')

# # Root endpoint
# @app.route('/')
# def home():
#     return "Welcome to the Loan Default Prediction API. Use the `/predict` endpoint to make predictions."

# # Prediction endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Parse input JSON data
#         data = request.json
#         if not data:
#             return jsonify({'error': 'No input data provided'}), 400
        
#         # Convert input JSON to DataFrame
#         df = pd.DataFrame([data])  # Wrap in a list to create a single-row DataFrame
        
#         # Preprocess input data
#         features = preprocess_data(df, is_training=False)
        
#         # Make prediction
#         prediction = model.predict(features)
        
#         # Return prediction as JSON
#         return jsonify({'prediction': int(prediction[0])})
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
