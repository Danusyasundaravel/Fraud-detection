#!/usr/bin/env python
# coding: utf-8

# In[1]:



import tensorflow as tf



# In[2]:


from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import os # Import the os module

app = Flask(__name__)

# Load the trained model
model_path = '/content/fraud_detection_model (2).h5'
if os.path.exists(model_path): # Check if the model file exists
    try: # Use a try-except block to catch potential errors during model loading
        model = load_model(model_path)
    except OSError:
        print(f"Error loading the model. Please check if {model_path} is a valid HDF5 file.")
        # Handle the error appropriately, e.g., exit the program or use a default model
else:
    print(f"Error: Model file not found at {model_path}")
    # Handle the error appropriately, e.g., exit the program or use a default model

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Dummy encoder for pre-trained encoder (Replace with real values if needed)
label_encoder_fitted = {
    'type': label_encoder.fit(['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']),
    'branch': label_encoder.fit(['A', 'B', 'C']),
    'Time of day': label_encoder.fit(['Morning', 'Afternoon', 'Evening']),
    'Acct type': label_encoder.fit(['savings', 'current'])
}

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('/conent/index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    data = {
        'type': request.form['type'],
        'branch': request.form['branch'],
        'amount': float(request.form['amount']),
        'oldbalanceOrg': float(request.form['oldbalanceOrg']),
        'newbalanceOrig': float(request.form['newbalanceOrig']),
        'oldbalanceDest': float(request.form['oldbalanceDest']),
        'newbalanceDest': float(request.form['newbalanceDest']),
        'unusuallogin': int(request.form['unusuallogin']),
        'isFlaggedFraud': int(request.form['isFlaggedFraud']),
        'Acct type': request.form['Acct type'],
        'Time of day': request.form['Time of day']
    }

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Encode categorical features
    df['type'] = label_encoder_fitted['type'].transform(df['type'])
    df['branch'] = label_encoder_fitted['branch'].transform(df['branch'])
    df['Time of day'] = label_encoder_fitted['Time of day'].transform(df['Time of day'])
    df['Acct type'] = label_encoder_fitted['Acct type'].transform(df['Acct type'])

    # Drop unnecessary columns
    df.drop(['Date of transaction'], axis=1, inplace=True, errors='ignore')

    # Make prediction
    if 'model' in locals(): # Check if the model variable exists
        prediction = model.predict(df)
        result = 'Fraud' if prediction[0][0] > 0.5 else 'Not Fraud'
    else:
        result = "Model not loaded. Can't make prediction."

    return render_template('index.html', prediction_text=f'Transaction is {result}')

if __name__ == "__main__":
    app.run(debug=True)


# In[4]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install flask')
get_ipython().system('pip install pandas')
get_ipython().system('pip install scikit-learn')
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import os # Import the os module


# Removed the if __name__ == "__main__": block as it's not needed in this context


# In[12]:


app = Flask(__name__)

# Load the trained model
model_path = 'fraud_detection_model (2).h5' # Removed '/content/' as it's not needed in this context
if os.path.exists(model_path): # Check if the model file exists
    try: # Use a try-except block to catch potential errors during model loading
        model = load_model(model_path)
    except OSError:
        print(f"Error loading the model. Please check if {model_path} is a valid HDF5 file.")
        # Handle the error appropriately, e.g., exit the program or use a default model
else:
    print(f"Error: Model file not found at {model_path}")
    # Handle the error appropriately, e.g., exit the program or use a default model

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Dummy encoder for pre-trained encoder (Replace with real values if needed)
label_encoder_fitted = {
    'type': label_encoder.fit(['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']),
    'branch': label_encoder.fit(['A', 'B', 'C']),
    'Time of day': label_encoder.fit(['Morning', 'Afternoon', 'Evening']),
    'Acct type': label_encoder.fit(['savings', 'current'])
}

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    data = {
        'type': request.form['type'],
        'branch': request.form['branch'],
        'amount': float(request.form['amount']),
        'oldbalanceOrg': float(request.form['oldbalanceOrg']),
        'newbalanceOrig': float(request.form['newbalanceOrig']),
        'oldbalanceDest': float(request.form['oldbalanceDest']),
        'newbalanceDest': float(request.form['newbalanceDest']),
        'unusuallogin': int(request.form['unusuallogin']),
        'isFlaggedFraud': int(request.form['isFlaggedFraud']),
        'Acct type': request.form['Acct type'],
        'Time of day': request.form['Time of day']
    }

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Encode categorical features
    df['type'] = label_encoder_fitted['type'].transform(df['type'])
    df['branch'] = label_encoder_fitted['branch'].transform(df['branch'])
    df['Time of day'] = label_encoder_fitted['Time of day'].transform(df['Time of day'])
    df['Acct type'] = label_encoder_fitted['Acct type'].transform(df['Acct type'])

    # Drop unnecessary columns
    df.drop(['Date of transaction'], axis=1, inplace=True, errors='ignore')

    # Make prediction
    if 'model' in locals(): # Check if the model variable exists
        prediction = model.predict(df)
        result = 'Fraud' if prediction[0][0] > 0.5 else 'Not Fraud'
    else:
        result = "Model not loaded. Can't make prediction."

    return render_template('index.html', prediction_text=f'Transaction is {result}')

