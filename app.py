{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "yTlIh6kjymPU",
    "outputId": "facb76b2-9e07-4892-cc6d-567401a91dd3"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: tensorflow in /usr/local/lib/python3.10/dist-packages (2.17.0)\n",
      "Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.4.0)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.6.3)\n",
      "Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (24.3.25)\n",
      "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.6.0)\n",
      "Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.2.0)\n",
      "Requirement already satisfied: h5py>=3.10.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.11.0)\n",
      "Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (18.1.1)\n",
      "Requirement already satisfied: ml-dtypes<0.5.0,>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.4.0)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.3.0)\n",
      "Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from tensorflow) (24.1)\n",
      "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.20.3)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.32.3)\n",
      "Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow) (71.0.4)\n",
      "Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.4.0)\n",
      "Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (4.12.2)\n",
      "Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.64.1)\n",
      "Requirement already satisfied: tensorboard<2.18,>=2.17 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.17.0)\n",
      "Requirement already satisfied: keras>=3.2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.4.1)\n",
      "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.37.1)\n",
      "Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.26.4)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow) (0.44.0)\n",
      "Requirement already satisfied: rich in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (13.8.0)\n",
      "Requirement already satisfied: namex in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (0.0.8)\n",
      "Requirement already satisfied: optree in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (0.12.1)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.8)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (2.0.7)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (2024.8.30)\n",
      "Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (3.7)\n",
      "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (0.7.2)\n",
      "Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (3.0.4)\n",
      "Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.10/dist-packages (from werkzeug>=1.0.1->tensorboard<2.18,>=2.17->tensorflow) (2.1.5)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=3.2.0->tensorflow) (3.0.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=3.2.0->tensorflow) (2.16.1)\n",
      "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=3.2.0->tensorflow) (0.1.2)\n"
     ]
    }
   ],
   "source": [
    "!pip install tensorflow\n",
    "import tensorflow as tf # Import tensorflow\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "RnxG2Fh20jFg",
    "outputId": "6ec4b944-4508-4066-91d7-b96f8236a9ab"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Error loading the model. Please check if /content/fraud_detection_model.h5 is a valid HDF5 file.\n",
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
      " * Running on http://127.0.0.1:5000\n",
      "INFO:werkzeug:\u001b[33mPress CTRL+C to quit\u001b[0m\n",
      "INFO:werkzeug: * Restarting with stat\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, jsonify, render_template\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras import regularizers\n",
    "import os # Import the os module\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load the trained model\n",
    "model_path = '/content/fraud_detection_model.h5'\n",
    "if os.path.exists(model_path): # Check if the model file exists\n",
    "    try: # Use a try-except block to catch potential errors during model loading\n",
    "        model = load_model(model_path)\n",
    "    except OSError:\n",
    "        print(f\"Error loading the model. Please check if {model_path} is a valid HDF5 file.\")\n",
    "        # Handle the error appropriately, e.g., exit the program or use a default model\n",
    "else:\n",
    "    print(f\"Error: Model file not found at {model_path}\")\n",
    "    # Handle the error appropriately, e.g., exit the program or use a default model\n",
    "\n",
    "# Initialize LabelEncoder\n",
    "label_encoder = LabelEncoder()\n",
    "\n",
    "# Dummy encoder for pre-trained encoder (Replace with real values if needed)\n",
    "label_encoder_fitted = {\n",
    "    'type': label_encoder.fit(['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']),\n",
    "    'branch': label_encoder.fit(['A', 'B', 'C']),\n",
    "    'Time of day': label_encoder.fit(['Morning', 'Afternoon', 'Evening']),\n",
    "    'Acct type': label_encoder.fit(['savings', 'current'])\n",
    "}\n",
    "\n",
    "# Home route to render the HTML form\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "# Prediction route\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    # Retrieve form data\n",
    "    data = {\n",
    "        'type': request.form['type'],\n",
    "        'branch': request.form['branch'],\n",
    "        'amount': float(request.form['amount']),\n",
    "        'oldbalanceOrg': float(request.form['oldbalanceOrg']),\n",
    "        'newbalanceOrig': float(request.form['newbalanceOrig']),\n",
    "        'oldbalanceDest': float(request.form['oldbalanceDest']),\n",
    "        'newbalanceDest': float(request.form['newbalanceDest']),\n",
    "        'unusuallogin': int(request.form['unusuallogin']),\n",
    "        'isFlaggedFraud': int(request.form['isFlaggedFraud']),\n",
    "        'Acct type': request.form['Acct type'],\n",
    "        'Time of day': request.form['Time of day']\n",
    "    }\n",
    "\n",
    "    # Convert to DataFrame\n",
    "    df = pd.DataFrame([data])\n",
    "\n",
    "    # Encode categorical features\n",
    "    df['type'] = label_encoder_fitted['type'].transform(df['type'])\n",
    "    df['branch'] = label_encoder_fitted['branch'].transform(df['branch'])\n",
    "    df['Time of day'] = label_encoder_fitted['Time of day'].transform(df['Time of day'])\n",
    "    df['Acct type'] = label_encoder_fitted['Acct type'].transform(df['Acct type'])\n",
    "\n",
    "    # Drop unnecessary columns\n",
    "    df.drop(['Date of transaction'], axis=1, inplace=True, errors='ignore')\n",
    "\n",
    "    # Make prediction\n",
    "    if 'model' in locals(): # Check if the model variable exists\n",
    "        prediction = model.predict(df)\n",
    "        result = 'Fraud' if prediction[0][0] > 0.5 else 'Not Fraud'\n",
    "    else:\n",
    "        result = \"Model not loaded. Can't make prediction.\"\n",
    "\n",
    "    return render_template('index.html', prediction_text=f'Transaction is {result}')\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run(debug=True)"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "authorship_tag": "ABX9TyNRV/ACWVYv7JRN37C9k8hL",
   "include_colab_link": true,
   "mount_file_id": "1ijotCV2Balt2I6f_8GQz7svopviw-zuT",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
