# Fraud-detection
This project demonstrates a fraud detection system that uses a trained machine learning model to predict whether a transaction is fraudulent based on several features such as the transaction amount, account balances, and transaction types. The model is deployed using Flask as a web application.

## Features
Prediction of fraud: The system takes transaction details as input and predicts whether the transaction is fraudulent or not.

Web interface: A simple user interface is provided to input transaction data and view the prediction result.

Local deployment: The system can be run locally for testing.

Model: A trained model as a Pickle file (.pkl) for easy loading and inference.

## Technologies Used
Python

Flask: For creating the web application.

scikit-learn: For training and saving the fraud detection model.

Pandas: For handling data input and transformations.

index.html: Provides a simple interface for entering transaction details.

## Dataset

Dataset link:https://data.world/wayvy/synthetic-fraud-detection-dataset 

The model was trained using a Synthetic Fraud Detection Dataset from data.world that includes the following fields:


step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).

type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.

amount - amount of the transaction in local currency.

nameOrig - customer who started the transaction

oldbalanceOrg - initial balance before the transaction

newbalanceOrig - new balance after the transaction

nameDest - customer who is the recipient of the transaction

oldbalanceDest - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).

newbalanceDest - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).

isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.

isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.

Acct Type - The type of account either saving or current

Transaction Date- Date of transaction occured.
