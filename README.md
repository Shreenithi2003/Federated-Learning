# Federated-Learning
Privacy preserving Federated Learning Approach 
Implemented a privacy-preserving federated learning framework to detect subclinical mastitis in dairy cattle without sharing raw farm data. 
The dataset was partitioned across two clients, with each client independently training a local Logistic Regression model on its respective subset of mastitis-related data. To ensure data confidentiality, the clients did not transmit any raw data; instead, they shared only the model parameters (weights) with a central server. 
This server acted as an aggregator, combining the received parameters from both clients to update the global model over multiple federated training rounds. 
This approach preserved data privacy while enabling collaborative learning across farms. The final global model achieved an 88% detection accuracy, demonstrating the effectiveness of federated learning in sensitive agricultural health diagnostics.

# Features
1. Federated Learning Setup
2. Logistic Regression-based Detection
3. Decentralized Approach 
4. Privacy-preserving Model Training
5. Parameter Aggregation on Central Server
6. Achieved 88% Detection Accuracy

# Results
1. Model Accuracy: 88%
2. Privacy Status: Raw data never shared
3. Model: Logistic Regression

