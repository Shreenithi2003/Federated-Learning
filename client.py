import flwr as fl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import logging
import argparse
from models.model import SimpleModel

# Set up logging - FIX: Removing the client_id formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CLIENT - %(levelname)s - %(message)s',
)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, X_train, X_test, y_train, y_test):
        self.client_id = client_id
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = SimpleModel()
        # FIX: Removing the logger adapter that was causing issues
        
    def get_parameters(self, config):
        logging.info(f"Client {self.client_id}: Sending model parameters")
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        logging.info(f"Client {self.client_id}: Received fit request, training...")
        
        # Update model with global parameters
        self.model.set_weights(parameters)
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Return updated model parameters
        return self.model.get_weights(), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        logging.info(f"Client {self.client_id}: Received evaluate request")
        
        # Update model with global parameters
        self.model.set_weights(parameters)
        
        # Make predictions
        from sklearn.metrics import log_loss, accuracy_score
        y_pred = self.model.model.predict(self.X_test)
        y_prob = self.model.model.predict_proba(self.X_test)
        
        # Calculate loss and accuracy
        loss = log_loss(self.y_test, y_prob)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        logging.info(f"Client {self.client_id}: Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        
        # Return metrics
        return loss, len(self.X_test), {"accuracy": accuracy}

def load_and_preprocess_data(client_id=0):
    """Load the dataset and preprocess it for federated learning."""
    
    # Load dataset
    DATA_PATH = "datasets/cows_modified.csv"
    df = pd.read_csv(DATA_PATH)
    
    # Generate the target variable
    def generate_mastitis(row):
        if (5.5 <= row['EC'] <= 6.5 and 2.7 <= row['fat'] <= 3.4) or \
           (7.7 <= row['snf'] <= 8.7) or \
           (91.97 <= row['sodium'] <= 97.7 and row['chloride'] > 0.14 and row['ph'] > 6.69):
            return 1  
        else:
            return 0
    
    df["mastitis"] = df.apply(generate_mastitis, axis=1)
    
    # Prepare features and labels
    features = ["EC", "fat", "snf", "sodium", "chloride", "ph"]
    X = df[features].values
    y = df["mastitis"].values
    
    # For the simulation of federated learning, let's partition data by client
    # Each client gets a different random subset of the data
    np.random.seed(42 + client_id)
    indices = np.random.permutation(len(X))
    
    # Use 80% of the data based on client_id
    start_idx = int(client_id * 0.2 * len(X)) % len(X)
    end_idx = min(start_idx + int(0.8 * len(X)), len(X))
    client_indices = indices[start_idx:end_idx]
    
    X_client = X[client_indices]
    y_client = y[client_indices]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42+client_id
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def start_client(client_id=0, server_address="127.0.0.1:9091"):
    """Start a federated learning client."""
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(client_id)
    
    # Create and initialize a Flower client
    client = FlowerClient(client_id, X_train, X_test, y_train, y_test)
    
    # Start the client
    logging.info(f"Starting Flower client {client_id}...")
    
    # FIX: Using the correct method for newer versions of Flower
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),  # Convert NumPyClient to Client
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument("--client-id", type=int, default=0, help="Client ID")
    parser.add_argument("--server", type=str, default="127.0.0.1:9091", help="Server address")
    args = parser.parse_args()
    
    start_client(args.client_id, args.server)