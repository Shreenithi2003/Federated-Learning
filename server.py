import flwr as fl
import numpy as np
from models.model import SimpleModel

# Define Federated Strategy class
class FedAvgStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        model = SimpleModel()
        initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())
        super().__init__(
            initial_parameters=initial_parameters,
        )
        
    def aggregate_fit(self, rnd, results, failures):
        print(f"Round {rnd} - {len(results)} clients participated, {len(failures)} failures")
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters:
            print(f"✅ Round {rnd} aggregated successfully!")
        return aggregated_parameters
        
    def aggregate_evaluate(self, rnd, results, failures):
        """Aggregate evaluation results from clients."""
        if not results:
            return None
        
        # Weigh metrics by number of examples used
        accuracies = [r[1].metrics["accuracy"] * r[1].num_examples for r in results]
        examples = [r[1].num_examples for r in results]
        
        # Aggregate and print metrics
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {rnd} - Aggregated accuracy: {accuracy_aggregated:.4f}")
        
        return super().aggregate_evaluate(rnd, results, failures)

# Start Flower Server
def start_server(num_rounds=3, min_clients=2):
    print(f"Starting server... waiting for at least {min_clients} clients")
    strategy = FedAvgStrategy()
    
    # Configure server
    server_config = fl.server.ServerConfig(num_rounds=num_rounds)
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:9091",  # Define the server address
        config=server_config,
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server(num_rounds=3, min_clients=1)