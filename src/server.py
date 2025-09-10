import flwr as fl

def get_strategy():
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        on_evaluate_config_fn=lambda rnd: {"local_epochs": 1},  # keep simple
        on_fit_config_fn=lambda rnd: {"local_epochs": 1},
    )

def run_server(server_address: str = "0.0.0.0:8080", num_rounds: int = 3):
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=get_strategy(),
    )

if __name__ == "__main__":
    # Example: python server.py
    run_server()
