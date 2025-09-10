import flwr as fl

def aggregate_eval_metrics(metrics):
    # metrics is a list of dicts like {"test_accuracy": 0.97, ...}
    if not metrics:
        return {}
    avg_acc = sum(m.get("test_accuracy", 0.0) for m in metrics) / len(metrics)
    return {"test_accuracy": avg_acc}

def get_strategy():
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda rnd: {"local_epochs": 1},
        evaluate_metrics_aggregation_fn=lambda rnd, results, failures: (
            aggregate_eval_metrics([m for _, m in results])
        ),
    )

def run_server(server_address: str = "127.0.0.1:8080", num_rounds: int = 3):
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=get_strategy(),
    )
