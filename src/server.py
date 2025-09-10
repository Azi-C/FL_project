import flwr as fl

# Weighted-average client accuracies
def weighted_accuracy(results):
    if not results:
        return {}
    total_examples = sum(num_examples for num_examples, _ in results)
    if total_examples == 0:
        return {}
    acc_sum = sum(num_examples * m.get("test_accuracy", 0.0) for num_examples, m in results)
    avg_acc = acc_sum / total_examples
    return {"test_accuracy": avg_acc}

def get_strategy():
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda rnd: {"local_epochs": 1},
        evaluate_metrics_aggregation_fn=weighted_accuracy,  # ✅ updated signature
    )

def run_server(server_address: str = "127.0.0.1:8080", num_rounds: int = 3):
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=get_strategy(),
    )

if __name__ == "__main__":
    run_server()
