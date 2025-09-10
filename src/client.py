# client.py
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl

from model import create_model
from utils import load_partition_for_client, load_train_val, evaluate_accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def params_to_numpy(model: torch.nn.Module):
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]

def numpy_to_params(model: torch.nn.Module, parameters):
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state, strict=True)

def train_one_epoch(model, loader, device=DEVICE, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, num_clients: int = 10, epochs: int = 1, lr: float = 0.01):
        self.client_id = client_id
        self.num_clients = num_clients
        self.epochs = epochs
        self.lr = lr
        self.model = create_model().to(DEVICE)
        self.trainloader = load_partition_for_client(client_id=client_id, num_clients=num_clients)
        _, self.valloader = load_train_val()

    def get_parameters(self, config):
        return params_to_numpy(self.model)

    def fit(self, parameters, config):
        numpy_to_params(self.model, parameters)
        for _ in range(self.epochs):
            train_one_epoch(self.model, self.trainloader, DEVICE, lr=self.lr)
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        numpy_to_params(self.model, parameters)
        acc = evaluate_accuracy(self.model, self.valloader, DEVICE)
        return float(1.0 - acc), len(self.valloader.dataset), {"val_accuracy": acc}

def run_client(server_address: str = "0.0.0.0:8080", client_id: int = 0, num_clients: int = 2):
    fl.client.start_numpy_client(
        server_address=server_address,
        client=FlowerClient(client_id=client_id, num_clients=num_clients),
    )
