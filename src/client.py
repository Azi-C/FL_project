import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import flwr as fl
from utils import load_data, train_one_epoch, accuracy, DEVICE
from model import create_model


# --- Flower NumPyClient ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, local_epochs=1, lr=0.01):
        self.model = create_model().to(DEVICE)
        self.trainloader, self.testloader = load_data()
        self.local_epochs = local_epochs
        self.lr = lr

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for _, p in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # load global weights
        keys = list(self.model.state_dict().keys())
        state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(state, strict=True)
        # local training
        epochs = int(config.get("local_epochs", self.local_epochs))
        for _ in range(epochs):
            train_one_epoch(self.model, self.trainloader, lr=self.lr)

        # return updated weights + number of examples
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        keys = list(self.model.state_dict().keys())
        state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(state, strict=True)
        acc = accuracy(self.model, self.testloader)
        return float(1.0 - acc), len(self.testloader.dataset), {"test_accuracy": acc}

def run_client(server_address: str = "127.0.0.1:8080"):
    fl.client.start_client(
        server_address=server_address,
        client=FlowerClient().to_client(),
    )

if __name__ == "__main__":
    run_client()
