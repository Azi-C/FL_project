import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import flwr as fl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Simple CNN (works for MNIST) ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # -> 6x12x12
        x = self.pool(torch.relu(self.conv2(x)))  # -> 16x4x4
        x = torch.flatten(x, 1)                   # -> 256
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)                        # logits

# --- Data loaders (quick/easy) ---
def load_data(batch_size: int = 32):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST("./data", train=True,  download=True, transform=tf)
    testset  = datasets.MNIST("./data", train=False, download=True, transform=tf)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=2)
    testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

# --- Training/Eval helpers ---
def train_one_epoch(model, loader, lr=0.01):
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()

@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total else 0.0

# --- Flower NumPyClient ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, local_epochs=1, lr=0.01):
        self.model = Net().to(DEVICE)
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
        # (Optional) implement if you want server-side eval aggregation
        keys = list(self.model.state_dict().keys())
        state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(state, strict=True)
        acc = accuracy(self.model, self.testloader)
        return float(1.0 - acc), len(self.testloader.dataset), {"test_accuracy": acc}

def run_client(server_address: str = "127.0.0.1:8080"):
    fl.client.start_client(
        server_address=server_address,
        client=FlowerClient().to_client(),  # <- new API (no deprecation warning)
    )

if __name__ == "__main__":
    run_client()
