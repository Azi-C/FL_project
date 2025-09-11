# utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(batch_size: int = 32):
    """Load and return train/test DataLoaders for MNIST."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST("./data", train=True,  download=True, transform=tf)
    testset  = datasets.MNIST("./data", train=False, download=True, transform=tf)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=2)
    testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def load_train_val(batch_size: int = 32, val_split: float = 0.10, seed: int = 42):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    full = datasets.MNIST("./data", train=True, download=True, transform=tf)
    val_size = int(len(full) * val_split)
    train_size = len(full) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [train_size, val_size], generator=gen)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2)
    valloader   = DataLoader(val_set,  batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, valloader

def train_one_epoch(model: nn.Module, loader: DataLoader, lr: float = 0.01, device=DEVICE):
    """Train the model for one epoch."""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device=DEVICE) -> float:
    """Evaluate accuracy of the model."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0
