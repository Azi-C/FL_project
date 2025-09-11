import torch
from model import create_model
from utils import (
    load_partition_for_client,
    load_partition_for_client_non_iid,
    load_validation_loader,
    train_one_epoch,
    accuracy,
    DEVICE,
    assigned_labels_for_client
)

class Client:
    def __init__(
        self,
        cid: int,
        num_clients: int,
        lr: float = 0.01,
        local_epochs: int = 1,
        non_iid: bool = False,
        labels_per_client: int = 2,
        proportions=None,
        dirichlet_alpha=None
    ):
        self.cid = cid
        self.num_clients = num_clients
        self.lr = lr
        self.local_epochs = local_epochs
        self.model = create_model().to(DEVICE)

        if non_iid:
            self.trainloader = load_partition_for_client_non_iid(
                client_id=cid,
                num_clients=num_clients,
                labels_per_client=labels_per_client
            )
            self.assigned_labels = assigned_labels_for_client(cid, labels_per_client)
        else:
            self.trainloader = load_partition_for_client(
                client_id=cid,
                num_clients=num_clients,
                proportions=proportions,
                dirichlet_alpha=dirichlet_alpha
            )
            self.assigned_labels = None

        self.testloader = load_validation_loader()

    def num_examples(self):
        return len(self.trainloader.dataset)

    def get_params(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_params(self, parameters):
        state_dict = self.model.state_dict()
        for k, p in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(p)
        self.model.load_state_dict(state_dict, strict=True)

    def train_local(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.trainloader, criterion, optimizer)

    def evaluate(self):
        return accuracy(self.model, self.testloader)

    def evaluate_on(self, loader):
        return accuracy(self.model, loader)
