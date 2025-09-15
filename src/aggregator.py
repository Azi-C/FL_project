# src/aggregator.py
from client import Client

class Aggregator(Client):
    """Aggregator is just a Client that can be selected by strategy.
    All aggregation (FedAvg, consensus, storage I/O) is orchestrated in strategy.py.
    """
    pass
