# main.py
import threading
import time
from server import run_server
from client import run_client

def start_server():
    run_server(server_address="0.0.0.0:8080", num_rounds=3)

def start_client(cid):
    run_client(server_address="0.0.0.0:8080", client_id=cid, num_clients=2)

if __name__ == "__main__":
    # Start server in a thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Small delay so server starts before clients connect
    time.sleep(2)

    # Start two clients in their own threads
    client_threads = []
    for cid in range(2):
        t = threading.Thread(target=start_client, args=(cid,))
        t.start()
        client_threads.append(t)

    # Wait for clients to finish
    for t in client_threads:
        t.join()

    print("Federated Learning simulation completed.")
