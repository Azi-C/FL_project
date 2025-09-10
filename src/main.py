# main.py
import threading, time
from server import run_server
from client import run_client

ADDR = "127.0.0.1:8080"   # safer in Codespaces

def start_server():
    run_server(server_address=ADDR, num_rounds=3)

def start_client(cid):
    run_client(server_address=ADDR, client_id=cid, num_clients=2)

if __name__ == "__main__":
    s = threading.Thread(target=start_server, daemon=True)
    s.start()

    time.sleep(4)  # <-- increase boot delay

    clients = []
    for cid in range(2):
        t = threading.Thread(target=start_client, args=(cid,))
        t.start()
        clients.append(t)

    for t in clients:
        t.join()
