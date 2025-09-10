# main.py
import argparse
import flwr as fl
from server import run_server
from client import run_client

def main():
    parser = argparse.ArgumentParser(description="Federated Learning with Flower")
    subparsers = parser.add_subparsers(dest="role", required=True)

    # --- Server mode ---
    server_parser = subparsers.add_parser("server", help="Run Flower server")
    server_parser.add_argument(
        "--rounds", type=int, default=3, help="Number of FL rounds"
    )
    server_parser.add_argument(
        "--address", type=str, default="0.0.0.0:8080", help="Server address"
    )

    # --- Client mode ---
    client_parser = subparsers.add_parser("client", help="Run Flower client")
    client_parser.add_argument(
        "--id", type=int, default=0, help="Client partition ID"
    )
    client_parser.add_argument(
        "--num-clients", type=int, default=2, help="Total number of clients"
    )
    client_parser.add_argument(
        "--address", type=str, default="0.0.0.0:8080", help="Server address"
    )

    args = parser.parse_args()

    if args.role == "server":
        print(f"Starting server at {args.address} for {args.rounds} rounds...")
        run_server(server_address=args.address, num_rounds=args.rounds)
    elif args.role == "client":
        print(f"Starting client {args.id}/{args.num_clients} connecting to {args.address}...")
        run_client(
            server_address=args.address,
            client_id=args.id,
            num_clients=args.num_clients
        )

if __name__ == "__main__":
    main()
