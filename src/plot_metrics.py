# plot_metrics.py
"""
Read artifacts/metrics.csv and generate simple plots:
  1) Average accuracy vs. round
  2) Reputation per client vs. round

Outputs:
  artifacts/avg_acc.png
  artifacts/reputations.png

Usage (from project root):
  python src/plot_metrics.py
"""
import os
import json
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


CSV_PATH = "artifacts/metrics.csv"
OUT_DIR  = "artifacts"


def read_metrics(csv_path: str):
    rounds = []
    avg_acc = []
    reputations_over_time = defaultdict(list)  # cid -> list of (round, rep)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run strategy.py first.")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["round"])
            rounds.append(r)

            aa = float(row["avg_acc"])
            avg_acc.append(aa)

            # reputations is stored as JSON string, e.g. {"0": 1.0, "1": -0.0, ...}
            rep = json.loads(row["reputations"])
            for k, v in rep.items():
                cid = int(k)
                reputations_over_time[cid].append((r, float(v)))

    # Ensure each client's reputation list is sorted by round
    for cid in reputations_over_time:
        reputations_over_time[cid].sort(key=lambda x: x[0])

    return rounds, avg_acc, reputations_over_time


def plot_avg_acc(rounds, avg_acc, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds, avg_acc, marker="o")
    plt.title("Average Accuracy vs. Round")
    plt.xlabel("Round")
    plt.ylabel("Average Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    out_path = os.path.join(out_dir, "avg_acc.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved {out_path}")


def plot_reputations(reputations_over_time, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    # reputations_over_time: cid -> list of (round, rep)
    cids = sorted(reputations_over_time.keys())
    for cid in cids:
        pairs = reputations_over_time[cid]
        if not pairs:
            continue
        rs  = [r for r, _ in pairs]
        vals = [v for _, v in pairs]
        plt.plot(rs, vals, marker="o", label=f"Client {cid}")

    plt.title("Reputation per Client vs. Round")
    plt.xlabel("Round")
    plt.ylabel("Reputation r_i")
    plt.grid(True, linestyle="--", alpha=0.5)
    if cids:
        plt.legend()
    out_path = os.path.join(out_dir, "reputations.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved {out_path}")


def main():
    rounds, avg_acc, reputations_over_time = read_metrics(CSV_PATH)
    plot_avg_acc(rounds, avg_acc, OUT_DIR)
    plot_reputations(reputations_over_time, OUT_DIR)


if __name__ == "__main__":
    main()
