"""
Score new flows with the trained model.

Usage:
    python predict.py flows.csv            # CSV with CICFlowMeter columns
    python predict.py flows.csv --threshold 0.3

Prints one line per flow and writes <input>_scored.csv with two new
columns: botnet_score (0-1) and verdict (BOTNET / normal).
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Score network flows for botnet activity")
parser.add_argument("csv", help="CSV file of flows (CICFlowMeter column names)")
parser.add_argument("--threshold", type=float, default=0.5, help="score at or above this = botnet")
args = parser.parse_args()

model = joblib.load(os.path.join(HERE, "models", "botnet_rf.joblib"))
with open(os.path.join(HERE, "models", "feature_list.json")) as f:
    features = json.load(f)

flows = pd.read_csv(args.csv)
missing = [c for c in features if c not in flows.columns]
if missing:
    raise SystemExit(f"The CSV is missing these columns: {missing}")

X = flows[features].replace([np.inf, -np.inf], np.nan).fillna(0)
flows["botnet_score"] = model.predict_proba(X)[:, 1].round(4)
flows["verdict"] = np.where(flows["botnet_score"] >= args.threshold, "BOTNET", "normal")

for i, row in flows.iterrows():
    print(f"flow {i:>5}  score {row['botnet_score']:.4f}  {row['verdict']}")

out = os.path.splitext(args.csv)[0] + "_scored.csv"
flows.to_csv(out, index=False)
n_bot = int((flows["verdict"] == "BOTNET").sum())
print(f"\n{n_bot} of {len(flows)} flows flagged as botnet (threshold {args.threshold}). Saved {out}")
