# CTU-13 Botnet Detector

Detects botnet traffic in network flows using a Random Forest trained on the
CTU-13 dataset (real botnet and normal traffic captured at CTU University, Prague, 2011).


**Score your own flows from the command line:**

```bash
python predict.py my_flows.csv                  # CICFlowMeter column names
python predict.py my_flows.csv --threshold 0.3
```

## Results (held-out test set, 20,681 flows)

| Metric | Value |
|---|---|
| Accuracy | 99.27% |
| Precision | 98.67% |
| Recall | 99.38% |
| F1 | 99.02% |
| ROC-AUC | 0.9994 |
| 5-fold CV accuracy | 99.20% ± 0.07% |

Confusion matrix: 12,890 normal correct, 103 false alarms, 48 missed bots, 7,640 bots caught.

## What changed from v1

| | v1 | v2 |
|---|---|---|
| Duplicate rows | Kept (9,491 copies could sit in both train and test) | Removed before splitting |
| `Init Bwd Win Byts` | Top feature | Left out: its main value, 64240, is the lab Windows machines' default TCP window, so it fingerprints the computers, not bot behaviour |
| Scaler | StandardScaler (not needed for trees) | Removed |
| Trees | 200 | 100 (same accuracy, half the size) |
| Evaluation | One test split | Test split + 5-fold CV + 4 comparison models + ROC/PR curves + permutation importance |
| Dashboard model | Replayed scores computed in Python | The real forest runs in the browser and matches Python's scores |
| Inspector | Hand-written if/else rules | The real model, with tree votes and a per-feature explanation |
| Title | "Forecasting" | "Detection" (the model classifies flows; it doesn't predict the future) |

Accuracy went from 99.68% to 99.27%. That small drop comes from no longer testing on duplicate flows and no longer relying on the lab fingerprint. What remains measures how well the model recognises bot behaviour.

## Top features (permutation importance)

1. **Flow IAT Std**: how regular the gaps between packets are. Bots check in on a timer.
2. **Down/Up Ratio**: balance of data received vs sent.
3. **Pkt Len Std**: how much packet sizes vary. Bots repeat the same message.

## Files

```
train_model.py              training + evaluation + dashboard export
predict.py                  score a CSV of new flows
dashboard.html              open in a browser
CTU13-CSV-Dataset/          the two input CSVs
models/botnet_rf.joblib     trained model
models/feature_list.json    feature order the model expects
report/*.json               metrics, CV, comparison, importance, cleaning
dashboard_data/             model + flows bundle and the chart library
samples/                    example CSVs to try in the dashboard
index.html                  opens the dashboard (GitHub Pages entry point)
```

## Limits

- The dashboard replays recorded test flows; it doesn't capture live traffic.
- CTU-13 comes from one lab in 2011, so the model would need retraining on current traffic before real use.
- The dataset has no IP addresses or scenario IDs, so a split by infected host (a harder, more realistic test) isn't possible with this version of the data.
