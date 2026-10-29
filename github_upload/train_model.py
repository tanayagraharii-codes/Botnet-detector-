"""
CTU-13 Botnet Traffic Detector  (v2)
====================================
Trains a Random Forest on real CTU-13 network flows (CICFlowMeter features)
to classify each flow as Botnet (1) or Normal (0).

What changed from v1
--------------------
1. Duplicate rows are removed BEFORE splitting. ~9,500 flows were exact
   copies; when a copy lands in both train and test, the model is tested
   on data it has already seen, which inflates the score.
2. "Init Bwd Win Byts" is dropped from the main model. Its top value
   (64240) is the default TCP window of the infected Windows lab machines,
   so it fingerprints the 2011 lab computers rather than bot behaviour.
   A model trained WITH it is still reported, for comparison.
3. The scaler is gone. Tree models don't need scaled inputs.
4. Evaluation is much fuller: 5-fold cross-validation, a comparison with
   simpler models, ROC / precision-recall curves, threshold analysis and
   permutation importance.
5. The trained forest is exported to JavaScript, so the dashboard runs the
   REAL model in the browser (live replay and the manual Inspector).

Outputs
-------
  models/botnet_rf.joblib          trained Random Forest
  models/feature_list.json         feature order the model expects
  report/metrics.json              test-set metrics + cross-validation
  report/model_comparison.json     baseline models vs the final model
  report/feature_importance.json   impurity + permutation importance
  report/data_cleaning.json        what was removed and why
  dashboard_data/dashboard_bundle.js  everything the dashboard needs

Run:  python train_model.py      (needs: pip install -r requirements.txt)
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "CTU13-CSV-Dataset")
OUT_MODELS = os.path.join(HERE, "models")
OUT_REPORT = os.path.join(HERE, "report")
OUT_DASH = os.path.join(HERE, "dashboard_data")
for d in (OUT_MODELS, OUT_REPORT, OUT_DASH):
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
THRESHOLD = 0.5          # score >= THRESHOLD  ->  botnet
STREAM_SIZE = 400        # flows replayed by the dashboard

# Behaviour features: things any flow monitor can measure, and that describe
# HOW the traffic behaves (size, speed, timing, TCP flags).
FEATURES = [
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Mean", "Bwd Pkt Len Mean",
    "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std",
    "Fwd IAT Mean", "Bwd IAT Mean",
    "Pkt Len Mean", "Pkt Len Std",
    "SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt", "FIN Flag Cnt",
    "Down/Up Ratio",
]
FINGERPRINT_FEATURE = "Init Bwd Win Byts"   # lab-environment shortcut (see note 2)


def make_forest():
    return RandomForestClassifier(
        n_estimators=100,        # 100 trees score the same as 200 here, at half the size
        max_depth=16,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def scores(y_true, proba, threshold=THRESHOLD):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
    }


def step(msg):
    print(f"\n== {msg}")


t0 = time.time()

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
step("Loading CTU-13 flows")
attack = pd.read_csv(os.path.join(DATA_DIR, "CTU13_Attack_Traffic.csv"))
normal = pd.read_csv(os.path.join(DATA_DIR, "CTU13_Normal_Traffic.csv"))
raw = pd.concat([attack, normal], ignore_index=True)
raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
n_raw = len(raw)
print(f"{len(attack):,} botnet + {len(normal):,} normal = {n_raw:,} flows")

# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
step("Cleaning")
df = raw.replace([np.inf, -np.inf], np.nan).dropna()
n_bad = n_raw - len(df)
dup_mask = df.duplicated()
dups_by_class = df[dup_mask]["Label"].value_counts().to_dict()
df = df[~dup_mask].reset_index(drop=True)
print(f"removed {n_bad} rows with inf/NaN and {int(dup_mask.sum()):,} duplicate rows "
      f"-> {len(df):,} unique flows")

fp_counts = {
    "normal": int((df.loc[df.Label == 0, FINGERPRINT_FEATURE] == 64240).sum()),
    "botnet": int((df.loc[df.Label == 1, FINGERPRINT_FEATURE] == 64240).sum()),
}
cleaning = {
    "raw_flows": n_raw,
    "raw_botnet": int(len(attack)),
    "raw_normal": int(len(normal)),
    "invalid_rows_removed": int(n_bad),
    "duplicates_removed": int(dup_mask.sum()),
    "duplicates_botnet": int(dups_by_class.get(1, 0)),
    "duplicates_normal": int(dups_by_class.get(0, 0)),
    "unique_flows": int(len(df)),
    "unique_botnet": int((df.Label == 1).sum()),
    "unique_normal": int((df.Label == 0).sum()),
    "fingerprint_feature": FINGERPRINT_FEATURE,
    "fingerprint_value": 64240,
    "fingerprint_value_counts": fp_counts,
}
with open(os.path.join(OUT_REPORT, "data_cleaning.json"), "w") as f:
    json.dump(cleaning, f, indent=2)

X_all = df[FEATURES + [FINGERPRINT_FEATURE]]
y = df["Label"].astype(int)

# ---------------------------------------------------------------------------
# 3. Split: 75% train / 25% held-out test (never used for training)
# ---------------------------------------------------------------------------
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
X_train, X_test = X_train_all[FEATURES], X_test_all[FEATURES]
print(f"train {len(X_train):,} flows | test {len(X_test):,} flows")

# ---------------------------------------------------------------------------
# 4. Cross-validation of the final model on the training set
# ---------------------------------------------------------------------------
step("5-fold cross-validation")
cv = cross_validate(make_forest(), X_train, y_train,
                    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
cv_summary = {}
for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    vals = cv[f"test_{k}"]
    cv_summary[k] = {"mean": round(float(vals.mean()), 4),
                     "std": round(float(vals.std()), 4),
                     "folds": [round(float(v), 4) for v in vals]}
    print(f"{k:10s} {vals.mean():.4f} ± {vals.std():.4f}")

# ---------------------------------------------------------------------------
# 5. Train the final model and evaluate on the held-out test set
# ---------------------------------------------------------------------------
step("Training final Random Forest")
clf = make_forest().fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= THRESHOLD).astype(int)
test_metrics = scores(y_test, proba)
cm = confusion_matrix(y_test, pred).tolist()
print(json.dumps(test_metrics), "confusion", cm)

metrics = {
    "model": "RandomForestClassifier",
    "params": {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 3,
               "class_weight": "balanced"},
    "features": FEATURES,
    "threshold": THRESHOLD,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    **test_metrics,
    "confusion_matrix": cm,
    "confusion_matrix_labels": ["Normal (0)", "Botnet (1)"],
    "cross_validation": cv_summary,
}
with open(os.path.join(OUT_REPORT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
joblib.dump(clf, os.path.join(OUT_MODELS, "botnet_rf.joblib"))
with open(os.path.join(OUT_MODELS, "feature_list.json"), "w") as f:
    json.dump(FEATURES, f, indent=2)

# ---------------------------------------------------------------------------
# 6. Compare with simpler models and with the v1 setup
# ---------------------------------------------------------------------------
step("Model comparison (same split)")
comparison = []


def compare(name, note, model, Xtr, Xte):
    t = time.time()
    model.fit(Xtr, y_train)
    p = model.predict_proba(Xte)[:, 1]
    row = {"model": name, "note": note, **scores(y_test, p),
           "train_seconds": round(time.time() - t, 2)}
    comparison.append(row)
    print(f"{name:34s} acc {row['accuracy']:.4f}  f1 {row['f1']:.4f}  auc {row['roc_auc']:.4f}")


compare("Logistic Regression", "Straight-line boundary; needs scaling",
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        X_train, X_test)
compare("Single Decision Tree", "One tree, same depth as the forest's trees",
        DecisionTreeClassifier(max_depth=16, min_samples_leaf=3, class_weight="balanced",
                               random_state=RANDOM_STATE), X_train, X_test)
compare("Gradient Boosting", "Trees built one after another, each fixing the last",
        HistGradientBoostingClassifier(random_state=RANDOM_STATE), X_train, X_test)
compare("Random Forest + lab fingerprint", f"Same forest, also given '{FINGERPRINT_FEATURE}'",
        make_forest(), X_train_all, X_test_all)
comparison.append({"model": "Random Forest (final)", "note": "100 trees, behaviour features only",
                   **test_metrics, "train_seconds": None, "final": True})

# v1 setup: duplicates kept, fingerprint feature used, 200 trees
raw_clean = raw.replace([np.inf, -np.inf], np.nan).dropna()
Xa, Xb, ya, yb = train_test_split(raw_clean[FEATURES + [FINGERPRINT_FEATURE]], raw_clean["Label"],
                                  test_size=0.25, random_state=RANDOM_STATE,
                                  stratify=raw_clean["Label"])
v1 = RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=3,
                            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE).fit(Xa, ya)
v1_scores = scores(yb, v1.predict_proba(Xb)[:, 1])
comparison.append({"model": "v1 setup (old script)", "note": "Duplicates kept + lab fingerprint; "
                   "scored on a different test set", **v1_scores, "train_seconds": None})
print(f"{'v1 setup (old script)':34s} acc {v1_scores['accuracy']:.4f}")
with open(os.path.join(OUT_REPORT, "model_comparison.json"), "w") as f:
    json.dump(comparison, f, indent=2)

# ---------------------------------------------------------------------------
# 7. Feature importance: impurity (built in) + permutation (on test data)
# ---------------------------------------------------------------------------
step("Feature importance")
rng = np.random.RandomState(RANDOM_STATE)
sample = rng.choice(len(X_test), size=min(6000, len(X_test)), replace=False)
perm = permutation_importance(clf, X_test.iloc[sample], y_test.iloc[sample],
                              scoring="f1", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
importance = sorted(
    [{"feature": f,
      "impurity": round(float(i), 4),
      "permutation": round(float(p), 4),
      "permutation_std": round(float(s), 4)}
     for f, i, p, s in zip(FEATURES, clf.feature_importances_,
                           perm.importances_mean, perm.importances_std)],
    key=lambda r: r["impurity"], reverse=True)
for r in importance[:6]:
    print(f"{r['feature']:20s} impurity {r['impurity']:.3f}  permutation {r['permutation']:.3f}")
with open(os.path.join(OUT_REPORT, "feature_importance.json"), "w") as f:
    json.dump(importance, f, indent=2)

# ---------------------------------------------------------------------------
# 8. Curves for the dashboard
# ---------------------------------------------------------------------------
step("Building dashboard bundle")


def thin(xs, ys, n=120):
    idx = np.unique(np.linspace(0, len(xs) - 1, n).astype(int))
    return [[round(float(xs[i]), 4), round(float(ys[i]), 4)] for i in idx]


fpr, tpr, _ = roc_curve(y_test, proba)
prec, rec, _ = precision_recall_curve(y_test, proba)
BINS = 100
edges = np.linspace(0, 1, BINS + 1)
hist_normal = np.histogram(proba[y_test.values == 0], bins=edges)[0].tolist()
hist_botnet = np.histogram(proba[y_test.values == 1], bins=edges)[0].tolist()

# ---------------------------------------------------------------------------
# 9. Export the forest so the browser can run it
#    sklearn compares float32(x) <= threshold; the JS mirrors that exactly.
# ---------------------------------------------------------------------------
trees = []
for est in clf.estimators_:
    t = est.tree_
    leaf_p = t.value[:, 0, 1] / t.value[:, 0, :].sum(axis=1)
    trees.append({
        "f": t.feature.astype(int).tolist(),               # feature index, -2 = leaf
        "t": [float(v) for v in t.threshold],
        "l": t.children_left.astype(int).tolist(),
        "r": t.children_right.astype(int).tolist(),
        "p": [round(float(v), 6) for v in leaf_p],          # botnet probability at node
    })
nodes = sum(len(tr["f"]) for tr in trees)

# ---------------------------------------------------------------------------
# 10. Live stream: held-out test flows in a realistic order
#     (normal background traffic with bursts of botnet activity)
# ---------------------------------------------------------------------------
rng = np.random.RandomState(7)
test_idx = np.arange(len(X_test))
norm_idx = rng.permutation(test_idx[y_test.values == 0])
bot_idx = rng.permutation(test_idx[y_test.values == 1])
n_bot = int(STREAM_SIZE * 0.3)
norm_take, bot_take = list(norm_idx[:STREAM_SIZE - n_bot]), list(bot_idx[:n_bot])
order = []
while norm_take or bot_take:
    calm = rng.randint(4, 14)
    order += [norm_take.pop() for _ in range(min(calm, len(norm_take)))]
    burst = rng.randint(2, 9)
    order += [bot_take.pop() for _ in range(min(burst, len(bot_take)))]

stream = []
for i in order:
    row = X_test.iloc[i]
    stream.append({
        "x": [float(v) for v in row.values],
        "y": int(y_test.iloc[i]),
        "py": round(float(proba[i]), 6),    # Python's score, used to verify the JS
    })

# Typical values per class, for Inspector presets and explanations
typical = {
    "normal_median": [float(v) for v in X_train[y_train == 0].median().values],
    "botnet_median": [float(v) for v in X_train[y_train == 1].median().values],
}


def example(label):
    """A real test flow the model scores confidently for this class."""
    mask = (y_test.values == label) & ((proba > 0.95) if label else (proba < 0.05))
    pick = np.flatnonzero(mask)
    pick = pick[[s not in set(order) for s in pick]]
    return [float(v) for v in X_test.iloc[pick[0]].values]


bundle = {
    "generated": time.strftime("%Y-%m-%d %H:%M"),
    "features": FEATURES,
    "threshold": THRESHOLD,
    "cleaning": cleaning,
    "metrics": metrics,
    "comparison": comparison,
    "importance": importance,
    "curves": {"roc": thin(fpr, tpr), "pr": thin(rec, prec),
               "hist_normal": hist_normal, "hist_botnet": hist_botnet},
    "typical": typical,
    "examples": {"normal": example(0), "botnet": example(1)},
    "forest": {"n_trees": len(trees), "nodes": nodes, "trees": trees},
    "stream": stream,
}
path = os.path.join(OUT_DASH, "dashboard_bundle.js")
with open(path, "w") as f:
    f.write("// Generated by train_model.py - do not edit by hand\n")
    f.write("window.BOTNET = ")
    json.dump(bundle, f, separators=(",", ":"))
    f.write(";\n")
print(f"forest: {len(trees)} trees, {nodes:,} nodes | stream: {len(stream)} flows")
print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
print(f"\nDone in {time.time() - t0:.0f}s. Open dashboard.html in a browser.")
