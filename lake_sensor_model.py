"""
Lake State + Foaming Risk model for a 4-sensor ESP32 node
=========================================================
Sensors on the node:
    - Analog pH Sensor Kit (E-201-C, BNC)          -> pH
    - Analog Turbidity Sensor (R-0913)             -> turbidity (NTU)
    - Analog TDS Sensor (R-0228)                   -> TDS (ppm, 25 C compensated)
    - DS18B20 waterproof temperature sensor        -> temperature (C)

Outputs for any reading:
    1. State        : "Clean" / "Moderately Polluted"
    2. Foaming risk : "Low" / "Medium" / "High"  (+ a 0-100 Foaming Risk Index)

KEY DESIGN IDEA (why this is not circular)
------------------------------------------
Labels are built from ALL eight lab parameters in the dataset
(TN, TP, CODMn/PI, DOC, DO, Tur, EC, pH), but the Random Forests are trained
ONLY on what the node can actually measure (pH, EC<-TDS, turbidity).
So the model learns to infer the hidden nutrient / organic load that drives
pollution and foaming from cheap sensor signals. That is the real ML task.

(If you train on the same columns you used to build the label, the model
just re-learns your formula, which is why accuracy looks near-perfect.)

Dataset units (Luan et al., Figshare 2025):
    Tur in JTU (~NTU), EC in S/m, TN/TP/CODMn/DOC/DO in mg/L.
    Sensor TDS (ppm) -> EC(uS/cm) = TDS / 0.5 -> EC(S/m) = uS/cm * 1e-4

Temperature is NOT in the dataset. It is used (a) to temperature-compensate
EC if the TDS reading is not already compensated, and (b) as a transparent
rule-based modifier on foaming risk ("if current conditions persist", warm
water accelerates organic decomposition / algal growth). State this as a
heuristic in the paper; retrain with temperature once you log real data.

Usage:
    python lake_sensor_model.py train --csv lake_2023_12.csv
    python lake_sensor_model.py predict --temp 29 --tds 320 --tur 4.5 --ph 8.3
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

MODEL_PATH = "lake_sensor_models.pkl"
SENSOR_FEATURES = ["pH", "pH_dev", "Cond", "Tur"]   # what the node can measure
STATE_LABELS = ["Clean", "Moderately Polluted"]
FOAM_LABELS = ["Low", "Medium", "High"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def robust_minmax(s, lo=None, hi=None):
    """Min-max using 1st/99th percentiles so outliers don't squash the scale."""
    lo = s.quantile(0.01) if lo is None else lo
    hi = s.quantile(0.99) if hi is None else hi
    return ((s - lo) / (hi - lo)).clip(0, 1)


def add_sensor_features(df):
    df = df.copy()
    df["pH_dev"] = (df["pH"] - 7.75).abs()      # distance from healthy 7.0-8.5 band
    return df


def tds_to_cond_sm(tds_ppm, temp_c=None, k=0.5, compensated=True):
    """TDS (ppm) -> EC in S/m (dataset units). Compensates to 25 C if needed."""
    ec_us_cm = tds_ppm / k
    if not compensated and temp_c is not None:
        ec_us_cm = ec_us_cm / (1.0 + 0.02 * (temp_c - 25.0))
    return ec_us_cm * 1e-4


# ---------------------------------------------------------------------------
# STEP 1: Labels from all 8 lab parameters
# ---------------------------------------------------------------------------
def build_labels(df):
    s = pd.DataFrame(index=df.index)

    # --- State: equal-weight composite pollution score (WQI-style) ---
    for c in ["TN", "TP", "Tur", "PI", "Cond"]:
        s[c] = robust_minmax(df[c])
    s["DO"] = 1 - robust_minmax(df["DO"])
    s["pH"] = robust_minmax((df["pH"] - 7.75).abs())
    df["PollutionScore"] = s.mean(axis=1)
    median = df["PollutionScore"].median()
    df["State"] = np.where(df["PollutionScore"] <= median,
                           STATE_LABELS[0], STATE_LABELS[1])

    # --- Foaming potential: weighted towards foam drivers ---
    # Urban-lake foam = surfactants + phosphate-rich sewage + organic load
    # + low oxygen. TP and CODMn(PI) are the closest proxies available.
    foam_w = {"TP": 0.30, "PI": 0.25, "TN": 0.15, "DOC": 0.10,
              "DO_def": 0.10, "Tur": 0.10}
    f = pd.DataFrame(index=df.index)
    for c in ["TP", "PI", "TN", "DOC", "Tur"]:
        f[c] = robust_minmax(df[c])
    f["DO_def"] = 1 - robust_minmax(df["DO"])
    df["FoamScore"] = sum(f[c] * w for c, w in foam_w.items())
    df["FoamRisk"] = pd.qcut(df["FoamScore"], q=3, labels=FOAM_LABELS).astype(str)
    return df


# ---------------------------------------------------------------------------
# STEP 2: Train two Random Forests on sensor-observable features only
# ---------------------------------------------------------------------------
def train(csv_path):
    df = pd.read_csv(csv_path).dropna()
    print("Loaded:", df.shape)
    df = build_labels(add_sensor_features(df))
    df.to_csv("lake_2023_12_labeled_sensor.csv", index=False)

    print("\nState balance:\n", df["State"].value_counts())
    print("\nFoaming balance:\n", df["FoamRisk"].value_counts())

    X = df[SENSOR_FEATURES]
    bundle = {"features": SENSOR_FEATURES,
              "train_range": {c: (float(df[c].min()), float(df[c].max()))
                              for c in ["pH", "Cond", "Tur"]}}

    for target, labels, key in [("State", STATE_LABELS, "state_model"),
                                ("FoamRisk", FOAM_LABELS, "foam_model")]:
        y = df[target]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=16, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=42)
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X_te)

        print(f"\n===== {target} model (inputs: {SENSOR_FEATURES}) =====")
        print(classification_report(y_te, pred, labels=labels, digits=3))
        print("Confusion matrix (rows=true, cols=pred):", labels)
        print(confusion_matrix(y_te, pred, labels=labels))
        print("Feature importance:",
              dict(zip(SENSOR_FEATURES, rf.feature_importances_.round(3))))
        bundle[key] = rf

    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved models -> {MODEL_PATH}")


# ---------------------------------------------------------------------------
# STEP 3: Inference for one live sensor reading
# ---------------------------------------------------------------------------
def predict(temp_c, tds_ppm, turbidity_ntu, ph,
            tds_compensated=True, bundle=None):
    bundle = bundle or joblib.load(MODEL_PATH)
    cond = tds_to_cond_sm(tds_ppm, temp_c, compensated=tds_compensated)
    row = add_sensor_features(pd.DataFrame(
        [{"pH": ph, "Cond": cond, "Tur": turbidity_ntu}]))

    # Out-of-range check: RF cannot extrapolate beyond training data
    warnings = []
    for c, (lo, hi) in bundle["train_range"].items():
        v = float(row[c].iloc[0])
        if v < lo or v > hi:
            warnings.append(f"{c}={v:.4g} outside training range "
                            f"[{lo:.4g}, {hi:.4g}]; prediction is extrapolated")
    X = row[bundle["features"]]

    # State
    sm = bundle["state_model"]
    sp = dict(zip(sm.classes_, sm.predict_proba(X)[0]))
    state = max(sp, key=sp.get)

    # Foaming: model probabilities -> 0-100 index -> temperature persistence
    fm = bundle["foam_model"]
    fp = dict(zip(fm.classes_, fm.predict_proba(X)[0]))
    base_index = 100 * (0.5 * fp.get("Medium", 0) + 1.0 * fp.get("High", 0))
    # Heuristic (not learned): +1.5% risk per C above 25 C, bounded
    temp_factor = float(np.clip(1 + 0.015 * (temp_c - 25), 0.85, 1.25))
    fri = float(np.clip(base_index * temp_factor, 0, 100))
    foam = "Low" if fri < 100 / 3 else ("Medium" if fri < 200 / 3 else "High")

    return {
        "inputs": {"temp_c": temp_c, "tds_ppm": tds_ppm,
                   "turbidity_ntu": turbidity_ntu, "pH": ph,
                   "cond_S_per_m": round(cond, 5)},
        "state": state,
        "state_confidence": round(sp[state], 3),
        "foaming_risk": foam,
        "foaming_risk_index": round(fri, 1),
        "foam_class_probabilities": {k: round(v, 3) for k, v in fp.items()},
        "temperature_factor": round(temp_factor, 3),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--csv", default="lake_2023_12.csv")
    p = sub.add_parser("predict")
    p.add_argument("--temp", type=float, required=True, help="Water temp, C")
    p.add_argument("--tds", type=float, required=True, help="TDS, ppm")
    p.add_argument("--tur", type=float, required=True, help="Turbidity, NTU")
    p.add_argument("--ph", type=float, required=True)
    p.add_argument("--tds-uncompensated", action="store_true",
                   help="Set if TDS was NOT already compensated to 25 C")
    a = ap.parse_args()

    if a.cmd == "train":
        train(a.csv)
    else:
        print(json.dumps(predict(a.temp, a.tds, a.tur, a.ph,
                                 tds_compensated=not a.tds_uncompensated),
                         indent=2))
