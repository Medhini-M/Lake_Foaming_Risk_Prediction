"""
Interactive predictor for the lake pollution + foaming risk models.

Run lake_sensor_model.py train first so that lake_sensor_models.pkl exists
in the same folder, then:
    python predict_lake.py
"""
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "lake_sensor_models.pkl"


def ask_float(prompt, lo, hi):
    """Keep asking until the user types a number within [lo, hi]."""
    while True:
        raw = input(prompt).strip()
        try:
            v = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if not lo <= v <= hi:
            print(f"  Value must be between {lo} and {hi}.")
            continue
        return v


def ask_yes_no(prompt, default=True):
    raw = input(prompt).strip().lower()
    if raw == "":
        return default
    return raw.startswith("y")


def predict(bundle, temp_c, tds_ppm, turbidity_ntu, ph, tds_compensated=True):
    # TDS (ppm) -> EC (uS/cm) -> S/m, the unit used in the training dataset
    ec_us_cm = tds_ppm / 0.5
    if not tds_compensated:
        ec_us_cm /= 1.0 + 0.02 * (temp_c - 25.0)
    cond = ec_us_cm * 1e-4

    row = pd.DataFrame([{"pH": ph, "Cond": cond, "Tur": turbidity_ntu}])
    row["pH_dev"] = (row["pH"] - 7.75).abs()

    warnings = []
    for c, (lo, hi) in bundle["train_range"].items():
        v = float(row[c].iloc[0])
        if v < lo or v > hi:
            warnings.append(f"{c} = {v:.4g} is outside the training range "
                            f"[{lo:.4g}, {hi:.4g}]; result is extrapolated.")
    X = row[bundle["features"]]

    sm = bundle["state_model"]
    sp = dict(zip(sm.classes_, sm.predict_proba(X)[0]))
    state = max(sp, key=sp.get)

    fm = bundle["foam_model"]
    fp = dict(zip(fm.classes_, fm.predict_proba(X)[0]))
    base_index = 100 * (0.5 * fp.get("Medium", 0) + fp.get("High", 0))
    temp_factor = float(np.clip(1 + 0.015 * (temp_c - 25), 0.85, 1.25))
    fri = float(np.clip(base_index * temp_factor, 0, 100))
    foam = "Low" if fri < 100 / 3 else ("Medium" if fri < 200 / 3 else "High")

    return state, sp[state], foam, fri, fp, warnings


def main():
    try:
        bundle = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"'{MODEL_PATH}' not found. Run: python lake_sensor_model.py train")
        return

    print("=" * 55)
    print(" Lake Pollution & Foaming Early Warning - Manual Input")
    print("=" * 55)

    while True:
        temp = ask_float("Water temperature (C)      : ", 0, 50)
        tds = ask_float("TDS (ppm)                  : ", 0, 5000)
        tur = ask_float("Turbidity (NTU)            : ", 0, 3000)
        ph = ask_float("pH                         : ", 0, 14)
        comp = ask_yes_no("Is TDS already temperature-compensated? [Y/n]: ")

        state, s_conf, foam, fri, fp, warnings = predict(
            bundle, temp, tds, tur, ph, tds_compensated=comp)

        print("\n----------------- RESULT -----------------")
        print(f" Water state     : {state}  (confidence {s_conf:.0%})")
        print(f" Foaming risk    : {foam}")
        print(f" Foaming index   : {fri:.1f} / 100")
        print(" Foam class probs: " +
              ", ".join(f"{k} {v:.0%}" for k, v in fp.items()))
        for w in warnings:
            print(f" WARNING: {w}")
        print("------------------------------------------\n")

        if not ask_yes_no("Check another reading? [Y/n]: "):
            print("Done.")
            break


if __name__ == "__main__":
    main()
