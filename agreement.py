import os
import glob
import json
import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

LABEL_TO_IDX = {
    "WELL_SPECIFIED": 0,
    "REASONABLY_SPECIFIED": 1,
    "VAGUE": 2,
    "IMPOSSIBLE_TO_SOLVE": 3,
}


def load_model_preds(json_dir: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    pat = os.path.join(json_dir, "input_bouncer_*.json")
    for fn in glob.glob(pat):
        model = os.path.basename(fn).split("input_bouncer_")[1].split(".json")[0]
        with open(fn, "r") as f:
            data = json.load(f)

        preds = {}
        for iid, info in data.items():
            raw = info["label"]
            preds[iid] = LABEL_TO_IDX.get(raw, -1)  # -1 => unknown
        out[model] = preds
    return out


def percentage_agreement(y_pred: pd.Series, y_true: pd.Series) -> float:
    idx = y_true.index.intersection(y_pred.index)
    if len(idx) == 0:
        return float("nan")
    return (y_true.loc[idx] == y_pred.loc[idx]).mean()


def kappa_and_rho(y_pred: pd.Series, y_true: pd.Series) -> tuple[float, float]:
    idx = y_true.index.intersection(y_pred.index)
    if len(idx) == 0:
        return float("nan"), float("nan")
    kappa = cohen_kappa_score(y_true.loc[idx], y_pred.loc[idx])
    rho, _ = spearmanr(y_true.loc[idx], y_pred.loc[idx])
    return kappa, rho


def main(input_csv: str, json_dir: str) -> None:
    # human labels (0-3)
    gt = (
        pd.read_csv(input_csv)
        .set_index("instance_id")["underspecified"]
        .astype(int)
    )

    # model predictions
    preds_by_model = load_model_preds(json_dir)

    print("Agreement with human annotations\n")
    for model, preds in preds_by_model.items():
        y_pred = pd.Series(preds)
        agree = percentage_agreement(y_pred, gt)
        kappa, rho = kappa_and_rho(y_pred, gt)

        print(f"- {model}")
        print(f"    Agreement = {agree:.3f}")
        print(f"    Cohen kappa = {kappa:.3f}")
        print(f"    Spearman rho = {rho:.3f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="./dataset/input_bouncer.csv",
                        help="CSV with ground-truth labels")
    parser.add_argument("--json_dir", default="outputs",
                        help="folder holding input_bouncer_*.json files")
    args = parser.parse_args()
    main(args.input_csv, args.json_dir)