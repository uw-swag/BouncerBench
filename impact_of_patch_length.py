import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_output_preds(json_dir):
    out = {}
    pattern = os.path.join(json_dir, "output_bouncer_*.json")
    for fn in glob.glob(pattern):
        if "codex_" in fn:
            continue
        model = os.path.basename(fn).split("output_bouncer_")[1].split(".json")[0]
        with open(fn) as f:
            data = json.load(f)
        preds = {
            iid: (
                1 if info["label"] in ["BROAD_MISSING_KEY_ASPECTS", "INCORRECT"] else 0
            )
            for iid, info in data.items()
        }
        out[model] = preds
    return out


def bin_char_counts(df):
    df = df.copy()
    df["char_count"] = df["patch"].str.len()
    # new bins at 1000 and 2500
    bins = [0, 1000, 2500, np.inf]
    labels = ["<=1000", "1001–2500", ">2500"]
    df["length_bin"] = pd.cut(
        df["char_count"], bins=bins, labels=labels, include_lowest=True
    )
    return df


def gather_data(output_csv, json_dir):
    df = pd.read_csv(output_csv).set_index("instance_id")
    df = bin_char_counts(df)

    def parse_frac(ts):
        d = eval(ts)
        succ = len(d["FAIL_TO_PASS"]["success"]) + len(d["PASS_TO_PASS"]["success"])
        total = (
            succ + len(d["FAIL_TO_PASS"]["failure"]) + len(d["PASS_TO_PASS"]["failure"])
        )
        return succ / total if total > 0 else 0

    df["pass_frac"] = df["tests_status"].apply(parse_frac)
    df["true_bounce"] = (df["pass_frac"] < 1.0).astype(int)

    preds = load_output_preds(json_dir)
    rows = []
    for model, mp in preds.items():
        for iid, pred in mp.items():
            if iid not in df.index:
                continue
            entry = df.loc[iid]
            if pred == 1:
                rows.append(
                    {
                        "model": model,
                        "length_bin": entry["length_bin"],
                        "correct": int(entry["true_bounce"] == 1),
                        "incorrect": int(entry["true_bounce"] == 0),
                    }
                )
    rem_df = pd.DataFrame(rows)
    return df, rem_df


def compute_rates(df_full, rem_df):
    bins = ["<=1000", "1001–2500", ">2500"]
    # P = # bad patches per bin
    P = (
        df_full.groupby("length_bin", observed=False)["true_bounce"]
        .sum()
        .reindex(bins, fill_value=0)
    )
    # N = # good patches per bin
    total = df_full["length_bin"].value_counts().reindex(bins, fill_value=0)
    bad_frac = P / total
    print(f"Bad patch fractions by bin:\n{bad_frac}")
    N = total - P

    grp = (
        rem_df.groupby(["length_bin", "model"], observed=False)
        .agg(correct=("correct", "sum"), incorrect=("incorrect", "sum"))
        .reset_index()
    )

    rates = []
    for _, row in grp.iterrows():
        b, m = row["length_bin"], row["model"]
        tp = row["correct"] / P[b] if P[b] > 0 else 0
        fp = row["incorrect"] / N[b] if N[b] > 0 else 0
        rates.append({"length_bin": b, "model": m, "tp_rate": tp, "fp_rate": fp})
    return pd.DataFrame(rates)


def plot_rates(rates_df, df_full, out_file):
    bins = ["<=1000", "1001–2500", ">2500"]
    order = [
        "gemma3_27b-it-q8_0",
        "claude-3.7-sonnet",
        "gpt-4.1",
        "qwen3_32b-q8_0",
        "o4-mini",
        "codex",
    ]
    models = [m for m in order if m in rates_df["model"].unique()]

    x = np.arange(len(bins))
    width = 0.8 / len(models)
    total = df_full["length_bin"].value_counts().reindex(bins, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(models):
        sub = rates_df[rates_df["model"] == m].set_index("length_bin")
        tp = sub["tp_rate"].reindex(bins, fill_value=0).values
        fp = -sub["fp_rate"].reindex(bins, fill_value=0).values
        ax.bar(x + i * width, tp, width, label=f"{m} TPR")
        ax.bar(x + i * width, fp, width, label=f"{m} FPR", alpha=0.7)
    _remap_labels = {
        "<=1000": "≤100",
        "1001–2500": "≤2500",
        ">2500": ">2500",
    }
    ax.axhline(0, color="gray")
    labels = [f"{_remap_labels[b]} (N={total[b]})" for b in bins]
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Patch length (chars)")
    ax.set_ylabel("Rate")
    ax.legend(ncol=3)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    output_csv = "./dataset/random_sample_bouncer.csv"
    output_json_dir = "outputs"
    out_image = "output_bouncer_fp_tp_rate.pdf"

    df_full, rem_df = gather_data(output_csv, output_json_dir)
    rates_df = compute_rates(df_full, rem_df)
    plot_rates(rates_df, df_full, out_image)
    print(f"Saved chart to {out_image}")


if __name__ == "__main__":
    main()
