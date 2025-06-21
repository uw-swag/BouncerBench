import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_input_preds(json_dir):
    out = {}
    for fn in glob.glob(os.path.join(json_dir, "input_bouncer_*.json")):
        if "codex_" in fn:
            continue
        model = os.path.basename(fn).split("input_bouncer_")[1].split(".json")[0]
        with open(fn) as f:
            data = json.load(f)
        preds = {
            iid: (1 if info["label"] in ["VAGUE", "IMPOSSIBLE_TO_SOLVE"] else 0)
            for iid, info in data.items()
        }
        out[model] = preds
    return out


def bin_word_counts(df):
    df = df.copy()
    df["word_count"] = df["problem_statement"].str.split().str.len()
    bins = [0, 100, 200, 500, np.inf]
    labels = ["<=100", "101–200", "201–500", ">500"]
    df["length_bin"] = pd.cut(
        df["word_count"], bins=bins, labels=labels, include_lowest=True
    )
    return df


def gather_data(input_csv, input_json_dir):
    df_full = pd.read_csv(input_csv).set_index("instance_id")
    df_full = bin_word_counts(df_full)
    df_full["true_bounce"] = (df_full["underspecified"] >= 2).astype(int)
    preds = load_input_preds(input_json_dir)
    rows = []
    for model, mp in preds.items():
        for iid, pred in mp.items():
            if iid not in df_full.index:
                continue
            entry = df_full.loc[iid]
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
    return df_full, rem_df


def compute_rates(df_full, rem_df):
    tot_remove = df_full.groupby("length_bin", observed=False)["true_bounce"].sum()
    tot_keep = df_full.groupby("length_bin", observed=False).size() - tot_remove

    grp = (
        rem_df.groupby(["length_bin", "model"], observed=False)
        .agg(correct=("correct", "sum"), incorrect=("incorrect", "sum"))
        .reset_index()
    )

    rates = []
    for _, row in grp.iterrows():
        b, m = row["length_bin"], row["model"]
        tp = row["correct"] / tot_remove[b] if tot_remove[b] > 0 else 0
        fp = row["incorrect"] / tot_keep[b] if tot_keep[b] > 0 else 0
        rates.append({"length_bin": b, "model": m, "tp_rate": tp, "fp_rate": fp})
    return pd.DataFrame(rates)


def plot_rates(rates_df, df_full, out_file):
    bins = ["<=100", "101–200", "201–500", ">500"]
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
    totals = df_full["length_bin"].value_counts().reindex(bins, fill_value=0)
    tot_remove = (
        df_full.groupby("length_bin", observed=False)["true_bounce"]
        .sum()
        .reindex(bins, fill_value=0)
    )
    print(tot_remove)
    ambig_frac = tot_remove / totals
    print(f"Ambiguous fraction by bin:\n{ambig_frac}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(models):
        sub = rates_df[rates_df["model"] == m].set_index("length_bin")
        tp = sub["tp_rate"].reindex(bins, fill_value=0).values
        fp = -sub["fp_rate"].reindex(bins, fill_value=0).values
        ax.bar(x + i * width, tp, width, label=f"{m} TPR")
        ax.bar(x + i * width, fp, width, label=f"{m} FPR", alpha=0.7)

    ax.axhline(0, color="gray")
    # ax.plot(
    #     x + width*(len(models)-1)/2,
    #     ambig_frac.values,
    #     marker='o',
    #     linestyle='--',
    #     linewidth=1.5,
    #     label='Ambiguous fraction'
    # )
    _remap_labels = {
        "<=100": "≤100",
        "101–200": "≤200",
        "201–500": "≤500",
        ">500": ">500",
    }
    labels = [f"{_remap_labels[b]} (N={totals[b]})" for b in bins]
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Rate")
    ax.set_xlabel("Issue Length (word count)")
    ax.legend(ncol=2)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    df_full, rem_df = gather_data("./dataset/input_bouncer.csv", "outputs")
    rates_df = compute_rates(df_full, rem_df)
    plot_rates(rates_df, df_full, "input_bouncer_fp_tp_rate.pdf")
    print("Saved chart to input_bouncer_fp_tp_rate.pdf")


if __name__ == "__main__":
    main()
