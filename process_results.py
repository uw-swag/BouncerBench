import os
import json
import glob
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def load_input_preds(json_dir):
    out = {}
    for fn in glob.glob(os.path.join(json_dir, 'input_bouncer_*.json')):
        if "codex_" in fn:
            continue
        model = os.path.basename(fn).split('input_bouncer_')[1].split('.json')[0]
        with open(fn) as f:
            data = json.load(f)
        preds = {
            iid: (1 if info['label'] in ['VAGUE','IMPOSSIBLE_TO_SOLVE'] else 0)
            for iid, info in data.items()
        }
        out[model] = preds
    return out


def load_output_preds(json_dir):
    out = {}
    for fn in glob.glob(os.path.join(json_dir, 'output_bouncer_*.json')):
        if "codex_" in fn:
            continue
        model = os.path.basename(fn).split('output_bouncer_')[1].split('.json')[0]
        with open(fn) as f:
            data = json.load(f)
        preds = {
            iid: (1 if info['label'] in ['BROAD_MISSING_KEY_ASPECTS','INCORRECT'] else 0)
            for iid, info in data.items()
        }
        out[model] = preds
    return out


def compute_i_score(y_pred, y_true):
    N = len(y_true)
    s = 0.0
    for iid in y_true.index:
        b = y_pred.get(iid, 0)
        lbl = y_true.loc[iid]
        # If you bounce (bounce = 1), (-1)**0 = 1
        # If you accept (bounce = 0), (-1)**1 = -1
        # For a well-specified case (label = 0 or 1), (label – 1.5)<0 -> bouncing gives negative score
        # For an ambiguous case (label = 2 or 3), (label – 1.5)>0 -> bouncing gives positive score
        s += ((-1)**(1 - b)) * (lbl - 1.5)
    return (2/3) * (s / N)


def compute_o_score(y_pred, y_true, pass_frac):
    N = len(y_true)
    s = 0.0
    for iid in y_true.index:
        # b is 1 if you bounced (rejected) the patch, 0 if you accepted it.
        b = y_pred.get(iid, 0)
        # c is 1 if the patch is incorrect (pass_frac < 1.0), 0 if it is correct (pass_frac = 1.0).
        c = y_true.loc[iid]
        pf = pass_frac.loc[iid]
        s += ((-1)**(b + c)) * pf
    return s / N


def main(input_csv, output_csv, input_json_dir, output_json_dir):
    # --- Input Bouncer Metrics ---
    inp_df = pd.read_csv(input_csv).set_index('instance_id')
    true_spec = inp_df['underspecified'].astype(int)
    inp_preds = load_input_preds(input_json_dir)
    print("Input Bouncer Metrics")
    for m, pred in inp_preds.items():
        y_pred = pd.Series(pred)
        y_true = (true_spec >= 2).astype(int)
        idx = y_true.index.intersection(y_pred.index)

        # Precision and recall per class
        prec_accept = precision_score(y_true.loc[idx], y_pred.loc[idx], pos_label=0)
        rec_accept  = recall_score(y_true.loc[idx], y_pred.loc[idx], pos_label=0)
        f_accept    = f1_score(y_true.loc[idx], y_pred.loc[idx], pos_label=0)

        prec_bounce = precision_score(y_true.loc[idx], y_pred.loc[idx], pos_label=1)
        rec_bounce  = recall_score(y_true.loc[idx], y_pred.loc[idx], pos_label=1)
        f_bounce    = f1_score(y_true.loc[idx], y_pred.loc[idx], pos_label=1)

        # Macro F1 (average of both classes)
        f_macro = f1_score(y_true.loc[idx], y_pred.loc[idx], average='macro')

        i_score = compute_i_score(pred, true_spec.loc[idx])

        print(f"- {m}\n"
              f"    Accept (0): Prec={prec_accept:.3f}, Rec={rec_accept:.3f}, F1={f_accept:.3f}\n"
              f"    Bounce (1): Prec={prec_bounce:.3f}, Rec={rec_bounce:.3f}, F1={f_bounce:.3f}\n"
              f"    Overall F1 (macro)={f_macro:.3f}, I-Score={i_score:.3f}\n")

    # --- Output Bouncer Metrics ---
    out_df = pd.read_csv(output_csv).set_index('instance_id')
    def parse_frac(ts):
        d = eval(ts)
        succ = len(d['FAIL_TO_PASS']['success']) + len(d['PASS_TO_PASS']['success'])
        total = (succ
                 + len(d['FAIL_TO_PASS']['failure'])
                 + len(d['PASS_TO_PASS']['failure']))
        return succ/total if total>0 else 0.0

    pass_frac = out_df['tests_status'].apply(parse_frac)
    true_correct = (pass_frac < 1.0).astype(int)  # 1 => patch is incorrect => bounce
    out_preds = load_output_preds(output_json_dir)
    print("\nOutput Bouncer Metrics")
    for m, pred in out_preds.items():
        y_pred = pd.Series(pred)
        idx = true_correct.index.intersection(y_pred.index)

        # Precision and recall per class
        prec_accept = precision_score(true_correct.loc[idx], y_pred.loc[idx], pos_label=0)
        rec_accept  = recall_score(true_correct.loc[idx], y_pred.loc[idx], pos_label=0)
        f_accept    = f1_score(true_correct.loc[idx], y_pred.loc[idx], pos_label=0)

        prec_bounce = precision_score(true_correct.loc[idx], y_pred.loc[idx], pos_label=1)
        rec_bounce  = recall_score(true_correct.loc[idx], y_pred.loc[idx], pos_label=1)
        f_bounce    = f1_score(true_correct.loc[idx], y_pred.loc[idx], pos_label=1)

        # Macro F1
        f_macro = f1_score(true_correct.loc[idx], y_pred.loc[idx], average='macro')

        o_score = compute_o_score(pred, true_correct.loc[idx], pass_frac.loc[idx])

        print(f"- {m}\n"
              f"    Accept (0): Prec={prec_accept:.3f}, Rec={rec_accept:.3f}, F1={f_accept:.3f}\n"
              f"    Bounce (1): Prec={prec_bounce:.3f}, Rec={rec_bounce:.3f}, F1={f_bounce:.3f}\n"
              f"    Overall F1 (macro)={f_macro:.3f}, O-Score={o_score:.3f}\n")

if __name__ == '__main__':
    main(
        input_csv="./dataset/input_bouncer.csv",
        output_csv="./dataset/random_sample_bouncer.csv",
        input_json_dir="outputs",
        output_json_dir="outputs",
    )
