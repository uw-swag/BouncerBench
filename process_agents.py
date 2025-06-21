import json
import pandas as pd

def load_output_preds(json_file):
    with open(json_file) as f:
        data = json.load(f)
    preds = {
        iid: (1 if info['label'] in ['BROAD_MISSING_KEY_ASPECTS','INCORRECT'] else 0)
        for iid, info in data.items()
    }
    return preds

def load_input_preds(json_file):
    with open(json_file) as f:
        data = json.load(f)
    preds = {
        iid: (1 if info['label'] in ['VAGUE','IMPOSSIBLE_TO_SOLVE'] else 0)
        for iid, info in data.items()
    }
    return preds


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


def main(output_csv, output_json):
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
    # out of the 648 evaluatable cases
    pre_number_of_patches = true_correct.count()
    pre_number_of_correct_patches = true_correct[true_correct == 0].count()

    pred = load_output_preds(output_json)
    input_pred = load_input_preds("outputs/input_bouncer_o4-mini.json")
    y_pred = pd.Series(pred)
    yi_pred = pd.Series(input_pred)
    yi_pred = yi_pred[yi_pred == 0]
    idx = true_correct.index.intersection(y_pred.index)
    idx = idx.intersection(yi_pred.index)
    y_pred = y_pred.loc[idx]
    true_correct = true_correct.loc[idx]
    number_of_patches = true_correct.count()
    number_of_correct_patches = true_correct[true_correct == 0].count()
    
    # filter ypred == 0 (accepted patches)
    y_pred = y_pred[y_pred == 0]
    _idx = y_pred.index.intersection(true_correct.index)
    true_correct = true_correct.loc[_idx]
    post_number_of_patches = true_correct.count()
    post_number_of_correct_patches = true_correct[true_correct == 0].count()

    # average length of the patch
    avg_length = out_df.loc[idx, 'patch'].apply(len).mean()
    _number_of_patches = len(out_df.loc[idx, 'patch'])
    print(f"Average length of the patch: {avg_length:.2f} characters over {_number_of_patches} patches")

    print(f"Pre-filtering: {pre_number_of_patches} patches, {pre_number_of_correct_patches} correct patches (Percentage: {pre_number_of_correct_patches/pre_number_of_patches:.2%})")
    print(f"After Input Bouncer: {number_of_patches} patches, {number_of_correct_patches} correct patches (Percentage: {number_of_correct_patches/number_of_patches:.2%})")
    print(f"After Output Bouncer: {post_number_of_patches} patches, {post_number_of_correct_patches} correct patches (Percentage: {post_number_of_correct_patches/post_number_of_patches:.2%})")
    print(f"Percentage of Correct Patches lost: {1 - post_number_of_correct_patches/pre_number_of_correct_patches:.2%}")
if __name__ == '__main__':
    AGENTS = {
        "OpenHands":"20241103_OpenHands-CodeAct-2.1-sonnet-20241022",
        "amazon-q":"20250131_amazon-q-developer-agent-20241202-dev",
        "sweagent":"20250227_sweagent-claude-3-7-20250219",
        "random_sample": "codex"
    }
    for _name, _agent in AGENTS.items():
        if _agent == "codex":
            output_json = "./outputs/output_bouncer_codex.json"
            _csv = "./dataset/random_sample_bouncer.csv"
        else:
            output_json = f"./outputs/output_bouncer_codex_{_agent}.json"
            _csv = f"./dataset/agent_output/{_name}_bouncer.csv"
        print(f"Processing {_name}")
        main(
            output_csv=_csv,
            output_json=output_json
        )
