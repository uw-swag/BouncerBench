import json
import pandas as pd

def parse_frac(ts):
    d = eval(ts)
    succ = len(d['FAIL_TO_PASS']['success']) + len(d['PASS_TO_PASS']['success'])
    total = succ + len(d['FAIL_TO_PASS']['failure']) + len(d['PASS_TO_PASS']['failure'])
    return succ/total if total > 0 else 0.0

def load_ground_truth(csv_file, input_bouncer=False):
    out_df = pd.read_csv(csv_file).set_index('instance_id')
    if input_bouncer:
        return (out_df['underspecified'] >= 2).astype(int)
    else:
        pass_frac = out_df['tests_status'].apply(parse_frac)
        # 1 => patch is incorrect (should bounce), 0 => patch correct (should accept)
        return (pass_frac < 1.0).astype(int)

def load_output_preds(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return {
        iid: (1 if info['label'] in ['BROAD_MISSING_KEY_ASPECTS','INCORRECT'] else 0)
        for iid, info in data.items()
    }

def load_input_preds(json_file):
    with open(json_file) as f:
        data = json.load(f)
    preds = {
        iid: (1 if info['label'] in ['VAGUE','IMPOSSIBLE_TO_SOLVE'] else 0)
        for iid, info in data.items()
    }
    return preds

def count_function_calls(trace):
    return sum(1 for step in trace if step.get('type') == 'function_call')

def compare_flips(_csv, o4_json, codex_json, input_bouncer=False):
    if input_bouncer:
        print("Input Bouncer Analysis")
        gts = load_ground_truth(_csv, input_bouncer=True)
        pred_o4 = load_input_preds(o4_json)
        pred_codex = load_input_preds(codex_json)
    else:
        print("Output Bouncer Analysis")
        gts = load_ground_truth(_csv)
        pred_o4 = load_output_preds(o4_json)
        pred_codex = load_output_preds(codex_json)
    codex_data = json.load(open(codex_json))

    all_ids = set(gts.index) & set(pred_o4) & set(pred_codex)

    total_cases = len(all_ids)
    cases_with_fc = sum(
        1
        for iid in all_ids
        if count_function_calls(codex_data[iid].get('trace', [])) >= 1
    )
    print(f"Across all {total_cases} cases:")
    print(f"  • cases with ≥1 function_call in trace: {cases_with_fc} "
          f"({cases_with_fc/total_cases:.1%})\n")

    rows = []
    for iid in all_ids:
        p_o4 = pred_o4[iid]
        p_cx = pred_codex[iid]
        if p_o4 != p_cx:
            gt = gts.loc[iid]
            mistake = (p_cx != gt)
            fcalls = count_function_calls(codex_data[iid].get('trace', []))
            rows.append({
                'instance_id':  iid,
                'o4_pred':      p_o4,
                'codex_pred':   p_cx,
                'ground_truth': gt,
                'mistake':      mistake,
                'func_calls':   fcalls
            })

    df = pd.DataFrame(rows).set_index('instance_id')

    total_flips = len(df)
    total_mistakes = df['mistake'].sum()
    total_correct = total_flips - total_mistakes

    print(f"Total flips (o4 → codex): {total_flips}")
    print(f"  • mistakes introduced: {total_mistakes} ({total_mistakes/total_flips:.1%})")
    print(f"  • corrections      : {total_correct} ({total_correct/total_flips:.1%})\n")

    mistakes_df = df[df['mistake']]
    mm = len(mistakes_df)
    mm_fc = (mistakes_df['func_calls'] >= 1).sum()
    print(f"Mistaken flips with ≥1 function_call: {mm_fc} / {mm} ({mm_fc/mm:.1%})")

    correct_df = df[~df['mistake']]
    cc = len(correct_df)
    cc_fc = (correct_df['func_calls'] >= 1).sum()
    print(f"Correct flips with ≥1 function_call: {cc_fc} / {cc} ({cc_fc/cc:.1%})\n")

    print("→ Mistake flips by type with ≥1 function_call:")
    print(
        mistakes_df[mistakes_df['func_calls'] >= 1]
        .groupby(['o4_pred','codex_pred'])
        .size()
    )
    print("\n→ Correct flips by type with ≥1 function_call:")
    print(
        correct_df[correct_df['func_calls'] >= 1]
        .groupby(['o4_pred','codex_pred'])
        .size()
    )

if __name__ == "__main__":
    compare_flips(
        "./dataset/input_bouncer.csv",
        "./outputs/input_bouncer_o4-mini.json",
        "./outputs/input_bouncer_codex.json",
        input_bouncer = True
    )

    compare_flips(
        "./dataset/random_sample_bouncer.csv",
        "./outputs/output_bouncer_o4-mini.json",
        "./outputs/output_bouncer_codex.json",
    )