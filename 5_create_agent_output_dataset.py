import os
import pandas as pd

AGENTS = {
    "OpenHands":"20241103_OpenHands-CodeAct-2.1-sonnet-20241022",
    "amazon-q":"20250131_amazon-q-developer-agent-20241202-dev",
    "sweagent":"20250227_sweagent-claude-3-7-20250219"
}

OUTPUT_FOLDER = "./dataset/agent_output"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

output_bouncer_df = pd.read_csv("./dataset/random_sample_bouncer.csv")
input_bouncer_df = pd.read_csv("./dataset/input_bouncer.csv")
evaluatable_df = input_bouncer_df[
    (input_bouncer_df["false_negative"] != 2.0)
    & (input_bouncer_df["false_negative"] != 3.0)
    & (input_bouncer_df["other_major_issues"] != 1.0)
]
def get_problem_statement(instance_id):
    return input_bouncer_df[input_bouncer_df['instance_id'] == instance_id]['problem_statement'].values[0]
def get_repo(instance_id):
    return input_bouncer_df[input_bouncer_df['instance_id'] == instance_id]['repo'].values[0]
def get_base_commit(instance_id):
    return input_bouncer_df[input_bouncer_df['instance_id'] == instance_id]['base_commit'].values[0]

def get_folders(dir):
    """
    Returns a list of folders in the given directory.
    """
    return [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]

for _name, _agent in AGENTS.items():
    df = pd.read_csv("./data/all_patches.csv")
    df = df[df['benchmark'] == "test"]
    df = df[df['agent'] == _agent]
    # Filter out rows where 'instance_id' is not in 'output_bouncer_df'
    df = df[df["instance_id"].isin(evaluatable_df["instance_id"])]

    df['problem_statement'] = df['instance_id'].apply(get_problem_statement)
    df['repo'] = df['instance_id'].apply(get_repo)
    df["base_commit"] = df["instance_id"].apply(get_base_commit)
    df.to_csv(os.path.join(OUTPUT_FOLDER, f'{_name}_bouncer.csv'), index=False)
    print(f"Saved {_name} bouncer dataset with {len(df)} rows to {OUTPUT_FOLDER}/{_name}_bouncer.csv")
    