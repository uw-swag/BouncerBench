import os
import json
import pandas as pd
from tqdm import tqdm

BENCHMARKS = ["verified", "test", "lite"]

def get_folders(dir):
    """
    Returns a list of folders in the given directory.
    """
    return [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]

data = []

for _benchmark in tqdm(BENCHMARKS, desc="Processing benchmarks"):
    experiment_folder = "../experiments/evaluation/" + _benchmark
    for folder in tqdm(get_folders(experiment_folder), desc="Processing folders", leave=False):
        logs_folder = os.path.join(experiment_folder, folder, "logs")
        if not os.path.exists(logs_folder):
            continue
        # iterate through folders in logs_folder with tqdm
        for log in tqdm(get_folders(logs_folder), desc="Processing logs", leave=False):
            report_path = os.path.join(logs_folder, log, "report.json")
            try:
                with open(report_path) as f:
                    report = json.load(f)[log]
            except Exception:
                continue
            if not report["patch_successfully_applied"]:
                continue
            _resolved = report["resolved"]
            _patch_path = os.path.join(logs_folder, log, "patch.diff")
            _patch = open(_patch_path, "r").read()
            _tests_status = report["tests_status"]
            data.append({
                "instance_id": log,
                "agent": folder,
                "patch": _patch,
                "resolved": _resolved,
                "tests_status": _tests_status,
                "benchmark": _benchmark,
            })
            
df = pd.DataFrame(data)
# save to csv
df.to_csv("./data/all_patches.csv", index=False)