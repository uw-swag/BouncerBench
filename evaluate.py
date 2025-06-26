import json
import sys
import shutil
import yaml
import os
import re
from typing import Optional, Any
from pydantic import BaseModel, ValidationError
from rich.prompt import Confirm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

console = Console()

# --- Pydantic Models ---

class InputBounce(BaseModel):
    input_bounce: bool
    trace: Optional[Any]

class OutputBounce(BaseModel):
    output_bounce: bool
    trace: Optional[Any]

class Lite(BaseModel):
    input_bounce: bool
    output_bounce: bool
    trace: Any

# --- Metadata Parsing ---

def slugify(s):
  s = s.lower().strip()
  s = re.sub(r'[^\w\s-]', '', s)
  s = re.sub(r'[\s_-]+', '-', s)
  s = re.sub(r'^-+|-+$', '', s)
  return s

def parse_metadata(path="config.yaml"):
    try:
        with open(path, "r") as f:
            meta = yaml.safe_load(f)
        if meta is None:
            console.print("[bold red]Error:[/bold red] config file is empty or invalid")
            sys.exit(1)
        
        required = ["name", "oss", "site", "entries", "icon"]
        for field in required:
            if field not in meta or not meta[field]:
                console.print(f"[bold red]Error:[/bold red] config file missing required field: [yellow]{field}[/yellow]")
                sys.exit(1)
        if not os.path.exists(f"static/images/{meta['icon']}"):
            console.print(f"[bold red]Error:[/bold red] Icon file [yellow]{meta['icon']}[/yellow] does not exist at [yellow]static/images/[/yellow]")
            sys.exit(1)

        any_required = ["lite", "input", "output"]
        if not any(meta["entries"].get(field, False) for field in any_required):
            console.print("[bold red]Error:[/bold red] config file must specify at least one entry type: [yellow]lite[/yellow], [yellow]input[/yellow], or [yellow]output[/yellow]")
            sys.exit(1)
        should_overwrite = False
        for leaderboard in ["lite", "input", "output"]:
            name = slugify(meta["name"])        
            trace_dir = f"evaluation/{leaderboard}/{name}"
            if os.path.exists(trace_dir):
                # console.print(f"[bold red]Error:[/bold red] Submission ID [yellow]{meta['name']}[/yellow] is taken. Please choose a different name.")
                # sys.exit(1)
                if not should_overwrite:
                    should_overwrite = Confirm.ask(
                        f"Traces for [yellow]{meta['name']}[/yellow] already exist in [yellow]{trace_dir}[/yellow]. Do you want to overwrite them?",
                        default=False
                    )
                    if not should_overwrite:
                        console.print("[bold red]Error:[/bold red] Please choose a different submission name")
                        sys.exit(1)
                    else:
                        console.print(f"[bold yellow]Warning:[/bold yellow] Overwriting existing traces for [yellow]{meta['name']}[/yellow] in [yellow]{trace_dir}[/yellow]")
                        shutil.rmtree(trace_dir, ignore_errors=True)
                else:
                    console.print(f"[bold yellow]Warning:[/bold yellow] Overwriting existing traces for [yellow]{meta['name']}[/yellow] in [yellow]{trace_dir}[/yellow]")
                    shutil.rmtree(trace_dir, ignore_errors=True)
        return meta
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Config file [yellow]{path}[/yellow] not found")
        sys.exit(1)

# --- Trace Dumping ---

def dump_trace(trace, path_base):
    os.makedirs(os.path.dirname(path_base), exist_ok=True)
    if isinstance(trace, str):
        trace = {"trace": trace}
    with open(path_base + ".json", "w") as f:
        json.dump(trace, f, indent=2)
    return path_base + ".json"

# --- Data Validation & Trace Extraction ---

def validate_and_dump(data, leaderboard, meta):
    name = slugify(meta["name"])
    Model = {"input": InputBounce, "output": OutputBounce, "lite": Lite}[leaderboard]
    valid = 0
    os.makedirs(f"evaluation/{leaderboard}/{name}/traces", exist_ok=True)
    for instance_id, entry in data.items():
        try:
            obj = Model.model_validate(entry)
            trace = obj.trace
            trace_base = f"evaluation/{leaderboard}/{name}/traces/{instance_id}/trace"
            if trace is not None:
                dump_trace(trace, trace_base)
                valid += 1
        except ValidationError as e:
            console.print(f"[red]Validation error for {instance_id}:[/red] {e}")
            raise
    return valid

# --- Metrics Computation ---

def compute_i_score(y_pred, y_true):
    N = len(y_true)
    s = 0.0
    for iid in y_true.index:
        b = y_pred.get(iid, 0)
        lbl = y_true.loc[iid]
        s += ((-1)**(1 - b)) * (lbl - 1.5)
    return (2/3) * (s / N)

def compute_o_score(y_pred, y_true, pass_frac):
    N = len(y_true)
    s = 0.0
    for iid in y_true.index:
        b = y_pred.get(iid, 0)
        c = y_true.loc[iid]
        pf = pass_frac.loc[iid]
        s += ((-1)**(b + c)) * pf
    return s / N

def run_evaluation(data, leaderboard):
    input_metrics = output_metrics = None
    df = pd.DataFrame(data).T
    y_pred_input = None
    if leaderboard in ("input", "lite"):
        ds_input = load_dataset("uw-swag/input-bouncer", split="test")
        inp_df = pd.DataFrame(ds_input).set_index('instance_id')
        y_true_input = inp_df['input_bounce'].astype(int)
        y_pred_input = df['input_bounce'].astype(int)
        idx = y_true_input.index.intersection(y_pred_input.index)
        input_metrics = {
            "prec_accept": precision_score(y_true_input.loc[idx], y_pred_input.loc[idx], pos_label=0, zero_division=0),
            "rec_accept": recall_score(y_true_input.loc[idx], y_pred_input.loc[idx], pos_label=0, zero_division=0),
            "f_accept": f1_score(y_true_input.loc[idx], y_pred_input.loc[idx], pos_label=0, zero_division=0),
            "prec_bounce": precision_score(y_true_input.loc[idx], y_pred_input.loc[idx], pos_label=1, zero_division=0),
            "rec_bounce": recall_score(y_true_input.loc[idx], y_pred_input.loc[idx], pos_label=1, zero_division=0),
            "f_bounce": f1_score(y_true_input.loc[idx], y_pred_input.loc[idx], pos_label=1, zero_division=0),
            "f_macro": f1_score(y_true_input.loc[idx], y_pred_input.loc[idx], average='macro', zero_division=0),
            "i_score": compute_i_score(y_pred_input.loc[idx], inp_df['input_quality'].loc[idx])
        }

    if leaderboard in ("output", "lite"):
        ds_output = load_dataset("uw-swag/output-bouncer", split="test")
        out_df = pd.DataFrame(ds_output).set_index('instance_id')
        y_true_output = out_df['output_bounce'].astype(int)

        y_pred_output = df['output_bounce'].astype(int)
        idx_out = y_true_output.index.intersection(y_pred_output.index)
        output_metrics = {
            "prec_accept": precision_score(y_true_output.loc[idx_out], y_pred_output.loc[idx_out], pos_label=0, zero_division=0),
            "rec_accept": recall_score(y_true_output.loc[idx_out], y_pred_output.loc[idx_out], pos_label=0, zero_division=0),
            "f_accept": f1_score(y_true_output.loc[idx_out], y_pred_output.loc[idx_out], pos_label=0, zero_division=0),
            "prec_bounce": precision_score(y_true_output.loc[idx_out], y_pred_output.loc[idx_out], pos_label=1, zero_division=0),
            "rec_bounce": recall_score(y_true_output.loc[idx_out], y_pred_output.loc[idx_out], pos_label=1, zero_division=0),
            "f_bounce": f1_score(y_true_output.loc[idx_out], y_pred_output.loc[idx_out], pos_label=1, zero_division=0),
            "f_macro": f1_score(y_true_output.loc[idx_out], y_pred_output.loc[idx_out], average='macro', zero_division=0),
            "o_score": compute_o_score(y_pred_output.loc[idx_out], y_true_output.loc[idx_out], out_df['output_quality'].loc[idx_out])
        }

    return input_metrics, output_metrics

def print_summary(leaderboard, meta, valid, total, input_metrics, output_metrics):
    table = Table(title="Submission Summary")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("Name", meta["name"])
    table.add_row("OSS", str(meta["oss"]))
    table.add_row("Site", meta["site"])
    table.add_row("Valid Entries", f"{valid}/{total}")
    if valid < total:
        console.print(f"[bold yellow]Warning:[/bold yellow] Some entries do not have valid traces ({total - valid}/{total}). Please fix them before submitting.")
    table.add_row("Icon", f"./static/images/{meta['icon']}")
    
    # Create metrics JSON data
    metrics_json = {
        "model": meta["name"],
        "img_src": f"static/images/{meta['icon']}",
        "link": meta["site"]
    }
    
    if input_metrics:
        table.add_row("I-Score", f"{input_metrics['i_score']:.3f}")
        table.add_row("Input Macro F1", f"{input_metrics['f_macro']:.3f}")
        table.add_row("Input Bounce Recall (%)", f"{input_metrics['rec_bounce'] * 100:.1f}%")
        table.add_row("Input FNR_a (%)", f"{(100 - input_metrics['rec_accept'] * 100):.1f}%")
        _metrics_json = {}
        _metrics_json["I_Score"] = round(input_metrics['i_score'], 3)
        _metrics_json["F_m"] = round(input_metrics['f_macro'], 3)
        _metrics_json["R_b"] = round(input_metrics['rec_bounce'] * 100, 1)
        _metrics_json["FNR_a"] = round((100 - input_metrics['rec_accept'] * 100), 1)
        if leaderboard == "lite":
            metrics_json["input_metrics"] = _metrics_json
        else:
            metrics_json.update(_metrics_json)
        
    if output_metrics:
        table.add_row("O-Score", f"{output_metrics['o_score']:.3f}")
        table.add_row("Output Macro F1", f"{output_metrics['f_macro']:.3f}")
        table.add_row("Output Bounce Recall (%)", f"{output_metrics['rec_bounce'] * 100:.1f}%")
        table.add_row("Output FNR_a (%)", f"{(100 - output_metrics['rec_accept'] * 100):.1f}%")
        _metrics_json = {}
        _metrics_json["O_Score"] = round(output_metrics['o_score'], 3)
        _metrics_json["F_m"] = round(output_metrics['f_macro'], 3)
        _metrics_json["R_b"] = round(output_metrics['rec_bounce'] * 100, 1)
        _metrics_json["FNR_a"] = round((100 - output_metrics['rec_accept'] * 100), 1)
        if leaderboard == "lite":
            metrics_json["output_metrics"] = _metrics_json
        else:
            metrics_json.update(_metrics_json)
    
    # Dump metrics to JSON file
    name = slugify(meta["name"])
    metrics_dir = f"evaluation/{leaderboard}/{name}"
    os.makedirs(metrics_dir, exist_ok=True)
    with open(f"{metrics_dir}/metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=4)
    
    console.print(Panel(table, title=f"{leaderboard} Submission", expand=False))

# --- Main ---
def main():
    meta = parse_metadata()
    try:
        for leaderboard, file_loc in meta["entries"].items():
            if file_loc is None:
                continue
            with open(f"data/{file_loc}", "r") as f:
                data = json.load(f)
            valid = validate_and_dump(data, leaderboard, meta)
            input_metrics, output_metrics = run_evaluation(data, leaderboard)
            print_summary(leaderboard, meta, valid, len(data), input_metrics, output_metrics)
    except Exception as e:
        # delete entry in evaluation directory if it exists
        name = slugify(meta["name"])
        for leaderboard in ["lite", "input", "output"]:
            trace_dir = f"evaluation/{leaderboard}/{name}/traces"
            if os.path.exists(trace_dir):
                # remove the directory and its contents
                shutil.rmtree(trace_dir, ignore_errors=True)
        if not isinstance(e, ValidationError):
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()