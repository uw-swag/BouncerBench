import json
import pandas as pd
import plotly.graph_objects as go

# --- CONFIG: adjust these paths/models as needed ---
INPUT_CSV = "./dataset/input_bouncer.csv"
INPUT_PRED_JSON = "outputs/input_bouncer_o4-mini.json"
OUTPUT_CSV = "./dataset/random_sample_bouncer.csv"
OUTPUT_PRED_JSON = "outputs/output_bouncer_codex.json"

df_in = pd.read_csv(INPUT_CSV).set_index("instance_id")
truth_in = (df_in["underspecified"] >= 2).astype(int)  # 1=should bounce
with open(INPUT_PRED_JSON) as f:
    inp_data = json.load(f)
pred_in = pd.Series({
    iid: int(info["label"] in ["VAGUE","IMPOSSIBLE_TO_SOLVE"])
    for iid, info in inp_data.items()
})

df_out = pd.read_csv(OUTPUT_CSV).set_index("instance_id")
def parse_frac(ts):
    d = eval(ts)
    succ = len(d["FAIL_TO_PASS"]["success"]) + len(d["PASS_TO_PASS"]["success"])
    tot = succ + len(d["FAIL_TO_PASS"]["failure"]) + len(d["PASS_TO_PASS"]["failure"])
    return succ/tot if tot>0 else 0.0

pass_frac = df_out["tests_status"].apply(parse_frac)
truth_out = (pass_frac < 1.0).astype(int)  # 1=incorrect→should bounce
with open(OUTPUT_PRED_JSON) as f:
    out_data = json.load(f)
pred_out  = pd.Series({
    iid: int(info["label"] in ["BROAD_MISSING_KEY_ASPECTS","INCORRECT"])
    for iid, info in out_data.items()
})

common    = list(
    set(truth_in.index) & set(pred_in.index) &
    set(truth_out.index)& set(pred_out.index)
)
truth_in  = truth_in.loc[common]
pred_in   = pred_in.loc[common]
truth_out = truth_out.loc[common]
pred_out  = pred_out.loc[common]
total     = len(common)

eventual_ok = (truth_out == 0)  # True if patch passes all tests

IB_ok   = ((pred_in==1)& eventual_ok).sum()
IB_fail = ((pred_in==1)&~eventual_ok).sum()
IP_ok   = ((pred_in==0)& eventual_ok).sum()
IP_fail = ((pred_in==0)&~eventual_ok).sum()

IBC = ((pred_in==1)&(truth_in==1)).sum()
IBW = ((pred_in==1)&(truth_in==0)).sum()
IPC = ((pred_in==0)&(truth_in==0)).sum()
IPW = ((pred_in==0)&(truth_in==1)).sum()

print(f"Input Bounced: {IB_ok} correct, {IB_fail} wrong")

passed_ids = pred_in[pred_in==0].index
ti         = truth_out.loc[passed_ids]
pi         = pred_out.loc[passed_ids]

OB_ok   = ((pi==1)&(ti==0)).sum()
OB_fail = ((pi==1)&(ti==1)).sum()
OP_ok   = ((pi==0)&(ti==0)).sum()
OP_fail = ((pi==0)&(ti==1)).sum()

_nodes = [
    (f"All\n({total})", 0, 0.2),
    (f"Input Bounce</br></br>Clear:{IBW}</br>Vague:{IBC}", 0.1, 0.6),
    (f"Input Pass</br></br>Clear:{IPC}</br>Vague:{IPW}", 0.1, 0.4),
    # (f"Projected Agent Solved:{IB_ok}  Unsolved:{IB_fail}", 0.5, 0.6),
    (f"Agent</br></br>Resolved:{IP_ok}</br>Unresolved:{IP_fail}", 0.4, 0.5),
    (f"Output Bounce</br></br>Correct:{OB_ok}</br>Wrong:{OB_fail}", 0.65, 0.5),
    (f"Output Pass</br></br>Correct:{OP_ok}</br>Wrong:{OP_fail}", 0.65, 0.2),
    (f"User\nCorrect\n({OP_ok})", 1, 0.2),
    (f"User\nWrong\n({OP_fail})", 1, 0.2),
]
labels, x, y = map(list, zip(*_nodes))
ALL, IB, IP, AG, OB, OP, UC, UW = range(len(labels))
# Soft pastel colors
blue_color  = 'rgba(135,206,250,0.4)'
orange_color = 'rgba(255,160,122,0.4)'
# Greenish and reddish colors
green_color = 'rgba(168,231,207,0.4)'
red_color   = 'rgba(255,182,193,0.4)'

entries = [
    (ALL, IB, IBC, blue_color),
    (ALL, IB, IBW, orange_color),
    (ALL, IP, IPC, blue_color),
    (ALL, IP, IPW, orange_color),
    # (IB, HA, IB_ok, green_color),
    # (IB, HA, IB_fail, red_color),
    (IP, AG, IP_ok, green_color),
    (IP, AG, IP_fail, red_color),
    (AG, OB, OB_ok, green_color),
    (AG, OB, OB_fail, red_color),
    (AG, OP, OP_ok, green_color),
    (AG, OP, OP_fail, red_color),
    (OP, UC, OP_ok, green_color),
    (OP, UW, OP_fail, red_color),
]

sources, targets, values, colors = map(list, zip(*entries))
fig = go.Figure(go.Sankey(
    arrangement='snap',
    orientation='h',
    node=dict(
        label=labels,
        pad=15,
        thickness=10,
        color='black',
        line=dict(color='grey', width=0.5)
    ),
    link=dict(source=sources, target=targets, value=values, color=colors, line=dict(color='black', width=0.5))
))

legend_items = [
    dict(name="Clear Ticket", marker=dict(color=blue_color, size=15, symbol="square", line=dict(color="black", width=1))),
    dict(name="Vague Ticket", marker=dict(color=orange_color, size=15, symbol="square", line=dict(color="black", width=1))),
    dict(name="Correct Patch", marker=dict(color=green_color, size=15, symbol="square", line=dict(color="black", width=1))),
    dict(name="Wrong Patch", marker=dict(color=red_color, size=15, symbol="square", line=dict(color="black", width=1)))
]

for item in legend_items:
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=item['marker'],
        name=item['name'],
        showlegend=True
    ))

fig.update_layout(
    # title_text="Pipeline Flow – All → Input → Output → User",
    font=dict(size=18, color='black', weight='bold'),
    # , shadow="1px 1px 2px yellow"
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(
        x=0.8,
        y=0.01,
        xanchor='left',
        yanchor='bottom',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='grey',
        borderwidth=1
    )
)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.data[0].node.x = x
fig.data[0].node.y = y
fig.write_image("sankey_bouncer.png", width=1200, height=400)