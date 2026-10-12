"""Render figures and the README results block from reports/metrics.json.

    uv run --group viz python scripts/make_report.py
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics.json"
FIG_DIR = ROOT / "docs" / "figures"
README = ROOT / "README.md"

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODELS = {  # fixed categorical order: series identity never changes between charts
    "lightgbm": ("Current model (Dec 2025-Apr 2026)", "#2a78d6"),
    "lightgbm_trained_2024": ("Trained on 2024", "#eb6834"),
    "lightgbm_trained_2025_jan_sep": ("Trained on Jan-Sep 2025", "#1baf7a"),
}
NEWLINE = "\n"
EVENTS = [("2025-01-05", "MTA congestion fee"), ("2025-12-01", "Flex Fare price change")]
SEGMENTS = [
    ("all", "All trips"),
    ("no_congestion_fee", "No congestion\nfee"),
    ("congestion_fee_charged", "Congestion fee\ncharged"),
    ("metered", "Metered"),
    ("flex_fare", "Flex Fare"),
    ("airport", "Airport"),
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def timeline_chart(m: dict, metric: str, ylabel: str, title: str, path: Path) -> None:
    months = [np.datetime64(r["month"]) for r in m["timeline"]]
    x = np.array(months, dtype="datetime64[D]").astype("O")
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    _style(ax)
    t0, t1 = m["test_window"]
    ax.axvspan(np.datetime64(t0, "D").astype("O"), (np.datetime64(t1) + 1).astype("datetime64[D]").astype("O"),
               color="#efeeea", zorder=0)
    ax.text(np.datetime64(t0, "D").astype("O"), 0.02, " test", transform=ax.get_xaxis_transform(),
            color=TEXT_2, fontsize=8, va="bottom")
    for date, label in EVENTS:
        d = np.datetime64(date).astype("O")
        ax.axvline(d, color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(d, 1.0, f" {label}", transform=ax.get_xaxis_transform(), color=TEXT_2, fontsize=8, va="top")
    if metric == "bias":
        ax.axhline(0, color=TEXT_2, linewidth=1)
    for key, (label, color) in MODELS.items():
        if key not in m["timeline"][0]:
            continue
        y = [r[key][metric] for r in m["timeline"]]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=3.5, label=label)
        ax.annotate(label, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    color=TEXT, fontsize=8.5, va="center")
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, pad=22)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.08), ncol=3, frameon=False, fontsize=8.5,
              labelcolor=TEXT, handlelength=1.5)
    fig.subplots_adjust(right=0.78, top=0.82)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def segment_chart(m: dict, path: Path) -> None:
    keys = [k for k in MODELS if k in m["segments"]["all"]]
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    _style(ax)
    width, gap = 0.26, 0.02
    xs = np.arange(len(SEGMENTS))
    for i, key in enumerate(keys):
        label, color = MODELS[key]
        vals = [m["segments"][s][key]["mae"] for s, _ in SEGMENTS]
        pos = xs + (i - (len(keys) - 1) / 2) * (width + gap)
        ax.bar(pos, vals, width=width, color=color, label=label, zorder=2)
        for p, v in zip(pos, vals, strict=True):
            ax.text(p, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_2)
    ax.set_xticks(xs, [name for _, name in SEGMENTS])
    ax.set_ylabel("MAE ($)", color=TEXT_2, fontsize=9)
    ax.set_title(f"Test MAE by trip segment ({m['test_window'][0]} to {m['test_window'][1]})",
                 loc="left", color=TEXT, fontsize=11, pad=22)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.08), ncol=3, frameon=False, fontsize=8.5, labelcolor=TEXT)
    fig.subplots_adjust(top=0.82)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def results_markdown(m: dict) -> str:
    mod, seg, iv, lat = m["models"], m["segments"], m["intervals"], m["latency"]
    names = {
        "lightgbm": "**LightGBM, current regime** (Dec 2025-Apr 2026)",
        "elasticnet": "ElasticNet (same features)",
        "zone_pair_median": "Historical zone-pair median",
        "lightgbm_trained_2025_jan_sep": "LightGBM trained Jan-Sep 2025",
        "lightgbm_trained_2024": "LightGBM trained 2024",
    }
    d = m["data"]
    lines = [
        f"Out-of-time test set: **all {m['n_test']:,} cleaned trips from {m['test_window'][0]} to "
        f"{m['test_window'][1]}**. Target is the total charge excluding tip. Data: "
        f"{d['raw_rows']:,} raw trips ({d['months'][0]} to {d['months'][1]}), {d['clean_rows']:,} after cleaning.",
        "",
        "| Model | RMSE ($) | MAE ($) | Median AE ($) | Bias ($) |",
        "|---|---|---|---|---|",
    ]
    for key, label in names.items():
        r = mod[key]
        lines.append(f"| {label} | {r['rmse']:.2f} | {r['mae']:.2f} | {r['medae']:.2f} | {r['bias']:+.2f} |")
    conf = iv["confidence_level"]
    lines += [
        "",
        f"**{conf:.0%} conformal intervals** (calibrated on {m['model_metadata']['calibration_window'][0]}):",
        "",
        "| Interval | Coverage (all) | Mean width | Coverage, Flex Fare | Coverage, airport |",
        "|---|---|---|---|---|",
    ]
    for kind, label in (("fixed", "Fixed width (± q)"), ("adaptive", "Adaptive (± q x predicted error)")):
        s = iv[kind]["segments"]
        lines.append(f"| {label} | {s['all']['coverage']:.1%} | ${s['all']['mean_width']:.2f} | "
                     f"{s['flex_fare']['coverage']:.1%} | {s['airport']['coverage']:.1%} |")
    lines += [
        "",
        f"**Inference latency** (one request: feature build + prediction + interval, single thread, "
        f"{lat['n_requests']:,} requests): p50 {lat['p50_ms']:.2f} ms, p95 {lat['p95_ms']:.2f} ms, "
        f"p99 {lat['p99_ms']:.2f} ms.",
        "",
        "**Drift study, MAE ($) on the same test trips:**",
        "",
        "| Segment | Share | " + " | ".join(MODELS[k][0] for k in MODELS if k in seg["all"]) + " |",
        "|---|---|" + "---|" * sum(k in seg["all"] for k in MODELS),
    ]
    for s, name in SEGMENTS:
        vals = " | ".join(f"{seg[s][k]['mae']:.2f}" for k in MODELS if k in seg[s])
        lines.append(f"| {name.replace(NEWLINE, ' ')} | {seg[s]['share']:.0%} | {vals} |")
    return "\n".join(lines)


def walkforward_markdown(wf: dict) -> str:
    labels = {
        "static": "Static (current-regime model, unchanged)",
        "bias_fix": "**Monthly bias correction** (served by the API)",
        "retrain": "Monthly retrain (previous 4 months)",
    }
    months = list(wf["months"])
    lines = [
        f"**Model maintenance, walk-forward over {months[0]} to {months[-1]}** "
        "(assumes last month's trips are available at the start of each month):",
        "",
        "| Strategy | MAE ($) | RMSE ($) | Bias ($) | Flex Fare MAE | Metered MAE | "
        + " | ".join(f"MAE {mo}" for mo in months) + " | Coverage |",
        "|---|---|---|---|---|---|" + "---|" * len(months) + "---|",
    ]
    for key, label in labels.items():
        p = wf["pooled"][key]
        per_month = " | ".join(f"{wf['months'][mo][key]['all']['mae']:.2f}" for mo in months)
        lines.append(f"| {label} | {p['all']['mae']:.2f} | {p['all']['rmse']:.2f} | {p['all']['bias']:+.2f} | "
                     f"{p['flex_fare']['mae']:.2f} | {p['metered']['mae']:.2f} | {per_month} | "
                     f"{p['all']['coverage']:.1%} |")
    return "\n".join(lines)


def main() -> None:
    m = json.loads(METRICS.read_text())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    timeline_chart(m, "mae", "MAE ($)", "Monthly MAE: models trained on different pricing regimes",
                   FIG_DIR / "drift_timeline_mae.png")
    timeline_chart(m, "bias", "Mean bias ($, prediction - actual)", "Monthly bias by training regime",
                   FIG_DIR / "drift_timeline_bias.png")
    segment_chart(m, FIG_DIR / "drift_mae_by_segment.png")

    block = results_markdown(m)
    wf_path = METRICS.parent / "walkforward.json"
    if wf_path.exists():
        block += "\n\n" + walkforward_markdown(json.loads(wf_path.read_text()))
    text = README.read_text(encoding="utf-8")
    new = re.sub(r"(<!-- RESULTS:START -->\n).*?(\n<!-- RESULTS:END -->)", lambda g: g[1] + block + g[2],
                 text, flags=re.S)
    if new == text and "<!-- RESULTS:START -->" not in text:
        print(block)
    README.write_text(new, encoding="utf-8")
    print(f"figures -> {FIG_DIR}; README results block updated")


if __name__ == "__main__":
    main()
