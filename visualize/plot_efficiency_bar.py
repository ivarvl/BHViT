"""Horizontal bar-chart visualization of the BHViT width sweep.

Reads every config in configs/sweep/d*/config.json, profiles it analytically
via scripts/profile_model.py, and emits two horizontal bar charts:

  1. weight memory in MB   (binary @ 1 bit + residual FP @ 16 bit, paper convention)
  2. total OPs in MOPs     (FLOPs + BOPs/64)

Each chart also shows an FP32 baseline (same architecture at d=64, w/a=32/32)
as the topmost bar AND as a vertical reference line for easy comparison.

Output is written to visualize/output/ as PNG and PDF.

Usage:
    python visualize/plot_sweep.py
    python visualize/plot_sweep.py --num-classes 1000
"""

import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from profile_model import Counts, profile  # noqa: E402

SWEEP_DIR = os.path.join(REPO, "configs", "sweep")
OUT_DIR = os.path.join(HERE, "output")

FP_BYTES = 2  # FP16 — matches paper-reported sizes
BOP_DIVISOR = 64  # 1 BOP = 1/64 FLOP


def discover_widths():
    # dirs = [d for d in os.listdir(SWEEP_DIR) if re.fullmatch(r"d\d+", d)]
    # return sorted(int(d[1:]) for d in dirs)
    #
    return [64, 96, 128, 160, 192, 224, 256]


def profile_width(d, num_classes, wbits=1, abits=1, some_fp=False, fp_bytes=FP_BYTES):
    with open(os.path.join(SWEEP_DIR, f"d{d}", "config.json")) as f:
        cfg = json.load(f)
    cfg.update(
        {
            "num_classes": num_classes,
            "weight_bits": wbits,
            "input_bits": abits,
            "some_fp": some_fp,
            "shift3": True,
            "shift5": True,
            "disable_layerscale": False,
        }
    )
    c = Counts()
    profile(cfg, c)

    size_mb = (c.bin_weight_params / 8 + c.fp_weight_params * fp_bytes) / 1024 / 1024
    ops_mops = (c.fp_macs + c.bin_macs / BOP_DIVISOR) / 1e6
    return size_mb, ops_mops


def annotate_h(ax, bars, fmt, xmax):
    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            fmt.format(w),
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#222222",
        )


def style_axes(ax, title, xlabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Stage-0 hidden dim $d$", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    widths = discover_widths()
    sizes_mb, ops_mops = [], []
    some_fp = True
    for d in widths:
        s, o = profile_width(d, args.num_classes, some_fp=some_fp)
        sizes_mb.append(s)
        ops_mops.append(o)
        print(f"  d={d:<4d}  size={s:6.2f} MB   ops={o:7.2f} MOPs")

    # FP32 baseline at the smallest width (d=64) — same architecture,
    # all conv/linear weights at 32-bit, all activations at FP.
    fp_d = widths[0]
    fp_size, fp_ops = profile_width(
        fp_d,
        args.num_classes,
        wbits=32,
        abits=32,
        fp_bytes=4,
    )
    fp_partial_size, fp_partial_ops = profile_width(
        fp_d,
        args.num_classes,
        wbits=32,
        abits=1,
        fp_bytes=4,
    )
    print(f"\nFP32 baseline (d={fp_d}):  size={fp_size:.2f} MB   ops={fp_ops:.1f} MOPs")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    # Order on y-axis (top → bottom):  FP32, d=256, d=240, …, d=64.
    # In matplotlib barh, the first row plotted goes to y=0 (bottom), so we
    # arrange the data with d=64 first and the FP32 baseline last.
    bin_labels = [str(d) for d in widths]
    y_labels = bin_labels + [f"FP32\n(d={fp_d})"] + [f"Weights-Only FP32\n(d={fp_d})"]
    y_pos = np.arange(len(y_labels))

    bin_color = "#2E86AB"
    bin_color_top = "#1F6E91"
    fp_color = "#9B2D20"
    ref_color = "#444444"

    def plot_panel(ax, bin_vals, fp_val, fp2_val, title, xlabel, fmt):
        colors = [bin_color] * len(bin_vals) + [fp_color] * 2
        values = list(bin_vals) + [fp_val, fp2_val]
        bars = ax.barh(
            y_pos, values, color=colors, edgecolor="white", linewidth=0.8, zorder=3
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        style_axes(ax, title, xlabel)

        xmax = max(values) * 1.18
        ax.set_xlim(0, xmax)
        annotate_h(ax, bars, fmt, xmax)

        # vertical reference line at the FP32 value
        ax.axvline(
            fp_val,
            color=ref_color,
            linestyle="--",
            linewidth=1.4,
            zorder=4,
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7.0))

    plot_panel(
        ax1,
        sizes_mb,
        fp_size,
        fp_partial_size,
        "Model weight memory",
        "Weight size (MB)",
        "{:.2f}",
    )
    plot_panel(
        ax2,
        ops_mops,
        fp_ops,
        fp_partial_ops,
        "Total compute  (FLOPs + BOPs/64)",
        "OPs (MOPs)",
        "{:.1f}",
    )

    # Shared figure-level legend below the panels — avoids overlapping the
    # FP32 reference line that sits near the right edge of each axes.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=bin_color, label="binary (w/a=1/1)"),
        Patch(facecolor=fp_color, label=f"FP32 @ d={fp_d}"),
        Line2D(
            [0],
            [0],
            color=ref_color,
            linestyle="--",
            linewidth=1.4,
            label="FP32 reference",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.suptitle(
        f"BHViT | w/a = 1/1, {'some FP16' if some_fp else 'fully binary'}, {args.num_classes} classes",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    png = os.path.join(args.out_dir, "sweep_bars.png")
    pdf = os.path.join(args.out_dir, "sweep_bars.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"\nSaved: {png}\n       {pdf}")


if __name__ == "__main__":
    main()
