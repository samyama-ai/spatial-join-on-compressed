"""Regenerate the paper figures from results/ CSVs. Deterministic."""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)


def _load_pairs():
    p = os.path.join(OUT, "all_pairs.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def fig1_margin_law(df):
    """Per-pair decoded fraction vs |margin| (the H1' law), TIGER + synthetic."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for fam, col, mk in [("tiger", "#1f77b4", "o"), ("synthetic", "#ff7f0e", ".")]:
        d = df[df["family"] == fam]
        if len(d) == 0:
            continue
        ax.scatter(d["abs_margin"].clip(1e-6), d["phi_pair"], s=8, alpha=0.35,
                   c=col, marker=mk, label=f"{fam} (n={len(d)})")
    # binned median trend on TIGER
    d = df[df["family"] == "tiger"].copy()
    if len(d) > 20:
        d["lm"] = np.log10(d["abs_margin"].clip(1e-6))
        bins = np.linspace(d["lm"].min(), d["lm"].max(), 12)
        d["b"] = np.digitize(d["lm"], bins)
        g = d.groupby("b").agg(x=("abs_margin", "median"), y=("phi_pair", "median"))
        ax.plot(g["x"], g["y"], "k-", lw=2, label="TIGER median trend")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|{\rm margin}|$  (signed clearance, degrees)")
    ax.set_ylabel(r"decoded fraction  $\varphi_{\rm pair}$")
    ax.set_title("Decode-work law: decode is governed by the margin")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_margin_law.png"), dpi=150)
    plt.close(fig)


def fig2_teeth(summary):
    """naive/prog and brink/prog decode ratios vs translation delta on TIGER."""
    tg = summary[(summary["family"] == "tiger") & summary["separation"].notna()].copy()
    if len(tg) == 0:
        return
    tg["county"] = tg["workload"].str.extract(r"tiger_(\d+)_")
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for cty, g in tg.groupby("county"):
        g = g.sort_values("separation")
        ax.plot(g["separation"], g["ratio_naive_over_prog"], "-o", ms=4,
                label=f"{cty}: vs naive-refine")
    ax.axhline(2.0, color="r", ls="--", lw=1, label="H3 bar (2x)")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"translation $\delta$ (fraction of median polygon diagonal)")
    ax.set_ylabel("decode-vertex reduction  (x)")
    ax.set_title("Teeth: progressive vs naive-refine (provably-exact)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_teeth.png"), dpi=150)
    plt.close(fig)


def fig3_regime(summary):
    """Regime map: decoded fraction vs selectivity, with the crossover band."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for fam, col in [("synthetic", "#ff7f0e"), ("tiger", "#1f77b4")]:
        d = summary[summary["family"] == fam]
        if len(d) == 0:
            continue
        ax.scatter(d["selectivity"], d["phi_prog"], s=30, c=col, alpha=0.7,
                   label=fam)
    ax.axhspan(0.0, 0.10, color="green", alpha=0.08)
    ax.axhspan(0.70, 1.0, color="red", alpha=0.08)
    ax.text(0.02, 0.05, "big-win regime", fontsize=8, color="green")
    ax.text(0.02, 0.93, "fall-back regime", fontsize=8, color="red")
    ax.set_xlabel("join selectivity")
    ax.set_ylabel(r"workload decoded fraction  $\varphi$")
    ax.set_title("Regime map (decode axis)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_regime.png"), dpi=150)
    plt.close(fig)


def main():
    df = _load_pairs()
    summ = pd.read_csv(os.path.join(OUT, "all_summary.csv"))
    if df is not None:
        fig1_margin_law(df)
    fig2_teeth(summ)
    fig3_regime(summ)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
