from pathlib import Path
import os
import shutil
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(r"C:\Users\15220\Desktop\article") / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(r"C:\Users\15220\Desktop\article")
SOURCE = ROOT / "flashweave_reference_edge_retention_absr010_permutation999"
CACHE = SOURCE / "computed_flashweave_retention_absr010_unique_edges_permutation999_cache.pkl"
OUT = ROOT / "flashweave_module_retention_complete_summary_okabe_ito_soft"
PDF = OUT / "figures_pdf"
PNG = OUT / "figures_png"
CODE = OUT / "code"
for d in [OUT, PDF, PNG, CODE]:
    d.mkdir(parents=True, exist_ok=True)

GROUPS = ["HC", "MDDNSI", "MDDSI"]
SHORT = {"HC": "HC", "MDDNSI": "NSI", "MDDSI": "SI"}
COL = {"HC": "#56B4E9", "MDDNSI": "#F0B94D", "MDDSI": "#D99BC5"}
GREEN = "#009E73"
GOLD = "#F0B94D"
GREY = "#D9D9D9"
DARK = "#333842"
BLUE = "#D99A25"
RED = "#B86A9E"
CMAP_RET = LinearSegmentedColormap.from_list(
    "soft_retention", ["#F8F8F4", "#E4ECD9", "#BFD5C6", "#82AEB7", "#4F7698"]
)
CMAP_LOST = LinearSegmentedColormap.from_list(
    "soft_lost", ["#FBF8F3", "#ECDCCB", "#D2B494", "#A97D65"]
)
CMAP_DELTA = LinearSegmentedColormap.from_list(
    "soft_delta", ["#4E6F91", "#B8CBD4", "#F7F6F1", "#DAB9A6", "#A96363"]
)
MODULES = [f"SM{i:02d}" for i in range(1, 13)]
R_KEEP = 0.10
N_PERM = 999
N_BOOT = 999


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def save(fig, stem):
    fig.savefig(PDF / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PNG / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def fmt_p(p):
    if not np.isfinite(p):
        return "p=NA"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def boot_ci(boot, cols, val):
    ci = boot.groupby(cols)[val].quantile([0.025, 0.5, 0.975]).unstack().reset_index()
    ci.columns = list(cols) + ["ci_low", "median", "ci_high"]
    return ci


def prep(payload):
    obs_overall = payload["obs_overall"].copy()
    obs_pairs = payload["obs_pairs"].copy()
    boot = payload["boot"].copy()
    boot_pairs = payload["boot_pairs"].copy()
    perm = payload["perm"].copy()
    pair_perm = payload["pair_perm"].copy()
    diff = payload["diff"].copy()
    members = payload["members"].copy()
    edges = payload["edges"].copy()

    for df in [obs_overall, obs_pairs, boot, boot_pairs]:
        df["lost_rate"] = df["lost_or_nonsignificant_edges"] / df["reference_edges"]
    single = obs_pairs[(obs_pairs["pair_type"] == "intra_module") & obs_pairs["module_a"].isin(MODULES)].copy()
    single["module"] = single["module_a"]
    single_boot = boot_pairs[(boot_pairs["pair_type"] == "intra_module") & boot_pairs["module_a"].isin(MODULES)].copy()
    single_boot["module"] = single_boot["module_a"]
    single_perm = pair_perm[pair_perm["pair"].isin([f"{m}-{m}" for m in MODULES])].copy()
    single_perm["module"] = single_perm["pair"].str.slice(0, 4)

    module_sizes = members.groupby("module")["species"].nunique().rename("module_species").reset_index()
    single = single.merge(module_sizes, on="module", how="left")
    return obs_overall, obs_pairs, boot, boot_pairs, perm, pair_perm, diff, single, single_boot, single_perm, edges, members


def plot_overall_count_status(obs):
    setup()
    parts = ["intra_module", "inter_module"]
    labels = ["within modules", "between modules"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), sharey=True)
    for ax, part, label in zip(axes, parts, labels):
        dd = obs[obs["network_part"] == part].set_index("group").reindex(GROUPS)
        x = np.arange(3)
        retained = dd["retained_same_direction_edges"].to_numpy()
        reversed_edges = dd["reversed_direction_edges"].to_numpy()
        lost = dd["lost_or_nonsignificant_edges"].to_numpy()
        ax.bar(x, retained, color=GREEN, edgecolor="white", linewidth=0.45, label="retained")
        ax.bar(x, reversed_edges, bottom=retained, color=GOLD, edgecolor="white", linewidth=0.45, label="reversed")
        ax.bar(x, lost, bottom=retained + reversed_edges, color=GREY, edgecolor="white", linewidth=0.45, label="lost/weak")
        ax.set_xticks(x); ax.set_xticklabels([SHORT[g] for g in GROUPS])
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.grid(axis="y", color="#E6E8EF", linewidth=0.55)
        for i, total in enumerate(dd["reference_edges"]):
            ax.text(i, total + max(dd["reference_edges"]) * 0.025, f"n={int(total)}", ha="center", va="bottom", fontsize=6)
    axes[0].set_ylabel("reference edge count")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "01_overall_within_between_retained_reversed_lost_counts")


def plot_overall_rate_status(obs, perm):
    setup()
    parts = ["intra_module", "inter_module"]
    labels = ["within modules", "between modules"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), sharey=True)
    for ax, part, label in zip(axes, parts, labels):
        dd = obs[obs["network_part"] == part].set_index("group").reindex(GROUPS)
        x = np.arange(3)
        retained = dd["retention_rate"].to_numpy()
        reversed_rate = dd["reversal_rate"].to_numpy()
        lost = dd["lost_rate"].to_numpy()
        ax.bar(x, retained, color=GREEN, edgecolor="white", linewidth=0.45, label="retained")
        ax.bar(x, reversed_rate, bottom=retained, color=GOLD, edgecolor="white", linewidth=0.45, label="reversed")
        ax.bar(x, lost, bottom=retained + reversed_rate, color=GREY, edgecolor="white", linewidth=0.45, label="lost/weak")
        ax.set_xticks(x); ax.set_xticklabels([SHORT[g] for g in GROUPS])
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="#E6E8EF", linewidth=0.55)
        text_y = 1.02
        for c in ["MDDNSI_vs_HC", "MDDSI_vs_HC", "MDDSI_vs_MDDNSI"]:
            p = perm[(perm["contrast"] == c) & (perm["network_part"] == part)]["permutation_p"]
            if len(p):
                lab = c.replace("MDDNSI", "NSI").replace("MDDSI", "SI").replace("_vs_", " vs ")
                ax.text(0.5, text_y, f"{lab}: {fmt_p(float(p.iloc[0]))}", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=5.8)
                text_y += 0.075
    axes[0].set_ylabel("fraction of reference edges")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "02_overall_within_between_retained_reversed_lost_rates")


def plot_overall_retained_count_bars(obs, boot, perm):
    setup()
    ci = boot_ci(boot, ["group", "network_part"], "retained_same_direction_edges")
    parts = ["intra_module", "inter_module"]
    x = np.arange(2)
    width = 0.23
    fig, ax = plt.subplots(figsize=(5.8, 3.45))
    for k, g in enumerate(GROUPS):
        dd = obs[obs["group"] == g].set_index("network_part").reindex(parts)
        cc = ci[ci["group"] == g].set_index("network_part").reindex(parts)
        xpos = x + (k - 1) * width
        ax.bar(xpos, dd["retained_same_direction_edges"], width=width, color=COL[g], edgecolor="white", linewidth=0.45, label=SHORT[g])
        yerr = np.vstack([
            np.maximum(dd["retained_same_direction_edges"].to_numpy() - cc["ci_low"].to_numpy(), 0),
            np.maximum(cc["ci_high"].to_numpy() - dd["retained_same_direction_edges"].to_numpy(), 0),
        ])
        ax.errorbar(xpos, dd["retained_same_direction_edges"], yerr=yerr, fmt="none", ecolor=DARK, elinewidth=0.65, capsize=2)
        for i, part in enumerate(parts):
            ax.text(xpos[i], dd.loc[part, "retained_same_direction_edges"] + 12,
                    str(int(dd.loc[part, "retained_same_direction_edges"])), ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(x); ax.set_xticklabels(["within modules", "between modules"])
    ax.set_ylabel("same-direction retained edge count")
    ax.grid(axis="y", color="#E6E8EF", linewidth=0.55)
    ax.legend(ncol=3, loc="upper right")
    ymax = obs["retained_same_direction_edges"].max() * 1.38
    ax.set_ylim(0, ymax)
    for i, part in enumerate(parts):
        y = obs[obs["network_part"] == part]["retained_same_direction_edges"].max() * 1.12
        for j, c in enumerate(["MDDNSI_vs_HC", "MDDSI_vs_HC", "MDDSI_vs_MDDNSI"]):
            p = perm[(perm["contrast"] == c) & (perm["network_part"] == part)]["permutation_p"]
            if len(p):
                lab = c.replace("MDDNSI", "NSI").replace("MDDSI", "SI").replace("_vs_", " vs ")
                ax.text(i, y + j * ymax * 0.07, f"{lab}: {fmt_p(float(p.iloc[0]))}", ha="center", fontsize=5.8)
    fig.tight_layout()
    save(fig, "03_overall_within_between_retained_edge_counts")


def plot_overall_retention_rate_bars(obs, boot, perm):
    setup()
    ci = boot_ci(boot, ["group", "network_part"], "retention_rate")
    parts = ["intra_module", "inter_module"]
    x = np.arange(2)
    width = 0.23
    fig, ax = plt.subplots(figsize=(5.8, 3.45))
    for k, g in enumerate(GROUPS):
        dd = obs[obs["group"] == g].set_index("network_part").reindex(parts)
        cc = ci[ci["group"] == g].set_index("network_part").reindex(parts)
        xpos = x + (k - 1) * width
        ax.bar(xpos, dd["retention_rate"], width=width, color=COL[g], edgecolor="white", linewidth=0.45, label=SHORT[g])
        yerr = np.vstack([
            np.maximum(dd["retention_rate"].to_numpy() - cc["ci_low"].to_numpy(), 0),
            np.maximum(cc["ci_high"].to_numpy() - dd["retention_rate"].to_numpy(), 0),
        ])
        ax.errorbar(xpos, dd["retention_rate"], yerr=yerr, fmt="none", ecolor=DARK, elinewidth=0.65, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(["within modules", "between modules"])
    ax.set_ylabel("same-direction retained edge fraction")
    ax.grid(axis="y", color="#E6E8EF", linewidth=0.55)
    ax.legend(ncol=3, loc="upper right")
    ax.set_ylim(0, max(0.72, obs["retention_rate"].max() * 1.38))
    fig.tight_layout()
    save(fig, "04_overall_within_between_retention_rates")


def plot_single_module_counts(single, single_boot, single_perm):
    setup()
    ci = boot_ci(single_boot, ["group", "module"], "retained_same_direction_edges")
    x = np.arange(12)
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for k, g in enumerate(GROUPS):
        dd = single[single["group"] == g].set_index("module").reindex(MODULES)
        cc = ci[ci["group"] == g].set_index("module").reindex(MODULES)
        xpos = x + (k - 1) * width
        ax.bar(xpos, dd["retained_same_direction_edges"], width=width, color=COL[g], edgecolor="white", linewidth=0.45, label=SHORT[g])
        yerr = np.vstack([
            np.maximum(dd["retained_same_direction_edges"].to_numpy() - cc["ci_low"].to_numpy(), 0),
            np.maximum(cc["ci_high"].to_numpy() - dd["retained_same_direction_edges"].to_numpy(), 0),
        ])
        ax.errorbar(xpos, dd["retained_same_direction_edges"], yerr=yerr, fmt="none", ecolor=DARK, elinewidth=0.55, capsize=1.8)
    ymax = max(10, single["retained_same_direction_edges"].max() * 1.35)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x); ax.set_xticklabels(MODULES)
    ax.set_ylabel("within-module retained edge count")
    ax.grid(axis="y", color="#E6E8EF", linewidth=0.55)
    ax.legend(ncol=3, loc="upper right")
    annotate_single_p(ax, single, single_perm, "retained_same_direction_edges", ymax)
    fig.tight_layout()
    save(fig, "05_single_module_retained_edge_counts")


def annotate_single_p(ax, single, single_perm, value_col, ymax):
    for i, m in enumerate(MODULES):
        p_nsi = single_perm[(single_perm["module"] == m) & (single_perm["contrast"] == "MDDNSI_vs_HC")]["permutation_p"]
        p_si = single_perm[(single_perm["module"] == m) & (single_perm["contrast"] == "MDDSI_vs_HC")]["permutation_p"]
        p_si_nsi = single_perm[(single_perm["module"] == m) & (single_perm["contrast"] == "MDDSI_vs_MDDNSI")]["permutation_p"]
        if len(p_nsi) and float(p_nsi.iloc[0]) < 0.05:
            y = single[(single["module"] == m) & (single["group"] == "MDDNSI")][value_col].iloc[0] + ymax * 0.035
            ax.text(i, y, stars(float(p_nsi.iloc[0])), ha="center", va="bottom", color=COL["MDDNSI"], fontsize=8, fontweight="bold")
        mark = ""
        if len(p_si):
            mark += stars(float(p_si.iloc[0]))
        if len(p_si_nsi) and float(p_si_nsi.iloc[0]) < 0.05:
            mark += "#"
        if mark:
            y = min(ymax * 0.96, single[single["module"] == m][value_col].max() + ymax * 0.045)
            ax.text(i, y, mark, ha="center", va="bottom", color=RED, fontsize=8, fontweight="bold")
    ax.text(0.01, 0.985, "orange * NSI vs HC; purple * SI vs HC; # SI vs NSI, permutation p<0.05",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.1,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88))


def plot_single_module_rates(single, single_boot, single_perm):
    setup()
    ci = boot_ci(single_boot, ["group", "module"], "retention_rate")
    x = np.arange(12)
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for k, g in enumerate(GROUPS):
        dd = single[single["group"] == g].set_index("module").reindex(MODULES)
        cc = ci[ci["group"] == g].set_index("module").reindex(MODULES)
        xpos = x + (k - 1) * width
        ax.bar(xpos, dd["retention_rate"], width=width, color=COL[g], edgecolor="white", linewidth=0.45, label=SHORT[g])
        yerr = np.vstack([
            np.maximum(dd["retention_rate"].to_numpy() - cc["ci_low"].to_numpy(), 0),
            np.maximum(cc["ci_high"].to_numpy() - dd["retention_rate"].to_numpy(), 0),
        ])
        ax.errorbar(xpos, dd["retention_rate"], yerr=yerr, fmt="none", ecolor=DARK, elinewidth=0.55, capsize=1.8)
    ymax = min(1.08, max(0.72, single["retention_rate"].max() * 1.08))
    ax.set_ylim(0, ymax)
    ax.set_xticks(x); ax.set_xticklabels(MODULES)
    ax.set_ylabel("within-module retained edge fraction")
    ax.grid(axis="y", color="#E6E8EF", linewidth=0.55)
    ax.legend(ncol=3, loc="upper right")
    annotate_single_p(ax, single, single_perm, "retention_rate", ymax)
    fig.tight_layout()
    save(fig, "06_single_module_retention_rates")


def plot_single_module_lost_counts(single):
    setup()
    mat = single.pivot(index="module", columns="group", values="lost_or_nonsignificant_edges").reindex(index=MODULES, columns=GROUPS)
    fig, ax = plt.subplots(figsize=(3.6, 4.6))
    im = ax.imshow(mat.to_numpy(), cmap=CMAP_LOST, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels([SHORT[g] for g in GROUPS])
    ax.set_yticks(range(12)); ax.set_yticklabels(MODULES)
    for i, m in enumerate(MODULES):
        for j, g in enumerate(GROUPS):
            ax.text(j, i, str(int(mat.loc[m, g])), ha="center", va="center", fontsize=6,
                    color="white" if mat.loc[m, g] > np.nanmax(mat.to_numpy()) * 0.55 else DARK)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("lost/weak edge count")
    fig.tight_layout()
    save(fig, "07_single_module_lost_or_weak_edge_counts")


def plot_single_module_status_rates(single):
    setup()
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 5.8), sharex=True, sharey=True)
    for ax, g in zip(axes, GROUPS):
        dd = single[single["group"] == g].set_index("module").reindex(MODULES)
        x = np.arange(12)
        retained = dd["retention_rate"].to_numpy()
        reversed_rate = dd["reversal_rate"].to_numpy()
        lost = dd["lost_rate"].to_numpy()
        ax.bar(x, retained, color=GREEN, edgecolor="white", linewidth=0.35, label="retained")
        ax.bar(x, reversed_rate, bottom=retained, color=GOLD, edgecolor="white", linewidth=0.35, label="reversed")
        ax.bar(x, lost, bottom=retained + reversed_rate, color=GREY, edgecolor="white", linewidth=0.35, label="lost/weak")
        ax.set_ylabel(SHORT[g])
        ax.grid(axis="y", color="#E6E8EF", linewidth=0.45)
        ax.set_ylim(0, 1)
    axes[-1].set_xticks(range(12)); axes[-1].set_xticklabels(MODULES)
    axes[0].legend(ncol=3, loc="upper right")
    fig.text(0.005, 0.5, "fraction of within-module reference edges", va="center", rotation=90, fontsize=7)
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    save(fig, "08_single_module_retained_reversed_lost_rates")


def pair_matrix(obs_pairs, group, value, pair_type=None):
    dd = obs_pairs[obs_pairs["group"] == group]
    if pair_type:
        dd = dd[dd["pair_type"] == pair_type]
    mat = pd.DataFrame(np.nan, index=MODULES, columns=MODULES)
    for _, r in dd.iterrows():
        a, b = r["module_a"], r["module_b"]
        if a in MODULES and b in MODULES:
            mat.loc[a, b] = r[value]
            mat.loc[b, a] = r[value]
    return mat


def plot_between_count_matrices(obs_pairs):
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.25), sharex=True, sharey=True)
    vmax = obs_pairs[obs_pairs["pair_type"] == "inter_module"]["retained_same_direction_edges"].max()
    for ax, g in zip(axes, GROUPS):
        mat = pair_matrix(obs_pairs, g, "retained_same_direction_edges", "inter_module").to_numpy(copy=True)
        np.fill_diagonal(mat, np.nan)
        im = ax.imshow(mat, cmap=CMAP_RET, vmin=0, vmax=vmax)
        ax.set_title(SHORT[g], color=COL[g], fontsize=8, fontweight="bold")
        ax.set_xticks(range(12)); ax.set_xticklabels(MODULES, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(12)); ax.set_yticklabels(MODULES, fontsize=5.5)
    cb = plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.02)
    cb.set_label("retained edge count")
    save(fig, "09_between_module_retained_edge_count_matrices")


def plot_between_rate_matrices(obs_pairs):
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.25), sharex=True, sharey=True)
    vmax = obs_pairs[obs_pairs["pair_type"] == "inter_module"]["retention_rate"].max()
    for ax, g in zip(axes, GROUPS):
        mat = pair_matrix(obs_pairs, g, "retention_rate", "inter_module").to_numpy(copy=True)
        np.fill_diagonal(mat, np.nan)
        im = ax.imshow(mat, cmap=CMAP_RET, vmin=0, vmax=vmax)
        ax.set_title(SHORT[g], color=COL[g], fontsize=8, fontweight="bold")
        ax.set_xticks(range(12)); ax.set_xticklabels(MODULES, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(12)); ax.set_yticklabels(MODULES, fontsize=5.5)
    cb = plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.02)
    cb.set_label("retained edge fraction")
    save(fig, "10_between_module_retention_rate_matrices")


def plot_between_lost_count_matrices(obs_pairs):
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.25), sharex=True, sharey=True)
    vmax = obs_pairs[obs_pairs["pair_type"] == "inter_module"]["lost_or_nonsignificant_edges"].max()
    for ax, g in zip(axes, GROUPS):
        mat = pair_matrix(obs_pairs, g, "lost_or_nonsignificant_edges", "inter_module").to_numpy(copy=True)
        np.fill_diagonal(mat, np.nan)
        im = ax.imshow(mat, cmap=CMAP_LOST, vmin=0, vmax=vmax)
        ax.set_title(SHORT[g], color=COL[g], fontsize=8, fontweight="bold")
        ax.set_xticks(range(12)); ax.set_xticklabels(MODULES, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(12)); ax.set_yticklabels(MODULES, fontsize=5.5)
    cb = plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.02)
    cb.set_label("lost/weak edge count")
    save(fig, "11_between_module_lost_or_weak_edge_count_matrices")


def plot_between_delta_rate_si_nsi(obs_pairs, pair_perm):
    setup()
    nsi = pair_matrix(obs_pairs, "MDDNSI", "retention_rate", "inter_module")
    si = pair_matrix(obs_pairs, "MDDSI", "retention_rate", "inter_module")
    delta = si - nsi
    vmax_rate = max(np.nanmax(nsi.to_numpy()), np.nanmax(si.to_numpy()), 0.01)
    vmax_delta = max(np.nanmax(np.abs(delta.to_numpy())), 0.01)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), sharex=True, sharey=True)
    for ax, mat, title, color in [
        (axes[0], nsi, "NSI retention rate", COL["MDDNSI"]),
        (axes[1], si, "SI retention rate", COL["MDDSI"]),
    ]:
        arr = mat.to_numpy(copy=True)
        np.fill_diagonal(arr, np.nan)
        im0 = ax.imshow(arr, cmap=CMAP_RET, vmin=0, vmax=vmax_rate)
        ax.set_title(title, color=color, fontsize=8, fontweight="bold")
        ax.set_xticks(range(12)); ax.set_xticklabels(MODULES, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(12)); ax.set_yticklabels(MODULES, fontsize=5.5)
    delta_arr = delta.to_numpy(copy=True)
    np.fill_diagonal(delta_arr, np.nan)
    im1 = axes[2].imshow(delta_arr, cmap=CMAP_DELTA, vmin=-vmax_delta, vmax=vmax_delta)
    axes[2].set_title("SI - NSI", fontsize=8, fontweight="bold")
    axes[2].set_xticks(range(12)); axes[2].set_xticklabels(MODULES, rotation=60, ha="right", fontsize=5.5)
    axes[2].set_yticks(range(12)); axes[2].set_yticklabels(MODULES, fontsize=5.5)
    pp = pair_perm[(pair_perm["contrast"] == "MDDSI_vs_MDDNSI") & (pair_perm["pair_type"] == "inter_module")]
    for _, r in pp.iterrows():
        if r["module_a"] in MODULES and r["module_b"] in MODULES and r["permutation_p"] < 0.05:
            i = MODULES.index(r["module_a"]); j = MODULES.index(r["module_b"])
            axes[2].text(j, i, "*", ha="center", va="center", fontsize=7, color=DARK)
            axes[2].text(i, j, "*", ha="center", va="center", fontsize=7, color=DARK)
    cb0 = plt.colorbar(im0, ax=axes[:2].ravel().tolist(), fraction=0.026, pad=0.02)
    cb0.set_label("retained edge fraction")
    cb1 = plt.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.02)
    cb1.set_label("delta retained fraction")
    fig.text(0.075, 0.965, "Between-module retention rate matrices; * on SI-NSI indicates permutation p<0.05", fontsize=6.2)
    save(fig, "12_between_module_SI_NSI_retention_rate_comparison")
    return nsi, si, delta


def plot_between_delta_count_pairwise(obs_pairs, pair_perm):
    setup()
    contrasts = [("MDDNSI_vs_HC", "NSI - HC"), ("MDDSI_vs_HC", "SI - HC"), ("MDDSI_vs_MDDNSI", "SI - NSI")]
    mats = []
    for c, _ in contrasts:
        g1, g0 = c.split("_vs_")
        m1 = pair_matrix(obs_pairs, g1, "retained_same_direction_edges", "inter_module")
        m0 = pair_matrix(obs_pairs, g0, "retained_same_direction_edges", "inter_module")
        mat = m1 - m0
        arr = mat.to_numpy(copy=True)
        np.fill_diagonal(arr, np.nan)
        mat = pd.DataFrame(arr, index=MODULES, columns=MODULES)
        mats.append(mat)
    vmax = max(np.nanmax(np.abs(np.stack([m.to_numpy() for m in mats]))), 1)
    fig, axes = plt.subplots(1, 3, figsize=(9.7, 3.25), sharex=True, sharey=True)
    for ax, mat, (c, title) in zip(axes, mats, contrasts):
        im = ax.imshow(mat.to_numpy(), cmap=CMAP_DELTA, vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xticks(range(12)); ax.set_xticklabels(MODULES, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(12)); ax.set_yticklabels(MODULES, fontsize=5.5)
        pp = pair_perm[(pair_perm["contrast"] == c) & (pair_perm["pair_type"] == "inter_module")]
        for _, r in pp.iterrows():
            if r["module_a"] in MODULES and r["module_b"] in MODULES and r["permutation_p"] < 0.05:
                i = MODULES.index(r["module_a"]); j = MODULES.index(r["module_b"])
                ax.text(j, i, "*", ha="center", va="center", fontsize=7, color=DARK)
                ax.text(i, j, "*", ha="center", va="center", fontsize=7, color=DARK)
    cb = plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.02)
    cb.set_label("delta retained edge count")
    fig.text(0.075, 0.965, "* permutation p<0.05; cells are between-module FlashWeave reference edges", fontsize=6.2)
    save(fig, "13_between_module_pairwise_delta_retained_edge_counts")


def write_outputs(payload, tables, nsi_rate, si_rate, si_minus_nsi):
    obs_overall, obs_pairs, boot, boot_pairs, perm, pair_perm, diff, single, single_boot, single_perm, edges, members = tables
    overview = pd.DataFrame([
        ["reference_network", "Fixed HC FlashWeave reference network. Group-specific networks were not rebuilt."],
        ["edge_retained", f"Group CLR Pearson correlation has the same sign as FlashWeave weight and |r| >= {R_KEEP}."],
        ["edge_reversed", f"Group CLR Pearson correlation has the opposite sign and |r| >= {R_KEEP}."],
        ["edge_lost_or_weak", f"Reference edge does not satisfy same-direction |r| >= {R_KEEP}."],
        ["within_module", "Both edge endpoints are assigned to the same refined species module."],
        ["between_module", "The two edge endpoints are assigned to different refined species modules."],
        ["bootstrap", f"{N_BOOT} within-group bootstrap resamples for 95% CI."],
        ["permutation", f"{N_PERM} pairwise label permutations preserving group sizes."],
    ], columns=["item", "definition"])

    out_xlsx = OUT / "flashweave_module_retention_complete_summary_results.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="00_method_overview", index=False)
        obs_overall.to_excel(writer, sheet_name="01_overall_within_between", index=False)
        perm.to_excel(writer, sheet_name="02_overall_perm999", index=False)
        single.to_excel(writer, sheet_name="03_single_module_internal", index=False)
        single_perm.to_excel(writer, sheet_name="04_single_module_perm999", index=False)
        obs_pairs.to_excel(writer, sheet_name="05_module_pair_all", index=False)
        pair_perm.to_excel(writer, sheet_name="06_module_pair_perm999", index=False)
        diff.to_excel(writer, sheet_name="07_module_pair_delta_rate", index=False)
        nsi_rate.to_excel(writer, sheet_name="08_NSI_between_rate_matrix")
        si_rate.to_excel(writer, sheet_name="09_SI_between_rate_matrix")
        si_minus_nsi.to_excel(writer, sheet_name="10_SI_minus_NSI_rate_matrix")
        edges.drop(columns=["ia", "ib"], errors="ignore").to_excel(writer, sheet_name="11_reference_edges", index=False)
        members.to_excel(writer, sheet_name="12_module_membership", index=False)

    summary = obs_overall[["group", "network_part", "reference_edges", "retained_same_direction_edges",
                           "reversed_direction_edges", "lost_or_nonsignificant_edges", "retention_rate",
                           "reversal_rate", "lost_rate"]]
    single_tab = single.pivot(index="module", columns="group", values="retained_same_direction_edges").reindex(MODULES)
    report = OUT / "flashweave_module_retention_complete_summary_report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# FlashWeave module edge retention complete summary\n\n")
        f.write("This folder summarizes retained, reversed, and lost/weak HC FlashWeave reference edges at three levels: overall within/between modules, single-module internal edges, and between-module module-pair matrices.\n\n")
        f.write("## Method\n\n")
        f.write(f"Retained edge = same-direction group CLR Pearson correlation with |r| >= {R_KEEP}. Bootstrap={N_BOOT}; permutation={N_PERM}.\n\n")
        f.write("## Overall within/between summary\n\n")
        f.write(summary.to_csv(index=False))
        f.write("\n## Single-module retained edge counts\n\n")
        f.write(single_tab.reset_index().to_csv(index=False))
        sig = pair_perm[(pair_perm["contrast"] == "MDDSI_vs_MDDNSI") & (pair_perm["pair_type"] == "inter_module") & (pair_perm["permutation_p"] < 0.05)]
        f.write("\n## SI vs NSI between-module significant pairs\n\n")
        if sig.empty:
            f.write("No between-module pair reached permutation p<0.05 for SI vs NSI.\n")
        else:
            f.write(sig.to_csv(index=False))
    return out_xlsx, report


def combine_pdfs():
    from pypdf import PdfWriter
    names = [
        "01_overall_within_between_retained_reversed_lost_counts",
        "02_overall_within_between_retained_reversed_lost_rates",
        "03_overall_within_between_retained_edge_counts",
        "04_overall_within_between_retention_rates",
        "05_single_module_retained_edge_counts",
        "06_single_module_retention_rates",
        "07_single_module_lost_or_weak_edge_counts",
        "08_single_module_retained_reversed_lost_rates",
        "09_between_module_retained_edge_count_matrices",
        "10_between_module_retention_rate_matrices",
        "11_between_module_lost_or_weak_edge_count_matrices",
        "12_between_module_SI_NSI_retention_rate_comparison",
        "13_between_module_pairwise_delta_retained_edge_counts",
    ]
    writer = PdfWriter()
    for n in names:
        writer.append(str(PDF / f"{n}.pdf"))
    out = PDF / "FlashWeave_module_retention_complete_summary_combined.pdf"
    with open(out, "wb") as fh:
        writer.write(fh)
    return out


def main():
    warnings.filterwarnings("ignore")
    if not CACHE.exists():
        raise FileNotFoundError(CACHE)
    payload = pd.read_pickle(CACHE)
    tables = prep(payload)
    obs_overall, obs_pairs, boot, boot_pairs, perm, pair_perm, diff, single, single_boot, single_perm, edges, members = tables

    plot_overall_count_status(obs_overall)
    plot_overall_rate_status(obs_overall, perm)
    plot_overall_retained_count_bars(obs_overall, boot, perm)
    plot_overall_retention_rate_bars(obs_overall, boot, perm)
    plot_single_module_counts(single, single_boot, single_perm)
    plot_single_module_rates(single, single_boot, single_perm)
    plot_single_module_lost_counts(single)
    plot_single_module_status_rates(single)
    plot_between_count_matrices(obs_pairs)
    plot_between_rate_matrices(obs_pairs)
    plot_between_lost_count_matrices(obs_pairs)
    nsi_rate, si_rate, si_minus_nsi = plot_between_delta_rate_si_nsi(obs_pairs, pair_perm)
    plot_between_delta_count_pairwise(obs_pairs, pair_perm)
    xlsx, report = write_outputs(payload, tables, nsi_rate, si_rate, si_minus_nsi)
    combined = combine_pdfs()
    shutil.copy2(__file__, CODE / Path(__file__).name)
    print(OUT)
    print(combined)
    print(xlsx)
    print(report)


if __name__ == "__main__":
    main()
