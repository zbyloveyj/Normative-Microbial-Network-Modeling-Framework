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


ROOT = Path(r"C:\Users\15220\Desktop\article")
SOURCE = ROOT / "flashweave_reference_edge_retention_absr010_permutation999"
CACHE = SOURCE / "computed_flashweave_retention_absr010_unique_edges_permutation999_cache.pkl"
OUT = ROOT / "flashweave_single_module_internal_metrics_absr010_permutation999"
FIG = OUT / "figures_pdf"
PNG = OUT / "figures_png"
CODE = OUT / "code"
for d in [OUT, FIG, PNG, CODE]:
    d.mkdir(parents=True, exist_ok=True)

GROUPS = ["HC", "MDDNSI", "MDDSI"]
SHORT = {"HC": "HC", "MDDNSI": "NSI", "MDDSI": "SI"}
COL = {"HC": "#5BAA7D", "MDDNSI": "#4C78A8", "MDDSI": "#D55E5E"}
DARK = "#2F3742"
BLUE = "#3B5B92"
RED = "#C44E52"
GOLD = "#D8A03D"
GREY = "#A7ADB8"
R_KEEP = 0.10
N_BOOT = 999
N_PERM = 999
MODULES = [f"SM{i:02d}" for i in range(1, 13)]


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


def fmt_p(p):
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PNG / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def load_cache():
    if not CACHE.exists():
        raise FileNotFoundError(f"Missing source cache: {CACHE}")
    return pd.read_pickle(CACHE)


def make_single_module_tables(payload):
    obs_pairs = payload["obs_pairs"].copy()
    boot_pairs = payload["boot_pairs"].copy()
    pair_ci = payload["pair_ci"].copy()
    pair_perm = payload["pair_perm"].copy()
    diff = payload["diff"].copy()
    members = payload["members"].copy()
    groups = payload["groups"]
    edges = payload["edges"].copy()

    intra_obs = obs_pairs[(obs_pairs["pair_type"] == "intra_module") & (obs_pairs["module_a"].isin(MODULES))].copy()
    intra_obs["module"] = intra_obs["module_a"]
    intra_obs["loss_rate"] = intra_obs["lost_or_nonsignificant_edges"] / intra_obs["reference_edges"]

    intra_ci = pair_ci[(pair_ci["pair_type"] == "intra_module") & (pair_ci["module_a"].isin(MODULES))].copy()
    intra_ci["module"] = intra_ci["module_a"]

    intra_boot = boot_pairs[(boot_pairs["pair_type"] == "intra_module") & (boot_pairs["module_a"].isin(MODULES))].copy()
    intra_boot["module"] = intra_boot["module_a"]
    intra_boot["loss_rate"] = intra_boot["lost_or_nonsignificant_edges"] / intra_boot["reference_edges"]

    intra_perm = pair_perm[pair_perm["pair"].isin([f"{m}-{m}" for m in MODULES])].copy()
    intra_perm["module"] = intra_perm["pair"].str.slice(0, 4)

    intra_diff = diff[diff["pair"].isin([f"{m}-{m}" for m in MODULES])].copy()
    intra_diff["module"] = intra_diff["pair"].str.slice(0, 4)

    module_size = members.groupby("module")["species"].nunique().rename("module_species").reset_index()
    ref_edges = edges[edges["pair_type"] == "intra_module"].groupby("module_a").size().rename("reference_edges_from_edge_table").reset_index().rename(columns={"module_a": "module"})

    single = intra_obs.merge(module_size, on="module", how="left").merge(ref_edges, on="module", how="left")

    overview = pd.DataFrame([
        ["source_network", "Fixed HC FlashWeave reference network; no group-specific network was rebuilt."],
        ["module_unit", "SM01-SM12 single-module internal edges only; each row summarizes edges whose two endpoints are in the same module."],
        ["retention", f"Retained edge = group CLR correlation has the same sign as the HC FlashWeave edge and |r| >= {R_KEEP}."],
        ["loss", f"Loss = edge does not satisfy same-direction |r| >= {R_KEEP}; this includes weak or direction-inconsistent edges."],
        ["reversal", f"Reversal = group CLR correlation has the opposite sign and |r| >= {R_KEEP}."],
        ["bootstrap", f"{N_BOOT} within-group bootstrap resamples were used for retention-rate 95% CI."],
        ["permutation", f"{N_PERM} pairwise label permutations preserving group sizes were used for p values."],
        ["samples", ", ".join([f"{g}={(groups == g).sum()}" for g in GROUPS])],
    ], columns=["item", "definition"])
    return overview, single, intra_ci, intra_perm, intra_diff, intra_boot


def plot_single_module_retention(single, ci, perm):
    setup()
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    x = np.arange(len(MODULES))
    width = 0.24
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
    ax.set_xticks(x)
    ax.set_xticklabels(MODULES, rotation=0)
    ax.set_ylabel("within-module retained fraction")
    ax.grid(axis="y", color="#E6E8EF", linewidth=0.55)
    ax.legend(ncol=3, loc="upper right")

    for i, m in enumerate(MODULES):
        p_nsi_hc = perm[(perm["module"] == m) & (perm["contrast"] == "MDDNSI_vs_HC")]["permutation_p"]
        p_si_hc = perm[(perm["module"] == m) & (perm["contrast"] == "MDDSI_vs_HC")]["permutation_p"]
        p_nsi = perm[(perm["module"] == m) & (perm["contrast"] == "MDDSI_vs_MDDNSI")]["permutation_p"]
        if len(p_nsi_hc) and float(p_nsi_hc.iloc[0]) < 0.05:
            y = min(ymax * 0.93, single[(single["module"] == m) & (single["group"] == "MDDNSI")]["retention_rate"].iloc[0] + ymax * 0.035)
            ax.text(i, y, stars(float(p_nsi_hc.iloc[0])), ha="center", va="bottom", color=COL["MDDNSI"], fontsize=8, fontweight="bold")
        mark = ""
        if len(p_si_hc):
            mark += stars(float(p_si_hc.iloc[0]))
        if len(p_nsi) and float(p_nsi.iloc[0]) < 0.05:
            mark += "#"
        if mark:
            y = min(ymax * 0.97, single[single["module"] == m]["retention_rate"].max() + ymax * 0.035)
            ax.text(i, y, mark, ha="center", va="bottom", color=RED, fontsize=8, fontweight="bold")
    ax.text(0.01, 0.985, "blue * NSI vs HC; red * SI vs HC; # SI vs NSI, permutation p<0.05", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.2,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88))
    fig.tight_layout()
    save(fig, "FigureSM1_single_module_retention_bootstrap_permutation")


def plot_single_module_status(single):
    setup()
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 5.8), sharex=True, sharey=True)
    status_cols = [
        ("retention_rate", "retained", "#3B7A57"),
        ("reversal_rate", "reversed", "#C98B2C"),
        ("loss_rate", "lost or weak", "#C7CCD6"),
    ]
    for ax, g in zip(axes, GROUPS):
        dd = single[single["group"] == g].set_index("module").reindex(MODULES)
        bottom = np.zeros(len(MODULES))
        for col, lab, color in status_cols:
            vals = dd[col].to_numpy()
            ax.bar(MODULES, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.35, label=lab)
            bottom += vals
        ax.set_ylim(0, 1)
        ax.set_ylabel(SHORT[g])
        ax.grid(axis="y", color="#E6E8EF", linewidth=0.45)
    axes[-1].set_xticklabels(MODULES, rotation=0)
    axes[0].legend(ncol=3, loc="upper right")
    fig.text(0.005, 0.5, "fraction of within-module reference edges", va="center", rotation=90, fontsize=7)
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    save(fig, "FigureSM2_single_module_edge_status_stacked")


def plot_delta_heatmap(intra_diff, intra_perm):
    setup()
    contrasts = [("MDDNSI_vs_HC", "NSI-HC"), ("MDDSI_vs_HC", "SI-HC"), ("MDDSI_vs_MDDNSI", "SI-NSI")]
    mat = pd.DataFrame(index=[c[1] for c in contrasts], columns=MODULES, dtype=float)
    sig = pd.DataFrame("", index=[c[1] for c in contrasts], columns=MODULES)
    for c, lab in contrasts:
        dd = intra_diff[intra_diff["contrast"] == c].set_index("module")
        pp = intra_perm[intra_perm["contrast"] == c].set_index("module")
        for m in MODULES:
            if m in dd.index:
                mat.loc[lab, m] = dd.loc[m, "delta_retention_rate"]
            if m in pp.index:
                sig.loc[lab, m] = stars(float(pp.loc[m, "permutation_p"]))
    vmax = np.nanmax(np.abs(mat.to_numpy()))
    vmax = max(vmax, 0.01)
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(MODULES)))
    ax.set_xticklabels(MODULES)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i, row in enumerate(mat.index):
        for j, m in enumerate(MODULES):
            val = mat.loc[row, m]
            label = "" if not np.isfinite(val) else f"{val:+.2f}"
            ax.text(j, i, label + sig.loc[row, m], ha="center", va="center", fontsize=6.2,
                    color="white" if abs(val) > vmax * 0.55 else DARK)
    cb = plt.colorbar(im, ax=ax, fraction=0.032, pad=0.015)
    cb.set_label("delta retained fraction")
    fig.tight_layout()
    save(fig, "FigureSM3_single_module_delta_retention_heatmap")


def plot_module_ranking(single, intra_perm):
    setup()
    pivot = single.pivot(index="module", columns="group", values="retention_rate").reindex(MODULES)
    rank = pivot.copy()
    rank["SI_minus_HC"] = rank["MDDSI"] - rank["HC"]
    rank["SI_minus_NSI"] = rank["MDDSI"] - rank["MDDNSI"]
    rank["reference_edges"] = single[single["group"] == "HC"].set_index("module").reindex(MODULES)["reference_edges"]
    rank = rank.sort_values("SI_minus_HC")
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    y = np.arange(len(rank))
    ax.hlines(y, 0, rank["SI_minus_HC"], color="#CAD0DA", linewidth=1.1)
    ax.scatter(rank["SI_minus_HC"], y, s=22 + rank["reference_edges"].fillna(0) * 0.8,
               c=np.where(rank["SI_minus_HC"] < 0, BLUE, RED), edgecolor="white", linewidth=0.45)
    pp = intra_perm[intra_perm["contrast"] == "MDDSI_vs_HC"].set_index("module")
    for i, m in enumerate(rank.index):
        if m in pp.index and float(pp.loc[m, "permutation_p"]) < 0.05:
            ax.text(rank.loc[m, "SI_minus_HC"] - 0.018 if rank.loc[m, "SI_minus_HC"] < 0 else rank.loc[m, "SI_minus_HC"] + 0.018,
                    i, stars(float(pp.loc[m, "permutation_p"])), ha="right" if rank.loc[m, "SI_minus_HC"] < 0 else "left",
                    va="center", fontsize=7, fontweight="bold")
    ax.axvline(0, color=DARK, linewidth=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(rank.index)
    ax.set_xlabel("SI - HC within-module retained fraction")
    ax.grid(axis="x", color="#E6E8EF", linewidth=0.5)
    ax.text(0.02, 0.98, "point size = within-module reference edge count", transform=ax.transAxes,
            va="top", ha="left", fontsize=6.1,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#D1D5DB", lw=0.5))
    fig.tight_layout()
    save(fig, "FigureSM4_single_module_SI_vs_HC_rank")


def plot_module_ranking_nsi(single, intra_perm):
    setup()
    pivot = single.pivot(index="module", columns="group", values="retention_rate").reindex(MODULES)
    rank = pivot.copy()
    rank["NSI_minus_HC"] = rank["MDDNSI"] - rank["HC"]
    rank["reference_edges"] = single[single["group"] == "HC"].set_index("module").reindex(MODULES)["reference_edges"]
    rank = rank.sort_values("NSI_minus_HC")
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    y = np.arange(len(rank))
    ax.hlines(y, 0, rank["NSI_minus_HC"], color="#CAD0DA", linewidth=1.1)
    ax.scatter(rank["NSI_minus_HC"], y, s=22 + rank["reference_edges"].fillna(0) * 0.8,
               c=np.where(rank["NSI_minus_HC"] < 0, BLUE, RED), edgecolor="white", linewidth=0.45)
    pp = intra_perm[intra_perm["contrast"] == "MDDNSI_vs_HC"].set_index("module")
    for i, m in enumerate(rank.index):
        if m in pp.index and float(pp.loc[m, "permutation_p"]) < 0.05:
            ax.text(rank.loc[m, "NSI_minus_HC"] - 0.018 if rank.loc[m, "NSI_minus_HC"] < 0 else rank.loc[m, "NSI_minus_HC"] + 0.018,
                    i, stars(float(pp.loc[m, "permutation_p"])), ha="right" if rank.loc[m, "NSI_minus_HC"] < 0 else "left",
                    va="center", fontsize=7, fontweight="bold")
    ax.axvline(0, color=DARK, linewidth=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(rank.index)
    ax.set_xlabel("NSI - HC within-module retained fraction")
    ax.grid(axis="x", color="#E6E8EF", linewidth=0.5)
    ax.text(0.02, 0.98, "point size = within-module reference edge count", transform=ax.transAxes,
            va="top", ha="left", fontsize=6.1,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#D1D5DB", lw=0.5))
    fig.tight_layout()
    save(fig, "FigureSM5_single_module_NSI_vs_HC_rank")


def write_outputs(overview, single, ci, perm, diff, boot):
    out_xlsx = OUT / "flashweave_single_module_internal_metrics_absr010_permutation999_results.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="00_method_overview", index=False)
        single.to_excel(writer, sheet_name="01_single_module_metrics", index=False)
        ci.to_excel(writer, sheet_name="02_bootstrap_CI", index=False)
        perm.to_excel(writer, sheet_name="03_permutation999_p", index=False)
        diff.to_excel(writer, sheet_name="04_single_module_deltas", index=False)
        boot.to_excel(writer, sheet_name="05_bootstrap_raw", index=False)

    summary = single.pivot(index="module", columns="group", values="retention_rate").reindex(MODULES)
    summary["SI_minus_HC"] = summary["MDDSI"] - summary["HC"]
    summary["SI_minus_NSI"] = summary["MDDSI"] - summary["MDDNSI"]
    report = OUT / "flashweave_single_module_internal_metrics_absr010_permutation999_report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Single-module internal FlashWeave edge retention\n\n")
        f.write("This analysis keeps the same HC FlashWeave reference network used in the previous global module analysis. It focuses only on edges whose two endpoints belong to the same refined species module (SM01-SM12).\n\n")
        f.write(f"Retained edge = same-direction group-level CLR correlation with |r| >= {R_KEEP}. Bootstrap={N_BOOT}; permutation={N_PERM}.\n\n")
        f.write("## Retention summary\n\n")
        f.write(summary.reset_index().to_csv(index=False))
        f.write("\n## Permutation p values\n\n")
        cols = ["contrast", "module", "reference_edges", "observed_abs_delta_retention", "permutation_p"]
        f.write(perm[cols].to_csv(index=False))
    return out_xlsx, report


def combine():
    from pypdf import PdfWriter
    names = [
        "FigureSM1_single_module_retention_bootstrap_permutation",
        "FigureSM2_single_module_edge_status_stacked",
        "FigureSM3_single_module_delta_retention_heatmap",
        "FigureSM4_single_module_SI_vs_HC_rank",
        "FigureSM5_single_module_NSI_vs_HC_rank",
    ]
    writer = PdfWriter()
    for name in names:
        writer.append(str(FIG / f"{name}.pdf"))
    out = FIG / "FlashWeave_single_module_internal_metrics_absr010_permutation999_combined.pdf"
    with open(out, "wb") as fh:
        writer.write(fh)
    return out


def main():
    warnings.filterwarnings("ignore")
    payload = load_cache()
    overview, single, ci, perm, diff, boot = make_single_module_tables(payload)
    plot_single_module_retention(single, ci, perm)
    plot_single_module_status(single)
    plot_delta_heatmap(diff, perm)
    plot_module_ranking(single, perm)
    plot_module_ranking_nsi(single, perm)
    xlsx, report = write_outputs(overview, single, ci, perm, diff, boot)
    combined = combine()
    shutil.copy2(__file__, CODE / Path(__file__).name)
    print(OUT)
    print(combined)
    print(xlsx)
    print(report)


if __name__ == "__main__":
    main()
