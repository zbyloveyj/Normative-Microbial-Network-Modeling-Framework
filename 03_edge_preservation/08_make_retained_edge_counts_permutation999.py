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
OUT = ROOT / "flashweave_reference_edge_counts_absr010_permutation999"
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
GREY = "#C9CED8"
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


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PNG / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def load_payload():
    if not CACHE.exists():
        raise FileNotFoundError(f"Missing cache: {CACHE}")
    return pd.read_pickle(CACHE)


def bootstrap_count_ci(boot, group_cols, value_col):
    ci = boot.groupby(group_cols)[value_col].quantile([0.025, 0.5, 0.975]).unstack().reset_index()
    ci.columns = list(group_cols) + ["ci_low", "median", "ci_high"]
    return ci


def plot_overall_counts(obs, boot, perm):
    setup()
    ci = bootstrap_count_ci(boot, ["group", "network_part"], "retained_same_direction_edges")
    parts = ["intra_module", "inter_module"]
    labels = ["within modules", "between modules"]
    x = np.arange(len(parts))
    width = 0.23
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for k, g in enumerate(GROUPS):
        dd = obs[obs["group"] == g].set_index("network_part").reindex(parts)
        cc = ci[ci["group"] == g].set_index("network_part").reindex(parts)
        xpos = x + (k - 1) * width
        ax.bar(xpos, dd["retained_same_direction_edges"], width=width, color=COL[g],
               edgecolor="white", linewidth=0.45, label=SHORT[g])
        yerr = np.vstack([
            np.maximum(dd["retained_same_direction_edges"].to_numpy() - cc["ci_low"].to_numpy(), 0),
            np.maximum(cc["ci_high"].to_numpy() - dd["retained_same_direction_edges"].to_numpy(), 0),
        ])
        ax.errorbar(xpos, dd["retained_same_direction_edges"], yerr=yerr, fmt="none",
                    ecolor=DARK, elinewidth=0.65, capsize=2)
        for i, part in enumerate(parts):
            val = int(dd.loc[part, "retained_same_direction_edges"])
            ax.text(xpos[i], val + 12, str(val), ha="center", va="bottom", fontsize=6.2, color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("same-direction retained edge count")
    ax.grid(axis="y", color="#E6E8EF", linewidth=0.55)
    ax.legend(ncol=3, loc="upper right")
    ymax = max(obs["retained_same_direction_edges"]) * 1.38
    ax.set_ylim(0, ymax)
    for i, part in enumerate(parts):
        y = obs[obs["network_part"] == part]["retained_same_direction_edges"].max() * 1.12
        for j, contrast in enumerate(["MDDNSI_vs_HC", "MDDSI_vs_HC", "MDDSI_vs_MDDNSI"]):
            p = perm[(perm["contrast"] == contrast) & (perm["network_part"] == part)]["permutation_p"]
            if len(p):
                lab = contrast.replace("MDDNSI", "NSI").replace("MDDSI", "SI").replace("_vs_", " vs ")
                ax.text(i, y + j * ymax * 0.07, f"{lab}: {fmt_p(float(p.iloc[0]))}",
                        ha="center", va="bottom", fontsize=5.8)
    fig.tight_layout()
    save(fig, "FigureCount1_overall_intra_inter_retained_edge_counts")


def plot_overall_status_counts(obs):
    setup()
    parts = ["intra_module", "inter_module"]
    labels = ["within modules", "between modules"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    for ax, part, label in zip(axes, parts, labels):
        dd = obs[obs["network_part"] == part].set_index("group").reindex(GROUPS)
        x = np.arange(len(GROUPS))
        retained = dd["retained_same_direction_edges"].to_numpy()
        reversed_edges = dd["reversed_direction_edges"].to_numpy()
        lost = dd["lost_or_nonsignificant_edges"].to_numpy()
        ax.bar(x, retained, color="#3B7A57", edgecolor="white", linewidth=0.4, label="retained")
        ax.bar(x, reversed_edges, bottom=retained, color="#C98B2C", edgecolor="white", linewidth=0.4, label="reversed")
        ax.bar(x, lost, bottom=retained + reversed_edges, color=GREY, edgecolor="white", linewidth=0.4, label="lost/weak")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[g] for g in GROUPS])
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.grid(axis="y", color="#E6E8EF", linewidth=0.55)
        for i, total in enumerate(dd["reference_edges"]):
            ax.text(i, total + max(dd["reference_edges"]) * 0.025, f"n={int(total)}", ha="center", va="bottom", fontsize=6)
    axes[0].set_ylabel("reference edge count")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "FigureCount2_overall_edge_status_counts")


def plot_single_module_count_bars(obs_pairs, boot_pairs, pair_perm):
    setup()
    obs = obs_pairs[(obs_pairs["pair_type"] == "intra_module") & (obs_pairs["module_a"].isin(MODULES))].copy()
    obs["module"] = obs["module_a"]
    boot = boot_pairs[(boot_pairs["pair_type"] == "intra_module") & (boot_pairs["module_a"].isin(MODULES))].copy()
    boot["module"] = boot["module_a"]
    ci = bootstrap_count_ci(boot, ["group", "module"], "retained_same_direction_edges")
    perm = pair_perm[pair_perm["pair"].isin([f"{m}-{m}" for m in MODULES])].copy()
    perm["module"] = perm["pair"].str.slice(0, 4)

    x = np.arange(len(MODULES))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for k, g in enumerate(GROUPS):
        dd = obs[obs["group"] == g].set_index("module").reindex(MODULES)
        cc = ci[ci["group"] == g].set_index("module").reindex(MODULES)
        xpos = x + (k - 1) * width
        ax.bar(xpos, dd["retained_same_direction_edges"], width=width, color=COL[g],
               edgecolor="white", linewidth=0.45, label=SHORT[g])
        yerr = np.vstack([
            np.maximum(dd["retained_same_direction_edges"].to_numpy() - cc["ci_low"].to_numpy(), 0),
            np.maximum(cc["ci_high"].to_numpy() - dd["retained_same_direction_edges"].to_numpy(), 0),
        ])
        ax.errorbar(xpos, dd["retained_same_direction_edges"], yerr=yerr, fmt="none",
                    ecolor=DARK, elinewidth=0.55, capsize=1.8)
    ymax = max(10, obs["retained_same_direction_edges"].max() * 1.35)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(MODULES)
    ax.set_ylabel("within-module retained edge count")
    ax.grid(axis="y", color="#E6E8EF", linewidth=0.55)
    ax.legend(ncol=3, loc="upper right")
    for i, m in enumerate(MODULES):
        p_nsi = perm[(perm["module"] == m) & (perm["contrast"] == "MDDNSI_vs_HC")]["permutation_p"]
        p_si = perm[(perm["module"] == m) & (perm["contrast"] == "MDDSI_vs_HC")]["permutation_p"]
        p_si_nsi = perm[(perm["module"] == m) & (perm["contrast"] == "MDDSI_vs_MDDNSI")]["permutation_p"]
        if len(p_nsi) and float(p_nsi.iloc[0]) < 0.05:
            y = obs[(obs["module"] == m) & (obs["group"] == "MDDNSI")]["retained_same_direction_edges"].iloc[0] + ymax * 0.035
            ax.text(i, y, stars(float(p_nsi.iloc[0])), ha="center", va="bottom", color=COL["MDDNSI"], fontsize=8, fontweight="bold")
        mark = ""
        if len(p_si):
            mark += stars(float(p_si.iloc[0]))
        if len(p_si_nsi) and float(p_si_nsi.iloc[0]) < 0.05:
            mark += "#"
        if mark:
            y = min(ymax * 0.96, obs[obs["module"] == m]["retained_same_direction_edges"].max() + ymax * 0.045)
            ax.text(i, y, mark, ha="center", va="bottom", color=RED, fontsize=8, fontweight="bold")
    ax.text(0.01, 0.985, "blue * NSI vs HC; red * SI vs HC; # SI vs NSI, permutation p<0.05",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.1,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88))
    fig.tight_layout()
    save(fig, "FigureCount3_single_module_retained_edge_counts")
    return obs, ci, perm


def plot_pair_count_heatmaps(obs_pairs):
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.25), sharex=True, sharey=True)
    vmax = obs_pairs["retained_same_direction_edges"].quantile(0.98)
    for ax, g in zip(axes, GROUPS):
        dd = obs_pairs[obs_pairs["group"] == g].copy()
        mat = pd.DataFrame(0, index=MODULES, columns=MODULES, dtype=float)
        for _, r in dd.iterrows():
            if r["module_a"] in MODULES and r["module_b"] in MODULES:
                mat.loc[r["module_a"], r["module_b"]] = r["retained_same_direction_edges"]
                mat.loc[r["module_b"], r["module_a"]] = r["retained_same_direction_edges"]
        im = ax.imshow(mat.to_numpy(), cmap="YlGnBu", vmin=0, vmax=vmax)
        ax.set_title(SHORT[g], color=COL[g], fontsize=8, fontweight="bold")
        ax.set_xticks(range(12)); ax.set_xticklabels(MODULES, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(12)); ax.set_yticklabels(MODULES, fontsize=5.5)
    cb = plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.02)
    cb.set_label("retained edge count")
    save(fig, "FigureCount4_module_pair_retained_edge_count_heatmaps")


def plot_between_module_delta_count_heatmaps(obs_pairs, pair_perm):
    setup()
    contrasts = [
        ("MDDNSI_vs_HC", "NSI - HC"),
        ("MDDSI_vs_HC", "SI - HC"),
        ("MDDSI_vs_MDDNSI", "SI - NSI"),
    ]
    values = obs_pairs[obs_pairs["pair_type"] == "inter_module"].copy()
    lookup = values.set_index(["group", "module_a", "module_b"])["retained_same_direction_edges"].to_dict()
    mats = []
    for contrast, _ in contrasts:
        g1, g0 = contrast.split("_vs_")
        mat = pd.DataFrame(np.nan, index=MODULES, columns=MODULES)
        for _, r in values.drop_duplicates(["module_a", "module_b"]).iterrows():
            a, b = r["module_a"], r["module_b"]
            if a not in MODULES or b not in MODULES or a == b:
                continue
            v1 = lookup.get((g1, a, b), lookup.get((g1, b, a), np.nan))
            v0 = lookup.get((g0, a, b), lookup.get((g0, b, a), np.nan))
            if np.isfinite(v1) and np.isfinite(v0):
                mat.loc[a, b] = v1 - v0
                mat.loc[b, a] = v1 - v0
        mats.append(mat)
    vmax = max(1, np.nanmax(np.abs(np.stack([m.to_numpy() for m in mats]))))
    fig, axes = plt.subplots(1, 3, figsize=(9.7, 3.25), sharex=True, sharey=True)
    for ax, mat, (contrast, title) in zip(axes, mats, contrasts):
        im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xticks(range(12)); ax.set_xticklabels(MODULES, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(12)); ax.set_yticklabels(MODULES, fontsize=5.5)
        pp = pair_perm[(pair_perm["contrast"] == contrast) & (pair_perm["pair_type"] == "inter_module")]
        for _, r in pp.iterrows():
            if r["module_a"] in MODULES and r["module_b"] in MODULES and r["permutation_p"] < 0.05:
                i = MODULES.index(r["module_a"])
                j = MODULES.index(r["module_b"])
                ax.text(j, i, "*", ha="center", va="center", fontsize=7, color=DARK)
                ax.text(i, j, "*", ha="center", va="center", fontsize=7, color=DARK)
    cb = plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.02)
    cb.set_label("delta retained edge count")
    fig.text(0.075, 0.965, "* permutation p<0.05; cells are between-module FlashWeave reference edges", fontsize=6.2)
    save(fig, "FigureCount5_between_module_pairwise_delta_count_heatmaps")


def plot_between_module_raw_nsi_si_matrices(obs_pairs):
    setup()
    groups = ["MDDNSI", "MDDSI"]
    values = obs_pairs[obs_pairs["pair_type"] == "inter_module"].copy()
    vmax = max(1, values[values["group"].isin(groups)]["retained_same_direction_edges"].max())
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.25), sharex=True, sharey=True)
    for ax, g in zip(axes, groups):
        dd = values[values["group"] == g]
        mat = pd.DataFrame(np.nan, index=MODULES, columns=MODULES)
        for _, r in dd.iterrows():
            a, b = r["module_a"], r["module_b"]
            if a in MODULES and b in MODULES and a != b:
                mat.loc[a, b] = r["retained_same_direction_edges"]
                mat.loc[b, a] = r["retained_same_direction_edges"]
        im = ax.imshow(mat.to_numpy(), cmap="YlGnBu", vmin=0, vmax=vmax)
        ax.set_title(SHORT[g], color=COL[g], fontsize=8, fontweight="bold")
        ax.set_xticks(range(12)); ax.set_xticklabels(MODULES, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(12)); ax.set_yticklabels(MODULES, fontsize=5.5)
        for i in range(12):
            for j in range(12):
                val = mat.iloc[i, j]
                if np.isfinite(val) and val > 0:
                    ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=4.6,
                            color="white" if val > vmax * 0.55 else DARK)
    cb = plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.032, pad=0.02)
    cb.set_label("raw retained edge count")
    fig.text(0.08, 0.965, "Raw between-module matrices; no subtraction from HC and no rate normalization", fontsize=6.2)
    save(fig, "FigureCount6_between_module_raw_NSI_SI_count_matrices")


def plot_between_module_si_nsi_retention_rate_comparison(obs_pairs, pair_perm):
    setup()
    values = obs_pairs[obs_pairs["pair_type"] == "inter_module"].copy()

    def make_rate_matrix(group):
        dd = values[values["group"] == group]
        mat = pd.DataFrame(np.nan, index=MODULES, columns=MODULES)
        for _, r in dd.iterrows():
            a, b = r["module_a"], r["module_b"]
            if a in MODULES and b in MODULES and a != b:
                mat.loc[a, b] = r["retention_rate"]
                mat.loc[b, a] = r["retention_rate"]
        return mat

    nsi = make_rate_matrix("MDDNSI")
    si = make_rate_matrix("MDDSI")
    delta = si - nsi
    vmax_rate = max(0.01, np.nanmax([np.nanmax(nsi.to_numpy()), np.nanmax(si.to_numpy())]))
    vmax_delta = max(0.01, np.nanmax(np.abs(delta.to_numpy())))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), sharex=True, sharey=True)

    for ax, mat, title, color in [
        (axes[0], nsi, "NSI retention rate", COL["MDDNSI"]),
        (axes[1], si, "SI retention rate", COL["MDDSI"]),
    ]:
        im0 = ax.imshow(mat.to_numpy(), cmap="YlGnBu", vmin=0, vmax=vmax_rate)
        ax.set_title(title, color=color, fontsize=8, fontweight="bold")
        ax.set_xticks(range(12)); ax.set_xticklabels(MODULES, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(12)); ax.set_yticklabels(MODULES, fontsize=5.5)

    im1 = axes[2].imshow(delta.to_numpy(), cmap="RdBu_r", vmin=-vmax_delta, vmax=vmax_delta)
    axes[2].set_title("SI - NSI", fontsize=8, fontweight="bold")
    axes[2].set_xticks(range(12)); axes[2].set_xticklabels(MODULES, rotation=60, ha="right", fontsize=5.5)
    axes[2].set_yticks(range(12)); axes[2].set_yticklabels(MODULES, fontsize=5.5)
    pp = pair_perm[(pair_perm["contrast"] == "MDDSI_vs_MDDNSI") & (pair_perm["pair_type"] == "inter_module")]
    for _, r in pp.iterrows():
        if r["module_a"] in MODULES and r["module_b"] in MODULES and r["permutation_p"] < 0.05:
            i = MODULES.index(r["module_a"])
            j = MODULES.index(r["module_b"])
            axes[2].text(j, i, "*", ha="center", va="center", fontsize=7, color=DARK)
            axes[2].text(i, j, "*", ha="center", va="center", fontsize=7, color=DARK)

    cb0 = plt.colorbar(im0, ax=axes[:2].ravel().tolist(), fraction=0.026, pad=0.02)
    cb0.set_label("retained edge fraction")
    cb1 = plt.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.02)
    cb1.set_label("delta retained fraction")
    fig.text(0.075, 0.965, "Between-module retention rate matrices; * on SI-NSI indicates permutation p<0.05", fontsize=6.2)
    save(fig, "FigureCount7_between_module_SI_NSI_retention_rate_comparison")
    return nsi, si, delta


def combine():
    from pypdf import PdfWriter
    names = [
        "FigureCount1_overall_intra_inter_retained_edge_counts",
        "FigureCount2_overall_edge_status_counts",
        "FigureCount3_single_module_retained_edge_counts",
        "FigureCount4_module_pair_retained_edge_count_heatmaps",
        "FigureCount5_between_module_pairwise_delta_count_heatmaps",
        "FigureCount6_between_module_raw_NSI_SI_count_matrices",
        "FigureCount7_between_module_SI_NSI_retention_rate_comparison",
    ]
    writer = PdfWriter()
    for name in names:
        writer.append(str(FIG / f"{name}.pdf"))
    out = FIG / "FlashWeave_reference_edge_counts_absr010_permutation999_combined.pdf"
    with open(out, "wb") as fh:
        writer.write(fh)
    return out


def main():
    warnings.filterwarnings("ignore")
    payload = load_payload()
    obs_overall = payload["obs_overall"].copy()
    obs_pairs = payload["obs_pairs"].copy()
    boot = payload["boot"].copy()
    boot_pairs = payload["boot_pairs"].copy()
    perm = payload["perm"].copy()
    pair_perm = payload["pair_perm"].copy()

    plot_overall_counts(obs_overall, boot, perm)
    plot_overall_status_counts(obs_overall)
    single_obs, single_ci, single_perm = plot_single_module_count_bars(obs_pairs, boot_pairs, pair_perm)
    plot_pair_count_heatmaps(obs_pairs)
    plot_between_module_delta_count_heatmaps(obs_pairs, pair_perm)
    plot_between_module_raw_nsi_si_matrices(obs_pairs)
    nsi_rate_mat, si_rate_mat, si_nsi_delta_rate_mat = plot_between_module_si_nsi_retention_rate_comparison(obs_pairs, pair_perm)
    combined = combine()

    overview = pd.DataFrame([
        ["main_quantity", "Counts are used instead of retention rates."],
        ["numerator", "retained_same_direction_edges = number of fixed HC FlashWeave reference edges retained in each group."],
        ["retained_edge", "same-direction group CLR Pearson correlation with |r| >= 0.10"],
        ["within_module", "both endpoints belong to the same SM module"],
        ["between_module", "two endpoints belong to different SM modules"],
        ["permutation", "999 pairwise label permutations preserving group sizes; p values inherited from the same fixed-edge test"],
    ], columns=["item", "definition"])
    with pd.ExcelWriter(OUT / "flashweave_reference_edge_counts_absr010_permutation999_results.xlsx", engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="00_method_overview", index=False)
        obs_overall.to_excel(writer, sheet_name="01_overall_counts", index=False)
        perm.to_excel(writer, sheet_name="02_overall_permutation999", index=False)
        obs_pairs.to_excel(writer, sheet_name="03_module_pair_counts", index=False)
        pair_perm.to_excel(writer, sheet_name="04_pair_permutation999", index=False)
        single_obs.to_excel(writer, sheet_name="05_single_module_counts", index=False)
        single_ci.to_excel(writer, sheet_name="06_single_module_boot_CI", index=False)
        single_perm.to_excel(writer, sheet_name="07_single_module_perm999", index=False)
        nsi_rate_mat.to_excel(writer, sheet_name="08_NSI_between_rate_matrix")
        si_rate_mat.to_excel(writer, sheet_name="09_SI_between_rate_matrix")
        si_nsi_delta_rate_mat.to_excel(writer, sheet_name="10_SI_minus_NSI_rate_matrix")

    report = OUT / "flashweave_reference_edge_counts_absr010_permutation999_report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# FlashWeave retained edge count analysis\n\n")
        f.write("This version uses retained edge counts, not retention rates. It therefore emphasizes the numerator: how many HC FlashWeave reference edges remain same-direction retained in each group.\n\n")
        f.write("## Overall counts\n\n")
        f.write(obs_overall.to_csv(index=False))
        f.write("\n## Pairwise permutation tests\n\n")
        f.write(perm.to_csv(index=False))

    shutil.copy2(__file__, CODE / Path(__file__).name)
    print(OUT)
    print(combined)
    print(OUT / "flashweave_reference_edge_counts_absr010_permutation999_results.xlsx")
    print(report)


if __name__ == "__main__":
    main()
