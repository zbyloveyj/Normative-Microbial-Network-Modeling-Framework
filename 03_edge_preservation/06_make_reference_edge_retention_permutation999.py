from pathlib import Path
import math
import os
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(r"C:\Users\15220\Desktop\article") / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\15220\Desktop\article")
DATA = ROOT / "complete_processed_data_package" / "MDD_SI_complete_processed_and_plotting_data.xlsx"
OUT = ROOT / "flashweave_reference_edge_retention_absr010_permutation999"
FIG = OUT / "figures_pdf"
PNG = OUT / "figures_png"
CODE = OUT / "code"
for d in [OUT, FIG, PNG, CODE]:
    d.mkdir(parents=True, exist_ok=True)

GROUPS = ["HC", "MDDNSI", "MDDSI"]
SHORT = {"HC": "HC", "MDDNSI": "NSI", "MDDSI": "SI"}
COL = {"HC": "#5BAA7D", "MDDNSI": "#4C78A8", "MDDSI": "#D55E5E"}
BLUE = "#3B5B92"
RED = "#C44E52"
TEAL = "#2A9D8F"
ORANGE = "#E69F00"
GREY = "#8C96A6"
DARK = "#2F3742"
RNG = np.random.default_rng(20260729)
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


def normal_p(z):
    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, np.nan)
    it = np.nditer(z, flags=["multi_index"])
    for val in it:
        out[it.multi_index] = math.erfc(abs(float(val)) / math.sqrt(2)) if np.isfinite(val) else np.nan
    return out


def corr_p(r, n):
    r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
    z = np.arctanh(r) * math.sqrt(max(n - 3, 1))
    return normal_p(z)


def fmt_p(p):
    if not np.isfinite(p):
        return "p=NA"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def load_inputs():
    species = pd.read_excel(DATA, sheet_name="01_all_samples_species_matrix")
    meta = pd.read_excel(DATA, sheet_name="03_all_sample_metadata")
    edges = pd.read_excel(DATA, sheet_name="F12_FlashWeave_edges")
    members = pd.read_excel(DATA, sheet_name="F12_module_membership_SM01_SM12")
    species = species.rename(columns={species.columns[0]: "sample"})

    edges = edges.rename(columns={edges.columns[0]: "a", edges.columns[1]: "b", edges.columns[2]: "weight"})
    module_map = dict(zip(members["species"].astype(str), members["module"].astype(str)))
    all_species = set(species.columns[1:])
    raw_reference_edges = len(edges)
    edges = edges[edges["a"].isin(all_species) & edges["b"].isin(all_species)].copy()
    edges["_edge_key"] = edges.apply(lambda r: "||".join(sorted([str(r["a"]), str(r["b"])])), axis=1)
    edges = edges.drop_duplicates("_edge_key").drop(columns="_edge_key").copy()
    species_available_edges = len(edges)
    edges = edges[edges["a"].isin(module_map) & edges["b"].isin(module_map)].copy()
    module_mapped_edges = len(edges)
    edges["module_a"] = edges["a"].map(module_map)
    edges["module_b"] = edges["b"].map(module_map)
    edges["module_lo"] = edges[["module_a", "module_b"]].min(axis=1)
    edges["module_hi"] = edges[["module_a", "module_b"]].max(axis=1)
    edges["pair"] = edges["module_lo"] + "-" + edges["module_hi"]
    edges["pair_type"] = np.where(edges["module_a"] == edges["module_b"], "intra_module", "inter_module")
    edges["ref_sign"] = np.sign(edges["weight"].astype(float))
    edges.loc[edges["ref_sign"] == 0, "ref_sign"] = 1

    taxa = sorted(set(edges["a"]).union(edges["b"]))
    taxon_index = {t: i for i, t in enumerate(taxa)}
    edges["ia"] = edges["a"].map(taxon_index).astype(int)
    edges["ib"] = edges["b"].map(taxon_index).astype(int)

    df = meta[["sample", "clinical_group"]].merge(species[["sample"] + taxa], on="sample", how="inner")
    df = df[df["clinical_group"].isin(GROUPS)].copy()
    X = df[taxa].astype(float).to_numpy()
    groups = df["clinical_group"].astype(str).to_numpy()
    samples = df["sample"].astype(str).to_numpy()
    audit = {
        "raw_flashweave_edge_rows": raw_reference_edges,
        "species_available_edges": species_available_edges,
        "module_mapped_edges_for_SM01_SM12": module_mapped_edges,
        "excluded_from_module_pair_analysis": species_available_edges - module_mapped_edges,
    }
    return X, groups, samples, taxa, edges, members, audit


def clr(X):
    positives = X[X > 0]
    pseudo = float(np.nanmin(positives) / 2) if positives.size else 1e-9
    L = np.log(X + pseudo)
    return L - L.mean(axis=1, keepdims=True)


def edge_correlations(X_clr, idx, edges):
    X = X_clr[idx, :]
    n = X.shape[0]
    X = X - X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = np.nan
    Z = X / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    a = edges["ia"].to_numpy()
    b = edges["ib"].to_numpy()
    r = np.sum(Z[:, a] * Z[:, b], axis=0) / max(n - 1, 1)
    r = np.clip(r, -1, 1)
    p = corr_p(r, n)
    sign = np.sign(r)
    ref = edges["ref_sign"].to_numpy()
    retained = (np.abs(r) >= R_KEEP) & (sign == ref)
    reversed_edge = (np.abs(r) >= R_KEEP) & (sign == -ref)
    active = np.abs(r) >= R_KEEP
    return pd.DataFrame({
        "edge_id": np.arange(len(edges)),
        "r": r,
        "p": p,
        "active": active,
        "retained_same_direction": retained,
        "reversed_direction": reversed_edge,
        "lost_or_nonsignificant": ~active,
    })


def summarize(edge_eval, edges, group, n):
    dat = edges[["pair_type", "pair", "module_lo", "module_hi"]].copy()
    dat = pd.concat([dat, edge_eval[["active", "retained_same_direction", "reversed_direction"]]], axis=1)
    rows = []
    for part in ["intra_module", "inter_module"]:
        sub = dat[dat["pair_type"] == part]
        ref = len(sub)
        retained = int(sub["retained_same_direction"].sum())
        active = int(sub["active"].sum())
        rev = int(sub["reversed_direction"].sum())
        rows.append({
            "group": group,
            "n": n,
            "network_part": part,
            "reference_edges": ref,
            "active_edges": active,
            "retained_same_direction_edges": retained,
            "reversed_direction_edges": rev,
            "lost_or_nonsignificant_edges": ref - active,
            "retention_rate": retained / max(ref, 1),
            "active_rate": active / max(ref, 1),
            "reversal_rate": rev / max(ref, 1),
        })
    return rows


def summarize_pairs(edge_eval, edges, group, n):
    dat = edges[["pair_type", "pair", "module_lo", "module_hi"]].copy()
    dat = pd.concat([dat, edge_eval[["active", "retained_same_direction", "reversed_direction"]]], axis=1)
    rows = []
    for (pair, lo, hi, part), sub in dat.groupby(["pair", "module_lo", "module_hi", "pair_type"], observed=True):
        ref = len(sub)
        retained = int(sub["retained_same_direction"].sum())
        active = int(sub["active"].sum())
        rev = int(sub["reversed_direction"].sum())
        rows.append({
            "group": group,
            "n": n,
            "pair": pair,
            "module_a": lo,
            "module_b": hi,
            "pair_type": part,
            "reference_edges": ref,
            "active_edges": active,
            "retained_same_direction_edges": retained,
            "reversed_direction_edges": rev,
            "lost_or_nonsignificant_edges": ref - active,
            "retention_rate": retained / max(ref, 1),
            "active_rate": active / max(ref, 1),
            "reversal_rate": rev / max(ref, 1),
        })
    return rows


def evaluate_groups(X_clr, groups, edges):
    edge_evals = []
    overall = []
    pairs = []
    for g in GROUPS:
        idx = np.where(groups == g)[0]
        ee = edge_correlations(X_clr, idx, edges)
        ee.insert(0, "group", g)
        edge_evals.append(ee)
        overall.extend(summarize(ee, edges, g, len(idx)))
        pairs.extend(summarize_pairs(ee, edges, g, len(idx)))
    return pd.concat(edge_evals, ignore_index=True), pd.DataFrame(overall), pd.DataFrame(pairs)


def bootstrap_ci(X_clr, groups, edges):
    rows = []
    pair_rows = []
    for g in GROUPS:
        idx = np.where(groups == g)[0]
        for b in range(N_BOOT):
            bidx = RNG.choice(idx, size=len(idx), replace=True)
            ee = edge_correlations(X_clr, bidx, edges)
            rows.extend([{**r, "bootstrap": b} for r in summarize(ee, edges, g, len(idx))])
            pair_rows.extend([{**r, "bootstrap": b} for r in summarize_pairs(ee, edges, g, len(idx))])
    boot = pd.DataFrame(rows)
    boot_pairs = pd.DataFrame(pair_rows)
    ci = boot.groupby(["group", "network_part"])["retention_rate"].quantile([0.025, 0.5, 0.975]).unstack().reset_index()
    ci.columns = ["group", "network_part", "ci_low", "median", "ci_high"]
    pair_ci = boot_pairs.groupby(["group", "pair", "module_a", "module_b", "pair_type"])["retention_rate"].quantile([0.025, 0.5, 0.975]).unstack().reset_index()
    pair_ci.columns = ["group", "pair", "module_a", "module_b", "pair_type", "ci_low", "median", "ci_high"]
    return boot, boot_pairs, ci, pair_ci


def permutation_tests(X_clr, groups, edges, obs_overall, obs_pairs):
    contrasts = [("MDDNSI", "HC"), ("MDDSI", "HC"), ("MDDSI", "MDDNSI")]
    perm_rows = []
    perm_pair_rows = []
    for a, b in contrasts:
        keep = np.where((groups == a) | (groups == b))[0]
        labels = groups[keep].copy()
        n_a = int((labels == a).sum())
        for k in range(N_PERM):
            shuffled = RNG.permutation(labels)
            idx_a = keep[shuffled == a]
            idx_b = keep[shuffled == b]
            eval_a = edge_correlations(X_clr, idx_a, edges)
            eval_b = edge_correlations(X_clr, idx_b, edges)
            sum_a = pd.DataFrame(summarize(eval_a, edges, a, n_a)).set_index("network_part")
            sum_b = pd.DataFrame(summarize(eval_b, edges, b, len(labels) - n_a)).set_index("network_part")
            pair_a = pd.DataFrame(summarize_pairs(eval_a, edges, a, n_a)).set_index("pair")
            pair_b = pd.DataFrame(summarize_pairs(eval_b, edges, b, len(labels) - n_a)).set_index("pair")
            for part in ["intra_module", "inter_module"]:
                perm_rows.append({
                    "contrast": f"{a}_vs_{b}",
                    "network_part": part,
                    "perm_abs_delta": abs(sum_a.loc[part, "retention_rate"] - sum_b.loc[part, "retention_rate"]),
                })
            common = pair_a.index.intersection(pair_b.index)
            for pair in common:
                perm_pair_rows.append({
                    "contrast": f"{a}_vs_{b}",
                    "pair": pair,
                    "perm_abs_delta": abs(pair_a.loc[pair, "retention_rate"] - pair_b.loc[pair, "retention_rate"]),
                })

    obs_wide = obs_overall.pivot(index="network_part", columns="group", values="retention_rate")
    obs_pair_wide = obs_pairs.pivot(index="pair", columns="group", values="retention_rate")
    perm = pd.DataFrame(perm_rows)
    perm_pairs = pd.DataFrame(perm_pair_rows)
    out = []
    for (contrast, part), dd in perm.groupby(["contrast", "network_part"]):
        a, b = contrast.replace("_vs_", "|").split("|")
        obs_delta = abs(obs_wide.loc[part, a] - obs_wide.loc[part, b])
        out.append({
            "contrast": contrast,
            "network_part": part,
            "observed_abs_delta_retention": obs_delta,
            "permutation_p": (1 + (dd["perm_abs_delta"] >= obs_delta).sum()) / (len(dd) + 1),
            "n_permutation": len(dd),
        })
    pair_out = []
    pair_meta = obs_pairs.drop_duplicates("pair").set_index("pair")[["module_a", "module_b", "pair_type", "reference_edges"]]
    for (contrast, pair), dd in perm_pairs.groupby(["contrast", "pair"]):
        a, b = contrast.replace("_vs_", "|").split("|")
        if pair not in obs_pair_wide.index:
            continue
        obs_delta = abs(obs_pair_wide.loc[pair, a] - obs_pair_wide.loc[pair, b])
        row = {
            "contrast": contrast,
            "pair": pair,
            "observed_abs_delta_retention": obs_delta,
            "permutation_p": (1 + (dd["perm_abs_delta"] >= obs_delta).sum()) / (len(dd) + 1),
            "n_permutation": len(dd),
        }
        row.update(pair_meta.loc[pair].to_dict())
        pair_out.append(row)
    return pd.DataFrame(out), pd.DataFrame(pair_out)


def differential_pairs(obs_pairs):
    rows = []
    piv = obs_pairs.pivot(index="pair", columns="group", values="retention_rate")
    rev = obs_pairs.pivot(index="pair", columns="group", values="reversal_rate")
    meta = obs_pairs.drop_duplicates("pair").set_index("pair")[["module_a", "module_b", "pair_type", "reference_edges"]]
    for contrast, a, b in [("MDDNSI_vs_HC", "MDDNSI", "HC"), ("MDDSI_vs_HC", "MDDSI", "HC"), ("MDDSI_vs_MDDNSI", "MDDSI", "MDDNSI")]:
        for pair in piv.index:
            rows.append({
                "contrast": contrast,
                "pair": pair,
                "module_a": meta.loc[pair, "module_a"],
                "module_b": meta.loc[pair, "module_b"],
                "pair_type": meta.loc[pair, "pair_type"],
                "reference_edges": meta.loc[pair, "reference_edges"],
                "delta_retention_rate": piv.loc[pair, a] - piv.loc[pair, b],
                "delta_reversal_rate": rev.loc[pair, a] - rev.loc[pair, b],
                "group_a_retention": piv.loc[pair, a],
                "group_b_retention": piv.loc[pair, b],
            })
    return pd.DataFrame(rows)


def matrix_from_pairs(df, value):
    mods = [f"SM{i:02d}" for i in range(1, 13)]
    mat = np.full((12, 12), np.nan)
    for _, r in df.iterrows():
        i = mods.index(r["module_a"])
        j = mods.index(r["module_b"])
        mat[i, j] = r[value]
        mat[j, i] = r[value]
    return mat


def plot_reference_composition(edges):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), gridspec_kw={"width_ratios": [0.8, 1.4]})
    counts = edges["pair_type"].value_counts().reindex(["intra_module", "inter_module"])
    axes[0].bar([0, 1], counts.values, color=[TEAL, GREY], edgecolor="white")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["within\nmodules", "between\nmodules"])
    axes[0].set_ylabel("FlashWeave reference edge count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    pair_ref = edges.groupby(["pair", "module_lo", "module_hi", "pair_type"]).size().reset_index(name="reference_edges")
    mat = matrix_from_pairs(pair_ref.rename(columns={"module_lo": "module_a", "module_hi": "module_b"}), "reference_edges")
    im = axes[1].imshow(mat, cmap="YlGnBu")
    mods = [f"SM{i:02d}" for i in range(1, 13)]
    axes[1].set_xticks(range(12)); axes[1].set_xticklabels(mods, rotation=60, ha="right", fontsize=6)
    axes[1].set_yticks(range(12)); axes[1].set_yticklabels(mods, fontsize=6)
    cb = plt.colorbar(im, ax=axes[1], fraction=0.035, pad=0.02)
    cb.set_label("reference edges")
    fig.tight_layout()
    save(fig, "FigureFW1_reference_edge_module_composition")


def plot_overall(obs, ci, perm):
    setup()
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    parts = ["intra_module", "inter_module"]
    x = np.arange(len(parts))
    width = 0.23
    for k, g in enumerate(GROUPS):
        dd = obs[obs["group"] == g].set_index("network_part").loc[parts]
        cc = ci[ci["group"] == g].set_index("network_part").loc[parts]
        xpos = x + (k - 1) * width
        ax.bar(xpos, dd["retention_rate"], width=width, color=COL[g], edgecolor="white", linewidth=0.5, label=SHORT[g])
        yerr = np.vstack([
            np.maximum(dd["retention_rate"].to_numpy() - cc["ci_low"].to_numpy(), 0),
            np.maximum(cc["ci_high"].to_numpy() - dd["retention_rate"].to_numpy(), 0),
        ])
        ax.errorbar(xpos, dd["retention_rate"], yerr=yerr, fmt="none", ecolor=DARK, elinewidth=0.7, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["within modules", "between modules"])
    ax.set_ylabel("same-direction retained edge fraction")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.5)
    ax.legend(ncol=3, loc="upper right")
    ymax = max(0.08, obs["retention_rate"].max() * 1.55)
    ax.set_ylim(0, ymax)
    for i, part in enumerate(parts):
        y = obs[obs["network_part"] == part]["retention_rate"].max() * 1.10
        for j, contrast in enumerate(["MDDNSI_vs_HC", "MDDSI_vs_HC", "MDDSI_vs_MDDNSI"]):
            pval = perm[(perm["contrast"] == contrast) & (perm["network_part"] == part)]["permutation_p"]
            if len(pval):
                lab = contrast.replace("MDDNSI", "NSI").replace("MDDSI", "SI").replace("_vs_", " vs ")
                ax.text(i, y + j * ymax * 0.07, f"{lab}: {fmt_p(float(pval.iloc[0]))}", ha="center", va="bottom", fontsize=5.8)
    fig.tight_layout()
    save(fig, "FigureFW2_overall_intra_inter_retention_bootstrap_permutation")


def plot_module_pair_matrices(obs_pairs):
    setup()
    mods = [f"SM{i:02d}" for i in range(1, 13)]
    fig, axes = plt.subplots(1, 3, figsize=(9.7, 3.35), sharex=True, sharey=True)
    vmax = max(0.02, obs_pairs["retention_rate"].quantile(0.98))
    for ax, g in zip(axes, GROUPS):
        mat = matrix_from_pairs(obs_pairs[obs_pairs["group"] == g], "retention_rate")
        im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=vmax)
        ax.set_title(SHORT[g], color=COL[g], fontweight="bold")
        ax.set_xticks(range(12)); ax.set_xticklabels(mods, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(12)); ax.set_yticklabels(mods, fontsize=5.5)
    cb = plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cb.set_label("same-direction retained fraction")
    save(fig, "FigureFW3_module_pair_retention_matrices")


def plot_delta_heatmaps(diff, pair_perm):
    setup()
    mods = [f"SM{i:02d}" for i in range(1, 13)]
    contrasts = [("MDDSI_vs_HC", "SI - HC"), ("MDDSI_vs_MDDNSI", "SI - NSI")]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), sharex=True, sharey=True)
    mats = []
    for c, _ in contrasts:
        dd = diff[diff["contrast"] == c].rename(columns={"delta_retention_rate": "value"})
        mats.append(matrix_from_pairs(dd, "value"))
    vmax = max(0.01, np.nanmax(np.abs(np.stack(mats))))
    for ax, mat, (c, title) in zip(axes, mats, contrasts):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(12)); ax.set_xticklabels(mods, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(12)); ax.set_yticklabels(mods, fontsize=5.5)
        sig = pair_perm[(pair_perm["contrast"] == c) & (pair_perm["permutation_p"] < 0.05)]
        for _, r in sig.iterrows():
            i = mods.index(r["module_a"]); j = mods.index(r["module_b"])
            ax.text(j, i, "*", ha="center", va="center", fontsize=7, color="black")
            if i != j:
                ax.text(i, j, "*", ha="center", va="center", fontsize=7, color="black")
    cb = plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cb.set_label("delta retained fraction")
    fig.text(0.08, 0.97, "* pairwise permutation p<0.05 (999 permutations)", fontsize=6)
    save(fig, "FigureFW4_delta_retention_heatmaps_permutation")


def plot_within_module(obs_pairs, pair_perm):
    setup()
    mods = [f"SM{i:02d}" for i in range(1, 13)]
    dd = obs_pairs[obs_pairs["pair_type"] == "intra_module"].copy()
    mat = dd.pivot(index="module_a", columns="group", values="retention_rate").reindex(index=mods, columns=GROUPS)
    fig, ax = plt.subplots(figsize=(3.7, 4.3))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(3)); ax.set_xticklabels([SHORT[g] for g in GROUPS])
    ax.set_yticks(range(12)); ax.set_yticklabels(mods)
    for i, m in enumerate(mods):
        for j, g in enumerate(GROUPS):
            ax.text(j, i, f"{mat.loc[m, g]:.2f}", ha="center", va="center", fontsize=5.7)
        sig = ""
        p1 = pair_perm[(pair_perm["contrast"] == "MDDSI_vs_HC") & (pair_perm["pair"] == f"{m}-{m}")]["permutation_p"]
        p2 = pair_perm[(pair_perm["contrast"] == "MDDSI_vs_MDDNSI") & (pair_perm["pair"] == f"{m}-{m}")]["permutation_p"]
        if len(p1) and float(p1.iloc[0]) < 0.05:
            sig += "*"
        if len(p2) and float(p2.iloc[0]) < 0.05:
            sig += "#"
        if sig:
            ax.text(3.10, i, sig, ha="left", va="center", color=RED, fontweight="bold", fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("within-module retained fraction")
    ax.text(0.0, 1.02, "* SI vs HC; # SI vs NSI, permutation p<0.05", transform=ax.transAxes, fontsize=6)
    fig.tight_layout()
    save(fig, "FigureFW5_within_module_retention_heatmap")


def plot_top_changed(diff, pair_perm):
    setup()
    dd = diff[diff["contrast"] == "MDDSI_vs_MDDNSI"].copy()
    if "permutation_p" not in dd.columns:
        dd = dd.merge(pair_perm[pair_perm["contrast"] == "MDDSI_vs_MDDNSI"][["pair", "permutation_p"]], on="pair", how="left")
    elif "permutation_p_x" in dd.columns:
        dd["permutation_p"] = dd["permutation_p_x"]
    dd["abs_delta"] = dd["delta_retention_rate"].abs()
    dd = dd.sort_values("abs_delta", ascending=False).head(18).sort_values("delta_retention_rate")
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    y = np.arange(len(dd))
    ax.hlines(y, 0, dd["delta_retention_rate"], color="#C4C9D2", linewidth=1.1)
    colors = np.where(dd["delta_retention_rate"] >= 0, RED, BLUE)
    ax.scatter(dd["delta_retention_rate"], y, s=28 + dd["reference_edges"] * 0.55, c=colors, edgecolor="white", linewidth=0.45)
    for i, (_, r) in enumerate(dd.iterrows()):
        if np.isfinite(r["permutation_p"]) and r["permutation_p"] < 0.05:
            ax.text(r["delta_retention_rate"] + (0.01 if r["delta_retention_rate"] >= 0 else -0.01), i, stars(r["permutation_p"]),
                    ha="left" if r["delta_retention_rate"] >= 0 else "right", va="center", fontsize=7, fontweight="bold")
    ax.axvline(0, color=DARK, linewidth=0.7)
    ax.set_yticks(y); ax.set_yticklabels(dd["pair"], fontsize=6)
    ax.set_xlabel("SI - NSI retained fraction")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.5)
    ax.text(0.02, 0.98, "point size = reference edge count", transform=ax.transAxes, va="top", fontsize=6.2,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#D1D5DB", lw=0.5, alpha=0.9))
    fig.tight_layout()
    save(fig, "FigureFW6_top_SI_vs_NSI_changed_module_pairs")


def plot_reversal(obs_pairs):
    setup()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    parts = ["intra_module", "inter_module"]
    x = np.arange(len(parts))
    width = 0.23
    overall = obs_pairs.groupby(["group", "pair_type"]).agg(
        reference_edges=("reference_edges", "sum"),
        reversed_edges=("reversed_direction_edges", "sum"),
    ).reset_index()
    overall["reversal_rate"] = overall["reversed_edges"] / overall["reference_edges"]
    for k, g in enumerate(GROUPS):
        dd = overall[overall["group"] == g].set_index("pair_type").loc[parts]
        ax.bar(x + (k - 1) * width, dd["reversal_rate"], width=width, color=COL[g], edgecolor="white", linewidth=0.5, label=SHORT[g])
    ax.set_xticks(x); ax.set_xticklabels(["within modules", "between modules"])
    ax.set_ylabel("opposite-direction |r|>=0.1 edge fraction")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.5)
    ax.legend(ncol=3)
    fig.tight_layout()
    save(fig, "FigureFW7_direction_reversal_rate")


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PNG / f"{stem}.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def combine(names):
    from pypdf import PdfWriter
    writer = PdfWriter()
    for n in names:
        writer.append(str(FIG / f"{n}.pdf"))
    with open(FIG / "FlashWeave_reference_edge_retention_absr010_permutation999_combined.pdf", "wb") as fh:
        writer.write(fh)


def main():
    warnings.filterwarnings("ignore")
    cache = OUT / "computed_flashweave_retention_absr010_unique_edges_permutation999_cache.pkl"
    if cache.exists():
        payload = pd.read_pickle(cache)
        X, groups, samples, taxa, edges, members = payload["X"], payload["groups"], payload["samples"], payload["taxa"], payload["edges"], payload["members"]
        audit = payload.get("audit", {})
        edge_eval, obs_overall, obs_pairs = payload["edge_eval"], payload["obs_overall"], payload["obs_pairs"]
        boot, boot_pairs, ci, pair_ci = payload["boot"], payload["boot_pairs"], payload["ci"], payload["pair_ci"]
        perm, pair_perm, diff = payload["perm"], payload["pair_perm"], payload["diff"]
    else:
        X, groups, samples, taxa, edges, members, audit = load_inputs()
        X_clr = clr(X)
        edge_eval, obs_overall, obs_pairs = evaluate_groups(X_clr, groups, edges)
        boot, boot_pairs, ci, pair_ci = bootstrap_ci(X_clr, groups, edges)
        perm, pair_perm = permutation_tests(X_clr, groups, edges, obs_overall, obs_pairs)
        diff = differential_pairs(obs_pairs)
        diff = diff.merge(pair_perm[["contrast", "pair", "permutation_p"]], on=["contrast", "pair"], how="left")
        pd.to_pickle({
            "X": X, "groups": groups, "samples": samples, "taxa": taxa, "edges": edges, "members": members, "audit": audit,
            "edge_eval": edge_eval, "obs_overall": obs_overall, "obs_pairs": obs_pairs,
            "boot": boot, "boot_pairs": boot_pairs, "ci": ci, "pair_ci": pair_ci,
            "perm": perm, "pair_perm": pair_perm, "diff": diff,
        }, cache)

    names = [
        "FigureFW1_reference_edge_module_composition",
        "FigureFW2_overall_intra_inter_retention_bootstrap_permutation",
        "FigureFW3_module_pair_retention_matrices",
        "FigureFW4_delta_retention_heatmaps_permutation",
        "FigureFW5_within_module_retention_heatmap",
        "FigureFW6_top_SI_vs_NSI_changed_module_pairs",
        "FigureFW7_direction_reversal_rate",
    ]
    plot_reference_composition(edges)
    plot_overall(obs_overall, ci, perm)
    plot_module_pair_matrices(obs_pairs)
    plot_delta_heatmaps(diff, pair_perm)
    plot_within_module(obs_pairs, pair_perm)
    plot_top_changed(diff, pair_perm)
    plot_reversal(obs_pairs)
    combine(names)

    overview = pd.DataFrame([
        ["reference_network", "HC FlashWeave reference network, fixed edge set", f"{audit.get('raw_flashweave_edge_rows', 'NA')} raw rows; {len(edges)} module-mapped edges used for SM01-SM12 module-pair statistics"],
        ["edge_retention_definition", f"group CLR correlation has same direction as FlashWeave edge and |r|>={R_KEEP}", "This matches the previous edge-retention definition used in earlier figures."],
        ["edge_loss_definition", f"reference edge does not satisfy same-direction |r|>={R_KEEP} in the tested group", ""],
        ["edge_reversal_definition", f"group CLR correlation has opposite direction and |r|>={R_KEEP}", ""],
        ["bootstrap", f"{N_BOOT} within-group bootstrap resamples", "95% CI for retained fraction"],
        ["permutation", f"{N_PERM} pairwise label permutations preserving group sizes", "p values for retention-rate differences"],
        ["samples", ", ".join([f"{g}={(groups==g).sum()}" for g in GROUPS]), ""],
        ["taxa", f"{len(taxa)} species involved in module-mapped FlashWeave edges", ""],
        ["module_mapping_audit", str(audit), "Full FlashWeave network is 3493 edges in Figure 1; module-pair analysis uses edges with both endpoints assigned to SM01-SM12."],
    ], columns=["item", "definition", "note"])

    with pd.ExcelWriter(OUT / "flashweave_reference_edge_retention_absr010_permutation999_results.xlsx", engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="00_method_overview", index=False)
        edges.drop(columns=["ia", "ib"], errors="ignore").to_excel(writer, sheet_name="01_reference_edges_annotated", index=False)
        obs_overall.to_excel(writer, sheet_name="02_overall_retention", index=False)
        ci.to_excel(writer, sheet_name="03_overall_bootstrap_CI", index=False)
        perm.to_excel(writer, sheet_name="04_overall_permutation999", index=False)
        obs_pairs.to_excel(writer, sheet_name="05_module_pair_retention", index=False)
        pair_ci.to_excel(writer, sheet_name="06_pair_bootstrap_CI", index=False)
        pair_perm.to_excel(writer, sheet_name="07_pair_permutation999", index=False)
        diff.to_excel(writer, sheet_name="08_pair_deltas", index=False)
        edge_eval.to_excel(writer, sheet_name="09_edge_level_group_eval", index=False)
        members.to_excel(writer, sheet_name="10_module_membership", index=False)

    top = diff[diff["contrast"] == "MDDSI_vs_MDDNSI"].sort_values("delta_retention_rate", key=lambda s: s.abs(), ascending=False).head(10)
    report = OUT / "flashweave_reference_edge_retention_absr010_permutation999_report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# FlashWeave reference edge retention and module connectivity comparison\n\n")
        f.write("Fixed HC FlashWeave edges were used as the reference network. The analysis asks how many reference edges are retained in HC, MDDNSI and MDDSI, separately for within-module and between-module connections.\n\n")
        f.write(f"Retention = same-direction group CLR correlation with |r|>={R_KEEP}. Bootstrap n={N_BOOT}; permutation n={N_PERM}.\n\n")
        f.write("## Overall retained fractions\n\n")
        f.write(obs_overall.to_csv(index=False))
        f.write("\n## Overall permutation p values\n\n")
        f.write(perm.to_csv(index=False))
        f.write("\n## Top SI vs NSI module-pair changes\n\n")
        f.write(top.to_csv(index=False))

    import shutil
    shutil.copy2(__file__, CODE / Path(__file__).name)
    print(OUT)
    print(FIG / "FlashWeave_reference_edge_retention_absr010_permutation999_combined.pdf")
    print(OUT / "flashweave_reference_edge_retention_absr010_permutation999_results.xlsx")
    print(report)


if __name__ == "__main__":
    main()
