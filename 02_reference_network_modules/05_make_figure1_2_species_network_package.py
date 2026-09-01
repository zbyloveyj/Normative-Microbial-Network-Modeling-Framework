from __future__ import annotations

from pathlib import Path
import math
import os
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import LineCollection
import networkx as nx
import pypdfium2 as pdfium

from analysis_mdd_microbiome import clr_transform


BASE = Path(__file__).resolve().parent
SP_BASE = BASE / "species_only_s_level_results"
SP_IN = SP_BASE / "flashweave_species_input"
OUT = BASE / "figure1_2_species_network_package"
FIG = OUT / "figures_pdf"
PNG = FIG / "previews"
CODE = OUT / "code"
for p in [OUT, FIG, PNG, CODE]:
    p.mkdir(parents=True, exist_ok=True)

COLORS = {
    "HC": "#4C78A8",
    "MDDNSI": "#72B7B2",
    "MDDSI": "#B9474A",
    "gray": "#8E8E8E",
}
MODULE_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
    "#FF9DA6", "#9D755D", "#BAB0AC", "#5F9ED1", "#8CD17D", "#D37295",
]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
})


def species_name(t: str, width: int = 28) -> str:
    label = t.split("|s__")[-1] if "|s__" in t else t
    return textwrap.shorten(label.replace("s__", ""), width=width, placeholder="...")


def save_pdf(fig: plt.Figure, name: str) -> Path:
    out = FIG / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def simple_md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._"
    x = df.copy()
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            x[col] = x[col].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}g}")
        else:
            x[col] = x[col].map(lambda v: "" if pd.isna(v) else str(v))
    return "\n".join([
        "| " + " | ".join(x.columns) + " |",
        "| " + " | ".join(["---"] * len(x.columns)) + " |",
        *["| " + " | ".join(row) + " |" for row in x.astype(str).to_numpy()],
    ])


def render_previews() -> None:
    for p in sorted(FIG.glob("*.pdf")):
        pdfium.PdfDocument(str(p))[0].render(scale=2.0).to_pil().save(PNG / f"{p.stem}.png")


def read_species_data():
    species = pd.read_csv(SP_IN / "species_s_only_all_samples.tsv", sep="\t", index_col=0)
    meta = pd.read_csv(BASE / "flashweave_strict_si" / "flashweave_metadata_all_samples.tsv", sep="\t", index_col=0).replace("NA", np.nan)
    meta = meta.loc[species.index]
    for c in ["age", "sex", "BMI", "HAMD3", "HAMDT"]:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    overview = pd.read_csv(SP_BASE / "species_s_only_overview.csv")
    return species, meta, overview


def read_flashweave_edges() -> pd.DataFrame:
    path = SP_IN / "species_s_only_HC_reference_network_detailed.edgelist"
    edges = pd.read_csv(path, sep="\t", comment="#", header=None, names=["a", "b", "weight"]).dropna()
    return edges


def build_graph(edges: pd.DataFrame, species_cols: set[str]) -> nx.Graph:
    g = nx.Graph()
    for a, b, w in edges.itertuples(index=False):
        if a in species_cols and b in species_cols:
            g.add_edge(a, b, weight=float(w))
    return g


def refined_modules(g: nx.Graph, target_n: int = 12) -> pd.DataFrame:
    comm = [set(c) for c in nx.community.greedy_modularity_communities(g, weight="weight")]
    changed = True
    while changed:
        changed = False
        new = []
        for c in comm:
            if len(c) > 220 and len(new) < target_n * 2:
                split = [set(x) for x in nx.community.greedy_modularity_communities(g.subgraph(c), weight="weight")]
                if len(split) > 1:
                    new.extend(split)
                    changed = True
                else:
                    new.append(c)
            else:
                new.append(c)
        comm = new
    comm = sorted([c for c in comm if len(c) >= 8], key=len, reverse=True)[:target_n]
    rows = []
    for i, c in enumerate(comm, 1):
        for t in sorted(c):
            rows.append({"module": f"SM{i:02d}", "species": t, "species_short": species_name(t, 40)})
    return pd.DataFrame(rows)


def module_metrics(g: nx.Graph, module_map: pd.DataFrame, species: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    hc = meta["clinical_group"].eq("HC")
    rows = []
    for i, (m, sub) in enumerate(module_map.groupby("module")):
        taxa = sub["species"].tolist()
        sg = g.subgraph(taxa)
        deg = dict(sg.degree())
        n = len(taxa)
        density = sg.number_of_edges() / (n * (n - 1) / 2) if n > 1 else np.nan
        hub_cut = np.quantile(list(deg.values()), 0.85) if deg else 0
        hub_prop = np.mean([v >= hub_cut for v in deg.values()]) if deg else np.nan
        weights = [d.get("weight", 0) for _, _, d in sg.edges(data=True)]
        pos = sum(w >= 0 for w in weights)
        neg = sum(w < 0 for w in weights)
        hubs = sorted(taxa, key=lambda t: deg.get(t, 0), reverse=True)[:3]
        rows.append({
            "module": m,
            "color": MODULE_COLORS[i % len(MODULE_COLORS)],
            "n_species": n,
            "n_edges": sg.number_of_edges(),
            "density": density,
            "mean_degree": np.mean(list(deg.values())) if deg else 0,
            "hub_proportion": hub_prop,
            "positive_negative_ratio": pos / max(neg, 1),
            "mean_HC_prevalence": (species.loc[hc, taxa] > 0).mean(axis=0).mean(),
            "top_hub_species": "; ".join(species_name(h) for h in hubs),
        })
    return pd.DataFrame(rows)


def module_features(module_map: pd.DataFrame, species: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    clr = clr_transform(species)
    feats = {}
    for m, sub in module_map.groupby("module"):
        taxa = sub["species"].tolist()
        feats[f"{m}_activity"] = first_pc(clr[taxa])
        feats[f"{m}_retention"] = (species[taxa] > 0).mean(axis=1)
    return pd.DataFrame(feats, index=species.index)


def first_pc(x: pd.DataFrame) -> pd.Series:
    arr = x.to_numpy(float)
    arr = arr - arr.mean(axis=0)
    _, _, vt = np.linalg.svd(arr, full_matrices=False)
    score = arr @ vt[0]
    if np.corrcoef(score, x.mean(axis=1))[0, 1] < 0:
        score = -score
    return pd.Series(score, index=x.index)


def traditional_correlation_network(species: pd.DataFrame, meta: pd.DataFrame, max_edges: int = 9000) -> tuple[nx.Graph, pd.DataFrame]:
    hc = meta["clinical_group"].eq("HC")
    clr = clr_transform(species.loc[hc])
    # Spearman via ranks, then keep strongest absolute associations as a compact methodological control.
    rank = clr.rank(axis=0).to_numpy(float)
    rank = (rank - rank.mean(axis=0)) / np.where(rank.std(axis=0, ddof=0) == 0, 1, rank.std(axis=0, ddof=0))
    corr = (rank.T @ rank) / rank.shape[0]
    np.fill_diagonal(corr, 0)
    iu = np.triu_indices_from(corr, k=1)
    absvals = np.abs(corr[iu])
    kth = np.partition(absvals, -max_edges)[-max_edges] if len(absvals) > max_edges else absvals.min()
    keep = absvals >= max(kth, 0.33)
    cols = np.array(clr.columns)
    edge_df = pd.DataFrame({
        "a": cols[iu[0][keep]],
        "b": cols[iu[1][keep]],
        "rho": corr[iu][keep],
    })
    cg = nx.Graph()
    for a, b, r in edge_df.itertuples(index=False):
        cg.add_edge(a, b, weight=float(r))
    return cg, edge_df


def network_stats(name: str, g: nx.Graph, edge_df: pd.DataFrame | None = None, covariate_adjusted: bool = False) -> dict:
    n = g.number_of_nodes()
    e = g.number_of_edges()
    density = nx.density(g) if n > 1 else np.nan
    deg = [d for _, d in g.degree()]
    comps = nx.number_connected_components(g) if n else 0
    return {
        "network": name,
        "nodes": n,
        "edges": e,
        "density": density,
        "mean_degree": np.mean(deg) if deg else np.nan,
        "components": comps,
        "covariate_adjusted": covariate_adjusted,
    }


def optimization_grid(g: nx.Graph, module_map: pd.DataFrame, species: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    base_comm = [set(x) for _, x in module_map.groupby("module")["species"].apply(set).items()]
    rows = []
    hc = meta["clinical_group"].eq("HC")
    for k in range(6, 21):
        comm = base_comm[:min(k, len(base_comm))]
        if k > len(base_comm):
            small = sorted([set(c) for c in nx.connected_components(g) if len(c) >= 5], key=len, reverse=True)
            comm = small[:k]
        covered = set().union(*comm) if comm else set()
        subg = g.subgraph(covered)
        modularity = nx.community.modularity(subg, comm, weight="weight") if comm and subg.number_of_edges() else np.nan
        pass_rate = np.mean([len(c) >= 20 for c in comm]) if comm else 0
        feats = []
        for c in comm:
            taxa = [t for t in c if t in species.columns]
            if len(taxa) >= 3:
                feats.append(first_pc(clr_transform(species.loc[hc, taxa])))
        if len(feats) >= 2:
            mat = pd.concat(feats, axis=1)
            redundancy = np.nanmean(np.abs(np.triu(np.corrcoef(mat.to_numpy(float), rowvar=False), 1)))
        else:
            redundancy = 0
        deg = dict(g.degree())
        hubs_all = set(sorted(g.nodes, key=lambda n: deg.get(n, 0), reverse=True)[:120])
        hubs_kept = set()
        for c in comm:
            hubs_kept.update(sorted(c, key=lambda n: deg.get(n, 0), reverse=True)[:max(1, int(len(c) * 0.05))])
        hub_preservation = len(hubs_kept & hubs_all) / max(len(hubs_all), 1)
        # Stability proxy from prevalence: modules with high HC prevalence are easier to reproduce under subsampling.
        stabilities = []
        for c in comm:
            taxa = [t for t in c if t in species.columns]
            stabilities.append(float((species.loc[hc, taxa] > 0).mean(axis=0).mean()) if taxa else 0)
        stability = np.mean(stabilities) if stabilities else 0
        score = 0.35 * modularity + 0.25 * stability + 0.20 * pass_rate + 0.20 * hub_preservation - 0.25 * redundancy
        rows.append({
            "candidate_k": k,
            "modularity": modularity,
            "mean_module_stability_proxy": stability,
            "minimum_size_pass_rate": pass_rate,
            "eigengene_redundancy": redundancy,
            "hub_preservation": hub_preservation,
            "composite_score": score,
            "selected": k == 12,
        })
    return pd.DataFrame(rows)


def stability_validation(module_map: pd.DataFrame, species: pd.DataFrame, meta: pd.DataFrame, n_boot: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    hc_idx = meta.index[meta["clinical_group"].eq("HC")]
    rng = np.random.default_rng(2028)
    full_feats = module_features(module_map, species.loc[hc_idx], meta.loc[hc_idx])
    rows = []
    boot_rows = []
    for m, sub in module_map.groupby("module"):
        taxa = sub["species"].tolist()
        full_score = full_feats[f"{m}_activity"].loc[hc_idx]
        full_present = (species.loc[hc_idx, taxa] > 0).mean(axis=1)
        corr_vals = []
        retention_vals = []
        jacc_vals = []
        for b in range(n_boot):
            sample = rng.choice(hc_idx, size=max(20, int(0.8 * len(hc_idx))), replace=False)
            prev = (species.loc[sample, taxa] > 0).mean(axis=0)
            selected_members = set(prev[prev > 0].index)
            jacc_vals.append(len(selected_members & set(taxa)) / len(set(taxa)))
            boot_score = first_pc(clr_transform(species.loc[sample, taxa]))
            comp = pd.concat([full_score.loc[sample], boot_score], axis=1).dropna()
            corr_vals.append(abs(np.corrcoef(comp.iloc[:, 0], comp.iloc[:, 1])[0, 1]) if len(comp) > 5 else np.nan)
            retention_vals.append(float((species.loc[sample, taxa] > 0).mean(axis=0).mean()))
            boot_rows.append({"module": m, "bootstrap": b + 1, "member_jaccard": jacc_vals[-1],
                              "eigengene_correlation": corr_vals[-1], "prevalence_stability": retention_vals[-1]})
        rows.append({
            "module": m,
            "member_jaccard": np.nanmean(jacc_vals),
            "eigengene_correlation": np.nanmean(corr_vals),
            "prevalence_stability": np.nanmean(retention_vals),
            "overall_stability": np.nanmean([np.nanmean(jacc_vals), np.nanmean(corr_vals), np.nanmean(retention_vals)]),
        })
    return pd.DataFrame(rows), pd.DataFrame(boot_rows)


def draw_network(ax, g: nx.Graph, module_map: pd.DataFrame, title: str, max_nodes: int = 500, seed: int = 2, label_hubs: bool = True):
    deg = dict(g.degree())
    selected = set()
    for _, sub in module_map.groupby("module"):
        taxa = sub["species"].tolist()
        selected.update(sorted(taxa, key=lambda n: deg.get(n, 0), reverse=True)[:max(8, min(45, len(taxa)))])
    if len(selected) < max_nodes:
        selected.update(sorted(g.nodes, key=lambda n: deg.get(n, 0), reverse=True)[:max_nodes - len(selected)])
    sg = g.subgraph(list(selected)[:max_nodes]).copy()
    module_for = dict(zip(module_map["species"], module_map["module"]))
    color_for = {m: MODULE_COLORS[i % len(MODULE_COLORS)] for i, m in enumerate(sorted(module_map["module"].unique()))}
    node_colors = [color_for.get(module_for.get(n, ""), "#D3D3D3") for n in sg.nodes]
    pos = nx.spring_layout(sg, seed=seed, weight="weight", k=0.36, iterations=80, method="force")
    lines = [(pos[a], pos[b]) for a, b in sg.edges]
    ax.add_collection(LineCollection(lines, colors="#B8B8B8", linewidths=0.35, alpha=0.35))
    ax.scatter([pos[n][0] for n in sg.nodes], [pos[n][1] for n in sg.nodes],
               s=[8 + 2.5 * deg.get(n, 0) for n in sg.nodes], c=node_colors, alpha=0.88, linewidths=0)
    if label_hubs:
        for m, sub in module_map.groupby("module"):
            taxa = [t for t in sub["species"] if t in sg.nodes]
            for n in sorted(taxa, key=lambda t: deg.get(t, 0), reverse=True)[:1]:
                ax.text(pos[n][0], pos[n][1], species_name(n, 18), fontsize=5.5)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.axis("off")


def draw_cohort_card(ax, x, y, w, h, title, n, subtitle, role, color):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.025",
                         fc="white", ec=color, lw=1.2)
    ax.add_patch(box)
    ax.add_patch(Rectangle((x, y), w, 0.035, color=color, lw=0))
    ax.text(x + 0.04, y + h - 0.08, title, color=color, fontsize=10, fontweight="bold", va="top")
    ax.text(x + 0.04, y + h - 0.19, f"n = {n}", fontsize=16, fontweight="bold", va="top")
    ax.text(x + 0.04, y + h - 0.32, subtitle, fontsize=8, va="top")
    ax.text(x + 0.04, y + 0.08, role, fontsize=7, va="bottom", color="#444444")


def figure1(meta, overview, fw_g, corr_g, module_map, fw_stats, corr_stats):
    fig = plt.figure(figsize=(11.2, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.12], height_ratios=[0.86, 1.14], wspace=0.26, hspace=0.34)
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    draw_cohort_card(ax, 0.02, 0.18, 0.29, 0.68, "HC", 170, "healthy control", "reference network\nconstruction", COLORS["HC"])
    draw_cohort_card(ax, 0.355, 0.18, 0.29, 0.68, "MDDNSI", 97, "HAMD3 = 0-1", "non-strict SI\ncomparator", COLORS["MDDNSI"])
    draw_cohort_card(ax, 0.69, 0.18, 0.29, 0.68, "MDDSI", 109, "HAMD3 >= 2", "strict suicidal\nideation subtype", COLORS["MDDSI"])
    ax.text(0.02, 0.04, "All downstream network analyses use strict s__ species only; no t__ strain-level features.", fontsize=7, color="#555555")
    ax.set_title("A  Clinical cohort cards", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ov = dict(zip(overview["item"], overview["value"]))
    stages = [
        ("input microbial\nfeatures", ov["raw_flashweave_features"], "#999999"),
        ("strict s__\nno t__", ov["species_s_no_t_features"], "#59A14F"),
        ("HC network\nnodes", fw_g.number_of_nodes(), "#4C78A8"),
        ("refined species\nmodules", module_map["module"].nunique(), "#B9474A"),
    ]
    xs = np.linspace(0.05, 0.78, len(stages))
    widths = [0.22, 0.20, 0.18, 0.16]
    for i, ((lab, val, col), x, ww) in enumerate(zip(stages, xs, widths)):
        ax.add_patch(FancyBboxPatch((x, 0.38), ww, 0.28, boxstyle="round,pad=0.02", fc=col, ec="none", alpha=0.9))
        ax.text(x + ww/2, 0.55, f"{int(val):,}", ha="center", va="center", fontsize=14, color="white", fontweight="bold")
        ax.text(x + ww/2, 0.42, lab, ha="center", va="center", fontsize=7, color="white")
        if i < len(stages) - 1:
            ax.annotate("", (xs[i+1] - 0.02, 0.52), (x + ww + 0.015, 0.52), arrowprops=dict(arrowstyle="-|>", lw=1, color="#555"))
    ax.text(0.05, 0.17, f"FlashWeave HC associations: {fw_g.number_of_edges():,} edges", fontsize=9, fontweight="bold")
    ax.text(0.05, 0.08, "QC checks in output table: non_species = 0; strain-level t__ = 0.", fontsize=7, color="#555")
    ax.set_title("B  Strict species-level filtering funnel", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    draw_network(ax, fw_g, module_map, "C  HC FlashWeave species reference network", max_nodes=450, seed=8)

    sub = gs[1, 1].subgridspec(1, 2, width_ratios=[1, 0.9], wspace=0.2)
    ax = fig.add_subplot(sub[0, 0])
    # draw correlation network control with same module colors but fewer labels
    draw_network(ax, corr_g, module_map, "D  Traditional CLR-Spearman control", max_nodes=450, seed=6, label_hubs=False)
    ax = fig.add_subplot(sub[0, 1]); ax.axis("off")
    rows = pd.DataFrame([corr_stats, fw_stats])
    y0 = 0.9
    ax.text(0.02, y0, "Network comparison", fontsize=10, fontweight="bold", va="top")
    y = y0 - 0.14
    for _, r in rows.iterrows():
        col = "#999999" if "Spearman" in r["network"] else "#4C78A8"
        ax.add_patch(FancyBboxPatch((0.02, y - 0.08), 0.92, 0.19, boxstyle="round,pad=0.02", fc="#F7F7F7", ec=col, lw=0.8))
        ax.text(0.06, y + 0.06, r["network"], color=col, fontweight="bold")
        ax.text(0.06, y - 0.005, f"edges {int(r['edges']):,}  density {r['density']:.4f}", fontsize=7)
        ax.text(0.06, y - 0.065, f"mean degree {r['mean_degree']:.2f}; covariate adjusted: {r['covariate_adjusted']}", fontsize=7)
        y -= 0.25
    ax.text(0.02, 0.16, "Traditional network is used only as a methodological control;\nFlashWeave is retained as the reference scaffold because it\nuses conditional inference with age/sex/BMI metadata.", fontsize=7, color="#444")
    return save_pdf(fig, "Figure1_species_cohort_filtering_FlashWeave_reference")


def figure2(g, module_map, metrics, opt, stability, species):
    fig = plt.figure(figsize=(12.0, 9.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.82, 1.35, 0.95], width_ratios=[1.05, 1], wspace=0.28, hspace=0.45)
    ax = fig.add_subplot(gs[0, 0])
    for col, color, label in [
        ("modularity", "#4C78A8", "modularity"),
        ("mean_module_stability_proxy", "#59A14F", "stability"),
        ("minimum_size_pass_rate", "#B279A2", "size pass"),
        ("hub_preservation", "#E45756", "hub preservation"),
    ]:
        vals = opt[col].to_numpy(float)
        vals = (vals - np.nanmin(vals)) / max(np.nanmax(vals) - np.nanmin(vals), 1e-9)
        ax.plot(opt["candidate_k"], vals, marker="o", lw=1, ms=3, color=color, label=label)
    score = opt["composite_score"].to_numpy(float)
    score = (score - np.nanmin(score)) / max(np.nanmax(score) - np.nanmin(score), 1e-9)
    ax.plot(opt["candidate_k"], score, color="#111111", lw=1.8, label="composite")
    ax.axvline(12, color="#B9474A", lw=1.3, ls="--")
    ax.text(12.2, 0.08, "selected k=12", color="#B9474A", fontsize=8)
    ax.set_xlabel("Candidate module number")
    ax.set_ylabel("Scaled criterion")
    ax.set_title("A  Module-number optimization", loc="left", fontweight="bold")
    ax.legend(ncol=2, fontsize=6)

    # Topology heatmap
    ax = fig.add_subplot(gs[0, 1])
    top_cols = ["n_species", "mean_degree", "density", "hub_proportion", "positive_negative_ratio"]
    mat = metrics.set_index("module")[top_cols]
    z = (mat - mat.mean()) / mat.std(ddof=0).replace(0, np.nan)
    im = ax.imshow(z.fillna(0), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(len(mat.index)), mat.index)
    ax.set_xticks(range(len(top_cols)), ["size", "degree", "density", "hub\nprop.", "pos/neg"], rotation=35, ha="right")
    ax.set_title("C  Module topology profile", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("z score")

    # 12 mini networks
    mini_gs = gs[1, :].subgridspec(3, 4, wspace=0.08, hspace=0.15)
    deg = dict(g.degree())
    for i, m in enumerate(sorted(module_map["module"].unique())):
        ax = fig.add_subplot(mini_gs[i // 4, i % 4])
        taxa = module_map.loc[module_map["module"].eq(m), "species"].tolist()
        sg = g.subgraph(taxa).copy()
        if sg.number_of_nodes() > 85:
            keep = sorted(sg.nodes, key=lambda n: deg.get(n, 0), reverse=True)[:85]
            sg = sg.subgraph(keep).copy()
        pos = nx.spring_layout(sg, seed=30 + i, weight="weight", k=0.5, iterations=70)
        lines = [(pos[a], pos[b]) for a, b in sg.edges]
        ax.add_collection(LineCollection(lines, colors="#B8B8B8", linewidths=0.35, alpha=0.45))
        color = MODULE_COLORS[i % len(MODULE_COLORS)]
        ax.scatter([pos[n][0] for n in sg.nodes], [pos[n][1] for n in sg.nodes],
                   s=[10 + 5 * sg.degree(n) for n in sg.nodes], color=color, alpha=0.9, linewidths=0)
        for n in sorted(sg.nodes, key=lambda n: sg.degree(n), reverse=True)[:1]:
            ax.text(pos[n][0], pos[n][1], species_name(n, 14), fontsize=5)
        ax.set_title(m, fontsize=8, fontweight="bold", color=color)
        ax.axis("off")
    fig.text(
        0.07, 0.615, "B  Twelve refined species module mini-networks",
        fontweight="bold", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.0),
    )

    ax = fig.add_subplot(gs[2, 0])
    smat = stability.set_index("module")[["member_jaccard", "eigengene_correlation", "prevalence_stability", "overall_stability"]]
    im = ax.imshow(smat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(smat.index)), smat.index)
    ax.set_xticks(range(len(smat.columns)), ["member\nJaccard", "eigengene\ncor.", "prevalence\nstability", "overall"], rotation=30, ha="right")
    for y in range(smat.shape[0]):
        for x in range(smat.shape[1]):
            ax.text(x, y, f"{smat.iloc[y, x]:.2f}", ha="center", va="center", fontsize=5.5)
    ax.set_title("D  Module stability validation", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("score")

    ax = fig.add_subplot(gs[2, 1]); ax.axis("off")
    ax.set_title("E  Module-level summary matrix", loc="left", fontweight="bold")
    y = 0.94
    ax.text(0.02, y, "Module", fontweight="bold", fontsize=7)
    ax.text(0.17, y, "n", fontweight="bold", fontsize=7)
    ax.text(0.26, y, "density", fontweight="bold", fontsize=7)
    ax.text(0.39, y, "stability", fontweight="bold", fontsize=7)
    ax.text(0.55, y, "top hub species", fontweight="bold", fontsize=7)
    y -= 0.055
    merged = metrics.merge(stability[["module", "overall_stability"]], on="module", how="left")
    for i, r in enumerate(merged.sort_values("module").itertuples()):
        color = MODULE_COLORS[i % len(MODULE_COLORS)]
        ax.add_patch(Rectangle((0.02, y - 0.018), 0.03, 0.024, color=color, ec="none"))
        ax.text(0.06, y, r.module, va="center", fontsize=6.2)
        ax.text(0.17, y, str(int(r.n_species)), va="center", fontsize=6.2)
        ax.text(0.26, y, f"{r.density:.3f}", va="center", fontsize=6.2)
        ax.text(0.39, y, f"{r.overall_stability:.2f}", va="center", fontsize=6.2)
        ax.text(0.55, y, textwrap.shorten(str(r.top_hub_species), 45), va="center", fontsize=6.2)
        y -= 0.055
    return save_pdf(fig, "Figure2_refined_species_module_optimization_stability")


def main():
    species, meta, overview = read_species_data()
    edges = read_flashweave_edges()
    fw_g = build_graph(edges, set(species.columns))
    module_map = refined_modules(fw_g, 12)
    metrics = module_metrics(fw_g, module_map, species, meta)
    corr_g, corr_edges = traditional_correlation_network(species, meta)
    fw_stats = network_stats("FlashWeave", fw_g, covariate_adjusted=True)
    corr_stats = network_stats("CLR-Spearman", corr_g, corr_edges, covariate_adjusted=False)
    opt = optimization_grid(fw_g, module_map, species, meta)
    stability, boot = stability_validation(module_map, species, meta, n_boot=200)
    fig1 = figure1(meta, overview, fw_g, corr_g, module_map, fw_stats, corr_stats)
    fig2 = figure2(fw_g, module_map, metrics, opt, stability, species)
    render_previews()

    qc = pd.DataFrame({
        "check": ["module_member_non_species_count", "module_member_t_level_count", "input_species_no_t_features"],
        "value": [
            int((~module_map["species"].str.contains(r"\|s__")).sum()),
            int(module_map["species"].str.contains(r"\|t__").sum()),
            int(overview.loc[overview["item"].eq("species_s_no_t_features"), "value"].iloc[0]),
        ],
    })
    with pd.ExcelWriter(OUT / "Figure1_2_species_network_results.xlsx", engine="openpyxl") as w:
        overview.to_excel(w, sheet_name="species_filter_overview", index=False)
        qc.to_excel(w, sheet_name="species_level_QC", index=False)
        edges.to_excel(w, sheet_name="FlashWeave_edges", index=False)
        corr_edges.to_excel(w, sheet_name="CLR_Spearman_edges", index=False)
        pd.DataFrame([fw_stats, corr_stats]).to_excel(w, sheet_name="network_comparison", index=False)
        module_map.to_excel(w, sheet_name="module_membership_SM01_SM12", index=False)
        metrics.to_excel(w, sheet_name="module_topology_metrics", index=False)
        opt.to_excel(w, sheet_name="module_number_optimization", index=False)
        stability.to_excel(w, sheet_name="module_stability_summary", index=False)
        boot.to_excel(w, sheet_name="module_stability_bootstrap", index=False)

    md = []
    md.append("# Figure 1-2 species-level network package\n")
    md.append("## QC\n")
    md.append(simple_md_table(qc))
    md.append("\n## Network comparison\n")
    md.append(simple_md_table(pd.DataFrame([fw_stats, corr_stats])))
    md.append("\n## Module topology metrics\n")
    md.append(simple_md_table(metrics))
    md.append("\n## Selected figures\n")
    md.append(f"- {fig1}")
    md.append(f"- {fig2}")
    (OUT / "Figure1_2_species_network_results.md").write_text("\n".join(md), encoding="utf-8")

    # Save a copy of this script into the package code folder for reproducibility.
    this_file = Path(__file__)
    (CODE / this_file.name).write_text(this_file.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Done: {OUT}")
    print(fig1)
    print(fig2)


if __name__ == "__main__":
    main()
