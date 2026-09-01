from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis_mdd_microbiome import clr_transform, first_pc_score, partial_spearman, short_taxon_name
from si_reference_network_analysis import groupwise_summary, md_table, ols_term


BASE = Path(__file__).resolve().parent
FW = BASE / "flashweave_strict_si"
XLSX = BASE / "microbiota file.xlsx"
OUT = FW / "postprocessed_results"
OUT.mkdir(exist_ok=True)


def read_flashweave_edges(path: Path) -> pd.DataFrame:
    # FlashWeave detailed output is tabular. Column names can differ slightly by version,
    # so parse flexibly and keep the first two taxa plus edge statistics.
    if path.suffix == ".edgelist":
        df = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            header=None,
            names=["taxon_a", "taxon_b", "weight"],
        )
        return df.dropna(subset=["taxon_a", "taxon_b"]).reset_index(drop=True)
    df = pd.read_csv(path, sep="\t")
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError(f"Unexpected FlashWeave edge table: {path}")
    a, b = cols[0], cols[1]
    out = pd.DataFrame({"taxon_a": df[a].astype(str), "taxon_b": df[b].astype(str)})
    for c in cols[2:]:
        out[c] = df[c]
    return out


def connected_modules(edges: pd.DataFrame) -> pd.DataFrame:
    nodes = sorted(set(edges["taxon_a"]) | set(edges["taxon_b"]))
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for _, e in edges.iterrows():
        union(e["taxon_a"], e["taxon_b"])
    comp = {}
    for n in nodes:
        comp.setdefault(find(n), []).append(n)
    comps = sorted(comp.values(), key=len, reverse=True)
    module = {}
    for i, c in enumerate([x for x in comps if len(x) >= 3], 1):
        for n in c:
            module[n] = f"Module_{i}"
    degree = {n: 0 for n in nodes}
    for _, e in edges.iterrows():
        degree[e["taxon_a"]] += 1
        degree[e["taxon_b"]] += 1
    nodes_df = pd.DataFrame({
        "taxon": nodes,
        "taxon_short": [short_taxon_name(n) for n in nodes],
        "module": [module.get(n, "unassigned") for n in nodes],
        "degree": [degree[n] for n in nodes],
    })
    nodes_df["hub"] = False
    for m, sub in nodes_df.groupby("module"):
        if m != "unassigned" and len(sub) >= 3:
            nodes_df.loc[sub.index, "hub"] = sub["degree"] >= sub["degree"].quantile(0.90)
    return nodes_df


def main():
    edge_path = FW / "flashweave_HC_reference_network_detailed.edgelist"
    if not edge_path.exists():
        edge_path = FW / "flashweave_HC_reference_network_detailed.tsv"
    if not edge_path.exists():
        raise FileNotFoundError(
            f"Missing {edge_path}. Run flashweave_strict_si/run_flashweave_HC_reference.jl first."
        )
    edges = read_flashweave_edges(edge_path)
    nodes = connected_modules(edges)

    abundance = pd.read_csv(FW / "flashweave_all_microbes_all_samples.tsv", sep="\t", index_col=0)
    meta = pd.read_csv(FW / "flashweave_metadata_all_samples.tsv", sep="\t", index_col=0)
    abundance = abundance.loc[meta.index]
    clr = clr_transform(abundance)

    scores = pd.DataFrame(index=meta.index)
    retention = pd.DataFrame(index=meta.index)
    for m, sub in nodes.groupby("module"):
        taxa = [t for t in sub["taxon"] if t in clr.columns]
        if m != "unassigned" and len(taxa) >= 3:
            scores[m] = first_pc_score(clr[taxa])
            retention[f"{m}_retention"] = (abundance[taxa] > 0).mean(axis=1)
    hc_scores = scores.loc[meta["clinical_group"] == "HC"]
    eco = pd.DataFrame({
        "ecological_deviation": np.sqrt((((scores - hc_scores.mean()) / hc_scores.std(ddof=0).replace(0, np.nan)) ** 2).sum(axis=1))
    }, index=meta.index)
    hubs = [t for t in nodes.loc[nodes["hub"], "taxon"] if t in clr.columns]
    hub = pd.DataFrame(index=meta.index)
    if hubs:
        hub["hub_abundance_score"] = clr[hubs].mean(axis=1)
        hub["hub_retention_score"] = (abundance[hubs] > 0).mean(axis=1)
    features = pd.concat([scores, retention, eco, hub], axis=1)

    meta_model = meta.replace("NA", np.nan).copy()
    for col in ["age", "sex", "BMI", "HAMD3", "HAMDT", "HAMDT_minus_HAMD3"]:
        if col in meta_model:
            meta_model[col] = pd.to_numeric(meta_model[col], errors="coerce")
    meta_model["SI_trend"] = meta_model["clinical_group"].map({"HC": 0, "MDDNSI": 1, "MDDSI": 2}).astype(float)
    meta_model["MDD_SI_binary"] = np.where(meta_model["disease"] == "MDD", (meta_model["clinical_group"] == "MDDSI").astype(float), np.nan)

    rows = []
    for f in features.columns:
        rows.append({"feature": f, **ols_term(features[f], meta_model.rename(columns={"sex": "gender", "BMI": "bmi"}), "SI_trend", ["age", "gender", "bmi"])})
    trend = pd.DataFrame(rows).sort_values("p")

    mdd = meta_model["disease"] == "MDD"
    rows = []
    rows_adj = []
    for f in features.columns:
        mm = meta_model.rename(columns={"sex": "gender", "BMI": "bmi"})
        rows.append({"feature": f, **ols_term(features.loc[mdd, f], mm.loc[mdd], "MDD_SI_binary", ["age", "gender", "bmi"])})
        rows_adj.append({"feature": f, **ols_term(features.loc[mdd, f], mm.loc[mdd], "MDD_SI_binary", ["age", "gender", "bmi", "HAMDT_minus_HAMD3"])})
    si = pd.DataFrame(rows).sort_values("p")
    si_adj = pd.DataFrame(rows_adj).sort_values("p")

    hamd3 = []
    for f in features.columns:
        r = partial_spearman(features.loc[mdd, [f]], meta_model.loc[mdd, "HAMD3"], meta_model.loc[mdd, ["age", "sex", "BMI"]].rename(columns={"sex": "gender", "BMI": "bmi"}))
        hamd3.append({"feature": f, "n": r.loc[0, "n"], "r": r.loc[0, "partial_spearman_r"], "p": r.loc[0, "p"]})
    hamd3 = pd.DataFrame(hamd3).sort_values("p")

    with pd.ExcelWriter(OUT / "flashweave_strict_SI_postprocessed_results.xlsx", engine="openpyxl") as w:
        edges.to_excel(w, sheet_name="FlashWeave_edges", index=False)
        nodes.to_excel(w, sheet_name="FlashWeave_nodes_modules", index=False)
        scores.to_excel(w, sheet_name="module_activity")
        retention.to_excel(w, sheet_name="module_retention")
        eco.to_excel(w, sheet_name="ecological_deviation")
        hub.to_excel(w, sheet_name="hub_scores")
        trend.to_excel(w, sheet_name="HC_MDDNSI_MDDSI_trend", index=False)
        si.to_excel(w, sheet_name="MDDSI_vs_MDDNSI", index=False)
        si_adj.to_excel(w, sheet_name="MDDSI_adj_depression", index=False)
        hamd3.to_excel(w, sheet_name="HAMD3_continuous", index=False)

    md = []
    md.append("# FlashWeave strict-SI postprocessed results\n")
    n_modules = nodes.query("module != 'unassigned'")["module"].nunique()
    md.append(f"- FlashWeave edges: {len(edges)}\n- Nodes: {len(nodes)}\n- Modules: {n_modules}\n")
    md.append("## Three-group trend\n")
    md.append(md_table(trend.head(15)))
    md.append("\n## Strict MDDSI vs MDDNSI\n")
    md.append(md_table(si.head(15)))
    md.append("\n## Adjusted for HAMD total excluding HAMD3\n")
    md.append(md_table(si_adj.head(15)))
    md.append("\n## HAMD3 continuous mapping\n")
    md.append(md_table(hamd3.head(15)))
    (OUT / "flashweave_strict_SI_postprocessed_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Done: {OUT}")


if __name__ == "__main__":
    main()
