from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


BASE = Path(__file__).resolve().parent
XLSX = BASE / "microbiota file.xlsx"
OUT = BASE / "flashweave_strict_si"
OUT.mkdir(exist_ok=True)


def clean_name(x: str) -> str:
    # FlashWeave/Julia tables are easier to handle when column names avoid spaces.
    return str(x).replace(" ", "_").replace(";", "_")


def main():
    meta = pd.read_excel(XLSX, sheet_name="MDD metadata")
    micro = pd.read_excel(XLSX, sheet_name="mdd microbiota").rename(columns={"clade_name": "NAME"})
    meta["NAME"] = meta["NAME"].astype(str)
    micro["NAME"] = micro["NAME"].astype(str)
    common = sorted(set(meta["NAME"]) & set(micro["NAME"]))
    meta = meta.set_index("NAME").loc[common].copy()
    micro = micro.set_index("NAME").loc[common].copy()

    h3 = pd.to_numeric(meta["HAMD3"], errors="coerce")
    meta["SI_group_strict"] = np.where(
        meta["Group"] == "HC",
        "HC",
        np.where(h3.fillna(0) >= 2, "MDDSI", "MDDNSI"),
    )
    meta["MDD_SI_binary_strict"] = np.where(meta["Group"] == "MDD", (h3 >= 2).astype(float), np.nan)
    meta["HAMDT_minus_HAMD3"] = pd.to_numeric(meta["HAMDT"], errors="coerce") - h3

    # "All microbes" here means all numeric microbial abundance columns in the microbiota sheet.
    # We remove only unusable features: all-zero / zero-variance columns.
    abundance = micro.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    nonzero = abundance.sum(axis=0) > 0
    variable = abundance.var(axis=0) > 0
    abundance = abundance.loc[:, nonzero & variable].copy()
    abundance.columns = [clean_name(c) for c in abundance.columns]
    abundance.index.name = "sample"

    fw_meta = pd.DataFrame(index=meta.index)
    fw_meta.index.name = "sample"
    fw_meta["clinical_group"] = meta["SI_group_strict"]
    fw_meta["disease"] = meta["Group"]
    fw_meta["HAMD3"] = h3
    fw_meta["HAMDT"] = pd.to_numeric(meta["HAMDT"], errors="coerce")
    fw_meta["HAMDT_minus_HAMD3"] = meta["HAMDT_minus_HAMD3"]
    fw_meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
    fw_meta["sex"] = pd.to_numeric(meta["gender"], errors="coerce")
    fw_meta["BMI"] = pd.to_numeric(meta["bmi"], errors="coerce")
    fw_meta = fw_meta.fillna("NA")

    hc_ids = meta.index[meta["SI_group_strict"] == "HC"]
    abundance_hc = abundance.loc[hc_ids].copy()
    fw_meta_hc = fw_meta.loc[hc_ids].copy()

    abundance.to_csv(OUT / "flashweave_all_microbes_all_samples.tsv", sep="\t")
    fw_meta.to_csv(OUT / "flashweave_metadata_all_samples.tsv", sep="\t")
    abundance_hc.to_csv(OUT / "flashweave_all_microbes_HC_only.tsv", sep="\t")
    fw_meta_hc.to_csv(OUT / "flashweave_metadata_HC_only.tsv", sep="\t")

    overview = pd.DataFrame(
        {
            "item": [
                "matched_samples",
                "HC",
                "MDDNSI_strict_HAMD3_0_1",
                "MDDSI_strict_HAMD3_ge_2",
                "raw_microbe_columns",
                "retained_microbe_columns_for_flashweave",
                "removed_all_zero_or_zero_variance",
            ],
            "value": [
                len(meta),
                int((meta["SI_group_strict"] == "HC").sum()),
                int((meta["SI_group_strict"] == "MDDNSI").sum()),
                int((meta["SI_group_strict"] == "MDDSI").sum()),
                micro.shape[1],
                abundance.shape[1],
                int(micro.shape[1] - abundance.shape[1]),
            ],
        }
    )
    overview.to_csv(OUT / "flashweave_strict_si_overview.csv", index=False)

    print(overview.to_string(index=False))
    print(f"Output directory: {OUT}")


if __name__ == "__main__":
    main()
