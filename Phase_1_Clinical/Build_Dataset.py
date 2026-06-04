"""
README - build_dataset.py
=========================

This script prepares the Phase 1 clinical-only dataset for the
MCI-to-dementia conversion prediction project.

It loads the raw ADNI clinical tables, creates pMCI/sMCI labels, extracts one
baseline clinical record per subject, merges all available clinical features,
and saves the raw metadata file for model training.

Important
---------
This script does not perform imputation.

Missing values are kept in final_metadata.csv. Imputation is handled later
inside the training pipeline, where the imputer is fitted only on the training
fold and then applied to validation/test data. This prevents data leakage.

What this script does
---------------------
1. Loads raw ADNI files such as DXSUM, PTDEMOG, MMSE, MOCA, ADAS, FAQ,
   GDSCALE, CDR, and NEUROBAT.
2. Builds pMCI and sMCI labels using baseline and follow-up diagnosis data.
3. Extracts one baseline row per subject from each clinical table.
4. Merges all available clinical feature blocks into one metadata file.
5. Creates derived clinical features using only available values.
6. Reports pre-imputation missingness and feature coverage.
7. Saves the final raw clinical dataset and label files.

Excluded from this script
-------------------------
APOE4
    Excluded because genetic testing may not be available in low-resource
    Indian clinical settings.

Clinical + MRI volumetric data
    Excluded because this script is only for Phase 1 clinical-only modelling.
    MRI-based features are handled separately in later phases.

Run
---
python preprocessing/build_dataset.py
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# PATHS
CSV_DIR = r"C:\Users\ASUS\Desktop\Research Resources\DATA\csv"

EXTRA_DIRS = [
    r"C:\Users\ASUS\Desktop\Research Resources\DATA",
    r"C:\Users\ASUS\Desktop\Research Resources\DATA\data",
]

ROOT = r"C:\Users\ASUS\Desktop\Research Resources\DementiaResearch"

OUT_META = os.path.join(ROOT, "data", "metadata", "final_metadata.csv")
OUT_LABELS = os.path.join(ROOT, "data", "labels", "pmci_smci_labels.csv")
OUT_DEBUG = os.path.join(ROOT, "data", "metadata", "dxsum_debug_baseline_followup.csv")


# UTILITIES
def sec(title):
    print(f"\n{'-' * 62}")
    print(f"  {title}")
    print(f"{'-' * 62}")


def fix_rid(df):
    """Convert RID to nullable Int64 to avoid int/float merge mismatches."""
    if "RID" in df.columns:
        df = df.copy()
        df["RID"] = pd.to_numeric(df["RID"], errors="coerce").astype("Int64")
    return df


def find(prefix, alts=None):
    """Find a CSV/XLSX file by prefix from the allowed data folders."""
    prefixes = [prefix] + (alts or [])

    for current_prefix in prefixes:
        current_prefix = current_prefix.lower()

        for folder in [CSV_DIR] + EXTRA_DIRS:
            if not os.path.isdir(folder):
                continue

            for ext in [".csv", ".xlsx"]:
                hits = [
                    file
                    for file in glob.glob(os.path.join(folder, f"*{ext}"))
                    if os.path.basename(file).lower().startswith(current_prefix)
                ]

                if hits:
                    return sorted(hits)[0]

    return None


def load(path, label):
    """Load one ADNI table and standardise column names."""
    if path is None:
        print(f"  MISS  {label}")
        return None

    try:
        if path.endswith(".xlsx"):
            df = pd.read_excel(path, engine="openpyxl")
        else:
            df = pd.read_csv(path, low_memory=False)

        df.columns = df.columns.str.upper().str.strip()
        df = fix_rid(df)

        print(
            f"  OK    {label:<20} {df.shape[0]:>7,} rows × {df.shape[1]:>4} cols"
            f"  [{os.path.basename(path)}]"
        )

        return df

    except Exception as error:
        print(f"  FAIL  {label}: {error}")
        return None


def first_col(df, candidates):
    """Return the first matching column from a candidate list."""
    return next((col for col in candidates if col in df.columns), None)


def get_baseline(df):
    """Return one baseline row per RID."""
    if "VISCODE" not in df.columns:
        return df.drop_duplicates("RID")

    viscode = df["VISCODE"].astype(str).str.lower().str.strip()

    baseline = df[viscode == "bl"]

    if baseline.empty:
        baseline = df[viscode == "sc"]

    if baseline.empty:
        baseline = df.sort_values("VISCODE").groupby("RID", as_index=False).first()

    return baseline.drop_duplicates("RID")


def first_per_rid(df):
    """Return the first available row per RID."""
    return df.drop_duplicates("RID", keep="first")


def pull(df, label, want, rename=None, demog=False):
    """Extract baseline features from one ADNI table without imputation."""
    rename = rename or {}

    if df is None:
        return None

    if demog:
        source = first_per_rid(df)
    else:
        source = get_baseline(df)

    found = [col for col in want if col in source.columns]
    missing = [col for col in want if col not in source.columns]

    if missing:
        print(f"         {label} absent cols: {missing}")

    if not found:
        print(f"  SKIP  {label}: none of {want} found")
        return None

    out = source[["RID"] + found].copy()
    out.rename(columns=rename, inplace=True)
    out = out.drop_duplicates("RID")

    clean_cols = list(dict.fromkeys([rename.get(col, col) for col in found]))
    out = out.dropna(subset=clean_cols, how="all")
    out = fix_rid(out)

    print(
        f"  OK    {label:<20} {len(out):>4,} subjects  "
        f"| cols {list(dict.fromkeys(clean_cols))}"
    )

    return out


# Step 1: Load raw ADNI files
sec("STEP 1 — Loading raw ADNI CSV files")
print(f"  Source: {CSV_DIR}\n")

raw = {
    "DXSUM": load(find("DXSUM"), "DXSUM"),
    "PTDEMOG": load(find("PTDEMOG"), "PTDEMOG"),
    "MMSE": load(find("MMSE"), "MMSE"),
    "MOCA": load(find("MOCA"), "MOCA"),
    "ADAS": load(find("ADAS"), "ADAS"),
    "FAQ": load(find("FAQ"), "FAQ"),
    "GDSCALE": load(find("GDSCALE"), "GDSCALE"),
    "CDR": load(find("CDR", ["CDR_", "CDRSUM"]), "CDR"),
    "NEUROBAT": load(find("NEUROBAT"), "NEUROBAT"),
}

if raw["DXSUM"] is None:
    raise FileNotFoundError(
        f"DXSUM not found in {CSV_DIR}\n"
        "Run convert_rda_to_csv.R first."
    )

if raw["CDR"] is None:
    print("\n  NOTE: CDR not found — CDRSB_BL will be absent.")
    print("  Fix:  In R run  load('CDR.rda')")
    print("        write.csv(get(ls()[1]), 'DATA/csv/CDR.csv', row.names=FALSE)")


# Try to recover AGE from ADNIMERGE
adnimerge_path = find("ADNIMERGE")

if adnimerge_path:
    adnimerge = load(adnimerge_path, "ADNIMERGE")

    age_cols = [
        col
        for col in adnimerge.columns
        if col.upper() in {"AGE", "AGE_BL", "PTAGE", "YEARS_BL"}
    ]

    if age_cols:
        print(f"  AGE columns found in ADNIMERGE: {age_cols}")
        raw["ADNIMERGE_AGE"] = adnimerge[["RID"] + age_cols]
    else:
        print("  ADNIMERGE found but no AGE column — columns:", list(adnimerge.columns[:10]))
else:
    print("  ADNIMERGE.csv not found — AGE will be unavailable")
    print("  Download from ADNI portal → Study Data → Key ADNI tables")


# Step 2: Inspect DXSUM
sec("STEP 2 — Inspecting DXSUM diagnosis column")

dx = raw["DXSUM"].copy()
vsc = first_col(dx, ["VISCODE", "VISCODE2"])
dxc = first_col(dx, ["DXCHANGE", "DXCURREN", "DIAGNOSIS", "DX"])

if not vsc:
    raise KeyError("No VISCODE in DXSUM")

if not dxc:
    raise KeyError(f"No diagnosis column in DXSUM. Cols: {list(dx.columns)}")

print(f"  Visit code column : {vsc}")
print(f"  Diagnosis column  : {dxc}")
print(f"  Unique values     : {sorted(dx[dxc].dropna().unique(), key=str)}")

num_ratio = pd.to_numeric(dx[dxc].dropna().head(300), errors="coerce").notna().mean()
print(f"  Numeric ratio     : {num_ratio:.1%}")


# Step 3: Build pMCI / sMCI labels
sec("STEP 3 — Building pMCI / sMCI labels")

BL = {"BL", "SC"}


def as_int_set(series):
    return set(pd.to_numeric(series, errors="coerce").dropna().astype(int).tolist())


if num_ratio >= 0.80:
    dx[dxc] = pd.to_numeric(dx[dxc], errors="coerce")
    max_code = int(dx[dxc].max(skipna=True))

    print(f"  Mode: NUMERIC  max_code={max_code}")

    if max_code <= 3:
        print("  Sub-mode: DXCURREN (1=Normal 2=MCI 3=Dementia)")

        baseline_dx = get_baseline(dx)
        mci_set = as_int_set(baseline_dx.loc[baseline_dx[dxc] == 2, "RID"])

        followup = dx[
            dx["RID"].isin(mci_set)
            & ~dx[vsc].astype(str).str.upper().str.strip().isin(BL)
        ]

        converters = as_int_set(followup.loc[followup[dxc] == 3, "RID"])

    else:
        print("  Sub-mode: DXCHANGE (1-9)")

        baseline_dx = get_baseline(dx)
        mci_set = as_int_set(baseline_dx.loc[baseline_dx[dxc].isin({2, 4, 7}), "RID"])

        followup = dx[
            dx["RID"].isin(mci_set)
            & ~dx[vsc].astype(str).str.upper().str.strip().isin(BL)
        ]

        converters = as_int_set(followup.loc[followup[dxc].isin({5}), "RID"])

else:
    print("  Mode: TEXT")

    dx[dxc] = (
        dx[dxc]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )

    print(f"  Normalised: {sorted(dx[dxc].dropna().unique(), key=str)}")

    def classify_diagnosis(value):
        if pd.isna(value):
            return "UNK"

        value = str(value).upper()

        if any(key in value for key in ["DEMENTIA", "ALZHEIMER", "DEMENTED"]):
            return "DEM"

        if any(key in value for key in ["MCI", "EMCI", "LMCI", "MILD COGNITIVE"]):
            return "MCI"

        return "OTHER"

    dx["_c"] = dx[dxc].apply(classify_diagnosis)

    baseline_dx = get_baseline(dx)
    mci_set = as_int_set(baseline_dx.loc[baseline_dx["_c"] == "MCI", "RID"])

    followup = dx[
        dx["RID"].isin(mci_set)
        & ~dx[vsc].astype(str).str.upper().str.strip().isin(BL)
    ]

    converters = as_int_set(followup.loc[followup["_c"] == "DEM", "RID"])


stable = mci_set - converters

print(f"\n  MCI at baseline   : {len(mci_set):,}")
print(f"  pMCI (converters) : {len(converters):,}")
print(f"  sMCI (stable)     : {len(stable):,}")

labels = pd.DataFrame(
    {
        "RID": sorted(mci_set),
        "LABEL": [1 if rid in converters else 0 for rid in sorted(mci_set)],
    }
)

labels["RID"] = labels["RID"].astype("Int64")
labels["LABEL"] = labels["LABEL"].astype(int)

for label_value, label_name in [(0, "sMCI"), (1, "pMCI")]:
    count = (labels["LABEL"] == label_value).sum()
    print(f"    {label_name}: {count:,}  ({100 * count / len(labels):.1f}%)")


# Step 4: Save debug table
sec("STEP 4 — Saving debug table")

rows = []

for rid in sorted(mci_set):
    subject_rows = dx[dx["RID"] == rid].sort_values(vsc)

    baseline_rows = subject_rows[
        subject_rows[vsc].astype(str).str.upper().str.strip().isin(BL)
    ]

    followup_rows = subject_rows[
        ~subject_rows[vsc].astype(str).str.upper().str.strip().isin(BL)
    ]

    rows.append(
        {
            "RID": rid,
            "baseline_diag": baseline_rows[dxc].iloc[0] if len(baseline_rows) else "NONE",
            "n_total_visits": len(subject_rows),
            "n_followup_visits": len(followup_rows),
            "all_viscodes": "|".join(subject_rows[vsc].astype(str).tolist()),
            "all_diag": "|".join(subject_rows[dxc].fillna("NA").astype(str).tolist()),
            "followup_diag": "|".join(followup_rows[dxc].fillna("NA").astype(str).tolist()),
            "LABEL": 1 if rid in converters else 0,
            "label_text": "pMCI" if rid in converters else "sMCI",
        }
    )

os.makedirs(os.path.dirname(OUT_DEBUG), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_DEBUG, index=False)

print(f"  Saved: {OUT_DEBUG}")


# Step 5: Extract baseline clinical features
sec("STEP 5 — Extracting baseline clinical features")

blocks = {}


# Demographics
print("\n  Demographics (PTDEMOG)")

ptd = raw["PTDEMOG"]

if ptd is not None:
    source = first_per_rid(ptd)

    print(f"    All {len(source.columns)} cols: {list(source.columns)}")

    want = {
        "PTGENDER": "PTGENDER",
        "PTDOBYY": "PTDOBYY",
        "PTEDUCAT": "PTEDUCAT",
        "AGE": "AGE",
    }

    present = [col for col in want if col in source.columns]

    if present:
        dem = source[["RID"] + present].copy()
        dem.rename(columns=want, inplace=True)
        dem = fix_rid(dem.drop_duplicates("RID"))

        if "PTDOBYY" in dem.columns and "AGE" not in dem.columns:
            dem["AGE"] = 2010 - pd.to_numeric(dem["PTDOBYY"], errors="coerce")
            valid_age = dem["AGE"].notna().sum()

            if valid_age == 0:
                print("    AGE computation attempted from PTDOBYY — result: 0 valid values")
                print("    PTDOBYY appears empty for MCI baseline subjects")
                print("    AGE will be dropped — check ADNIMERGE.csv for AGE_BL column")
                dem.drop(columns=["AGE", "PTDOBYY"], errors="ignore", inplace=True)
            else:
                print(f"    AGE computed from PTDOBYY — {valid_age} valid values")

        if "PTGENDER" in dem.columns:
            dem["PTGENDER"] = pd.to_numeric(dem["PTGENDER"], errors="coerce").map(
                {1: 0, 2: 1}
            )

        feature_cols = [
            value
            for key, value in want.items()
            if key in present and value in dem.columns
        ]

        dem = dem.dropna(subset=list(dict.fromkeys(feature_cols)), how="all")

        print(f"    → {len(dem):,} subjects | cols: {list(dem.columns[1:])}")

        blocks["demog"] = dem


# MMSE
print("\n  MMSE")

mmse = pull(
    raw["MMSE"],
    "MMSE",
    want=["MMSCORE", "MMTOTAL", "MMSE_TOTAL", "TOTALMMSE"],
    rename={
        "MMSCORE": "MMSE_BL",
        "MMTOTAL": "MMSE_BL",
        "MMSE_TOTAL": "MMSE_BL",
        "TOTALMMSE": "MMSE_BL",
    },
)

if mmse is not None:
    blocks["mmse"] = mmse.loc[:, ~mmse.columns.duplicated()]


# MoCA
print("\n  MoCA")

moca_df = raw["MOCA"]

if moca_df is not None:
    print(f"    MOCA cols: {list(moca_df.columns)}")

moca = pull(
    moca_df,
    "MOCA",
    want=[
        "MOCATOTS",
        "MOCA_TOTAL",
        "MOCASTOT",
        "MOCARESULT",
        "MOCATOT",
        "MOCAT",
        "TOTAL",
        "MOCA",
    ],
    rename={
        "MOCATOTS": "MOCA_BL",
        "MOCA_TOTAL": "MOCA_BL",
        "MOCASTOT": "MOCA_BL",
        "MOCARESULT": "MOCA_BL",
        "MOCATOT": "MOCA_BL",
        "MOCAT": "MOCA_BL",
        "TOTAL": "MOCA_BL",
        "MOCA": "MOCA_BL",
    },
)

if moca is not None:
    blocks["moca"] = moca.loc[:, ~moca.columns.duplicated()]


# ADAS-Cog
print("\n  ADAS-Cog")

adas = pull(
    raw["ADAS"],
    "ADAS",
    want=[
        "TOTSCORE",
        "TOTAL11",
        "TOTAL13",
        "TOTALMOD",
        "ADAS11",
        "ADAS13",
        "ADAS_TOTAL",
    ],
    rename={
        "TOTSCORE": "ADAS11_BL",
        "TOTAL11": "ADAS11_BL",
        "ADAS11": "ADAS11_BL",
        "ADAS_TOTAL": "ADAS11_BL",
        "TOTAL13": "ADAS13_BL",
        "TOTALMOD": "ADAS13_BL",
        "ADAS13": "ADAS13_BL",
    },
)

if adas is not None:
    blocks["adas"] = adas.loc[:, ~adas.columns.duplicated()]


# FAQ
print("\n  FAQ")

faq = pull(
    raw["FAQ"],
    "FAQ",
    want=["FAQTOTAL", "FAQ_TOTAL", "FAQSCORE", "TOTAL"],
    rename={
        "FAQTOTAL": "FAQ_BL",
        "FAQ_TOTAL": "FAQ_BL",
        "FAQSCORE": "FAQ_BL",
        "TOTAL": "FAQ_BL",
    },
)

if faq is not None:
    blocks["faq"] = faq.loc[:, ~faq.columns.duplicated()]


# GDS
print("\n  Geriatric Depression Scale")

gds = pull(
    raw["GDSCALE"],
    "GDSCALE",
    want=["GDTOTAL", "GD_TOTAL", "GDSSCORE", "TOTAL"],
    rename={
        "GDTOTAL": "GDS_BL",
        "GD_TOTAL": "GDS_BL",
        "GDSSCORE": "GDS_BL",
        "TOTAL": "GDS_BL",
    },
)

if gds is not None:
    blocks["gds"] = gds.loc[:, ~gds.columns.duplicated()]


# CDR
print("\n  CDR")

if raw["CDR"] is not None:
    print(f"    CDR cols: {list(raw['CDR'].columns)}")

    cdr = pull(
        raw["CDR"],
        "CDR",
        want=[
            "CDGLOBAL",
            "CDRGLOB",
            "CDR_GLOBAL",
            "CDRSUM",
            "CDRSOB",
            "CDRSB",
            "SUMBOX",
        ],
        rename={
            "CDGLOBAL": "CDR_GLOBAL_BL",
            "CDRGLOB": "CDR_GLOBAL_BL",
            "CDR_GLOBAL": "CDR_GLOBAL_BL",
            "CDRSUM": "CDRSB_BL",
            "CDRSOB": "CDRSB_BL",
            "CDRSB": "CDRSB_BL",
            "SUMBOX": "CDRSB_BL",
        },
    )

    if cdr is not None:
        blocks["cdr"] = cdr.loc[:, ~cdr.columns.duplicated()]


# NEUROBAT
print("\n  NEUROBAT (RAVLT + executive function)")

neuro = pull(
    raw["NEUROBAT"],
    "NEUROBAT",
    want=[
        "AVDELTOT",
        "AVTOT1",
        "AVDEL30MIN",
        "LIMMTOTAL",
        "LDELTOTAL",
        "DIGITSCOR",
        "TRABSCOR",
    ],
    rename={
        "AVDELTOT": "RAVLT_forgetting",
        "AVTOT1": "RAVLT_immediate",
        "AVDEL30MIN": "RAVLT_delayed",
        "LIMMTOTAL": "LogMem_immediate",
        "LDELTOTAL": "LogMem_delayed",
        "DIGITSCOR": "DigitSpan",
        "TRABSCOR": "TrailsB",
    },
)

if neuro is not None:
    blocks["neuro"] = neuro.loc[:, ~neuro.columns.duplicated()]

print("\n  UCSFFSX7 — EXCLUDED (Phase 1 = clinical-only, no 1.5_Clinical+MRI_VolumetricData)")


# Step 6: Merge all feature blocks
sec("STEP 6 — Merging onto label table")

meta = labels.copy()

print(f"  Start: {len(meta):,} subjects\n")

for name in ["demog", "mmse", "moca", "adas", "faq", "gds", "cdr", "neuro"]:
    block = blocks.get(name)

    if block is None:
        print(f"  --    {name:<10} not available")
        continue

    block = fix_rid(block)

    meta["RID"] = meta["RID"].astype("Int64")
    block["RID"] = block["RID"].astype("Int64")

    meta = meta.merge(block, on="RID", how="left")
    meta = meta.loc[:, ~meta.columns.duplicated()]

    new_cols = [col for col in block.columns if col != "RID"]

    if new_cols:
        matched = meta[new_cols[0]].notna().sum()

        print(
            f"  OK    {name:<10} merged | "
            f"{matched:>4}/{len(meta)} subjects matched "
            f"({100 * matched / len(meta):.0f}%) | shape {meta.shape}"
        )


# Step 7: Create derived features without filling missing values
sec("STEP 7 — Derived features  (NaN preserved)")

skip_str = {"RID", "LABEL", "PTETHCAT", "PTRACCAT", "PTMARRY"}

for col in meta.columns:
    if col not in skip_str:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")

if {"MMSE_BL", "FAQ_BL"}.issubset(meta.columns):
    meta["MMSE_FAQ_composite"] = (30 - meta["MMSE_BL"]) + meta["FAQ_BL"]
    print("  MMSE_FAQ_composite   =  (30 - MMSE_BL) + FAQ_BL")

if {"ADAS11_BL", "MMSE_BL"}.issubset(meta.columns):
    meta["ADAS_MMSE_gap"] = meta["ADAS11_BL"] - (30 - meta["MMSE_BL"])
    print("  ADAS_MMSE_gap        =  ADAS11_BL - (30 - MMSE_BL)")

if {"RAVLT_forgetting", "RAVLT_immediate"}.issubset(meta.columns):
    immediate = meta["RAVLT_immediate"].replace(0, np.nan)
    meta["RAVLT_forget_rate"] = meta["RAVLT_forgetting"] / immediate
    print("  RAVLT_forget_rate    =  RAVLT_forgetting / RAVLT_immediate")

print("\n  NOTE: No imputation done here. Imputation happens inside the")
print("        training pipeline, fitted on train data only.")


# Step 8: Report pre-imputation feature coverage
sec("STEP 8 — Pre-imputation coverage report")

print(
    f"\n  Subjects : {len(meta):,}  |  "
    f"pMCI: {(meta['LABEL'] == 1).sum()}  "
    f"sMCI: {(meta['LABEL'] == 0).sum()}"
)

print(f"  Columns  : {len(meta.columns)} (RID + LABEL + {len(meta.columns) - 2} features)\n")

print(f"  {'Column':<28} {'N present':>10}  {'Missing':>8}  {'Coverage':>9}")
print(f"  {'-' * 28} {'-' * 10}  {'-' * 8}  {'-' * 9}")

for col in meta.columns:
    if col == "RID":
        continue

    present = meta[col].notna().sum()
    missing = meta[col].isna().sum()
    coverage = 100 * present / len(meta) if len(meta) else 0

    flag = "  <- LOW, will be imputed in pipeline" if coverage < 50 else ""

    print(f"  {col:<28} {present:>10,}  {missing:>8,}  {coverage:>8.1f}%{flag}")


# Drop columns with no usable coverage
DROP_EMPTY = ["PTDOBYY", "AGE", "LogMem_immediate", "LogMem_delayed"]

dropped = [col for col in DROP_EMPTY if col in meta.columns]
meta.drop(columns=dropped, inplace=True)

print(f"\n  Dropped 0%-coverage columns: {dropped}")
print(f"  Final shape: {meta.shape}  (RID + LABEL + {meta.shape[1] - 2} features)")


# Step 9: Save raw outputs
sec("STEP 9 — Saving outputs  (NaN values preserved)")

for path in [OUT_META, OUT_LABELS]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

meta.to_csv(OUT_META, index=False)
meta[["RID", "LABEL"]].to_csv(OUT_LABELS, index=False)

print(f"  final_metadata.csv    {meta.shape}  ->  {OUT_META}")
print(f"  pmci_smci_labels.csv              ->  {OUT_LABELS}")
print(f"  dxsum_debug.csv                   ->  {OUT_DEBUG}")


sec("BUILD COMPLETE")

p_count = (meta["LABEL"] == 1).sum()
s_count = (meta["LABEL"] == 0).sum()

print(
    f"""
  Subjects  : {len(meta):,}
  pMCI (1)  : {p_count:,}  ({100 * p_count / len(meta):.1f}%)
  sMCI (0)  : {s_count:,}  ({100 * s_count / len(meta):.1f}%)
  Features  : {len(meta.columns) - 2}  (clinical only, raw NaN intact)

  Excluded
  ------------------------------------------------
  APOE4   — genetic testing unavailable in low-resource settings
  1.5_Clinical+MRI_VolumetricData     — Phase 2 only

  Pre-imputation missingness is what you report in the paper.
  Imputation happens inside training pipeline, with no leakage.

"""
)