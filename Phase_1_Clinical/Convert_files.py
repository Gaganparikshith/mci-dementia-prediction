import pandas as pd
from pathlib import Path
import pyreadr

rda_file = Path(r"C:\Users\ASUS\Desktop\Research Resources\DATA\data\CDR.rda")
out_csv = Path(r"C:\Users\ASUS\Desktop\Research Resources\DATA\CDR.csv")

print(f"Loading: {rda_file.name}")
result = pyreadr.read_r(str(rda_file))

keys = list(result.keys())
print(f"Objects in .rda: {keys}")

df = result[keys[0]]
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"Columns: {list(df.columns)}")

cdr_cols = [col for col in df.columns if "CDR" in col.upper()]
print(f"\nCDR-related columns: {cdr_cols}")

id_cols = [col for col in df.columns if col.upper() in {"RID", "PTID"}]
print(f"ID columns: {id_cols}")

df.to_csv(out_csv, index=False)

print(f"\nSaved: {out_csv}")
print("Done. Now rerun build_dataset.py")