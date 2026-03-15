"""
Create a small slice of Enron emails as .eml files in data/emails/.

Prerequisite: Download the Enron Email Dataset from Kaggle and place emails.csv
in this directory: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
"""
import pandas as pd
import os
import hashlib
from pathlib import Path

EMAILS_CSV = Path("emails.csv")
OUT_DIR = Path("data/emails")

if not EMAILS_CSV.exists():
    print("emails.csv not found.")
    print("Download from: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset")
    print("Place emails.csv in the project root, then run: python slice_data.py")
    raise SystemExit(1)

print("Loading emails.csv... (this may take ~1 min)")
df = pd.read_csv(EMAILS_CSV)
print(f"Total emails in dataset: {len(df)}")

# Pick emails from one mailbox; 100-300 messages for spec
mask = df["file"].str.startswith("allen-p/")
slice_df = df[mask].head(250)

OUT_DIR.mkdir(parents=True, exist_ok=True)
saved = 0

for _, row in slice_df.iterrows():
    try:
        raw = row["message"]
        if pd.isna(raw) or not str(raw).strip():
            continue
        fname = hashlib.md5(row["file"].encode()).hexdigest()[:12] + ".eml"
        filepath = OUT_DIR / fname
        with open(filepath, "w", encoding="utf-8", errors="replace") as f:
            f.write(str(raw))
        saved += 1
    except Exception as e:
        print(f"Skipped one file: {e}")

print(f"Done! Saved {saved} .eml files to {OUT_DIR}/")