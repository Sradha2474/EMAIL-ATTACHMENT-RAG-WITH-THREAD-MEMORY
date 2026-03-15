import pandas as pd
import os
import hashlib

print("Loading emails.csv... (this takes ~1 min)")
df = pd.read_csv("emails.csv")
print(f"Total emails in dataset: {len(df)}")

# Pick emails from one person's mailbox in early 2001
mask = df["file"].str.startswith("allen-p/")
slice_df = df[mask].head(200)

os.makedirs("data/emails", exist_ok=True)
saved = 0

for _, row in slice_df.iterrows():
    try:
        fname = hashlib.md5(row["file"].encode()).hexdigest()[:12] + ".eml"
        filepath = os.path.join("data", "emails", fname)
        with open(filepath, "w", encoding="utf-8", errors="replace") as f:
            f.write(row["message"])
        saved += 1
    except Exception as e:
        print(f"Skipped one file: {e}")

print(f"Done! Saved {saved} emails to data/emails/")