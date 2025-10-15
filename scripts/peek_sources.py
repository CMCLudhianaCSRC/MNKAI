# save as scripts/peek_sources.py and run: python scripts/peek_sources.py
import pandas as pd, pathlib, sys
p = pathlib.Path("data/metadata.csv")  # change if yours differs
df = pd.read_csv(p)
cols = [c.lower() for c in df.columns]
df.columns = cols
have = set(cols)
need = {"source","split","label"}
missing = need - have
if missing:
    print(f"CSV is missing columns: {missing}. Columns found: {cols}")
    sys.exit(1)

print("\n== Counts by source ==")
print(df.groupby("source").size().sort_values(ascending=False))

print("\n== Train/Valid by source ==")
print(df.groupby(["source","split"]).size())

print("\n== Class counts (train only) ==")
print(df[df.split=="train"].groupby("label").size().sort_values(ascending=False))
