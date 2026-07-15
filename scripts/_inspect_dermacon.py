"""Inspect DermaCon-IN coverage: classes, FST distribution, overlap with ISIC 8-class."""
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\AIFINAL\images\dermacon_in")

md = pd.read_csv(ROOT / "METADATA" / "Skin_Metadata.tab", sep="\t")
print(f"rows={len(md)}  columns={len(md.columns)}")
print()

# Counts by Main_class
print("=== Main_class ===")
print(md["Main_class"].value_counts())
print()

# Top Disease_label
print("=== top Disease_label (50) ===")
print(md["Disease_label"].value_counts().head(50))
print()

# Sub_class breakdown
print("=== Sub_class ===")
print(md["Sub_class"].value_counts().head(30))
print()

# Fitzpatrick distribution
print("=== Fitzpatrick (FST) ===")
print(md["Fitzpatrick"].value_counts())
print()

# Monk skin tone
print("=== Monk skin tone ===")
print(md["Monk_skin_tone"].value_counts())
print()

# Subject (patient) count for group-aware splits
print("=== unique Subject_IDs ===")
print(f"  unique subjects: {md['Subject_ID'].nunique()}")
print(f"  images per subject: median={md.groupby('Subject_ID').size().median()} "
      f"max={md.groupby('Subject_ID').size().max()}")
print()

# Overlap with ISIC 8-class — naive keyword scan
KEYWORDS = {
    "mel": ["melanoma"],
    "nv":  ["nevus", "naevus"],
    "bcc": ["basal cell"],
    "scc": ["squamous"],
    "df":  ["dermatofibroma"],
    "bkl": ["seborrheic keratosis", "solar lentigo", "benign keratosis"],
    "ak":  ["actinic keratosis", "keratoacanthoma"],
    "vasc": ["hemangioma", "angioma", "vascular"],
}
print("=== rough 8-class overlap (substring match on Disease_label) ===")
total_match = 0
for c, kws in KEYWORDS.items():
    pattern = "|".join(kws)
    matched = md["Disease_label"].str.contains(pattern, case=False, na=False).sum()
    total_match += matched
    print(f"  {c}: {matched} (kw={kws})")
print(f"  total matched: {total_match} / {len(md)}  ({100*total_match/max(1,len(md)):.1f}%)")
