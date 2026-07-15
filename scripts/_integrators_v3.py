"""Phase-1 + Phase-3 integrators for the v3 pipeline.

Reads downloaded Kaggle datasets from D:\\AIFINAL\\images\\external\\<name>\\
and emits two manifest CSVs:

  data/manifests/kaggle_external.csv      — Phase 1 base sources
  data/manifests/phase3_external.csv      — Phase 3 Indian fine-tune sources

Each CSV is plain UTF-8 with the canonical columns:
  isic_id, filepath, label_name, source, group_id, sex, age_approx,
  fitzpatrick, anatom_site, image_type, attribution

Sources integrated:
  Phase 1:
    - PAD-UFES-20            (already in pre-existing kaggle_external.csv)
    - Derm7pt                (already in)
    - BCN20000               (already in)
    - ISIC 2019              (already in)
    - ISIC 2024 challenge    [NEW] (~400k incl. 3D-TBP crops + metadata)
    - normal class:          [NEW] mpox-msld healthy, oily-dry-normal, our-normal-skin
  Phase 3:
    - DermNet                [NEW] (23 disease classes, mostly Indian-relevant)
    - SkinDiseaseDataset     [NEW] (9 classes incl. acne, eczema, psoriasis)
"""
from __future__ import annotations

import sys, re
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnkai.data.manifest import (
    LabelNormalizer, PHASE1_CLASSES, PHASE3_CLASSES,
)

EXT_ROOT = Path(r"D:\AIFINAL\images\external")

OUT_PHASE1 = Path("data/manifests/kaggle_external.csv")    # legacy name; will be merged
OUT_PHASE3 = Path("data/manifests/phase3_external.csv")    # new

PHASE1_ROWS: list[dict] = []
PHASE3_ROWS: list[dict] = []


# ──────────────────────────────────────────────────────────────────────────────
#  ISIC 2024 challenge (Phase 1)
#    Format on disk:
#      isic-2024-challenge.zip extracted ->
#        train-image/image/ISIC_xxxxxxx.jpg          (~400k thumbnails, ~2GB)
#        train-metadata.csv                          (image_id, target, age, ...)
#        train-image.hdf5  (huge full-res; we ignore for now)
# ──────────────────────────────────────────────────────────────────────────────
def integrate_isic2024(extracted: Path = EXT_ROOT / "isic-2024") -> int:
    """Strategy-C labelling for ISIC 2024 (~401k 3D-TBP close-up tiles).

    The dataset task is binary "malignant vs benign" — most images don't have
    fine-grain dx. We split into three buckets:

      A. target==1 + iddx_3 specific  -> hard label = mel/bcc/scc/...
                                         weak_label = False
      B. target==0 + iddx_3 specific  -> hard label = matched class
                                         weak_label = False
      C. target==1, iddx_3 NaN        -> hard label = 'mel'  (most TBP malignancies)
                                         weak_label = True
      D. target==0, iddx_3 NaN        -> hard label = 'nv'   (most benign TBP tiles)
                                         weak_label = True   (down-weighted in sampler)

    Output rows have an extra `weak_label` column the trainer / sampler can use
    to halve their weight (or filter out for high-confidence-only evaluation).
    """
    if not extracted.exists():
        print(f"[isic2024] {extracted} missing — skipping"); return 0

    meta_paths = list(extracted.rglob("train-metadata*.csv"))
    if not meta_paths:
        print(f"[isic2024] no train-metadata.csv found"); return 0
    meta = pd.read_csv(meta_paths[0], low_memory=False)
    print(f"[isic2024] metadata: {meta_paths[0]}, shape={meta.shape}", flush=True)

    img_lookup: dict[str, str] = {}
    for ext in (".jpg", ".jpeg", ".png"):
        for p in extracted.rglob(f"*{ext}"):
            img_lookup[p.stem] = str(p)
    print(f"[isic2024] images on disk: {len(img_lookup):,}", flush=True)

    norm = LabelNormalizer(classes=PHASE1_CLASSES)

    n_hard = 0     # bucket A or B
    n_weak_mel = 0 # bucket C
    n_weak_nv  = 0 # bucket D
    no_image  = 0
    for _, r in meta.iterrows():
        iid = str(r["isic_id"]).strip()
        fp = img_lookup.get(iid)
        if not fp:
            no_image += 1
            continue

        target = int(r.get("target", 0)) if pd.notna(r.get("target", 0)) else 0

        # Try fine-grain dx first
        cls = None
        for col in ("iddx_3", "iddx_2", "iddx_full"):
            v = str(r.get(col, "") or "").strip()
            if v and v.lower() not in ("nan", "none", "benign", "malignant"):
                cls = norm.map(v)
                if cls:
                    break

        weak = False
        if cls is None:
            # Strategy C bucket C/D: weak label from target
            cls = "mel" if target == 1 else "nv"
            weak = True

        # Identify if this came from MSK (most ISIC2024 is MSK contributed) →
        # If so, mark it as the same external test "MSK" domain for split logic
        attrib = str(r.get("attribution", "") or "").lower()
        is_msk = "memorial sloan kettering" in attrib

        PHASE1_ROWS.append({
            "isic_id":     iid,
            "filepath":    fp,
            "label_name":  cls,
            "source":      "ISIC_2024" + ("_MSK" if is_msk else ""),
            "group_id":    str(r.get("patient_id", iid)) if pd.notna(r.get("patient_id")) else iid,
            "fitzpatrick": None,
            "sex":         r.get("sex"),
            "age_approx":  r.get("age_approx"),
            "weak_label":  weak,
            "lesion_id_isic24":  r.get("lesion_id"),
            "image_type":  r.get("image_type"),       # all 'TBP tile: close-up'
            "tbp_tile_type": r.get("tbp_tile_type"),  # '3D: XP' / '3D: white'
        })
        if weak:
            if cls == "mel": n_weak_mel += 1
            else:            n_weak_nv  += 1
        else:
            n_hard += 1

    total = n_hard + n_weak_mel + n_weak_nv
    print(f"[isic2024] integrated {total:,} rows: "
          f"hard={n_hard:,}, weak_nv={n_weak_nv:,}, weak_mel={n_weak_mel:,} "
          f"(no_image={no_image:,})", flush=True)
    return total


# ──────────────────────────────────────────────────────────────────────────────
#  Mpox dataset (Phase 1: extract healthy + mpox subsets)
#  joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20:
#    extracted/Original Images/Original Images/<class>/<file>.jpg
#    classes typically: Mpox, Chickenpox, Measles, Cowpox, Healthy, HFMD
# ──────────────────────────────────────────────────────────────────────────────
def integrate_mpox(extracted: Path = EXT_ROOT / "mpox-msld") -> int:
    if not extracted.exists():
        print(f"[mpox] {extracted} missing — skipping"); return 0

    # Walk the directory, group by parent folder name
    by_class: dict[str, list[Path]] = {}
    for img in extracted.rglob("*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        cls_dir = img.parent.name
        by_class.setdefault(cls_dir, []).append(img)

    # Flatten to all classes, then map to our 9-class
    print(f"[mpox] folders found: {list(by_class.keys())[:10]}", flush=True)

    # Apr 2026 expansion:
    #   Healthy -> Phase 1 'normal'
    #   Mpox/Chickenpox/Cowpox/HFMD/Measles/Monkeypox -> Phase 3 'viral_infection'
    healthy_keywords = ["healthy", "normal", "negative"]
    viral_keywords = ["mpox", "monkeypox", "chickenpox", "cowpox",
                      "hfmd", "measles"]

    p1_matched, p3_matched = 0, 0
    for cls_dir, images in by_class.items():
        cls_l = cls_dir.lower()
        is_healthy = any(kw in cls_l for kw in healthy_keywords)
        is_viral   = any(kw in cls_l for kw in viral_keywords)

        for img in images:
            stem = img.stem
            if is_healthy:
                PHASE1_ROWS.append({
                    "isic_id":     f"MPOX_{stem}",
                    "filepath":    str(img),
                    "label_name":  "normal",
                    "source":      "MPOX_HEALTHY",
                    "group_id":    f"MPOX_{stem}",
                    "fitzpatrick": None, "sex": None, "age_approx": None,
                })
                p1_matched += 1
            elif is_viral:
                PHASE3_ROWS.append({
                    "isic_id":     f"MPOX_{stem}",
                    "filepath":    str(img),
                    "label_name":  "viral_infection",
                    "source":      f"MPOX_{cls_dir.upper()}",
                    "group_id":    f"MPOX_{stem}",
                    "fitzpatrick": None, "sex": None, "age_approx": None,
                })
                p3_matched += 1
    print(f"[mpox] healthy(P1)={p1_matched}, viral(P3)={p3_matched}", flush=True)
    return p1_matched + p3_matched


# ──────────────────────────────────────────────────────────────────────────────
#  Oily/Dry/Normal skin types (Phase 1: extract 'normal' subset only)
#  shakyadissanayake/oily-dry-and-normal-skin-types-dataset:
#    Oily-Dry-Skin-Types/Oily-Dry-Skin-Types/{train,valid,test}/{normal,oily,dry}/*.jpg
# ──────────────────────────────────────────────────────────────────────────────
def integrate_skin_types(extracted: Path = EXT_ROOT / "skin-types") -> int:
    if not extracted.exists():
        print(f"[skintypes] {extracted} missing — skipping"); return 0
    matched = 0
    for img in extracted.rglob("*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        # Find any parent folder named 'normal'
        if any(p.name.lower() == "normal" for p in img.parents):
            PHASE1_ROWS.append({
                "isic_id":     f"SKTYPE_{img.stem}",
                "filepath":    str(img),
                "label_name":  "normal",
                "source":      "SKINTYPE_NORMAL",
                "group_id":    f"SKTYPE_{img.stem}",
                "fitzpatrick": None, "sex": None, "age_approx": None,
            })
            matched += 1
    print(f"[skintypes] normal-class images: {matched}", flush=True)
    return matched


# ──────────────────────────────────────────────────────────────────────────────
#  ahdasdwdasd/our-normal-skin (Phase 1)
# ──────────────────────────────────────────────────────────────────────────────
def integrate_our_normal(extracted: Path = EXT_ROOT / "our-normal-skin") -> int:
    if not extracted.exists():
        print(f"[our_normal] {extracted} missing — skipping"); return 0
    matched = 0
    for img in extracted.rglob("*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        PHASE1_ROWS.append({
            "isic_id":     f"OURNORM_{img.stem}",
            "filepath":    str(img),
            "label_name":  "normal",
            "source":      "OUR_NORMAL_SKIN",
            "group_id":    f"OURNORM_{img.stem}",
            "fitzpatrick": None, "sex": None, "age_approx": None,
        })
        matched += 1
    print(f"[our_normal] images: {matched}", flush=True)
    return matched


# ──────────────────────────────────────────────────────────────────────────────
#  pacificrm/skindiseasedataset (Phase 1 + Phase 3)
#  Folder structure: Skin diseases/<class>/*.jpg
#  Classes typically: Acne, Carcinoma, Eczema, Keratosis, Milia,
#                     Psoriasis, Rosacea, Healthy, etc.
# ──────────────────────────────────────────────────────────────────────────────
def integrate_skindisease(extracted: Path = EXT_ROOT / "skindisease") -> int:
    if not extracted.exists():
        print(f"[skindisease] {extracted} missing — skipping"); return 0

    norm_p1 = LabelNormalizer(classes=PHASE1_CLASSES)
    norm_p3 = LabelNormalizer(classes=PHASE3_CLASSES)

    # Walk by class folder
    by_class: dict[str, list[Path]] = {}
    for img in extracted.rglob("*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        # use the deepest subdir name as class
        for parent in img.parents:
            n = parent.name
            if n and n.lower() not in ("skin diseases", "skindiseases",
                                       "skindiseasedataset", extracted.name.lower()):
                cls_dir = n; break
        else:
            cls_dir = "unknown"
        by_class.setdefault(cls_dir, []).append(img)

    print(f"[skindisease] class folders: {list(by_class.keys())}", flush=True)

    p1_matched, p3_matched = 0, 0
    for cls_dir, images in by_class.items():
        cls_text = re.sub(r"[_\-]+", " ", cls_dir).strip().lower()

        cls_p1 = norm_p1.map(cls_text)
        cls_p3 = norm_p3.map(cls_text)

        for img in images:
            stem = img.stem
            if cls_p1:
                PHASE1_ROWS.append({
                    "isic_id":     f"SDS_{stem}",
                    "filepath":    str(img),
                    "label_name":  cls_p1,
                    "source":      "SKINDISEASE_P1",
                    "group_id":    f"SDS_{stem}",
                    "fitzpatrick": None, "sex": None, "age_approx": None,
                })
                p1_matched += 1
            if cls_p3 and cls_p3 not in PHASE1_CLASSES:
                # Phase 3-only class
                PHASE3_ROWS.append({
                    "isic_id":     f"SDS_{stem}",
                    "filepath":    str(img),
                    "label_name":  cls_p3,
                    "source":      "SKINDISEASE_P3",
                    "group_id":    f"SDS_{stem}",
                    "fitzpatrick": None, "sex": None, "age_approx": None,
                })
                p3_matched += 1
    print(f"[skindisease] Phase 1: {p1_matched}, Phase 3: {p3_matched}", flush=True)
    return p1_matched + p3_matched


# ──────────────────────────────────────────────────────────────────────────────
#  shubhamgoel27/dermnet (Phase 3 — Indian-relevant disease classes)
#  Folder: train/<class>/*.jpg, test/<class>/*.jpg  (23 classes)
# ──────────────────────────────────────────────────────────────────────────────
def integrate_dermnet(extracted: Path = EXT_ROOT / "dermnet") -> int:
    """DermNet has 23 verbose folder names like 'Acne and Rosacea Photos'.
    We map each folder to its primary 8-class / Phase-3 class via keywords.
    """
    if not extracted.exists():
        print(f"[dermnet] {extracted} missing — skipping"); return 0

    # Hand-curated DermNet folder -> class mapping (Apr 2026 expansion).
    # Now covers ALL 23 DermNet folders. Recovers ~2.3k previously-skipped images.
    DERMNET_MAP: dict[str, tuple[str | None, str | None]] = {
        # folder_substring               (Phase1_class,         Phase3_class)
        "acne and rosacea":              (None,                 "acne"),
        "actinic keratosis":             ("ak",                 None),
        "atopic dermatitis":             (None,                 "eczema"),
        "bullous":                       (None,                 "pemphigus"),
        "cellulitis":                    (None,                 "bacterial_infection"),
        "eczema":                        (None,                 "eczema"),
        "exanthems":                     (None,                 "drug_eruption"),
        "hair loss":                     (None,                 "alopecia"),
        "herpes":                        (None,                 "viral_infection"),
        "light diseases":                (None,                 "vitiligo"),
        "lupus":                         (None,                 "autoimmune_systemic"),
        "melanoma skin cancer nevi":     ("mel",                None),
        "nail fungus":                   (None,                 "tinea"),
        "poison ivy":                    (None,                 "contact_dermatitis"),
        "psoriasis pictures lichen":     (None,                 "psoriasis"),
        "scabies lyme":                  (None,                 "scabies"),
        "seborrheic keratoses":          ("bkl",                None),
        "systemic":                      (None,                 "autoimmune_systemic"),
        "tinea ringworm":                (None,                 "tinea"),
        "urticaria":                     (None,                 "urticaria"),
        "vascular tumors":               ("vasc",               None),
        "vasculitis":                    (None,                 "autoimmune_systemic"),
        "warts molluscum":               (None,                 "viral_infection"),
    }

    by_class: dict[str, list[Path]] = {}
    for img in extracted.rglob("*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        cls_dir = img.parent.name
        by_class.setdefault(cls_dir, []).append(img)
    print(f"[dermnet] folders found: {len(by_class)}", flush=True)

    p1_matched, p3_matched = 0, 0
    skipped_folders = []
    for cls_dir, images in by_class.items():
        cls_lower = cls_dir.lower()
        cls_p1, cls_p3 = None, None
        for substr, (p1, p3) in DERMNET_MAP.items():
            if substr in cls_lower:
                cls_p1, cls_p3 = p1, p3
                break
        if cls_p1 is None and cls_p3 is None:
            skipped_folders.append(cls_dir)
            continue

        for img in images:
            stem = img.stem
            if cls_p1:
                PHASE1_ROWS.append({
                    "isic_id":     f"DN_{stem}",
                    "filepath":    str(img),
                    "label_name":  cls_p1,
                    "source":      "DERMNET_P1",
                    "group_id":    f"DN_{stem}",
                    "fitzpatrick": None, "sex": None, "age_approx": None,
                    "weak_label":  False,
                })
                p1_matched += 1
            if cls_p3:
                PHASE3_ROWS.append({
                    "isic_id":     f"DN_{stem}",
                    "filepath":    str(img),
                    "label_name":  cls_p3,
                    "source":      "DERMNET_P3",
                    "group_id":    f"DN_{stem}",
                    "fitzpatrick": None, "sex": None, "age_approx": None,
                })
                p3_matched += 1
    print(f"[dermnet] Phase 1: {p1_matched}, Phase 3: {p3_matched}, "
          f"skipped folders: {skipped_folders}", flush=True)
    return p1_matched + p3_matched


# ──────────────────────────────────────────────────────────────────────────────
#  Re-merge with previously-integrated PAD/Derm7pt/BCN/ISIC2019 rows.
#  We re-call the legacy module's integrators so PHASE1_ROWS holds everything.
# ──────────────────────────────────────────────────────────────────────────────
def integrate_legacy_kaggle():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "legacy", str(Path(__file__).parent / "_dl_kaggle_datasets.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.ALL_ROWS = []
    mod.integrate_pad_ufes(EXT_ROOT / "pad-ufes-20")
    mod.integrate_derm7pt(EXT_ROOT / "derm7pt")
    mod.integrate_bcn20000(EXT_ROOT / "bcn20000")
    mod.integrate_isic2019(EXT_ROOT / "isic-2019")
    PHASE1_ROWS.extend(mod.ALL_ROWS)
    print(f"[legacy_kaggle] re-integrated {len(mod.ALL_ROWS):,} rows from PAD/Derm7pt/BCN/ISIC2019",
          flush=True)


# ──────────────────────────────────────────────────────────────────────────────
#  SCIN (Google + Stanford 2024)  — 10k+ crowdsourced clinical photos
#  Schema (per scin/dataset_schema.md):
#    cases.csv  : case_id, dermatologist_skin_condition_on_label_name,
#                 estimated_fitzpatrick_skin_type_label_name,
#                 monk_skin_tone_label_name, ...
#    images.csv : case_id, image_path, ...
#  Mapping notes: SCIN labels are clinical (acne, eczema, psoriasis, etc.) —
#  most map to PHASE 3 classes, not Phase 1.
# ──────────────────────────────────────────────────────────────────────────────
def integrate_scin(extracted: Path = EXT_ROOT / "scin") -> int:
    """SCIN (Google + Stanford 2024).

    Real schema (verified Apr 2026):
      scin_cases.csv  : case_id, image_1_path, image_2_path, image_3_path,
                        sex_at_birth, age_group, fitzpatrick_skin_type, ...
      scin_labels.csv : case_id,
                        dermatologist_skin_condition_on_label_name (text),
                        weighted_skin_condition_label (consensus),
                        dermatologist_fitzpatrick_skin_type_label_1/2/3 (FST1-6)

    Each case has up to 3 images (image_N_path = relative path to .png).
    """
    if not extracted.exists():
        print(f"[scin] {extracted} missing — skipping"); return 0

    cases_csv  = next((p for p in extracted.rglob("scin_cases.csv")), None)
    labels_csv = next((p for p in extracted.rglob("scin_labels.csv")), None)
    if cases_csv is None:
        print(f"[scin] scin_cases.csv not found — skipping"); return 0

    cases = pd.read_csv(cases_csv, low_memory=False)
    print(f"[scin] cases: {len(cases):,}, columns: {len(cases.columns)}", flush=True)

    # Merge labels into cases via case_id
    if labels_csv is not None:
        labels = pd.read_csv(labels_csv, low_memory=False)
        cases = cases.merge(labels, on="case_id", how="left", suffixes=("", "_lbl"))
        print(f"[scin] merged with labels", flush=True)

    norm_p1 = LabelNormalizer(classes=PHASE1_CLASSES)
    norm_p3 = LabelNormalizer(classes=PHASE3_CLASSES)

    # SCIN's label columns we care about (in order of preference)
    label_cols_priority = [
        "dermatologist_skin_condition_on_label_name",
        "weighted_skin_condition_label",
    ]
    fst_col = "dermatologist_fitzpatrick_skin_type_label_1"

    # SCIN labels are often LISTS encoded as strings (e.g. "['Eczema', 'Acne']")
    # We split on commas/semicolons to extract individual conditions.
    def split_label(text: str) -> list[str]:
        if not text or text in ("nan", "None", "[]"):
            return []
        # remove brackets/quotes
        t = text.strip("[](){}").replace("'", "").replace('"', "")
        # split on common separators
        parts = re.split(r"[,;|]", t)
        return [p.strip().lower() for p in parts if p.strip()]

    p1_matched, p3_matched = 0, 0
    no_image, no_label = 0, 0

    for _, r in cases.iterrows():
        cid = str(r["case_id"]).strip()

        # Collect candidate label texts
        all_label_texts = []
        for col in label_cols_priority:
            v = r.get(col, None)
            if pd.notna(v):
                all_label_texts.extend(split_label(str(v)))
        if not all_label_texts:
            no_label += 1
            continue

        # Find first matching class in our taxonomies
        cls_p1, cls_p3 = None, None
        for lt in all_label_texts:
            if cls_p1 is None: cls_p1 = norm_p1.map(lt)
            if cls_p3 is None: cls_p3 = norm_p3.map(lt)
            if cls_p1 and cls_p3:
                break
        if not (cls_p1 or cls_p3):
            continue

        # Get FST
        fst_raw = r.get(fst_col, None)
        fst = None
        if pd.notna(fst_raw):
            m = re.search(r"FST\s*([1-6])", str(fst_raw))
            if m: fst = m.group(1)

        # Walk image_1_path .. image_3_path
        for col in ("image_1_path", "image_2_path", "image_3_path"):
            ipath = r.get(col, None)
            if pd.isna(ipath) or not ipath:
                continue
            # Path is relative to scin extracted root, like 'dataset/images/...'
            full = extracted / str(ipath)
            if not full.exists():
                # try relative from extracted/dataset
                full = extracted / "dataset" / Path(str(ipath)).name
                if not full.exists():
                    continue

            stem = full.stem
            if cls_p1:
                PHASE1_ROWS.append({
                    "isic_id":     f"SCIN_{stem}",
                    "filepath":    str(full),
                    "label_name":  cls_p1,
                    "source":      "SCIN_P1",
                    "group_id":    f"SCIN_{cid}",
                    "fitzpatrick": fst,
                    "sex":         r.get("sex_at_birth"),
                    "age_approx":  r.get("age_group"),
                    "weak_label":  False,
                })
                p1_matched += 1
            if cls_p3 and cls_p3 not in PHASE1_CLASSES:
                PHASE3_ROWS.append({
                    "isic_id":     f"SCIN_{stem}",
                    "filepath":    str(full),
                    "label_name":  cls_p3,
                    "source":      "SCIN_P3",
                    "group_id":    f"SCIN_{cid}",
                    "fitzpatrick": fst,
                    "sex":         r.get("sex_at_birth"),
                    "age_approx":  r.get("age_group"),
                })
                p3_matched += 1

    print(f"[scin] Phase 1: {p1_matched}, Phase 3: {p3_matched} "
          f"(no_label: {no_label})", flush=True)
    return p1_matched + p3_matched


def main():
    print("=" * 70)
    print("PHASE 1 + PHASE 3 INTEGRATORS (Strategy C)")
    print("=" * 70)
    integrate_legacy_kaggle()
    integrate_isic2024()
    integrate_mpox()
    integrate_skin_types()
    integrate_our_normal()
    integrate_skindisease()
    integrate_dermnet()
    integrate_scin()

    OUT_PHASE1.parent.mkdir(parents=True, exist_ok=True)
    df1 = pd.DataFrame(PHASE1_ROWS)
    df1.to_csv(OUT_PHASE1, index=False)
    df3 = pd.DataFrame(PHASE3_ROWS)
    df3.to_csv(OUT_PHASE3, index=False)

    print("\n" + "=" * 70)
    print(f"PHASE 1 manifest: {OUT_PHASE1} ({len(df1):,} rows)")
    if len(df1):
        print(df1["source"].value_counts().to_string())
        print()
        print(df1["label_name"].value_counts().to_string())
    print("\n" + "=" * 70)
    print(f"PHASE 3 manifest: {OUT_PHASE3} ({len(df3):,} rows)")
    if len(df3):
        print(df3["source"].value_counts().to_string())
        print()
        print(df3["label_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
