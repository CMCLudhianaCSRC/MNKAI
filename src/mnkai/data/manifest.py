"""Manifest construction + label normalization.

The legacy repo had label normalization scattered across train.py, train_isic.py,
and a YAML alias table that was loaded redundantly. This module owns it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger


# ------------------------------------------------------------------
# Phase-1 base taxonomy: ISIC 8-class + 'normal' (no-lesion) = 9 classes.
# This is what the production model ships with for general dermoscopy /
# clinical photo screening.
# ------------------------------------------------------------------
PHASE1_CLASSES = ["mel", "nv", "bcc", "scc", "df", "bkl", "ak", "vasc", "normal"]

# Legacy alias — same list, kept for backwards compatibility.
DEFAULT_CLASSES = list(PHASE1_CLASSES)

# ------------------------------------------------------------------
# Phase-3 expanded taxonomy: base 9 + 11 Indian-relevant conditions.
# Used only for the post-base Indian fine-tune stage on DermaCon-IN +
# DermNet + skindiseasedataset. NEVER mix into Phase 1 base training.
# ------------------------------------------------------------------
PHASE3_EXTRA_CLASSES = [
    # Original 11 Indian-relevant classes
    "vitiligo",
    "tinea",                  # all fungal skin: cruris/corporis/faciei/pedis/versicolor/candidiasis/onychomycosis
    "acne",
    "psoriasis",
    "eczema",                 # incl. atopic dermatitis, seborrheic
    "melasma",                # incl. post-inflammatory hyperpigmentation
    "lichen_planus",
    "pemphigus",              # autoimmune bullous (all bullous diseases)
    "steroid_damage",         # topical-steroid-induced (very Indian)
    "scabies",                # parasitic + insect bites
    "contact_dermatitis",
    # Expansion (Apr 2026): 7 new catch-all classes that recover ~10k previously-skipped images
    "rosacea",                # facial flushing, papules
    "bacterial_infection",    # cellulitis, impetigo, pyoderma, etc.
    "viral_infection",        # warts, molluscum, herpes, HPV, also mpox/chickenpox/cowpox/HFMD/measles
    "drug_eruption",          # exanthems, drug reactions
    "autoimmune_systemic",    # lupus, vasculitis, connective tissue diseases
    "urticaria",              # hives, pruritic urticarial papules
    "alopecia",               # hair loss, alopecia areata, etc.
]
PHASE3_CLASSES = PHASE1_CLASSES + PHASE3_EXTRA_CLASSES   # 27 classes total

DEFAULT_ALIASES: dict[str, list[str]] = {
    "mel": ["mel", "melanoma", "melanoma in situ", "melanoma invasive", "melanoma, nos",
            "melanoma metastasis", "superficial spreading melanoma", "nodular melanoma",
            "lentigo maligna", "lentigo malignant melanoma", "acral lentiginous melanoma",
            "mucosal melanoma", "mucosal melanotic macule", "mm", "melanocytic malignant",
            "atypical melanocytic neoplasm"],
    "nv":  ["nv", "nevus", "naevus", "melanocytic nevus", "common nevus", "atypical nevus",
            "dysplastic nevus", "compound nevus", "intradermal nevus", "junctional nevus",
            "melanocytic nevi"],
    "bcc": ["bcc", "basal cell carcinoma", "basal cell carcinoma, nos", "basalioma"],
    "scc": ["scc", "squamous cell carcinoma", "squamous cell carcinoma, nos",
            "squamous cell carcinoma, invasive", "squamous cell carcinoma in situ",
            "keratinocyte carcinoma", "squamous carcinoma", "akiec_scc"],
    "df":  ["df", "dermatofibroma", "histiocytoma"],
    "bkl": ["bkl", "benign keratosis", "seborrheic keratosis", "pigmented benign keratosis",
            "solar lentigo", "lentigo", "lentigo nos", "lentigo simplex",
            "lichen planus-like keratosis", "lichen planus like keratosis",
            "lplk", "sk", "sebk"],
    "ak":  ["ak", "actinic keratosis", "akiec", "actinic keratoses",
            "solar or actinic keratosis", "keratoacanthoma"],
    "vasc": ["vasc", "vascular", "angioma", "hemangioma", "pyogenic granuloma",
             "venous lake", "vascular lesion", "angiokeratoma",
             "arteriovenous malformation", "glomus tumor"],
    # Phase 1 — normal/healthy skin (no lesion present).
    "normal": ["normal", "healthy", "healthy skin", "no lesion", "control",
               "negative", "no_disease", "clear skin", "skin", "normalskin",
               "background"],
    # Phase 3 — Indian-relevant conditions.
    "vitiligo": ["vitiligo", "leucoderma"],
    "tinea":    ["tinea", "tinea cruris", "tinea corporis", "tinea faciei",
                 "tinea pedis", "tinea versicolor", "ringworm",
                 "steroid modified tinea", "athletes foot",
                 "fungal infection (tinea)",
                 # consolidated: candidiasis, onychomycosis are also fungal
                 "candidiasis", "candidal intertrigo", "candidal balanoposthitis",
                 "candida", "onychomycosis", "nail fungus",
                 "tinea ringworm candidiasis and other fungal infections"],
    "acne":     ["acne", "acne vulgaris", "acne conglobata", "acneiform",
                 "acne and rosacea photos"],   # DermNet folder is acne+rosacea combined
    "psoriasis":["psoriasis", "inverse psoriasis", "guttate psoriasis",
                 "plaque psoriasis", "psoriasis vulgaris",
                 "psoriasis pictures lichen planus and related diseases"],
    "eczema":   ["eczema", "atopic dermatitis", "atopic eczema", "atopic dermatitis photos",
                 "seborrheic dermatitis", "infected eczema",
                 "scrotal dermatitis", "xerosis", "eczema photos"],
    "melasma":  ["melasma", "post-inflammatory hyperpigmentation",
                 "post inflammatory hyperpigmentation", "pih", "chloasma",
                 "light diseases and disorders of pigmentation"],
    "lichen_planus": ["lichen planus", "lichen"],
    "pemphigus": ["pemphigus", "pemphigus vulgaris", "bullous pemphigoid",
                  "autoimmune bullous", "bullous", "bullous disease photos",
                  "bullous diseases"],
    "steroid_damage": ["steroid damage", "steroid modified", "topical steroid damaged face",
                       "topical steroid damaged skin", "tsdf"],
    "scabies":  ["scabies", "mites", "scabies lyme disease and other infestations and bites",
                 "infestations bites", "infestations_bites", "infestations and bites",
                 "lyme disease", "insect bites"],
    "contact_dermatitis": ["contact dermatitis", "allergic contact dermatitis",
                           "irritant contact dermatitis",
                           "poison ivy photos and other contact dermatitis"],
    # ── Expansion classes (Apr 2026) ────────────────────────────────────────
    "rosacea":  ["rosacea"],
    "bacterial_infection": [
        "bacterial infection", "bacterial",
        "cellulitis", "impetigo", "pyoderma", "folliculitis",
        "cellulitis impetigo and other bacterial infections",
        "scc_bacterial", "skin abscess",
    ],
    "viral_infection": [
        "viral infection", "viral",
        "warts", "verruca", "wart", "molluscum", "molluscum contagiosum",
        "herpes", "hpv", "herpes simplex", "herpes zoster", "shingles",
        "warts molluscum and other viral infections",
        "herpes hpv and other stds photos",
        # Mpox-MSLD non-healthy folders all collapse to viral_infection
        "monkeypox", "mpox", "chickenpox", "cowpox",
        "hfmd", "hand foot and mouth disease", "measles",
        # NOTE: the bare alias "exanthema" was previously ALSO listed here, but it
        # is canonically owned by `drug_eruption` (DermNet's "exanthems and drug
        # eruptions" folder). It was assigned to drug_eruption at runtime anyway
        # (later dict key wins), so removing it here changes no mapping — it only
        # removes the silent collision. Viral exanthems are still covered by the
        # specific disease aliases above (mpox/chickenpox/measles/HFMD).
    ],
    "drug_eruption": [
        "drug eruption", "drug reaction", "drugeruption",
        "exanthems and drug eruptions", "exanthem", "exanthema",
        "fixed drug eruption",
    ],
    "autoimmune_systemic": [
        "lupus", "lupus erythematosus", "discoid lupus",
        "lupus and other connective tissue diseases",
        "vasculitis", "vasculitis photos", "leukocytoclastic vasculitis",
        "connective tissue", "connective tissue disease",
        "systemic disease", "systemic lupus",
        "dermatomyositis", "scleroderma",
    ],
    "urticaria": [
        "urticaria", "urticaria hives", "hives",
        "pruritic urticarial papules", "pruritic urticarial papules and plaques",
        "puppp",
    ],
    "alopecia": [
        "alopecia", "alopecia areata", "androgenetic alopecia",
        "hair loss", "hair loss photos alopecia and other hair diseases",
        "telogen effluvium", "scarring alopecia",
    ],
}

DEFAULT_IGNORE = ["ns", "none", "undetermined", "other", "unknown"]
# Note: "normal" used to be in this ignore list (when the model was 8-class
# tumor-only). It's now the 9th class for the deployable Phase 1 model;
# DermaCon-IN's "no lesion present" cases also map to it via "no lesion" alias.


@dataclass
class LabelNormalizer:
    classes: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    aliases: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_ALIASES.items()})
    ignore: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE))

    def __post_init__(self):
        self.classes = [c.strip().lower() for c in self.classes]
        self._table: dict[str, str] = {c: c for c in self.classes}
        # Track which canonical class(es) claim each alias so we can warn on any
        # alias claimed by >1 canonical class (a silent mislabeling bug: the
        # later dict key wins, quietly overriding the earlier one).
        alias_claims: dict[str, list[str]] = {}
        for canon, alist in self.aliases.items():
            canon_l = canon.strip().lower()
            if canon_l not in self.classes:
                logger.warning(f"[labels] alias key '{canon}' not in classes; ignoring")
                continue
            for a in alist:
                a_l = a.strip().lower()
                claimants = alias_claims.setdefault(a_l, [])
                if canon_l not in claimants:
                    claimants.append(canon_l)
                self._table[a_l] = canon_l
        # Fail-loud-ish collision guard: warn (do not raise) so existing data
        # keeps loading, but the ambiguity is surfaced. `map()` resolves to the
        # last-claiming canonical class (dict insertion order), reported here.
        for alias, claimants in alias_claims.items():
            if len(claimants) > 1:
                logger.warning(
                    f"[labels] alias '{alias}' is claimed by {len(claimants)} canonical "
                    f"classes {claimants}; resolving to '{self._table[alias]}' (last wins). "
                    f"Assign it to exactly one class to remove this ambiguity."
                )
        self._alias_claims = alias_claims
        self._ignore = {x.strip().lower() for x in self.ignore}

    def map(self, raw: str) -> str | None:
        """Return canonical class label, or None if `raw` should be dropped."""
        if raw is None:
            return None
        s = str(raw).strip().lower()
        if s in self._ignore:
            return None
        if s in self._table:
            return self._table[s]
        # try with normalized whitespace / separators
        s2 = " ".join(s.replace("-", " ").replace("_", " ").replace("/", " ").replace("\\", " ").split())
        return self._table.get(s2)


def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Find canonical column names; returns a dict path/label/group/source/sex/age/site/fitzpatrick."""
    lc = {c.lower(): c for c in df.columns}
    pick = lambda *names: next((lc[n] for n in names if n in lc), None)
    return {
        "filepath": pick("filepath", "image", "image_path", "filename"),
        "label_name": pick("label_name", "label", "diagnosis", "dx"),
        "group_id": pick("group_id", "lesion_id", "patient_id"),
        "source": pick("source", "dataset", "attribution"),
        "sex": pick("sex", "gender"),
        "age_approx": pick("age_approx", "age"),
        "anatom_site_general": pick("anatom_site_general", "anatom_site", "site"),
        "fitzpatrick": pick("fitzpatrick", "fst", "skin_tone"),
    }


def normalize_manifest(df: pd.DataFrame, normalizer: LabelNormalizer | None = None) -> pd.DataFrame:
    """Return a copy of `df` with canonical columns and labels.

    Drops rows whose label maps to None (ignored) or that are missing a path.
    """
    norm = normalizer or LabelNormalizer()
    cols = detect_columns(df)
    if cols["filepath"] is None or cols["label_name"] is None:
        raise ValueError(f"manifest must have a path and a label column; got {list(df.columns)}")

    rename = {v: k for k, v in cols.items() if v is not None and v != k}
    out = df.rename(columns=rename).copy()

    # drop missing paths
    out = out.dropna(subset=["filepath"])

    # normalize labels
    raw = out["label_name"].astype(str).tolist()
    mapped = [norm.map(r) for r in raw]
    out["label_name"] = mapped
    n_drop = sum(1 for x in mapped if x is None)
    out = out.dropna(subset=["label_name"]).reset_index(drop=True)
    logger.info(f"[manifest] normalized: kept {len(out)} rows, dropped {n_drop}")

    # ensure a group_id column exists (fallback: each image its own group, which is
    # weak but correct — never silently treats different images as the same lesion).
    if "group_id" not in out.columns or out["group_id"].isna().any():
        if "group_id" not in out.columns:
            out["group_id"] = ["row_" + str(i) for i in range(len(out))]
        else:
            mask = out["group_id"].isna()
            out.loc[mask, "group_id"] = ["row_" + str(i) for i in out.index[mask]]
        logger.warning("[manifest] some rows had no group_id; using row_* fallback (no grouping)")

    return out


def class_counts(df: pd.DataFrame, classes: list[str]) -> list[int]:
    counts = df["label_name"].value_counts()
    return [int(counts.get(c, 0)) for c in classes]


__all__ = [
    "LabelNormalizer",
    "DEFAULT_CLASSES",
    "PHASE1_CLASSES",
    "PHASE3_CLASSES",
    "PHASE3_EXTRA_CLASSES",
    "DEFAULT_ALIASES",
    "DEFAULT_IGNORE",
    "detect_columns",
    "normalize_manifest",
    "class_counts",
]
