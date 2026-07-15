# MNKAI-III — Data Inventory (2026-07-05)

**Headline: we already have ~370k images — we're ALREADY inside your 300k–1M target.** Volume is not the
bottleneck. The bottleneck for the Indian / dark-skin metrics is *targeted* dark-skin data (~15k, scarce),
not total count.

## Physical images on disk: **369,905**
| Store | Count | Modality | Skin tone |
|---|---|---|---|
| ISIC-2024 | 157,619 | dermoscopy | light-skewed |
| Mpox / MSLD | 40,820 | clinical | diverse (viral only) |
| ISIC-2019 | 25,331 | dermoscopy | light |
| ISIC (shards/dir) | ~30,000 | dermoscopy | light |
| DermNet | 19,559 | clinical | mixed (scraped) |
| BCN20000 | 18,946 | dermoscopy | light |
| SkinDisease (Kaggle) | 15,444 | clinical | mixed |
| **Fitzpatrick17k (full)** | 16,577 | clinical | **has FST I–VI (2,168 dark)** |
| SCIN | 10,379 | clinical | diverse (Monk-labeled) |
| **DermaCon-IN** | 5,450 | clinical | **Indian (darker)** |
| Skin-types / PAD / Derm7pt / DDI / misc | ~10,000 | mixed | mixed (**DDI 207 dark**) |

## Labeled vs unlabeled
- **Labeled + de-duplicated (`master.csv`): ~266k** (ready for supervised heads now).
- **Total incl. extra/unlabeled on disk: ~370k** (usable for SSL).

## The split that actually matters
| Bucket | ~Count | Helps which metrics |
|---|---|---|
| **Dermoscopy (light)** | ~230k | 9-class + cancer/melanoma (dermoscopy) |
| **Clinical photo** | ~130k | 25-class Indian + clinical cancer |
| **Dark-skin / Indian (FST IV–VI, confirmable)** | **~15k** (DermaCon 5.5k + Fitz-dark ~5k + DDI ~0.4k + SCIN-dark) | Indian macro-F1 + fairness + DDI |

## What this means for the plan
1. **You don't need to "go get" 300k–1M — you have ~370k.** Reaching 1M would just add more *light-skin dermoscopy*
   (the ISIC archive has ~500k more) → helps cancer/9-class only, ~nothing for Indian/dark-skin.
2. **Lever A (now, days, no SSL, no restart):** run the **frozen-DINOv2@518 + probe** pattern on the full ~266k
   labeled corpus for the 9-class, 25-class, and cancer heads → direct metric gains from using *all* the data.
3. **Optional SSL (few days on A100):** continue-pretrain on the ~370k (labeled+unlabeled) → backbone bump.
4. **For Indian/dark-skin specifically:** the lever is **targeted data** (Stanford AIMI: MRA-MIDAS/DDI-2/ASAN;
   more Indian clinical), not volume — because dark-skin cancer imagery barely exists (DDI is essentially it).

**Bottom line:** volume is solved. The month is best spent on (A) using the 266k we have via frozen-probe,
(B) an optional short SSL run, and (C) chasing the scarce *targeted* dark-skin data — not on padding to 1M.
