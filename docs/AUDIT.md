# Diagnostic Audit (snapshot from 2026-04-28)

This document captures the issues identified in the legacy MNKAI codebase
(now archived under `*_legacy/`) and the corresponding fixes that landed in
the rebuild (Phases 0-1).

## Severity legend
- **Critical** — silently invalidates reported metrics
- **High** — measurable hit to model quality or reproducibility
- **Medium** — code-health / maintainability

## Findings

### CRITICAL: patient-level data leakage in train/val split
- The legacy splitter used `StratifiedShuffleSplit` on the label column
  ([src_legacy/train.py:108](../src_legacy/train.py)) and ignored the
  `group_id` column entirely (`group_column: ""` in
  [configs_legacy/mnkai_isic.yaml](../configs_legacy/mnkai_isic.yaml)).
- The labeled manifest has 4,716 unique `group_id`s for 48,705 images
  (top patient: 1,427 images). Same lesion almost certainly straddled
  train/val.
- **Fix**: [`mnkai.data.splits.make_splits`](../src/mnkai/data/splits.py)
  uses a greedy class-balanced GroupKFold; integration test
  `tests/test_splits.py::test_no_group_leakage_across_folds` enforces
  zero overlap.

### CRITICAL: `cb_focal` was configured but never implemented
- The legacy config requested
  `loss: cb_focal`, `class_balancing: effective_num`. The trainer only
  matched `loss == "focal"` ([src_legacy/train.py:406](../src_legacy/train.py))
  and silently fell through to plain CE.
- **Fix**: [`mnkai.losses.build_loss`](../src/mnkai/losses/focal.py) raises
  on unknown names. `ClassBalancedFocal` is a real implementation of Cui
  et al. 2019.

### HIGH: focal-loss `pt` was the soft-label dot product, not p_y
- Inline `FocalLoss` at [src_legacy/train.py:120](../src_legacy/train.py)
  computes `pt = (true_dist * probs).sum(-1)`. With label smoothing > 0
  this is mathematically wrong: the focal modulator should always come
  from the true-class probability.
- **Fix**: see new [`FocalLoss`](../src/mnkai/losses/focal.py); regression
  test `tests/test_losses.py::test_focal_pt_is_true_class_prob_not_soft_dot`.

### HIGH: hair-removal pipeline corrupted dark-skin samples
- Legacy [src_legacy/hair_removal.py](../src_legacy/hair_removal.py) used
  17×17 black-hat with `threshold=10` on raw grayscale. Threshold 10
  fires on melanin in Fitzpatrick V-VI skin and inpaints legitimate
  pigment.
- **Fix**: [`mnkai.data.transforms.remove_hair_safe`](../src/mnkai/data/transforms.py)
  gates the inpaint mask by an HSV "is-skin" mask AND uses multi-orientation
  black-hat with adaptive thresholding.

### HIGH: no color constancy
- Cross-source domain shift was untreated. Adding Shades-of-Gray
  (Finlayson, p=6) is a near-zero-cost domain normalizer that
  consistently lifts cross-source AUROC by 1-3 points in published
  dermoscopy studies.
- **Fix**: applied as the first step in both train and valid pipelines
  ([transforms.py](../src/mnkai/data/transforms.py)).

### HIGH: no statistical evaluation
- No bootstrapped confidence intervals, no McNemar, no DeLong; the
  legacy `eval/run40/robustness_breakdown.csv` had two rows (sex × {male, female}).
- **Fix**: [`mnkai.evaluation.bootstrap`](../src/mnkai/evaluation/bootstrap.py)
  and [`mnkai.evaluation.subgroups`](../src/mnkai/evaluation/subgroups.py).

### HIGH: no external test domain
- All metrics were on a single 80/20 split of the same manifest.
- **Fix**: split builder takes `--external-source MSK`; trainer never sees
  external rows; `scripts/evaluate.py --split external_test` is the
  external-test entrypoint.

### HIGH: two parallel training stacks
- `src_legacy/train.py` (timm) vs `src_legacy/utils/...` +
  `scripts_legacy/train_isic_stable.py` (multimodal) — drifted, used
  different manifests, different label maps, different eval shapes.
- **Fix**: single trainer in [`mnkai.training.train`](../src/mnkai/training/train.py);
  multimodality is a config flag (`tabular.enabled: true`), not a
  separate pipeline.

### HIGH: GPU mismatch documentation
- The original config targeted an 8 GB card. Actual hardware is
  RTX 4050 Laptop, **6 GB**. New configs are sized for that
  ([configs/convnext_tiny.yaml](../configs/convnext_tiny.yaml)
  uses batch=12 grad_accum=4 BF16 channels_last).

### MEDIUM: contradictory "best" reports
- `reports_legacy/best_metrics.json`: macro-F1=0.11, accuracy=22.8%.
  `eval_legacy/run40/metrics_overall.json`: macro-F1=0.69. The first is
  almost certainly from a collapsed early run; user has confirmed it can
  be deleted. The new pipeline writes `best_metrics.json` only on
  improvement and stamps it with the run's epoch + config snapshot.

### MEDIUM: `cudnn.benchmark`, deterministic seeding, worker generators
- Config requested `cudnn_benchmark: true` but `configure_torch_backends()`
  in the legacy code never applied it. Worker seeding was incomplete
  (no `torch.Generator` passed to the DataLoader).
- **Fix**: [`mnkai.utils.seed`](../src/mnkai/utils/seed.py) handles all of this;
  trainer wires `worker_init_fn=seed_worker, generator=torch_generator(seed)`.

### MEDIUM: 18 MB binary `isic.exe` and a 37 KB `temp.txt` were checked in
- Removed in Phase 0.

## Open issues for later phases

- **Datasets are gone**: the manifest references `D:\AIFINAL\fine_pool\...`
  paths that no longer exist on this machine. Phase 3 will re-download
  ISIC 2019/2020, Fitzpatrick17k, and (if accessible) DermaCon-IN onto
  the T7 SSD at `D:\AIFINAL\`.
- **No FST labels in the current manifest**: the project's headline goal
  (Indian skin tones) requires Fitzpatrick17k or a clinically labeled
  Indian set. Phase 3 will integrate them and rebuild the manifest with
  a `fitzpatrick` column.
- **Calibration set overlap**: the legacy `eval/run40/calibration.json`
  reports `ece_after_evalhalf = 0.0198` with `temperature = 0.7353` —
  needs verification that the half used for temperature fitting was
  truly held out from the half used for ECE measurement. Phase 5 will
  re-run calibration on a clean 50/50 split of the validation fold.
