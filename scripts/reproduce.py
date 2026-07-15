"""Single documented entrypoint for the frozen-probe pipeline (plan + provenance).

This script is the canonical, machine-readable description of *how the headline
numbers are produced* and *in what order*. It does three things — none of which
run training, feature extraction, or any heavy compute:

  * ``plan``     — print the ordered pipeline (each stage's command, seeds, inputs,
                   outputs, and the artifact it refreshes). This is the recipe.
  * ``check``    — verify every declared input exists and QC-validate the cached
                   feature matrices (delegates to ``qc_checks``); report input
                   content hashes. A fast pre-flight that would have caught the
                   all-zero-features and missing-cache failures.
  * ``manifest`` — write a run manifest (git SHA, seeds, config + input hashes,
                   package versions) per stage into ``eval/reproduce/`` so a run
                   is reconstructable after the fact.

It deliberately **does not execute** the pipeline stages. Re-running the science
(especially anything that regenerates ``models/clinical_25class.pkl`` or a metric)
is a controlled operation the operator performs by hand from the printed plan.

Two sub-pipelines are described:
  A. 25-class Indian frozen-probe + trust stack + deployable model.
  B. DDI dark-skin cancer frozen multi-backbone ensemble + bootstrap CIs.

Usage
-----
    python scripts/reproduce.py plan
    python scripts/reproduce.py check
    python scripts/reproduce.py manifest --out eval/reproduce
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Be robust to a legacy (cp1252) console: never crash on a stray non-ASCII glyph.
with contextlib.suppress(Exception):  # pragma: no cover - depends on the host console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Make sibling scripts and src/ importable when run directly.
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


# ============================================================================
# pipeline definition
# ============================================================================
@dataclass(frozen=True)
class Stage:
    """One reproducible pipeline step. Commands are documentation, not executed here."""

    key: str
    title: str
    command: str
    seeds: dict = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)      # files that must pre-exist
    outputs: list[str] = field(default_factory=list)     # artifacts the step writes
    validate_feats: list[str] = field(default_factory=list)  # caches `check` QC-validates
    note: str = ""


# Sub-pipeline A: 25-class Indian frozen-probe -> trust -> deployable model.
PIPELINE_25CLASS: list[Stage] = [
    Stage(
        key="cache_features",
        title="Extract frozen DINOv2@518 features for the 25-class Indian set",
        command=(
            "python scripts/_cache_features.py "
            "--config configs/phase3_indian_25class.yaml "
            "--backbone dinov2 --img-size 518 "
            "--out eval/probe/feats_25class_518.npz"
        ),
        seeds={"note": "extraction is deterministic (frozen backbone, no aug, eval mode)"},
        inputs=["configs/phase3_indian_25class.yaml"],
        outputs=["eval/probe/feats_25class_518.npz"],
        note="HEAVY (GPU forward pass over all images). Reads the image drive. "
             "Skip if the cache already exists and passes `reproduce.py check`.",
    ),
    Stage(
        key="probe_multiclass",
        title="Train the frozen-feature MLP probe (25-class macro-F1)",
        command=(
            "python scripts/_probe_multiclass.py "
            "--feats eval/probe/feats_25class_518.npz --head mlp "
            "--out eval/probe/probe_25class_518_mlp.json"
        ),
        seeds={"numpy": 0, "sklearn_random_state": 0},
        inputs=["eval/probe/feats_25class_518.npz"],
        outputs=["eval/probe/probe_25class_518_mlp.json"],
        validate_feats=["eval/probe/feats_25class_518.npz"],
        note="CPU sklearn on cached features. Fixed seeds required for the "
             "reported 0.573 macro-F1 to reproduce.",
    ),
    Stage(
        key="trust_refresh",
        title="Refresh the trust layer (temperature scaling + RAPS conformal + abstention)",
        command="python scripts/_trust_refresh_25class.py",
        seeds={"numpy": 0},
        inputs=["eval/probe/feats_25class_518.npz"],
        outputs=["eval/probe/trust_25class.json"],
        validate_feats=["eval/probe/feats_25class_518.npz"],
        note="Produces ECE 0.016 (post-temp), conformal set-size, abstention curve.",
    ),
    Stage(
        key="save_deployable",
        title="Fit + serialize the deployable classifier",
        command="python scripts/_save_deployable.py",
        seeds={"numpy": 0, "k_reg": 2, "lambda": 0.05},
        inputs=["eval/probe/feats_25class_518.npz", "eval/probe/trust_25class.json"],
        outputs=["models/clinical_25class.pkl"],
        validate_feats=["eval/probe/feats_25class_518.npz"],
        note="WRITES THE SHIPPED MODEL. Run only as a deliberate, controlled "
             "regeneration — never casually.",
    ),
]

# Sub-pipeline B: DDI dark-skin cancer frozen multi-backbone ensemble + CIs.
PIPELINE_DDI: list[Stage] = [
    Stage(
        key="ddi_ensemble",
        title="Extract + probe 6 frozen backbones on DDI; score-average ensemble",
        command="python scripts/_ddi_ensemble.py",
        seeds={"stratified_kfold_random_state": 0},
        inputs=["eval/probe/ddi_backbones/dinov2_b.npy",
                "eval/probe/ddi_backbones/dinov2_reg.npy",
                "eval/probe/ddi_backbones/dinov2_l.npy",
                "eval/probe/ddi_backbones/eva02_b.npy",
                "eval/probe/ddi_backbones/convnextv2_b.npy",
                "eval/probe/ddi_backbones/siglip_b.npy"],
        outputs=["eval/probe/ddi_ensemble.json"],
        validate_feats=["eval/probe/ddi_backbones/dinov2_b.npy",
                        "eval/probe/ddi_backbones/eva02_b.npy",
                        "eval/probe/ddi_backbones/convnextv2_b.npy",
                        "eval/probe/ddi_backbones/siglip_b.npy"],
        note="Feature extraction is HEAVY + reads the image drive; if the per-"
             "backbone .npy caches under ddi_backbones/ exist, only the CPU "
             "probe/ensemble step re-runs.",
    ),
    Stage(
        key="ddi_ensemble_ci",
        title="Bootstrap 95% CIs + paired-delta-AUROC + multi-seed stability",
        command="python scripts/_ddi_ensemble_ci.py",
        seeds={"point_estimate_seed": 0, "multiseed_stability": "seeds 0..4"},
        inputs=["eval/probe/ddi_backbones/dinov2_b.npy",
                "eval/probe/ddi_backbones/dinov2_reg.npy",
                "eval/probe/ddi_backbones/dinov2_l.npy",
                "eval/probe/ddi_backbones/eva02_b.npy",
                "eval/probe/ddi_backbones/convnextv2_b.npy",
                "eval/probe/ddi_backbones/siglip_b.npy"],
        outputs=["eval/probe/ddi_ensemble_ci.json"],
        validate_feats=["eval/probe/ddi_backbones/dinov2_b.npy"],
        note="Pure CPU on the cached .npy features. Produces the AUROC CIs and "
             "the ΔAUROC significance test.",
    ),
]

SUBPIPELINES: dict[str, list[Stage]] = {
    "25class": PIPELINE_25CLASS,
    "ddi": PIPELINE_DDI,
}


# ============================================================================
# commands
# ============================================================================
def _iter_stages(which: str):
    if which == "all":
        for name, stages in SUBPIPELINES.items():
            for st in stages:
                yield name, st
    else:
        for st in SUBPIPELINES[which]:
            yield which, st


def cmd_plan(args: argparse.Namespace) -> int:
    print("=" * 78)
    print("MNKAI-III frozen-probe reproduction plan (documentation only — nothing runs)")
    print("=" * 78)
    for name, stages in SUBPIPELINES.items():
        if args.which not in ("all", name):
            continue
        print(f"\n### Sub-pipeline: {name}")
        for i, st in enumerate(stages, 1):
            print(f"\n  [{i}] {st.key} — {st.title}")
            print(f"      seeds : {st.seeds}")
            if st.inputs:
                print(f"      in    : {', '.join(st.inputs)}")
            if st.outputs:
                print(f"      out   : {', '.join(st.outputs)}")
            print(f"      $ {st.command}")
            if st.note:
                print(f"      note  : {st.note}")
    print("\n" + "-" * 78)
    print("Order matters: A) cache_features -> probe_multiclass -> trust_refresh -> "
          "save_deployable;")
    print("               B) ddi_ensemble -> ddi_ensemble_ci.")
    print("Run `python scripts/reproduce.py check` first to validate inputs.")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    try:
        import qc_checks
    except Exception as e:  # pragma: no cover
        print(f"FAIL: cannot import qc_checks: {e}", file=sys.stderr)
        return 2

    ok = True
    seen_feats: set[str] = set()
    print("[check] verifying declared inputs exist + QC-validating feature caches\n")
    for name, st in _iter_stages(args.which):
        for rel in st.inputs:
            p = ROOT / rel
            status = "ok" if p.is_file() else "MISSING"
            if not p.is_file():
                ok = False
            print(f"  [{name}/{st.key}] input {rel}: {status}")
        for rel in st.validate_feats:
            if rel in seen_feats:
                continue
            seen_feats.add(rel)
            p = ROOT / rel
            if not p.is_file():
                print(f"  [{name}/{st.key}] validate {rel}: SKIP (missing)")
                continue
            try:
                rep = qc_checks.validate_feature_cache(qc_checks.load_features(p))
                verdict = "PASS" if rep.ok else "FAIL"
                if not rep.ok:
                    ok = False
                print(f"  [{name}/{st.key}] validate {rel}: {verdict} "
                      f"({rep.n_rows}x{rep.n_dims})")
                for prob in rep.problems:
                    print(f"        - {prob}")
            except Exception as e:
                ok = False
                print(f"  [{name}/{st.key}] validate {rel}: ERROR {e}")
    print(f"\n=> {'ALL INPUTS OK' if ok else 'PROBLEMS FOUND'}")
    return 0 if ok else 1


def cmd_manifest(args: argparse.Namespace) -> int:
    try:
        from mnkai.utils.run_manifest import write_run_manifest
    except Exception as e:  # pragma: no cover
        print(f"FAIL: cannot import mnkai.utils.run_manifest "
              f"(run with PYTHONPATH=src): {e}", file=sys.stderr)
        return 2

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, st in _iter_stages(args.which):
        cfg_path = None
        for rel in st.inputs:
            if rel.endswith((".yaml", ".yml")):
                cfg_path = str(ROOT / rel)
                break
        dest = out_dir / f"{name}__{st.key}.manifest.json"
        write_run_manifest(
            dest,
            stage=f"{name}/{st.key}",
            seeds=st.seeds,
            config_path=cfg_path,
            inputs=[str(ROOT / r) for r in st.inputs],
            outputs=[str(ROOT / r) for r in st.outputs],
            extra={"command": st.command, "title": st.title},
        )
        written.append(str(dest.relative_to(ROOT)))
        print(f"  wrote {dest.relative_to(ROOT)}")
    print(json.dumps({"manifests": written}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="reproduce",
        description="Document + pre-flight the frozen-probe pipeline. Never executes "
                    "training / feature-extraction / model regeneration.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = dict(choices=["all", "25class", "ddi"], default="all")

    p = sub.add_parser("plan", help="print the ordered pipeline (recipe)")
    p.add_argument("--which", **common)
    p.set_defaults(func=cmd_plan)

    c = sub.add_parser("check", help="verify inputs exist + QC-validate feature caches")
    c.add_argument("--which", **common)
    c.set_defaults(func=cmd_check)

    m = sub.add_parser("manifest", help="write per-stage run manifests")
    m.add_argument("--which", **common)
    m.add_argument("--out", default="eval/reproduce", help="output dir for manifests")
    m.set_defaults(func=cmd_manifest)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
