"""Calibration sub-package.

Conformal prediction (Angelopoulos & Bates 2021) lives here. The legacy
mnkai.evaluation.calibration module continues to host temperature scaling
and ECE metrics; importing both is fine.
"""
from .conformal import (
    ConformalCalibrator,
    APSScores,
    fit_split_conformal,
    coverage_report,
)

__all__ = [
    "ConformalCalibrator",
    "APSScores",
    "fit_split_conformal",
    "coverage_report",
]
