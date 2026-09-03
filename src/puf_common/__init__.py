"""Shared optical-PUF analysis primitives (Experiment 1 numerical definitions)."""

from puf_common.acf_features import extract_acf_features
from puf_common.ncc import zero_mean_ncc
from puf_common.metrics import auc_roc, d_prime, equal_error_rate, margin, robust_gap
from puf_common.envelope import (
    estimate_speckle_width,
    gaussian_blur,
    highpass_detail,
    local_ratio_detail,
)
from puf_common.intensity_features import extract_intensity_features
from puf_common.psd_features import extract_psd_features
from puf_common.texture_features import extract_texture_features
from puf_common.tqdm_progress import make_video_progress, mute_console_logging, stage_tqdm

__all__ = [
    "zero_mean_ncc",
    "d_prime",
    "equal_error_rate",
    "auc_roc",
    "robust_gap",
    "margin",
    "gaussian_blur",
    "local_ratio_detail",
    "highpass_detail",
    "estimate_speckle_width",
    "extract_acf_features",
    "extract_psd_features",
    "extract_intensity_features",
    "extract_texture_features",
    "stage_tqdm",
    "make_video_progress",
    "mute_console_logging",
]
