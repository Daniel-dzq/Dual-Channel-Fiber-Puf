"""Envelope removal — shared implementation."""

from puf_common.envelope import (
    estimate_speckle_width,
    gaussian_blur,
    highpass_detail,
    local_ratio_detail,
)

__all__ = [
    "gaussian_blur",
    "local_ratio_detail",
    "highpass_detail",
    "estimate_speckle_width",
]
