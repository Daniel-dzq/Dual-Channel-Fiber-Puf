"""Content provenance for full-recording green detail-response caches."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def response_provenance(video: Path, mask: np.ndarray, preprocessing) -> dict:
    import cv2
    mask = np.ascontiguousarray(mask, dtype=np.bool_)
    return {
        'schema': 1,
        'video_sha256': sha256_file(video),
        'valid_pixel_mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
        'valid_pixel_mask_shape': list(mask.shape),
        'valid_pixel_count': int(mask.sum()),
        'optical_channel': 'green_bgr1',
        'temporal_aggregation': 'pixelwise_median_all_decodable_frames',
        'head_trim_seconds': 0,
        'tail_trim_seconds': 0,
        'dark_mode': preprocessing.dark_mode,
        'envelope_sigma_pixels': float(preprocessing.envelope_sigma),
        'envelope_epsilon': float(preprocessing.envelope_epsilon),
        'envelope_border': 'BORDER_REFLECT101',
        'response_dtype': 'float32',
        'numpy_version': np.__version__,
        'opencv_version': cv2.__version__,
        'envelope_source_sha256': sha256_file(Path(__file__).parents[2] / 'puf_common' / 'envelope.py'),
        'channel_source_sha256': sha256_file(Path(__file__).parents[2] / 'puf_common' / 'channels.py'),
        'feature_source_sha256': sha256_file(Path(__file__).parents[2] / 'puf_common' / 'features.py'),
        'preprocessing_source_sha256': sha256_file(
            Path(__file__).parents[1] / 'ml_attack' / 'video_preprocessing.py'),
    }


def provenance_path(vector: Path) -> Path:
    return Path(vector).with_suffix('.provenance.json')


def cache_matches(vector: Path, expected: dict) -> bool:
    try:
        saved = json.loads(provenance_path(vector).read_text())
        if saved['input'] != expected or saved['response_sha256'] != sha256_file(vector):
            return False
        array = np.load(vector, allow_pickle=False, mmap_mode='r')
        return (array.shape == (expected['valid_pixel_count'],)
                and array.dtype == np.float32 and bool(np.isfinite(array).all()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def write_provenance(vector: Path, expected: dict) -> None:
    payload = {'input': expected, 'response_sha256': sha256_file(vector)}
    target = provenance_path(vector)
    with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, delete=False) as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        temporary = Path(stream.name)
    os.replace(temporary, target)
