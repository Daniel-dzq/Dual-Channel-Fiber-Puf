#!/usr/bin/env python3
"""Verify the public recording inventory and checksums without modifying the dataset.

This integrity check is separate from numerical reproduction of the publication.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

EXPECTED = {'formal_mechanical_reconfiguration': 20640, 'threshold_development': 765,
            'fiber_length': 475, 'fixed_state': 150, 'wavelength_pathway_control': 60,
            'macro_pixel_screening': 57}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def member(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f'Invalid release-relative path: {name}')
    return path


def verify(root):
    root = root.resolve()
    errors = []
    with (root / 'MANIFEST.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    counts = collections.Counter(r['dataset'] for r in rows)
    if dict(counts) != EXPECTED:
        errors.append({'quantity': 'recording_counts', 'expected': EXPECTED, 'observed': dict(counts)})
    paths = [r['release_path'] for r in rows]
    if len(paths) != len(set(paths)):
        errors.append({'quantity': 'unique_recording_paths', 'error': 'duplicate manifest paths'})
    for r in rows:
        p = member(root, r['release_path'])
        if not p.is_file() or p.stat().st_size != int(r['file_size']) or digest(p) != r['sha256']:
            errors.append({'quantity': 'recording_integrity', 'path': r['release_path']})
    files = {str(p.relative_to(root)) for p in (root / 'raw_data').rglob('*') if p.is_file()}
    if files != set(paths):
        errors.append({'quantity': 'manifest_coverage', 'unlisted': sorted(files-set(paths)),
                       'missing': sorted(set(paths)-files)})
    listed = set()
    for line in (root / 'CHECKSUMS.sha256').read_text().splitlines():
        if not line.strip():
            continue
        sha, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        p = member(root, name)
        if name in listed or not p.is_file() or digest(p) != sha:
            errors.append({'quantity': 'package_checksum', 'path': name})
        listed.add(name)
    expected_files = {str(p.relative_to(root)) for p in root.rglob('*')
                      if p.is_file() and p.name != 'CHECKSUMS.sha256'}
    if listed != expected_files:
        errors.append({'quantity': 'checksum_coverage', 'unlisted': sorted(expected_files-listed),
                       'missing': sorted(listed-expected_files)})
    return {'recordings': len(rows), 'recording_counts': dict(counts), 'errors': errors,
            'integrity_status': 'PASS' if not errors else 'FAIL',
            'numerical_reproduction': 'Run dataset-specific raw analyses and publication validation separately'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.data_root)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({'integrity_status': 'FAIL', 'error': str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result['integrity_status'] == 'PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
