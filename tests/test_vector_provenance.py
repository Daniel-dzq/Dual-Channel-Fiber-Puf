import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import os
import numpy as np

MODULE = Path(__file__).parents[1] / 'src/experiment4_security/identity_credential/vector_provenance.py'
spec = importlib.util.spec_from_file_location('vector_provenance', MODULE)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class CacheProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / 'recording.mp4'
        self.video.write_bytes(b'acquisition_A')
        self.vector = self.root / 'response.npy'
        self.mask = np.array([[True, False], [False, True]])
        self.preprocessing = SimpleNamespace(dark_mode='none', envelope_sigma=42, envelope_epsilon=1)
        self.expected = p.response_provenance(self.video, self.mask, self.preprocessing)
        np.save(self.vector, np.array([0.1, 0.2], dtype=np.float32))

    def test_untracked_array_is_not_reused(self):
        self.assertFalse(p.cache_matches(self.vector, self.expected))

    def test_matching_input_and_response_are_reused(self):
        p.write_provenance(self.vector, self.expected)
        self.assertTrue(p.cache_matches(self.vector, self.expected))

    def test_replacement_with_same_size_and_timestamp_is_detected(self):
        p.write_provenance(self.vector, self.expected)
        st = self.video.stat()
        self.video.write_bytes(b'acquisition_B')
        os.utime(self.video, ns=(st.st_atime_ns, st.st_mtime_ns))
        changed = p.response_provenance(self.video, self.mask, self.preprocessing)
        self.assertFalse(p.cache_matches(self.vector, changed))

    def test_mask_coordinates_matter_at_equal_pixel_count(self):
        p.write_provenance(self.vector, self.expected)
        changed = p.response_provenance(self.video, ~self.mask, self.preprocessing)
        self.assertFalse(p.cache_matches(self.vector, changed))

    def test_protocol_change_is_detected(self):
        p.write_provenance(self.vector, self.expected)
        self.preprocessing.envelope_sigma = 20
        changed = p.response_provenance(self.video, self.mask, self.preprocessing)
        self.assertFalse(p.cache_matches(self.vector, changed))

    def test_response_tampering_is_detected(self):
        p.write_provenance(self.vector, self.expected)
        np.save(self.vector, np.array([0.1, 0.3], dtype=np.float32))
        self.assertFalse(p.cache_matches(self.vector, self.expected))

    def test_nonfinite_response_is_rejected_even_with_matching_hash(self):
        np.save(self.vector, np.array([0.1, np.nan], dtype=np.float32))
        p.write_provenance(self.vector, self.expected)
        self.assertFalse(p.cache_matches(self.vector, self.expected))

if __name__ == '__main__':
    unittest.main()
