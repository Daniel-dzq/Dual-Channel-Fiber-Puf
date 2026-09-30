import unittest
import numpy as np
from puf_common.ncc import zero_mean_ncc
from experiment4_security.ml_attack.batch_eval import pairwise_zero_mean_ncc_blocked


class BlockedNCCTests(unittest.TestCase):
    def test_long_offset_vectors_match_scalar_and_chunk_sizes(self):
        rng = np.random.default_rng(173)
        a = (1000 + rng.normal(size=(3, 167619))).astype(np.float32)
        b = np.stack([a[1], 2000 - a[0], np.full(a.shape[1], 7)]).astype(np.float32)
        expected = np.array([[zero_mean_ncc(x, y) for y in b] for x in a])
        for size in (8191, 65536, 200000):
            observed = pairwise_zero_mean_ncc_blocked(a, b, block_size=size)
            np.testing.assert_allclose(observed, expected, atol=2e-14, rtol=2e-14)
            self.assertEqual(np.argmax(observed[1]), 0)
            np.testing.assert_array_equal(observed[:, 2], 0)


if __name__ == '__main__':
    unittest.main()
