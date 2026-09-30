import importlib.util
from pathlib import Path
import unittest
import pandas as pd

path = Path(__file__).parents[1] / 'scripts/reproduce_threshold_development.py'
spec = importlib.util.spec_from_file_location('threshold_publication', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ThresholdPublicationTests(unittest.TestCase):
    def test_incomplete_analysis_cannot_pass_the_publication_gate(self):
        table = pd.DataFrame({'device_id_a': ['F01']*848, 'device_id_b': ['F01']*848,
            'group': [m.GENUINE]*112 + [m.INTER_CHALLENGE]*736, 'score': [0.]*848})
        with self.assertRaisesRegex(ValueError, '120 genuine and 840'):
            m.development_operating_point(table)

    def test_duplicate_identities_cannot_make_counts_complete(self):
        table = pd.DataFrame({'device_id_a': ['F01']*960, 'device_id_b': ['F01']*960,
            'state_id_a': ['S0']*960, 'challenge_id_a': ['C01']*960,
            'group': [m.GENUINE]*120 + [m.INTER_CHALLENGE]*840, 'score': [0.]*960})
        with self.assertRaisesRegex(ValueError, 'repeated genuine'):
            m.development_operating_point(table)

if __name__ == '__main__':
    unittest.main()
