import unittest
from experiment4_security.ml_attack.track_a_database_auth import database_authentication_conclusion

class PublicationQualityTests(unittest.TestCase):
    def test_inclusive_robust_margin_boundary(self):
        self.assertTrue(database_authentication_conclusion(.05, .90, .10).endswith('_VALID'))

    def test_nan_eer_cannot_make_a_unit_valid(self):
        self.assertTrue(database_authentication_conclusion(.2, 1., float('nan')).endswith('_FAILED'))

    def test_partial_retains_positive_margin_and_retrieval_requirement(self):
        self.assertTrue(database_authentication_conclusion(.04, .5, .2).endswith('_PARTIAL'))

if __name__ == '__main__':
    unittest.main()
