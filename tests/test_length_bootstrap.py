import unittest
from itertools import combinations
import numpy as np
import pandas as pd
from experiment00.bootstrap import pair_multiplicities, bootstrap_replicates

class HierarchicalPairBootstrapTests(unittest.TestCase):
    def table(self):
        rows=[]
        for d in ['F01','F02']:
            for c in ['C01','C02']:rows.append(('S_intra',d,c,d,c,.8))
            rows.append(('S_inter_challenge',d,'C01',d,'C02',.1))
        for c in ['C01','C02']:rows.append(('S_inter_device','F01',c,'F02',c,.05))
        t=pd.DataFrame(rows,columns=['score_type','fiber_id','challenge','fiber_id_b','challenge_b','score']);t['length_cm']=9
        return t
    def test_weights_equal_explicit_expansion(self):
        t=self.table();dev=[0,0,1];draws=[[0,0,1],[0,1,1],[0,0,1]]
        w=pair_multiplicities(t,dev,draws,['F01','F02'],['C01','C02'])
        # Genuine F01:C1,C2=3,3; F02=2,1; within=2+2,2; between=3*2,3*1.
        np.testing.assert_array_equal(w,[3,3,4,2,1,2,6,3])
    def test_duplicated_identity_never_becomes_impostor(self):
        t=self.table();w=pair_multiplicities(t,[0,0],[[0,0],[0,0]],['F01','F02'],['C01','C02'])
        self.assertEqual(w[0],4);self.assertEqual(w[t.score_type!='S_intra'].sum(),0)
    def test_incomplete_replicate_not_declared_a_winner(self):
        b=bootstrap_replicates(self.table(),n_iterations=30,seed=4)
        self.assertTrue((~b.complete_replicate).any())
        self.assertTrue(b.loc[~b.complete_replicate,'selected_length_cm'].isna().all())
    def test_duplicate_pair_rejected(self):
        t=self.table();t=pd.concat([t,t.iloc[:1]])
        with self.assertRaises(ValueError):bootstrap_replicates(t,n_iterations=1)
if __name__=='__main__':unittest.main()
