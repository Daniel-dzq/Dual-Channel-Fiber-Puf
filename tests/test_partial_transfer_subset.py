import tempfile, unittest
from pathlib import Path
import numpy as np
from experiment4_security.ml_attack.track_c_pl_partial_leakage import run_device_source_state
from experiment4_security.ml_attack.partial_leakage_splits import LeakSplit
from experiment4_security.ml_attack.config import ModelsConfig
class TransferSubsetTests(unittest.TestCase):
    def test_transfer_only_keeps_source_genuine_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'vectors').mkdir();rng=np.random.default_rng(8);ids=['C001','C002','C003','C004']
            b={s:{c:rng.normal(size=12).astype('float32') for c in ids} for s in ['M0','M1']}
            for c in ids:np.save(root/'vectors'/f'F01_M0_A_{c}.npy',rng.normal(size=12).astype('float32'))
            args=dict(shared_cache_root=root,all_challenge_ids=ids,challenge_to_bank={c:'B01' for c in ids},challenge_features={c:np.zeros(2) for c in ids},hamming_matrix=None,splits_by_rep={0:{2:LeakSplit(8,2,tuple(ids[:2]),tuple(ids[2:]))}},models_cfg=ModelsConfig(),pca_dimension_max=1,b_vectors_all_states=b,model_keys=('PL0_mean_leaked_response',))
            both=run_device_source_state('F01','M0',states_for_transfer=['M0','M1'],**args)
            target=run_device_source_state('F01','M0',states_for_transfer=['M1'],**args)
            expected=both['transfer_summary'].query("target_state == 'M1'").iloc[0]
            observed=target['transfer_summary'].iloc[0]
            for col in ['median_S_A_partial','median_S_G_hidden','Top1_hidden','RG_A_partial']:
                self.assertEqual(expected[col],observed[col])
if __name__=='__main__':unittest.main()
