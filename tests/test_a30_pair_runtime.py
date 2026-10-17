import json
from pathlib import Path
import tempfile
import unittest
from reasoning.a30_pair_runtime import valid_checkpoint

class A30RecoveryTests(unittest.TestCase):
    def test_requires_all_rank_optimizer_and_rng_files(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'trainer_state.json').write_text(json.dumps({'global_step':50}))
            q=p/'global_step50';q.mkdir()
            (q/'mp_rank_00_model_states.pt').write_bytes(b'fixture')
            for rank in range(2):
                (q/f'zero_pp_rank_{rank}_mp_rank_00_optim_states.pt').write_bytes(b'fixture')
                (p/f'rng_state_{rank}.pth').write_bytes(b'fixture')
            self.assertTrue(valid_checkpoint(p))
            (p/'rng_state_1.pth').unlink()
            self.assertFalse(valid_checkpoint(p))
            (p/'rng_state_1.pth').write_bytes(b'fixture')
            (q/'zero_pp_rank_1_mp_rank_00_optim_states.pt').unlink()
            self.assertFalse(valid_checkpoint(p))

if __name__=='__main__':unittest.main()
