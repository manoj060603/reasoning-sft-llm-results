import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from reasoning.fast_2b200 import checkpoint_valid, register


class FastTwoB200Checks(unittest.TestCase):
    def test_budget_carries_spending_and_reserves_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = {'output_dir':tmp}; r = {'prior_charged_gpu_hours':8}
            for job in ('first', 'second'):
                with patch.dict(os.environ, SLURM_GPUS_ON_NODE='2', SLURM_JOB_ID=job):
                    register(c, r); register(c, r)
            p = Path(tmp)/'allocations.json'; before = p.read_text()
            self.assertEqual(json.loads(before)['training_job_ids'], ['first','second'])
            with patch.dict(os.environ, SLURM_GPUS_ON_NODE='2', SLURM_JOB_ID='third'):
                with self.assertRaises(AssertionError): register(c, r)
            self.assertEqual(p.read_text(), before)

    def test_recovery_requires_both_rng_ranks_and_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'trainer_state.json').write_text(json.dumps({'global_step':50}))
            for name in ('optimizer.pt','scheduler.pt','rng_state_0.pth','model.safetensors'):
                (p/name).write_bytes(b'fixture')
            self.assertFalse(checkpoint_valid(p))
            (p/'rng_state_1.pth').write_bytes(b'fixture')
            self.assertTrue(checkpoint_valid(p))
            (p/'optimizer.pt').write_bytes(b'')
            self.assertFalse(checkpoint_valid(p))

    def test_register_resumes_highest_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            for step in (50,100):
                p=run/f'checkpoint-{step}'; p.mkdir()
                (p/'trainer_state.json').write_text(json.dumps({'global_step':step}))
                for name in ('optimizer.pt','scheduler.pt','rng_state_0.pth','rng_state_1.pth','model.safetensors'):
                    (p/name).write_bytes(b'fixture')
            (run/'checkpoint-100/rng_state_1.pth').unlink()
            c={'output_dir':tmp}
            with patch.dict(os.environ, SLURM_GPUS_ON_NODE='2', SLURM_JOB_ID='first'):
                register(c, {'prior_charged_gpu_hours':8})
            self.assertEqual(c['resume_from_checkpoint'],str(run/'checkpoint-50'))


if __name__ == '__main__': unittest.main()
