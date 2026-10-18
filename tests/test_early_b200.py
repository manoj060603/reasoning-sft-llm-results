import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from reasoning.early_b200 import checkpoint_valid, register, verify_final


class EarlyB200Checks(unittest.TestCase):
    def test_budget_retains_spent_a30_hours_and_eval_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg={'output_dir':tmp}; receipt={'prior_charged_gpu_hours':8}
            for job in ('one','two','three'):
                with patch.dict(os.environ,SLURM_GPUS_ON_NODE='1',SLURM_JOB_ID=job):
                    register(cfg,receipt)
                    register(cfg,receipt)
            a=json.load(open(Path(tmp)/'allocations.json'))
            self.assertEqual(a['training_job_ids'],['one','two','three'])
            with patch.dict(os.environ,SLURM_GPUS_ON_NODE='1',SLURM_JOB_ID='four'):
                with self.assertRaises(AssertionError): register(cfg,receipt)
            self.assertEqual(json.load(open(Path(tmp)/'allocations.json')),a)

    def test_recovery_rejects_missing_native_state_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'trainer_state.json').write_text(json.dumps({'global_step':50}))
            self.assertFalse(checkpoint_valid(p))
            for n in ('optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors'):
                (p/n).write_bytes(b'x')
            self.assertTrue(checkpoint_valid(p))
            (p/'rng_state.pth').unlink()
            self.assertFalse(checkpoint_valid(p))

    def test_final_requires_exact_endpoint_and_all_finite_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            state={'global_step':5625,'epoch':3.,'log_history':[
                {'step':i,'loss':.5,'grad_norm':1.} for i in range(1,5626)]}
            (p/'trainer_state.json').write_text(json.dumps(state))
            for n in ('config.json','tokenizer_config.json'): (p/n).write_text('{}')
            save_file({'fixture':torch.tensor([1.,2.])},str(p/'model.safetensors'))
            verify_final(p)
            save_file({'fixture':torch.tensor([float('nan')])},str(p/'model.safetensors'))
            with self.assertRaises(AssertionError): verify_final(p)
            state['global_step']=200
            (p/'trainer_state.json').write_text(json.dumps(state))
            with self.assertRaises(AssertionError): verify_final(p)


if __name__=='__main__': unittest.main()
