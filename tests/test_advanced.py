import unittest
import numpy as np
from reasoning.routes import fingerprint,farthest_select,route_select,capacity_mix_select
from reasoning.concise import boxed_tail_chars,structurally_clean
from test_pipeline import example

class AdvancedTests(unittest.TestCase):
    def test_concise_structure_requires_single_final_box(self):
        row={'output':'<think>x</think>work\\n\\boxed{7}'}
        feature={'complete_box':True,'response_tokens':200,'repetition':.01}
        self.assertTrue(structurally_clean(row,feature,512,.1,16))
        self.assertEqual(boxed_tail_chars(row['output']),0)
        self.assertFalse(structurally_clean(
            {'output':row['output']+' extended prose after the final answer'},feature,512,.1,16))
        self.assertFalse(structurally_clean(
            {'output':'<think>x</think>\\boxed{7} and \\boxed{8}'},feature,512,.1,16))

    def test_extending_exclusions_preserves_heldout_membership(self):
        from reasoning.extend_exclusions import extend
        rows=[example(i) for i in range(6)]
        train,val=extend(rows[:4],rows[4:],[rows[1]['instruction'],rows[4]['instruction']])
        self.assertEqual([r['id'] for r in train],[rows[i]['id'] for i in (0,2,3)])
        self.assertEqual([r['id'] for r in val],[rows[5]['id']])
        self.assertFalse({r['id'] for r in train}&{r['id'] for r in rows[4:]})

    def test_route_order_changes_signature(self):
        a=fingerprint('Suppose this equation holds. Verify it. Alternatively try induction.')
        b=fingerprint('Alternatively try induction. Verify it. Suppose this equation holds.')
        self.assertFalse(np.array_equal(a,b))
        self.assertTrue(np.isfinite(fingerprint('')).all())

    def test_farthest_covers_distinct_directions(self):
        x=[[1,0],[.99,0],[-1,0],[0,1]]
        chosen=farthest_select(x,[1,.9,.8,.7],3)
        self.assertEqual(chosen[0],0)
        self.assertIn(2,chosen);self.assertIn(3,chosen)

    def test_route_selection_budget(self):
        rows=[example(i) for i in range(20)]
        fs=[dict(id=r['id'],cluster=i%4,total_tokens=1000,response_tokens=900,complete_box=True,repetition=.01)
            for i,r in enumerate(rows)]
        selected,pool,quotas=route_select(rows,fs,8,42,4096)
        self.assertEqual(len({r['id'] for r in selected}),8)
        self.assertEqual(sum(quotas.values()),8)
        self.assertEqual(selected,route_select(rows,fs,8,42,4096)[0])

    def test_small_budget_can_leave_clusters_unselected(self):
        rows=[example(i) for i in range(20)]
        fs=[dict(id=r['id'],cluster=i%4,total_tokens=1000,response_tokens=900,complete_box=True,repetition=.01)
            for i,r in enumerate(rows)]
        selected,_,quotas=route_select(rows,fs,2,42,4096)
        self.assertEqual(len({r['id'] for r in selected}),2)
        self.assertIn(0,quotas.values())

    def test_capacity_mix_is_distinct_deterministic_and_length_bounded(self):
        rows=[example(i) for i in range(30)]
        fs=[dict(id=r['id'],cluster=i%4,total_tokens=2000 if i<18 else 3600,
                 response_tokens=1700 if i<18 else 3300,complete_box=True,repetition=.01)
            for i,r in enumerate(rows)]
        selected,pool,quotas=capacity_mix_select(rows,fs,20,42,4096,3072,.6)
        self.assertEqual(len(selected),20)
        self.assertEqual(len({r['id'] for r in selected}),20)
        self.assertEqual(quotas['short_size'],12)
        fmap={f['id']:f for f in fs}
        self.assertGreaterEqual(sum(fmap[r['id']]['total_tokens']<=3040 for r in selected),12)
        self.assertEqual(selected,capacity_mix_select(rows,fs,20,42,4096,3072,.6)[0])
        self.assertEqual(len(pool),30)

    def test_lmf_selector_rejects_repetition_and_prefers_pass8(self):
        from reasoning.choose_lmf_control import choose_safe
        base=dict(pass_at_1=.20,pass_at_8=.45,truncated_samples=20,
                  mean_repeated_8gram_fraction=.05)
        rows={
            'early':dict(pass_at_1=.21,pass_at_8=.46,truncated_samples=25,
                         mean_repeated_8gram_fraction=.06),
            'best':dict(pass_at_1=.19,pass_at_8=.48,truncated_samples=30,
                        mean_repeated_8gram_fraction=.07),
            'looping':dict(pass_at_1=.30,pass_at_8=.60,truncated_samples=100,
                           mean_repeated_8gram_fraction=.30),
        }
        self.assertEqual(choose_safe(base,rows),'best')
        self.assertFalse(rows['looping']['passes_termination_screen'])

if __name__=='__main__':unittest.main()

class ObjectiveTests(unittest.TestCase):
    def test_dft_gradient_matches_detached_probability(self):
        import torch
        from reasoning.train_adapter import response_loss
        torch.manual_seed(8)
        x=torch.randn(1,5,7,requires_grad=True)
        labels=torch.tensor([[-100,-100,2,3,4]])
        loss=response_loss(x,labels,'dft',chunk=2)
        actual=torch.autograd.grad(loss,x)[0]
        ref=x.detach().clone().requires_grad_()
        ce=torch.nn.functional.cross_entropy(ref[:,:-1].reshape(-1,7),labels[:,1:].reshape(-1),
            ignore_index=-100,reduction='none')
        valid=labels[:,1:].reshape(-1)!=-100
        expected=torch.autograd.grad((ce[valid]*torch.exp(-ce[valid]).detach()).mean(),ref)[0]
        self.assertTrue(torch.allclose(actual,expected,atol=1e-7))
        self.assertTrue(torch.equal(actual[:,0],torch.zeros_like(actual[:,0])))

    def test_sft_chunked_equals_standard_ce(self):
        import torch
        from reasoning.train_adapter import response_loss
        x=torch.randn(2,6,9,requires_grad=True)
        labels=torch.tensor([[-100,-100,2,3,4,5],[-100,1,2,3,4,-100]])
        actual=response_loss(x,labels,'sft',chunk=2)
        expected=torch.nn.functional.cross_entropy(x[:,:-1].reshape(-1,9),labels[:,1:].reshape(-1),ignore_index=-100)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))

class AnchorTests(unittest.TestCase):
    def test_forward_kl_and_gradient_direction(self):
        import torch
        from reasoning.train_adapter import forward_kl
        p=torch.randn(1,4,5,requires_grad=True)
        ref=torch.randn(1,4,5,requires_grad=True)
        labels=torch.tensor([[-100,-100,2,3]])
        actual=forward_kl(p,ref,labels,chunk=1)
        logp=p[:,1:3].log_softmax(-1)
        logref=ref[:,1:3].detach().log_softmax(-1)
        expected=(logref.exp()*(logref-logp)).sum(-1).mean()
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        actual.backward()
        self.assertIsNone(ref.grad)
        self.assertTrue(torch.equal(p.grad[:,0],torch.zeros_like(p.grad[:,0])))
        self.assertAlmostEqual(float(forward_kl(p.detach(),p.detach(),labels)),0.,places=6)
