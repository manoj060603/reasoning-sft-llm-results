import unittest
import torch
from reasoning.memory_ce import causal_ce

class MemoryCETests(unittest.TestCase):
    def test_loss_and_gradient_match_pinned_transformers(self):
        from transformers.loss.loss_utils import ForCausalLMLoss
        for dtype in (torch.float32,torch.bfloat16):
            for explicit in (False,True):
                for shift in (False,True):
                    torch.manual_seed(42)
                    a=torch.randn(2,259,37,dtype=dtype,requires_grad=True)
                    b=a.detach().clone().requires_grad_()
                    labels=torch.randint(0,37,(2,259));labels[:,:17]=-100
                    labels[0,200:]=-100
                    kwargs={'num_items_in_batch':torch.tensor(600)} if explicit else {}
                    if shift:kwargs['shift_labels']=labels.clone()
                    x=ForCausalLMLoss(a,labels,37,**kwargs)
                    y=causal_ce(b,labels,37,**kwargs)
                    x.backward();y.backward()
                    torch.testing.assert_close(x,y,rtol=2e-6,atol=1e-6)
                    torch.testing.assert_close(a.grad,b.grad,rtol=.015 if dtype==torch.bfloat16 else 2e-5,atol=1e-5 if dtype==torch.bfloat16 else 1e-8)

    def test_global_token_denominator_shards(self):
        torch.manual_seed(7)
        a=torch.randn(4,137,19,requires_grad=True)
        labels=torch.randint(0,19,(4,137));labels[0,:100]=-100
        shifted=torch.nn.functional.pad(labels,(0,1),value=-100)[...,1:]
        count=(shifted!=-100).sum()
        full=causal_ce(a,labels,19,num_items_in_batch=count)
        pieces=sum(causal_ce(a[i:i+1],labels[i:i+1],19,num_items_in_batch=count) for i in range(4))
        torch.testing.assert_close(full,pieces)
        g=torch.autograd.grad(full,a,retain_graph=True)[0]
        h=torch.autograd.grad(pieces,a)[0]
        torch.testing.assert_close(g,h)

if __name__=='__main__':unittest.main()
