"""Isolated CPU prototype. No trainer integration or model/cache construction."""
import torch

def margin_sum(student, reference, mask, rho=1.):
    if student.shape != reference.shape or student.shape[:-1] != mask.shape:
        raise ValueError('cache/logit/mask shape mismatch')
    if rho < 0:
        raise ValueError('negative reference margin multiplier')
    if not torch.isfinite(reference).all():
        raise ValueError('nonfinite reference cache')
    ref = reference.detach().float()
    stu = student.float()
    rd = ref.unsqueeze(-1)-ref.unsqueeze(-2)
    sd = stu.unsqueeze(-1)-stu.unsqueeze(-2)
    oriented = rd > 0
    penalties = torch.relu(rho*rd-sd)*oriented
    counts = oriented.sum((-1,-2)).clamp_min(1)
    per_token = penalties.sum((-1,-2))/counts
    return (per_token*mask).sum()

def check():
    dtype_results={}
    for dtype,tol in [(torch.float32,1e-6),(torch.bfloat16,.01)]:
        ref=torch.tensor([[[3.,1.,0.],[2.,2.,1.]],[[1.,0.,-1.],[3.,1.,0.]]],dtype=dtype)
        stu=(ref*.5).clone().requires_grad_()
        mask=torch.tensor([[True,False],[True,True]])
        loss=margin_sum(stu,ref,mask)/mask.sum()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(stu.grad).all()
        assert stu.grad[0,1].abs().sum()==0
        assert torch.allclose(margin_sum(stu.detach()+4,ref,mask),margin_sum(stu.detach(),ref,mask),atol=tol)
        assert margin_sum(ref,ref,mask).item()==0
        assert margin_sum(ref*2,ref,mask).item()==0
        assert margin_sum(stu.detach()[...,:1],ref[...,:1],mask).item()==0
        assert margin_sum(stu.detach(),torch.ones_like(ref),mask).item()==0
        assert torch.allclose(sum(margin_sum(stu.detach()[i:i+1],ref[i:i+1],mask[i:i+1]) for i in range(2)),loss.detach()*mask.sum(),atol=tol)
        dtype_results[str(dtype)]='passed'
    ref=torch.tensor([[3.,1.,0.]],dtype=torch.double)
    x=torch.tensor([[.3,.2,.1]],requires_grad=True)
    mask=torch.tensor([True])
    loss=margin_sum(x,ref,mask); loss.backward()
    for i in range(3):
        plus=x.detach().clone();minus=plus.clone();plus[0,i]+=.001;minus[0,i]-=.001
        fd=(margin_sum(plus,ref,mask)-margin_sum(minus,ref,mask))/.002
        assert abs(fd-x.grad[0,i])<.001
    return {'isolated_srp_cpu_checks':dtype_results,'finite_difference':'passed','masked_zero_gradient':True,'teacher_ties_ignored':True,'satisfied_margin_zero':True,'shift_invariant':True,'rank_summed_denominator':'passed','cache_constructed':False,'trainer_integrated':False,'gpu_used':False}

if __name__=='__main__':
    import json
    print(json.dumps(check()))
