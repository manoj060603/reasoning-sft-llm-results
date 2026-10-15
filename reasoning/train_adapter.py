"""Single-GPU BF16 LoRA SFT/DFT runner for the existing ARC runtime.

This path avoids changing shared LLaMA-Factory environments. Checkpoints include
optimizer/scheduler/RNG state and an input manifest; evaluation uses a merged copy.
"""
import argparse
import json
import math
import time
from pathlib import Path
from .data import digest,read_rows,write_json
from .features import token_ids


def response_loss(logits, labels, objective='sft', chunk=128):
    import torch
    total=logits.new_zeros((),dtype=torch.float32)
    count=(labels[:,1:]!=-100).sum()
    if int(count)==0:raise ValueError('batch has no response targets')
    for start in range(0,logits.shape[1]-1,chunk):
        end=min(start+chunk,logits.shape[1]-1)
        target=labels[:,start+1:end+1].reshape(-1)
        ce=torch.nn.functional.cross_entropy(logits[:,start:end].reshape(-1,logits.shape[-1]).float(),
                                             target,ignore_index=-100,reduction='none')
        active=target!=-100
        if objective=='dft':ce=ce*torch.exp(-ce.detach())
        elif objective!='sft':raise ValueError(objective)
        total=total+ce[active].sum()
    return total/count


def forward_kl(logits, reference_logits, labels, chunk=128):
    """KL(reference || policy) on response positions; reference is always detached."""
    import torch
    total=logits.new_zeros((),dtype=torch.float32)
    count=(labels[:,1:]!=-100).sum()
    if int(count)==0:raise ValueError('no response targets for KL')
    for start in range(0,logits.shape[1]-1,chunk):
        end=min(start+chunk,logits.shape[1]-1)
        logp=torch.nn.functional.log_softmax(logits[:,start:end].float(),dim=-1)
        with torch.no_grad():
            logref=torch.nn.functional.log_softmax(reference_logits[:,start:end].float(),dim=-1)
        per_token=(logref.exp()*(logref-logp)).sum(-1)
        total=total+per_token[labels[:,start+1:end+1]!=-100].sum()
    return total/count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--objective',choices=['sft','dft','asft'],default='sft')
    p.add_argument('--kl',type=float,default=.03)
    p.add_argument('--cutoff',type=int,default=4096);p.add_argument('--steps',type=int,default=64)
    p.add_argument('--accumulation',type=int,default=8);p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--rank',type=int,default=64)
    p.add_argument('--save-every',type=int,default=32);p.add_argument('--resume',action='store_true')
    p.add_argument('--keep-every',type=int,default=0,
                   help='preserve adapter-only milestone every N steps; zero disables')
    a=p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import LoraConfig,get_peft_model,PeftModel
    if not torch.cuda.is_available():p.error('CUDA required')
    if min(a.steps,a.accumulation,a.save_every,a.cutoff,a.rank)<1 or a.keep_every<0:
        p.error('counts must be positive and keep-every nonnegative')
    set_seed(a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=dict(arguments={k:v for k,v in vars(a).items() if k!='resume'},data_sha256=digest(a.data),
             code_sha256=digest(__file__))
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=cfg:raise ValueError('resume configuration mismatch')
        if (out/'complete.json').exists():print('Already complete');return
        if not a.resume:raise ValueError('existing run needs --resume')
    else:write_json(out/'manifest.json',cfg)
    write_json(out/'hardware.json',dict(gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda))
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    records=read_rows(a.data); encoded=[];dropped=0
    for r in records:
        prompt,full=token_ids(tokenizer,r)
        if len(full)>a.cutoff:dropped+=1;continue
        labels=[-100]*len(prompt)+full[len(prompt):]
        encoded.append((full,labels))
    if not encoded:raise ValueError('all training rows exceed cutoff')
    write_json(out/'data_stats.json',dict(input=len(records),used=len(encoded),dropped_overlength=dropped,
               tokens=sum(len(x[0]) for x in encoded)))
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    ckpt=out/'checkpoint'
    previous=out/'checkpoint-previous'
    if not ckpt.exists() and previous.exists():previous.rename(ckpt)
    start=0;state=None
    if a.resume and (ckpt/'state.pt').exists():
        model=PeftModel.from_pretrained(model,str(ckpt/'adapter'),is_trainable=True)
        state=torch.load(ckpt/'state.pt',map_location='cpu',weights_only=False)
        start=state['step']
    else:
        model=get_peft_model(model,LoraConfig(r=a.rank,lora_alpha=2*a.rank,lora_dropout=.05,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    model=model.to('cuda');model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=.01)
    warmup=max(1,int(a.steps*.03))
    def schedule(s):
        return s/warmup if s<warmup else .5*(1+math.cos(math.pi*(s-warmup)/max(1,a.steps-warmup)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    generator=torch.Generator().manual_seed(a.seed)
    order=torch.randperm(len(encoded),generator=generator).tolist();cursor=0
    if state:
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        order=state['order'];cursor=state['cursor'];generator.set_state(state['generator'])
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
    began=time.monotonic();optimizer.zero_grad(set_to_none=True)
    for step in range(start,a.steps):
        total=0.;ntokens=0
        for micro in range(a.accumulation):
            if cursor==len(order):order=torch.randperm(len(encoded),generator=generator).tolist();cursor=0
            ids,labels=encoded[order[cursor]];cursor+=1
            ids=torch.tensor([ids],device='cuda');labels=torch.tensor([labels],device='cuda')
            reference=None
            if a.objective=='asft':
                model.eval()
                with model.disable_adapter(), torch.no_grad():
                    reference=model(input_ids=ids,use_cache=False).logits
                model.train()
            outputs=model(input_ids=ids,use_cache=False)
            loss=response_loss(outputs.logits,labels,'dft' if a.objective=='asft' else a.objective)
            if reference is not None:
                loss=loss+a.kl*forward_kl(outputs.logits,reference,labels)
            (loss/a.accumulation).backward();total+=float(loss.detach())/a.accumulation;ntokens+=ids.numel()
            del outputs,loss,ids,labels,reference
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        record=dict(step=step+1,objective_loss=total,tokens=ntokens,grad_norm=float(norm),
                    elapsed_seconds=time.monotonic()-began,lr=scheduler.get_last_lr()[0])
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if (step+1)%a.save_every==0 or step+1==a.steps:
            temp=out/'checkpoint-writing';temp.mkdir(exist_ok=True)
            model.save_pretrained(temp/'adapter');tokenizer.save_pretrained(temp/'adapter')
            torch.save(dict(step=step+1,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                order=order,cursor=cursor,generator=generator.get_state(),cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state()),temp/'state.pt')
            import shutil
            if previous.exists():shutil.rmtree(previous)
            if ckpt.exists():ckpt.rename(previous)
            temp.rename(ckpt)
            if previous.exists():shutil.rmtree(previous)
            if a.keep_every and ((step+1)%a.keep_every==0 or step+1==a.steps):
                milestone=out/'milestones'/f'step-{step+1:06d}'
                if not milestone.exists():
                    shutil.copytree(ckpt/'adapter',milestone/'adapter')
                    write_json(milestone/'receipt.json',dict(step=step+1,
                        data_sha256=cfg['data_sha256'],code_sha256=cfg['code_sha256']))
    model=model.merge_and_unload();model.config.use_cache=True
    model.save_pretrained(out/'merged',safe_serialization=True);tokenizer.save_pretrained(out/'merged')
    write_json(out/'complete.json',dict(steps=a.steps,objective=a.objective,merged=str(out/'merged'),
               training_seconds=time.monotonic()-began,peak_gpu_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':main()
