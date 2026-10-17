"""Complete original-solution coverage and three-epoch SFT preparation.

No student generations, demonstration categories or benchmark results are inputs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import tempfile
import unicodedata
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean
from .data import alpaca,digest,prompt_id,read_rows,write_json
from .select import allocate,eligible

POOL_SHA="c115f4331f485458616b8e29a3715422b5d4479396cf50c287183014820eabc9"
FEATURES_SHA="039cd63692d528dcce2caf2be2bfed85be917a66298a2532933ef1107f9db979"
CUTOFF=16384
SIZE=15000
UPDATES=5625
COLUMNS={"prompt":"instruction","query":"input","response":"output","system":"system"}

def write_dataset_info(out):
    write_json(Path(out)/"dataset_info.json",{"advanced":{"file_name":"advanced.json","columns":COLUMNS}})


def rank_id(row_id):
    return hashlib.sha256(("full-length-42:"+row_id).encode()).hexdigest()


def select(features,size=SIZE):
    if len({f['id'] for f in features})!=len(features):
        raise ValueError("duplicate feature IDs")
    groups=defaultdict(list)
    for f in features:
        if eligible(f,CUTOFF): groups[f['cluster']].append(f)
    quotas=allocate({k:len(v) for k,v in groups.items()},size)
    chosen=[]; strata_receipt={}
    for cluster in sorted(groups):
        if not quotas[cluster]: continue
        ordered=sorted(groups[cluster],key=lambda f:(f['response_tokens'],f['id']))
        strata=defaultdict(list)
        for i,f in enumerate(ordered): strata[min(3,4*i//len(ordered))].append(f)
        sq=allocate({k:len(v) for k,v in strata.items()},quotas[cluster])
        for q in sorted(strata):
            picked=sorted(strata[q],key=lambda f:(rank_id(f['id']),f['id']))[:sq[q]]
            chosen.extend({**f,'length_quartile':q} for f in picked)
        strata_receipt[str(cluster)]={"eligible":len(ordered),"selected":quotas[cluster],
                                    "quartile_quotas":sq}
    chosen.sort(key=lambda f:(rank_id(f['id']),f['id']))
    if len(chosen)!=size or len({f['id'] for f in chosen})!=size:
        raise ValueError("wrong unique selection size")
    return chosen,strata_receipt


def processor(model,directory):
    from transformers import AutoTokenizer
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments
    from llamafactory.data.parser import get_dataset_list
    from llamafactory.data.converter import get_dataset_converter
    tok=AutoTokenizer.from_pretrained(model,trust_remote_code=False)
    args=DataArguments(template='qwen',cutoff_len=CUTOFF,train_on_prompt=False,packing=False)
    template=get_template_and_fix_tokenizer(tok,args)
    attrs=get_dataset_list(["advanced"],str(directory))
    assert len(attrs)==1 and attrs[0].system=="system"
    converter=get_dataset_converter("alpaca",attrs[0],args)
    return tok,template,SupervisedDatasetProcessor(template=template,tokenizer=tok,
                                                  processor=None,data_args=args),converter


def check_row(row,tok,template,proc,converter):
    converted=converter(row)
    prompt=[{'role':'user','content':row['instruction']}]
    response=[{'role':'assistant','content':row['output']}]
    if converted["_system"]!=row["system"] or converted["_prompt"]!=prompt or converted["_response"]!=response:
        raise ValueError("actual dataset conversion changed system, prompt or original target")
    pairs=template.encode_multiturn(tok,prompt+response,row['system'],None)
    if len(pairs)!=1: raise ValueError("unexpected multiple turns")
    source,target=pairs[0]
    expected=target+([tok.eos_token_id] if template.efficient_eos else [])
    batch=proc.preprocess_dataset({k:[v] for k,v in converted.items()})
    if len(batch["input_ids"])!=1:raise ValueError("actual processor dropped a selected row")
    ids,labels=batch["input_ids"][0],batch["labels"][0]
    actual=[x for x in labels if x!=-100]
    if len(source)+len(expected)>CUTOFF or len(ids)>CUTOFF or actual!=expected:
        raise ValueError("response truncation or serialization drift")
    if ids[:len(source)]!=source or labels[:len(source)]!=[-100]*len(source):
        raise ValueError("prompt is not completely masked")
    decoded=tok.decode(actual,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    if tok.eos_token_id not in actual:
        raise ValueError("supervised EOS missing")
    # The pinned tokenizer canonically composes Unicode (e.g. y + macron).
    # Export bytes remain original; require complete untruncated original-token
    # labels above and only canonical NFC equivalence in this decoded check.
    if not decoded.startswith(row['output']) and not decoded.startswith(unicodedata.normalize('NFC',row['output'])):
        raise ValueError("decoded original response differs beyond canonical Unicode normalization")
    return len(ids),len(actual)


def synthetic_check(model):
    from .data import SYSTEM
    temp=tempfile.TemporaryDirectory(prefix="full-length-cpu-")
    write_dataset_info(temp.name)
    tok,template,proc,converter=processor(model,temp.name)
    for repeats in (2,4500):
        row={'instruction':'Synthetic serialization fixture: compute two plus three.','input':'',
             'system':SYSTEM,'output':('Reason carefully. '*repeats)+r'\boxed{5}'}
        total,_=check_row(row,tok,template,proc,converter)
        if repeats==4500 and total<=8192: raise ValueError("fixture did not exercise the new context range")
    row['output']='The symbol y\u0304 is canonically equivalent to y with macron. '+r'\boxed{5}'
    check_row(row,tok,template,proc,converter)
    row['output']='Reason carefully. '*18000+r'\boxed{5}'
    try: check_row(row,tok,template,proc,converter)
    except ValueError: pass
    else: raise ValueError("overlength original response was accepted")
    print("Actual 16K processor: complete long targets/EOS and rejection of truncation passed.",flush=True)


def prepare(model,out):
    out=Path(out)
    pool='artifacts/prepared_extended/pool.json'; features='artifacts/features_extended.json'
    if digest(pool)!=POOL_SHA or digest(features)!=FEATURES_SHA: raise ValueError("sealed source mismatch")
    rows,fs=read_rows(pool),read_rows(features)
    if len(rows)!=83037 or len(fs)!=83037: raise ValueError("wrong source coverage")
    by_id={r['id']:r for r in rows}
    if len(by_id)!=83037 or {f['id'] for f in fs}!=set(by_id): raise ValueError("source IDs differ")
    write_dataset_info(out)
    picked,quotas=select(fs);tok,template,proc,converter=processor(model,out)
    data=[]; receipt=[]
    for i,f in enumerate(picked):
        row=by_id[f['id']]
        if prompt_id(row['instruction'])!=f['id']: raise ValueError("normalized identity mismatch")
        total,supervised=check_row(row,tok,template,proc,converter)
        data.append(row)
        receipt.append({'id':f['id'],'cluster':f['cluster'],'length_quartile':f['length_quartile'],
                        'exact_tokens':total,'supervised_tokens':supervised,
                        'original_response_sha256':hashlib.sha256(row['output'].encode()).hexdigest(),
                        'source_nfc_composition':unicodedata.normalize('NFC',row['output'])!=row['output']})
        if (i+1)%500==0: print(f"Actual 16K original-target/EOS checks: {i+1}/{SIZE}",flush=True)
    long_count=sum(x['exact_tokens']>8192 for x in receipt)
    if not long_count: raise ValueError("candidate does not introduce longer complete supervision")
    write_json(out/'advanced.json',alpaca(data))
    write_dataset_info(out)
    manifest={'schema':'full-length-original-15k-16k-three-epoch-v1',
              'pool_sha256':POOL_SHA,'features_sha256':FEATURES_SHA,'module_sha256':digest(__file__),
              'allocation_module_sha256':digest('reasoning/select.py'),'model_manifest':json.load(open('artifacts/model.json')),
              'selected_ids':[r['id'] for r in receipt],'selected':receipt,'quotas':quotas,
              'eligible_examples':sum(eligible(f,CUTOFF) for f in fs),
              'selected_above_8k':long_count,'max_exact_tokens':max(r['exact_tokens'] for r in receipt),
              'mean_exact_tokens':mean(r['exact_tokens'] for r in receipt),
              'source_responses_with_nfc_composition':sum(r['source_nfc_composition'] for r in receipt),
              'supervised_tokens_per_epoch':sum(r['supervised_tokens'] for r in receipt),
              'cutoff':CUTOFF,'unique_prompts':SIZE,'epochs':3,'example_presentations':45000,'updates':UPDATES,
              'original_targets_complete':True,'prompt_tokens_masked':True,'eos_supervised':True,
              'actual_dataset_converter_verified':True,'system_column_preserved':True,
              'dataset_info_sha256':digest(out/'dataset_info.json'),
              'student_generations_used_for_selection':False,'public_scores_used_for_selection':False,
              'data_sha256':digest(out/'advanced.json'),
              'versions':{k:importlib.metadata.version(k) for k in ('torch','transformers')}}
    write_json(out/'selection_manifest.json',manifest)
    write_json(out/'preparation_receipt.json',{k:v for k,v in manifest.items() if k not in ('selected','selected_ids','quotas')})
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('selected','selected_ids','quotas')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--output',default='artifacts/datasets-full-length-15k')
    p.add_argument('--synthetic-check',action='store_true');args=p.parse_args()
    if args.synthetic_check:synthetic_check(args.model)
    else:prepare(args.model,args.output)

if __name__=='__main__':main()
