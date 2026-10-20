"""Run one stage of the tiny MLM-to-retrieval encoder lineage on CPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import jax.numpy as jnp
import numpy as np
import optax

from network import (
    ROOT, BidirectionalEncoder, SPECIAL, batch_tokens, init_params, load_data,
    load_params, make_model, make_tokenizer, read_jsonl, sha256_bytes, token_words,
    tree_bytes, tree_parameter_count,
)

CONFIG = json.loads((ROOT / 'config.json').read_text())
STAGE_NAMES = {1: 'masked-language-pretraining', 2: 'contrastive-continuation', 3: 'supervised-contrastive-adaptation', 4: 'held-out-retrieval-and-export'}


def load_candidate(path: str):
    spec = importlib.util.spec_from_file_location('learner_model', Path(path).resolve())
    if spec is None or spec.loader is None:
        raise ValueError(f'cannot load implementation: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def paths_for(output: Path, stage: int):
    return output / 'checkpoints' / f'stage-{stage}'


def source_identity(candidate):
    files=[ROOT/'network.py',ROOT/'run.py',ROOT/'probe_export.py',Path(candidate.__file__).resolve()]
    entries={path.name:sha256_bytes(path.read_bytes()) for path in files}
    return {'files':entries,'sha256':canonical_hash(entries)}


def validate_checkpoint_chain(output: Path, through: int, config: dict, tokenizer: dict,
                              data_hash: str, source_code: dict):
    previous=None; manifests=[]
    for stage in range(1,through+1):
        directory=paths_for(output,stage)
        manifest_path=directory/'manifest.json'; params_path=directory/'params.msgpack'
        if not manifest_path.is_file() or not params_path.is_file():
            raise ValueError(f'missing stage handoff checkpoint: {directory}')
        manifest=json.loads(manifest_path.read_text()); params=params_path.read_bytes()
        if sha256_bytes(params)!=manifest.get('params_sha256'):
            raise ValueError(f'stage {stage} parameter checksum mismatch')
        if manifest.get('stage')!=stage or manifest.get('parent_checkpoint_sha256')!=previous:
            raise ValueError(f'stage {stage} parent checkpoint lineage mismatch')
        if manifest.get('config_sha256')!=canonical_hash(config): raise ValueError(f'stage {stage} configuration mismatch')
        if manifest.get('tokenizer_sha256')!=canonical_hash(tokenizer): raise ValueError(f'stage {stage} tokenizer mismatch')
        if manifest.get('data_sha256')!=data_hash: raise ValueError(f'stage {stage} data identity mismatch')
        if manifest.get('source_code_identity')!=source_code: raise ValueError(f'stage {stage} source-code identity mismatch')
        payload={key:value for key,value in manifest.items() if key!='checkpoint_sha256'}
        expected=hashlib.sha256(params+canonical_hash(payload).encode()).hexdigest()
        if expected!=manifest.get('checkpoint_sha256'): raise ValueError(f'stage {stage} manifest checksum mismatch')
        previous=manifest['checkpoint_sha256']; manifests.append(manifest)
    return manifests


def validate_stale_parent_rejection(output: Path, config: dict, tokenizer: dict,
                                    data_hash: str, source_code: dict, actual_parent_hash: str):
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix='embedding-stale-parent-') as temp:
        target=Path(temp)/'stage-3'; target.mkdir()
        source=paths_for(output,3)
        (target/'params.msgpack').write_bytes((source/'params.msgpack').read_bytes())
        manifest=json.loads((source/'manifest.json').read_text())
        manifest['parent_checkpoint_sha256']='0'*64
        (target/'manifest.json').write_text(json.dumps(manifest))
        try: read_state(target,config,tokenizer,data_hash,actual_parent_hash,source_code)
        except ValueError as error:
            if 'parent' not in str(error): raise
        else: raise AssertionError('stale/swapped checkpoint parent was accepted')


def read_state(directory: Path, config: dict, tokenizer: dict, data_hash: str, expected_parent: str | None, source_code: dict):
    manifest_path = directory / 'manifest.json'
    parameter_path = directory / 'params.msgpack'
    if not manifest_path.is_file() or not parameter_path.is_file():
        raise ValueError(f'missing stage handoff checkpoint: {directory}')
    manifest = json.loads(manifest_path.read_text())
    params_bytes = parameter_path.read_bytes()
    if sha256_bytes(params_bytes) != manifest['params_sha256']:
        raise ValueError('checkpoint parameter checksum mismatch')
    if manifest['config_sha256'] != canonical_hash(config):
        raise ValueError('checkpoint configuration mismatch')
    if manifest['tokenizer_sha256'] != canonical_hash(tokenizer):
        raise ValueError('checkpoint tokenizer mismatch')
    if manifest['data_sha256'] != data_hash:
        raise ValueError('checkpoint data identity mismatch')
    if manifest.get('source_code_identity') != source_code:
        raise ValueError('checkpoint source-code identity mismatch')
    if manifest['parent_checkpoint_sha256'] != expected_parent:
        raise ValueError('checkpoint parent lineage mismatch')
    payload = {key: value for key, value in manifest.items() if key != 'checkpoint_sha256'}
    expected = hashlib.sha256(params_bytes + canonical_hash(payload).encode()).hexdigest()
    if expected != manifest['checkpoint_sha256']:
        raise ValueError('checkpoint manifest checksum mismatch')
    model = make_model(config, tokenizer)
    template = init_params(model, config['seed'], config['max_length'])
    return load_params(template, params_bytes), manifest


def save_state(directory: Path, params, *, stage: int, config: dict, tokenizer: dict, data_hash: str,
               parent: dict | None, metrics: dict, parameter_count: int, source_code: dict):
    directory.mkdir(parents=True, exist_ok=True)
    params_bytes = tree_bytes(params)
    metadata = {
        'stage': stage,
        'stage_name': STAGE_NAMES[stage],
        'data_sha256': data_hash,
        'source_sha256': json.loads((ROOT / 'data/manifest.json').read_text())['source_sha256'],
        'source_code_identity': source_code,
        'config_sha256': canonical_hash(config),
        'tokenizer_sha256': canonical_hash(tokenizer),
        'parent_stage': None if parent is None else parent['stage'],
        'parent_checkpoint_sha256': None if parent is None else parent['checkpoint_sha256'],
        'parameter_count': parameter_count,
        'optimizer_state_reset_at_stage_boundary': stage > 1,
        'metrics': metrics,
    }
    metadata['params_sha256'] = sha256_bytes(params_bytes)
    metadata['checkpoint_sha256'] = hashlib.sha256(params_bytes + canonical_hash(metadata).encode()).hexdigest()
    (directory / 'params.msgpack').write_bytes(params_bytes)
    (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n')
    (directory / 'manifest.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    return metadata


def corrupted_batch(tokens: np.ndarray, rng: np.random.Generator, probability: float):
    valid = (tokens >= 5)
    selected = (rng.random(tokens.shape) < probability) & valid
    for row in range(len(tokens)):
        if np.any(valid[row]) and not np.any(selected[row]):
            selected[row, np.flatnonzero(valid[row])[0]] = True
    targets = np.where(selected, tokens, 0).astype(np.int32)
    corrupted = np.where(selected, SPECIAL['<mask>'], tokens).astype(np.int32)
    return corrupted, targets, selected


def view_batch(tokens: np.ndarray, rng: np.random.Generator, probability: float):
    result = tokens.copy()
    valid = result >= 5
    result[(rng.random(result.shape) < probability) & valid] = SPECIAL['<mask>']
    return result


def chunks(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        if len(batch) < batch_size:
            batch = batch + items[:batch_size - len(batch)]
        yield batch


def mlm_eval(candidate, model, params, docs, tokenizer, config, seed=17):
    rng = np.random.default_rng(seed)
    values=[]; counts=[]
    for batch in chunks(docs, config['batch_size']):
        tokens, _, _ = batch_tokens([doc['text'] for doc in batch], tokenizer, config['max_length'])
        corrupted, targets, selected = corrupted_batch(tokens, rng, config['mask_probability'])
        loss = candidate.masked_lm_loss(model, params, corrupted, targets, selected)
        count = int(selected.sum())
        values.append(float(loss) * count); counts.append(count)
    return sum(values) / max(sum(counts), 1)


def retrieval_metrics(candidate, model, params, docs, queries, qrels, tokenizer, config):
    if not queries:
        return {'queries': 0, 'recall_at_1': None, 'mrr': None, 'ndcg_at_5': None}
    doc_tokens, _, _ = batch_tokens([doc['text'] for doc in docs], tokenizer, config['max_length'])
    query_tokens, _, _ = batch_tokens([query['text'] for query in queries], tokenizer, config['max_length'])
    doc_vectors = np.asarray(candidate.encode(model, params, doc_tokens), dtype=np.float64)
    query_vectors = np.asarray(candidate.encode(model, params, query_tokens), dtype=np.float64)
    doc_by_id = {doc['id']: i for i, doc in enumerate(docs)}
    judged = {}
    for item in qrels:
        judged.setdefault(item['query_id'], {})[item['doc_id']] = item['relevance']
    recalls=[]; mrr=[]; ndcgs=[]
    for query, vector in zip(queries, query_vectors):
        scores = doc_vectors @ vector
        ranking = np.argsort(-scores)
        relevant = judged.get(query['id'], {})
        ranked_ids = [docs[index]['id'] for index in ranking]
        positions = [ranked_ids.index(doc_id) + 1 for doc_id, rel in relevant.items() if rel > 0 and doc_id in doc_by_id]
        recalls.append(float(any(position == 1 for position in positions)))
        mrr.append(1.0 / min(positions) if positions else 0.0)
        dcg = sum((2 ** relevant.get(doc_id, 0) - 1) / np.log2(rank + 1) for rank, doc_id in enumerate(ranked_ids[:5], start=1))
        ideal = sorted(relevant.values(), reverse=True)[:5]
        idcg = sum((2 ** rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
        ndcgs.append(dcg / idcg if idcg else 0.0)
    return {'queries': len(queries), 'recall_at_1': float(np.mean(recalls)), 'mrr': float(np.mean(mrr)), 'ndcg_at_5': float(np.mean(ndcgs))}


def lexical_metrics(docs, queries, qrels):
    if not queries:
        return {'queries': 0, 'recall_at_1': None, 'mrr': None, 'ndcg_at_5': None}
    token_lists = [token_words(doc['text']) for doc in docs]
    doc_freq = {}
    for words in token_lists:
        for word in set(words): doc_freq[word] = doc_freq.get(word, 0) + 1
    avg_len = np.mean([len(words) for words in token_lists])
    n = len(docs); judged = {}
    for item in qrels: judged.setdefault(item['query_id'], {})[item['doc_id']] = item['relevance']
    recalls=[]; mrr=[]; ndcgs=[]
    for query in queries:
        qterms=token_words(query['text']); scores=[]
        for words in token_lists:
            tf={}
            for word in words: tf[word]=tf.get(word,0)+1
            score=0.0
            for word in qterms:
                df=doc_freq.get(word,0)
                if not df: continue
                inverse=np.log(1+(n-df+0.5)/(df+0.5))
                frequency=tf.get(word,0)
                score += inverse * frequency * 2.2 / (frequency + 1.2*(0.25+0.75*len(words)/max(avg_len,1)))
            scores.append(score)
        order=np.argsort(-np.asarray(scores)); ranked=[docs[index]['id'] for index in order]
        rels=judged.get(query['id'],{})
        positions=[ranked.index(doc_id)+1 for doc_id,rel in rels.items() if rel>0 and doc_id in ranked]
        recalls.append(float(1 in positions)); mrr.append(1/min(positions) if positions else 0.0)
        dcg=sum((2**rels.get(doc_id,0)-1)/np.log2(rank+1) for rank,doc_id in enumerate(ranked[:5],start=1))
        ideal=sorted(rels.values(),reverse=True)[:5]
        idcg=sum((2**rel-1)/np.log2(rank+1) for rank,rel in enumerate(ideal,start=1))
        ndcgs.append(dcg/idcg if idcg else 0.0)
    return {'queries':len(queries),'recall_at_1':float(np.mean(recalls)),'mrr':float(np.mean(mrr)),'ndcg_at_5':float(np.mean(ndcgs))}


def load_or_create_tokenizer(docs, output, config):
    path=output/'tokenizer.json'
    if path.exists():
        tokenizer=json.loads(path.read_text())
        if tokenizer['fit_split']!='train': raise ValueError('tokenizer was not fit on training split')
        return tokenizer
    tokenizer=make_tokenizer(docs, config['tokenizer']['max_vocabulary'])
    output.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(tokenizer,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    return tokenizer


def train_mlm(candidate, model, params, docs, tokenizer, config):
    optimizer=optax.adam(config['pretrain_learning_rate']); opt_state=optimizer.init(params)
    rng=np.random.default_rng(config['seed']+1); history=[]
    def loss_fn(p, corrupted, targets, selected): return candidate.masked_lm_loss(model,p,corrupted,targets,selected)
    @jax.jit
    def step(p, state, corrupted, targets, selected):
        loss, grads=jax.value_and_grad(loss_fn)(p,corrupted,targets,selected)
        grad_norm=optax.global_norm(grads)
        finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads)]))
        updates,state=optimizer.update(grads,state,p)
        return optax.apply_updates(p,updates),state,loss,grad_norm,finite
    for epoch in range(config['pretrain_epochs']):
        order=rng.permutation(len(docs)); epoch_loss=[]; epoch_grad_norms=[]
        ordered=[docs[int(index)] for index in order]
        for batch in chunks(ordered,config['batch_size']):
            tokens,_,_=batch_tokens([doc['text'] for doc in batch],tokenizer,config['max_length'])
            corrupted,targets,selected=corrupted_batch(tokens,rng,config['mask_probability'])
            params,opt_state,loss,grad_norm,finite=step(params,opt_state,corrupted,targets,selected)
            loss=float(loss); grad_norm=float(grad_norm)
            if not np.isfinite(loss): raise ValueError('MLM batch loss is nonfinite; selected-token mask may be empty')
            if not bool(finite) or not np.isfinite(grad_norm): raise ValueError('MLM gradients are nonfinite')
            epoch_loss.append(loss); epoch_grad_norms.append(grad_norm)
        history.append({'epoch':epoch+1,'train_mlm_nll':float(np.mean(epoch_loss)),
                        'updates':len(epoch_loss),'gradient_norm_mean':float(np.mean(epoch_grad_norms)),
                        'gradient_norm_max':float(np.max(epoch_grad_norms))})
    return params,history


def train_contrastive(candidate, model, params, docs, tokenizer, config):
    optimizer=optax.adam(config['contrastive_learning_rate']); opt_state=optimizer.init(params)
    rng=np.random.default_rng(config['seed']+2); history=[]
    def loss_fn(p,left,right): return candidate.contrastive_loss(model,p,left,right,config['temperature'])
    @jax.jit
    def step(p,state,left,right):
        loss,grads=jax.value_and_grad(loss_fn)(p,left,right)
        grad_norm=optax.global_norm(grads)
        finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads)]))
        updates,state=optimizer.update(grads,state,p)
        return optax.apply_updates(p,updates),state,loss,grad_norm,finite
    for update in range(config['contrastive_updates']):
        indexes=rng.choice(len(docs),size=config['batch_size'],replace=len(docs)<config['batch_size'])
        batch=[docs[int(index)] for index in indexes]
        tokens,_,_=batch_tokens([doc['text'] for doc in batch],tokenizer,config['max_length'])
        left=view_batch(tokens,rng,config['view_mask_probability']); right=view_batch(tokens,rng,config['view_mask_probability'])
        params,opt_state,loss,grad_norm,finite=step(params,opt_state,left,right)
        loss=float(loss); grad_norm=float(grad_norm)
        if not np.isfinite(loss): raise ValueError('contrastive update produced a nonfinite loss')
        if not bool(finite) or not np.isfinite(grad_norm): raise ValueError('contrastive gradients are nonfinite')
        history.append({'step':len(history)+1,'loss':loss,'gradient_norm':grad_norm})
    return params,history


def train_adaptation(candidate, model, params, docs, queries, qrels, tokenizer, config):
    optimizer=optax.adam(config['adaptation_learning_rate']); opt_state=optimizer.init(params)
    rng=np.random.default_rng(config['seed']+3); history=[]
    doc_by_id={doc['id']:doc for doc in docs}
    query_by_id={query['id']:query for query in queries}
    pairs=[(query_by_id[item['query_id']],doc_by_id[item['doc_id']]) for item in qrels
           if item['query_id'] in query_by_id and item['doc_id'] in doc_by_id]
    if len(pairs)<2: raise ValueError('adaptation requires at least two curated training query-positive pairs')
    def loss_fn(p,tokens,labels):
        vectors,_=model.apply({'params':p},jnp.asarray(tokens,dtype=jnp.int32))
        return candidate.supervised_contrastive_loss(vectors,labels,config['temperature'])
    @jax.jit
    def step(p,state,tokens,labels):
        loss,grads=jax.value_and_grad(loss_fn)(p,tokens,labels)
        grad_norm=optax.global_norm(grads)
        finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads)]))
        updates,state=optimizer.update(grads,state,p)
        return optax.apply_updates(p,updates),state,loss,grad_norm,finite
    for _ in range(config['adaptation_updates']):
        pair_ids=rng.choice(len(pairs),size=4,replace=len(pairs)<4)
        texts=[]; labels=[]
        for pair_id in pair_ids:
            query,positive=pairs[int(pair_id)]
            texts.extend([query['text'],positive['text']])
            labels.extend([int(pair_id),int(pair_id)])
        tokens,_,_=batch_tokens(texts,tokenizer,config['max_length'])
        params,opt_state,loss,grad_norm,finite=step(params,opt_state,tokens,np.asarray(labels,dtype=np.int32))
        loss=float(loss); grad_norm=float(grad_norm)
        if not np.isfinite(loss): raise ValueError('supervised contrastive update produced a nonfinite loss')
        if not bool(finite) or not np.isfinite(grad_norm): raise ValueError('supervised contrastive gradients are nonfinite')
        history.append({'step':len(history)+1,'loss':loss,'gradient_norm':grad_norm})
    return params,history


def validate_objectives(candidate, model, params, tokenizer, config, objectives):
    docs,_,_,_=load_data(); sample=docs[0:4]
    tokens,_,_=batch_tokens([doc['text'] for doc in sample],tokenizer,config['max_length'])
    result={}
    if 'mlm' in objectives:
        corrupted,targets,selected=corrupted_batch(tokens,np.random.default_rng(99),config['mask_probability'])
        observed=float(candidate.masked_lm_loss(model,params,corrupted,targets,selected))
        _,logits=model.apply({'params':params},jnp.asarray(corrupted))
        logits=np.asarray(logits); shifted=logits-logits.max(axis=-1,keepdims=True)
        logsum=np.log(np.exp(shifted).sum(axis=-1))
        gold=np.take_along_axis(shifted,targets[...,None],axis=-1)[...,0]
        oracle=float(np.sum((logsum-gold)*selected)/selected.sum())
        error=float(abs(observed-oracle))
        if not np.isclose(observed,oracle,rtol=1e-5,atol=1e-5): raise AssertionError('MLM loss disagrees with independent NumPy oracle')
        empty=np.zeros_like(selected)
        try: empty_result=float(candidate.masked_lm_loss(model,params,corrupted,targets,empty))
        except (ValueError, AssertionError): empty_result=float('nan')
        if np.isfinite(empty_result): raise AssertionError('empty MLM target selection must fail or return a nonfinite sentinel')
        result['mlm_numpy_abs_error']=error
        result['empty_mlm_selection_rejected']=True
    if 'contrastive' in objectives:
        left=view_batch(tokens,np.random.default_rng(12),.1); right=view_batch(tokens,np.random.default_rng(13),.1)
        observed=float(candidate.contrastive_loss(model,params,left,right,config['temperature']))
        left_vectors=np.asarray(candidate.encode(model,params,left)); right_vectors=np.asarray(candidate.encode(model,params,right))
        similarity=left_vectors@right_vectors.T/config['temperature']
        def ce(matrix):
            z=matrix-matrix.max(axis=1,keepdims=True)
            return np.mean(np.log(np.exp(z).sum(axis=1))-np.diag(z))
        oracle=float(.5*(ce(similarity)+ce(similarity.T)))
        error=float(abs(observed-oracle))
        if not np.isclose(observed,oracle,rtol=1e-5,atol=1e-5): raise AssertionError('contrastive loss disagrees with independent NumPy oracle')
        result['contrastive_numpy_abs_error']=error
    if 'supcon' in objectives:
        labels=np.asarray([0,0,1,1]); vectors=np.asarray(candidate.encode(model,params,tokens))
        observed=float(candidate.supervised_contrastive_loss(vectors,labels,config['temperature']))
        vectors=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-8)
        sims=vectors@vectors.T/config['temperature']; nonself=~np.eye(len(labels),dtype=bool); positive=(labels[:,None]==labels[None,:])&nonself
        logits=np.where(nonself,sims,-1e9); shifted=logits-logits.max(axis=1,keepdims=True)
        log_probability=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
        oracle=float(np.mean([-np.mean(log_probability[row,positive[row]]) for row in range(len(labels))]))
        error=float(abs(observed-oracle))
        if not np.isclose(observed,oracle,rtol=1e-5,atol=1e-5): raise AssertionError('supervised contrastive loss disagrees with independent NumPy oracle')
        result['supervised_contrastive_numpy_abs_error']=error
    result['objectives_checked']=list(objectives)
    return result


def run_stage(stage: int, candidate, output: Path):
    docs,queries,qrels,data_hash=load_data(); config=CONFIG
    code_identity=source_identity(candidate)
    training=[doc for doc in docs if doc['split']=='train']; validation=[doc for doc in docs if doc['split']=='validation']
    validation_queries=[query for query in queries if query['split']=='validation']
    validation_qrels=[item for item in qrels if item['query_id'].startswith('val-')]
    tokenizer=load_or_create_tokenizer(docs,output,config)
    model=make_model(config,tokenizer)
    if stage==1:
        params=init_params(model,config['seed'],config['max_length']); parent=None
        params,history=train_mlm(candidate,model,params,training,tokenizer,config)
        objective_checks=validate_objectives(candidate,model,params,tokenizer,config,('mlm',))
        metrics={'train_mlm_history':history,'optimizer_monitoring':{'learning_rate':config['pretrain_learning_rate'],'actual_updates':sum(row['updates'] for row in history),'finite_loss_and_gradients':True},'objective_checks':objective_checks,'validation_mlm_nll':mlm_eval(candidate,model,params,validation,tokenizer,config),
                 'validation_retrieval':retrieval_metrics(candidate,model,params,validation,validation_queries,validation_qrels,tokenizer,config),
                 'parameter_count':tree_parameter_count(params),'train_documents':len(training),'validation_documents':len(validation)}
    elif stage in (2,3):
        if stage==2:
            chain=validate_checkpoint_chain(output,1,config,tokenizer,data_hash,code_identity)
            parent_params,parent=read_state(paths_for(output,1),config,tokenizer,data_hash,None,code_identity)
            params,history=train_contrastive(candidate,model,parent_params,training,tokenizer,config)
            objective_checks=validate_objectives(candidate,model,params,tokenizer,config,('contrastive',))
            metrics={'objective_checks':objective_checks,'train_contrastive_loss_initial':history[0]['loss'],'train_contrastive_loss_final':history[-1]['loss'],
                     'train_contrastive_history':history,'optimizer_monitoring':{'learning_rate':config['contrastive_learning_rate'],'actual_updates':len(history),'finite_loss_and_gradients':True},'validation_mlm_nll':mlm_eval(candidate,model,params,validation,tokenizer,config),
                     'validation_retrieval':retrieval_metrics(candidate,model,params,validation,validation_queries,validation_qrels,tokenizer,config)}
        else:
            chain=validate_checkpoint_chain(output,2,config,tokenizer,data_hash,code_identity)
            parent_params,parent=read_state(paths_for(output,2),config,tokenizer,data_hash,chain[0]['checkpoint_sha256'],code_identity)
            train_queries=[query for query in queries if query['split']=='train']
            train_qrels=[item for item in qrels if item['query_id'].startswith('train-')]
            params,history=train_adaptation(candidate,model,parent_params,training,train_queries,train_qrels,tokenizer,config)
            metrics={'train_supervised_contrastive_initial':history[0]['loss'],'train_supervised_contrastive_final':history[-1]['loss'],
                     'train_supervised_contrastive_history':history,'optimizer_monitoring':{'learning_rate':config['adaptation_learning_rate'],'actual_updates':len(history),'finite_loss_and_gradients':True},'validation_mlm_nll':mlm_eval(candidate,model,params,validation,tokenizer,config),
                     'validation_retrieval':retrieval_metrics(candidate,model,params,validation,validation_queries,validation_qrels,tokenizer,config)}
            metrics['objective_checks']=validate_objectives(candidate,model,params,tokenizer,config,('supcon',))
    elif stage==4:
        chain=validate_checkpoint_chain(output,3,config,tokenizer,data_hash,code_identity)
        params,parent=read_state(paths_for(output,3),config,tokenizer,data_hash,chain[1]['checkpoint_sha256'],code_identity)
        test=[doc for doc in docs if doc['split']=='test']; test_queries=[query for query in queries if query['split']=='test']
        test_qrels=[item for item in qrels if item['query_id'].startswith('test-')]
        _, test_oov, test_truncations=batch_tokens([doc['text'] for doc in test],tokenizer,config['max_length'])
        metrics={'test_dense':retrieval_metrics(candidate,model,params,test,test_queries,test_qrels,tokenizer,config),
                 'test_bm25':lexical_metrics(test,test_queries,test_qrels),
                 'validation_dense':retrieval_metrics(candidate,model,params,validation,validation_queries,validation_qrels,tokenizer,config),
                 'validation_bm25':lexical_metrics(validation,validation_queries,validation_qrels),
                 'parameter_count':tree_parameter_count(params),'test_passages':len(test),'test_queries':len(test_queries),
                 'test_oov_word_fraction':test_oov,'test_passages_truncated':test_truncations,
                 'limitations':'Single novel, chapter-group split, five manually authored qrels; pilot only.'}
        export(output,model,params,tokenizer,config,parent,data_hash,code_identity)
        validate_export_fresh_process(output,code_identity)
        validate_export_rejections(output,config,code_identity)
        validate_stale_parent_rejection(output,config,tokenizer,data_hash,code_identity,chain[1]['checkpoint_sha256'])
    else: raise ValueError('stage must be 1, 2, 3, or 4')
    if stage != 4:
        parent_saved=save_state(paths_for(output,stage),params,stage=stage,config=config,tokenizer=tokenizer,data_hash=data_hash,
                                parent=parent,metrics=metrics,parameter_count=tree_parameter_count(params),source_code=code_identity)
        metrics['checkpoint_sha256']=parent_saved['checkpoint_sha256']
        (paths_for(output,stage)/'metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
    else:
        metrics['parent_checkpoint_sha256']=parent['checkpoint_sha256']
        (output/'evaluation.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'stage':stage,'name':STAGE_NAMES[stage],'output':str(output),'metrics':metrics},indent=2,sort_keys=True))
    return metrics


def export(output,model,params,tokenizer,config,parent,data_hash,source_code):
    directory=output/'export'; directory.mkdir(parents=True,exist_ok=True)
    probe='A curious girl asks the white rabbit where the garden path goes.'
    probe_tokens,_,_=batch_tokens([probe],tokenizer,config['max_length'])
    probe_vector=np.asarray(model.apply({'params':params},jnp.asarray(probe_tokens))[0],dtype=np.float32)
    (directory/'expected-probe.json').write_text(json.dumps({'embedding_sha256':hashlib.sha256(probe_vector.tobytes()).hexdigest()},indent=2)+'\n')
    params_bytes=tree_bytes(params)
    tokenizer_bytes=(json.dumps(tokenizer,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
    config_bytes=(json.dumps(config,sort_keys=True,indent=2)+'\n').encode()
    (directory/'params.msgpack').write_bytes(params_bytes)
    (directory/'tokenizer.json').write_bytes(tokenizer_bytes)
    (directory/'config.json').write_bytes(config_bytes)
    manifest={'format':'jaxpathways-bidirectional-encoder-v1','model_family':'tiny-bidirectional-self-attention',
              'source_sha256':json.loads((ROOT/'data/manifest.json').read_text())['source_sha256'],'data_sha256':data_hash,
              'parent_checkpoint_sha256':parent['checkpoint_sha256'],'parameter_count':tree_parameter_count(params),
              'tokenizer_sha256':canonical_hash(tokenizer),'config_sha256':canonical_hash(config),
              'source_code_identity':source_code,
              'files':{'params.msgpack':sha256_bytes(params_bytes),'tokenizer.json':sha256_bytes(tokenizer_bytes),'config.json':sha256_bytes(config_bytes),
                      'expected-probe.json':sha256_bytes((directory/'expected-probe.json').read_bytes())},
              'claims':{'benchmark':False,'300m_parameters':False,'natural_language_generalization':False}}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')


def verify_export(directory: Path, expected_config: dict, expected_source_code: dict):
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['format']!='jaxpathways-bidirectional-encoder-v1': raise ValueError('unsupported export format')
    if json.loads((directory/'config.json').read_text()) != expected_config: raise ValueError('export configuration mismatch')
    for name,digest in manifest['files'].items():
        if sha256_bytes((directory/name).read_bytes()) != digest: raise ValueError('export file checksum mismatch')
    tokenizer=json.loads((directory/'tokenizer.json').read_text())
    if canonical_hash(tokenizer) != manifest.get('tokenizer_sha256'):
        raise ValueError('export tokenizer mismatch')
    if canonical_hash(expected_config) != manifest.get('config_sha256'):
        raise ValueError('export manifest configuration mismatch')
    if manifest.get('source_code_identity') != expected_source_code:
        raise ValueError('export source-code identity mismatch')
    return manifest,tokenizer


def validate_export_fresh_process(output: Path, source_code: dict):
    directory=output/'export'; expected=json.loads((directory/'expected-probe.json').read_text()) if (directory/'expected-probe.json').exists() else None
    manifest,tokenizer=verify_export(directory,CONFIG,source_code)
    probe='A curious girl asks the white rabbit where the garden path goes.'
    command=[sys.executable,str(ROOT/'probe_export.py'),'--export',str(directory),'--text',probe]
    first=subprocess.run(command,check=True,capture_output=True,text=True)
    observed=json.loads(first.stdout)
    second=subprocess.run(command,check=True,capture_output=True,text=True)
    repeated=json.loads(second.stdout)
    if observed['embedding_sha256'] != repeated['embedding_sha256']:
        raise AssertionError('fresh-process exported encoder is not deterministic')
    if expected is not None and observed['embedding_sha256'] != expected['embedding_sha256']:
        raise AssertionError('fresh-process embedding differs from in-process expected vector')
    (output/'export'/'reload-receipt.json').write_text(json.dumps({'python':sys.version.split()[0],'processes':2,'embedding_sha256':observed['embedding_sha256'],'deterministic':True},indent=2)+'\n')


def validate_export_rejections(output: Path, config: dict, source_code: dict):
    import shutil
    from tempfile import TemporaryDirectory
    directory=output/'export'
    with TemporaryDirectory(prefix='embedding-export-check-') as temp:
        bad=Path(temp)/'bad-config'; bad.mkdir()
        (bad/'manifest.json').write_bytes((directory/'manifest.json').read_bytes())
        for name in json.loads((directory/'manifest.json').read_text())['files']:
            (bad/name).write_bytes((directory/name).read_bytes())
        altered=json.loads((bad/'config.json').read_text()); altered['width']+=1
        (bad/'config.json').write_text(json.dumps(altered))
        try: verify_export(bad,config,source_code)
        except ValueError as error:
            if 'configuration' not in str(error): raise
        else: raise AssertionError('changed export config was accepted')
        corrupt=Path(temp)/'corrupt'; corrupt.mkdir()
        (corrupt/'manifest.json').write_bytes((directory/'manifest.json').read_bytes())
        for name in json.loads((directory/'manifest.json').read_text())['files']:
            (corrupt/name).write_bytes((directory/name).read_bytes())
        payload=bytearray((corrupt/'params.msgpack').read_bytes()); payload[-1]^=1
        (corrupt/'params.msgpack').write_bytes(payload)
        try: verify_export(corrupt,config,source_code)
        except ValueError as error:
            if 'checksum' not in str(error): raise
        else: raise AssertionError('corrupted parameter bytes were accepted')
        changed_loader=Path(temp)/'changed-loader'; changed_loader.mkdir()
        loader_source=(ROOT/'probe_export.py').read_bytes()+b'\n# deliberate changed-loader rejection fixture\n'
        (changed_loader/'probe_export.py').write_bytes(loader_source)
        shutil.copyfile(ROOT/'network.py',changed_loader/'network.py')
        command=[sys.executable,str(changed_loader/'probe_export.py'),'--export',str(directory),
                 '--text','A curious girl asks the white rabbit where the garden path goes.']
        rejected=subprocess.run(command,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        if rejected.returncode==0 or 'loader source-code identity mismatch' not in rejected.stderr:
            raise AssertionError('changed export loader was not rejected against the recorded source identity')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--implementation',required=True)
    parser.add_argument('--stage',type=int,choices=range(1,5),required=True)
    parser.add_argument('--output',default='runs/embedding-lineage/run')
    args=parser.parse_args()
    candidate=load_candidate(args.implementation)
    run_stage(args.stage,candidate,Path(args.output))

if __name__=='__main__': main()
