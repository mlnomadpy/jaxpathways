"""ModelOps: own, approve, roll back and retire a model: worked experiments and reference solutions. CPU checks."""



# Local ModelOps lifecycle. Single writer, local filesystem, no authentication or authorization.
# Artifacts are immutable JSON files; consumer predictions come from the selected artifact.
import hashlib, json, tempfile
from pathlib import Path

def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def write_json(path,value): Path(path).write_text(canonical(value))
def read_json(path): return json.loads(Path(path).read_text())

def artifact_prediction(artifact,x): return artifact['slope']*x+artifact['intercept']
def make_bundle(version,artifact,target='cpu-demo'):
    return {'version':version,'artifact':artifact,'artifact_hash':digest(artifact),
            'evaluation_hash':digest({'fixture':'consumer-predictions-v1','version':version}),
            'owner':'course-team','target':target,'passed':True}
def approval_for(bundle,actor,issued,expires):
    return {'bundle_hash':digest(bundle),'reviewer':actor,'issued_at':issued,'expires_at':expires}
def verify(bundle,approval,now,target,tombstones,expected_version):
    if bundle['version']!=expected_version: raise ValueError('requested version does not match approved bundle')
    if bundle['version'] in tombstones: raise ValueError('retired target')
    if bundle['target']!=target or not bundle['passed']: raise ValueError('target or evaluation rejected')
    if digest(bundle['artifact'])!=bundle['artifact_hash']: raise ValueError('artifact checksum mismatch')
    if approval.get('bundle_hash')!=digest(bundle): raise ValueError('stale approval for another bundle')
    if not approval.get('reviewer') or not approval['issued_at']<=now<approval['expires_at']:
        raise ValueError('expired or invalid approval')

def state(store):
    p=Path(store)/'active.json'
    return read_json(p) if p.exists() else {'current':None,'previous':None}
def consumer_predict(store,x,cache,force_reload=False):
    """Resolve selection and verify content identity on both cache and disk paths."""
    store=Path(store); selected_state=state(store); selected=selected_state['current']
    if selected is None: raise ValueError('no selected model')
    tombstones=read_json(store/'tombstones.json')
    if selected in tombstones: raise ValueError('selected artifact is retired')
    bundle_hash=selected_state.get('current_bundle_hash')
    expected_artifact_hash=selected_state.get('current_artifact_hash')
    if not bundle_hash or not expected_artifact_hash: raise ValueError('selection is missing content identity')
    cache_key=(selected,bundle_hash,expected_artifact_hash)
    if force_reload or cache_key not in cache:
        artifact=read_json(store/'artifacts'/f'{selected}.json')
        if digest(artifact)!=expected_artifact_hash: raise ValueError('reloaded artifact bytes do not match approved bundle')
        cache[cache_key]=artifact
    else:
        artifact=cache[cache_key]
        if digest(artifact)!=expected_artifact_hash: raise ValueError('cached artifact does not match approved bundle')
    return artifact_prediction(artifact,x)

def activate(store,version,bundle,approval,now,target,audit_actor,tombstones):
    store=Path(store)
    try:
        verify(bundle,approval,now,target,tombstones,version)
        artifact_path=store/'artifacts'/f'{version}.json'
        if not artifact_path.exists() or digest(read_json(artifact_path))!=bundle['artifact_hash']:
            raise ValueError('stored artifact bytes do not match approved bundle')
    except ValueError as error:
        store.mkdir(parents=True,exist_ok=True)
        append_audit(store,{'action':'activation-rejected','version':version,'bundle_hash':digest(bundle),
                            'actor':audit_actor,'at':now,'reason':str(error)})
        raise
    before=state(store)
    new={'current':version,'previous':before['current'],
         'current_bundle_hash':digest(bundle),'current_artifact_hash':bundle['artifact_hash'],
         'previous_bundle_hash':before.get('current_bundle_hash'),
         'previous_artifact_hash':before.get('current_artifact_hash')}
    write_json(store/'active.json',new)
    append_audit(store,{'action':'activate','version':version,'bundle_hash':digest(bundle),
                        'actor':audit_actor,'at':now})
    return new

def append_audit(store,event):
    path=Path(store)/'audit.jsonl'
    with path.open('a') as f: f.write(canonical(event)+'\n')

def rollback(store,now,target,audit_actor,approvals,bundles,tombstones):
    before=state(store); version=before['previous']
    if version is None or version in tombstones:
        append_audit(store,{'action':'rollback-rejected','version':version,'bundle_hash':digest(bundles[version]) if version in bundles else None,
                            'actor':audit_actor,'at':now,'reason':'rollback target is not eligible'})
        raise ValueError('rollback target is not eligible')
    bundle=bundles[version]
    try:
        verify(bundle,approvals[version],now,target,tombstones,version)
        artifact_path=Path(store)/'artifacts'/f'{version}.json'
        if not artifact_path.exists() or digest(read_json(artifact_path))!=bundle['artifact_hash']:
            raise ValueError('rollback artifact bytes do not match approved bundle')
    except ValueError as error:
        append_audit(store,{'action':'rollback-rejected','version':version,'bundle_hash':digest(bundle),
                            'actor':audit_actor,'at':now,'reason':str(error)})
        raise
    write_json(Path(store)/'active.json',{'current':version,'previous':before['current'],
         'current_bundle_hash':digest(bundle),'current_artifact_hash':bundle['artifact_hash'],
         'previous_bundle_hash':digest(bundles[before['current']]) if before['current'] in bundles else None,
         'previous_artifact_hash':bundles[before['current']]['artifact_hash'] if before['current'] in bundles else None})
    append_audit(store,{'action':'rollback','version':version,'bundle_hash':digest(bundle),
                        'actor':audit_actor,'at':now})

def clear_fallback(store,actor,now):
    before=state(store)
    write_json(Path(store)/'active.json',{'current':before['current'],'previous':None,
         'current_bundle_hash':before.get('current_bundle_hash'),'current_artifact_hash':before.get('current_artifact_hash'),
         'previous_bundle_hash':None,'previous_artifact_hash':None})
    append_audit(store,{'action':'clear-fallback','version':before['current'],'actor':actor,'at':now})

def retire(store,version,actor,now,tombstones,bundles):
    selected=state(store)
    if version in (selected['current'],selected['previous']): raise ValueError('active or fallback version is not retireable')
    if version not in bundles: raise ValueError('unknown version')
    tombstones=set(tombstones); tombstones.add(version)
    write_json(Path(store)/'tombstones.json',sorted(tombstones))
    append_audit(store,{'action':'retire','version':version,'bundle_hash':digest(bundles[version]),
                        'actor':actor,'at':now})
    return tombstones

# Three valid artifacts intentionally produce different consumer predictions at x=1.
artifacts={'A':{'format':'linear-v1','slope':2.0,'intercept':1.0},
           'B':{'format':'linear-v1','slope':2.4,'intercept':1.0},
           'C':{'format':'linear-v1','slope':3.0,'intercept':1.0}}
bundles={name:make_bundle(name,a) for name,a in artifacts.items()}
approvals={name:approval_for(bundle,'reviewer-'+name,100,500) for name,bundle in bundles.items()}
with tempfile.TemporaryDirectory(prefix='modelops-lifecycle-') as work:
    root=Path(work); (root/'artifacts').mkdir(); write_json(root/'tombstones.json',[])
    for name,artifact in artifacts.items(): write_json(root/'artifacts'/f'{name}.json',artifact)
    cache={}
    # Establish initial A, then observe before activation, after B activation and after rollback to A.
    activate(root,'A',bundles['A'],approvals['A'],150,'cpu-demo','reviewer-A',[])
    before=consumer_predict(root,1.,cache)
    assert before==3.
    activate(root,'B',bundles['B'],approvals['B'],150,'cpu-demo','reviewer-B',[])
    after_b=consumer_predict(root,1.,cache)
    assert after_b==3.4
    rollback(root,150,'cpu-demo','operator-r',approvals,bundles,[])
    after_rollback=consumer_predict(root,1.,cache)
    assert after_rollback==3.
    # Rejected stale, expired and wrong-target activation has no pointer or prediction side effect.
    stable_pointer=(root/'active.json').read_bytes(); stable_prediction=consumer_predict(root,1.,cache)
    checks=[True]
    stale=dict(approvals['C'],bundle_hash=approvals['A']['bundle_hash'])
    expired=approval_for(bundles['C'],'reviewer-C',100,150)
    for approval,target in ((stale,'cpu-demo'),(expired,'cpu-demo'),(approvals['C'],'edge-demo')):
        try: activate(root,'C',bundles['C'],approval,150,target,'operator-x',[])
        except ValueError: checks.append(False)
        else: raise AssertionError('invalid activation accepted')
        assert (root/'active.json').read_bytes()==stable_pointer
        assert consumer_predict(root,1.,cache)==stable_prediction
    assert checks==[True,False,False,False]
    # A is now fallback and cannot be retired. Activate C, then reject B as fallback too.
    activate(root,'C',bundles['C'],approvals['C'],160,'cpu-demo','reviewer-C',[])
    assert consumer_predict(root,1.,cache)==4.
    # Negative control 1: the caller cannot name C while supplying B's valid approval.
    stable_pointer=(root/'active.json').read_bytes(); stable_cache=dict(cache)
    try: activate(root,'C',bundles['B'],approvals['B'],160,'cpu-demo','operator-x',[])
    except ValueError as error: assert 'version' in str(error)
    else: raise AssertionError('bundle B activated under caller-supplied version C')
    assert (root/'active.json').read_bytes()==stable_pointer and cache==stable_cache
    assert consumer_predict(root,1.,cache)==4.
    # Negative control 2: disk bytes changed under C fail a forced reload before prediction/cache write.
    artifact_path=root/'artifacts'/'C.json'; original_c_bytes=artifact_path.read_bytes()
    write_json(artifact_path,{'format':'linear-v1','slope':99.,'intercept':1.})
    stable_cache=dict(cache)
    try: consumer_predict(root,1.,cache,force_reload=True)
    except ValueError as error: assert 'do not match approved bundle' in str(error)
    else: raise AssertionError('changed C artifact was served')
    assert (root/'active.json').read_bytes()==stable_pointer and cache==stable_cache
    # A corrupted cache entry is checked against the selected artifact digest too.
    c_key=('C',digest(bundles['C']),bundles['C']['artifact_hash'])
    cache[c_key]={'format':'linear-v1','slope':99.,'intercept':1.}; stable_cache=dict(cache)
    try: consumer_predict(root,1.,cache)
    except ValueError as error: assert 'cached artifact' in str(error)
    else: raise AssertionError('changed cached artifact was served')
    assert (root/'active.json').read_bytes()==stable_pointer and cache==stable_cache
    cache[c_key]=artifacts['C']; artifact_path.write_bytes(original_c_bytes)
    try: retire(root,'A','operator-x',161,[],bundles)
    except ValueError: pass
    else: raise AssertionError('fallback retired')
    try: retire(root,'C','operator-x',161,[],bundles)
    except ValueError: pass
    else: raise AssertionError('active version retired')
    # With explicit fallback clearing, A/B become ineligible and can be tombstoned.
    clear_fallback(root,'operator-x',162)
    tombstones=retire(root,'A','operator-x',163,[],bundles)
    tombstones=retire(root,'B','operator-x',164,tombstones,bundles)
    assert set(tombstones)=={'A','B'}
    try: activate(root,'A',bundles['A'],approvals['A'],170,'cpu-demo','operator-x',tombstones)
    except ValueError as e: assert str(e)=='retired target'
    else: raise AssertionError('retired target reactivated')
    # Even cached artifacts cannot serve after tombstone; forced reload also cannot resurrect one.
    for reload in (False,True):
        try: consumer_predict(root,1.,cache,force_reload=reload) # active C remains eligible
        except ValueError: raise AssertionError('eligible active C rejected')
    # Select B via a rollback-shaped attempt is forbidden because tombstoned, no side effects.
    write_json(root/'active.json',{'current':'B','previous':'C','current_bundle_hash':digest(bundles['B']),'current_artifact_hash':bundles['B']['artifact_hash'],'previous_bundle_hash':digest(bundles['C']),'previous_artifact_hash':bundles['C']['artifact_hash']}) # simulate stale pointer discovered on reload
    for reload in (False,True):
        try: consumer_predict(root,1.,cache,force_reload=reload)
        except ValueError as e: assert str(e)=='selected artifact is retired'
        else: raise AssertionError('cache/reload served tombstoned artifact')
    write_json(root/'active.json',{'current':'C','previous':'B','current_bundle_hash':digest(bundles['C']),'current_artifact_hash':bundles['C']['artifact_hash'],'previous_bundle_hash':digest(bundles['B']),'previous_artifact_hash':bundles['B']['artifact_hash']})
    before_rollback_reject=(root/'active.json').read_bytes()
    try: rollback(root,170,'cpu-demo','operator-r',approvals,bundles,tombstones)
    except ValueError: pass
    else: raise AssertionError('rollback to retired fallback accepted')
    assert (root/'active.json').read_bytes()==before_rollback_reject
    audit=[json.loads(line) for line in (root/'audit.jsonl').read_text().splitlines()]
    assert all(event.get('actor') for event in audit)
    assert next(e for e in audit if e['action']=='retire' and e['version']=='A')['bundle_hash']==digest(bundles['A'])
    assert next(e for e in audit if e['action']=='activate' and e['version']=='B')['bundle_hash']==digest(bundles['B'])
    assert next(e for e in audit if e['action']=='activation-rejected' and e['version']=='C')['bundle_hash']==digest(bundles['C'])
    assert next(e for e in audit if e['action']=='rollback-rejected')['bundle_hash']==digest(bundles['B'])
print('consumer x=1 predictions: before B activation A=',before,'after B activation=',after_b,'after rollback=',after_rollback)
print('rejected stale/expired/wrong-target and mismatched-version activations preserved pointer/cache/output')
print('changed disk and cache artifacts rejected before prediction or cache mutation')
print('retirement: active/fallback rejected; A and B tombstoned after fallback clear:',sorted(tombstones))
print('retired target activation and rollback rejected; stale pointer denied with cache and forced reload')
print('audit events bind actor and exact bundle hash; local single-writer fixture, no authentication')

# Figure data experiment
visual_data={'kind':'bar','labels':['reviewed','stale approval','expired approval','wrong target'],'xlabel':'attempted transition','ylabel':'accepted (1) or rejected (0)','series':[{'label':'observed policy result','y':[int(v) for v in checks]}]}

# Experiment: Test the exact expiry boundary without mutating a registry
boundary_approval=approval_for(bundles['C'],'boundary-reviewer',100,200)
boundary_outcomes=[]
for now in (199,200,201):
    try:verify(bundles['C'],boundary_approval,now,'cpu-demo',[],expected_version='C')
    except ValueError as error:
        assert 'expired' in str(error)
        boundary_outcomes.append(False)
    else:boundary_outcomes.append(True)
assert boundary_outcomes==[True,False,False]
assert boundary_approval['expires_at']==200
print('Approval validity at 199, 200, 201:',boundary_outcomes)


# Reference solution. Try the exercise before reading this.
# Repeat the identity checks in fresh local state.
with tempfile.TemporaryDirectory(prefix='modelops-identity-exercise-') as folder:
    check_root=Path(folder); (check_root/'artifacts').mkdir()
    write_json(check_root/'tombstones.json',[])
    for name,artifact_value in artifacts.items():
        write_json(check_root/'artifacts'/f'{name}.json',artifact_value)
    check_cache={}
    activate(check_root,'C',bundles['C'],approvals['C'],150,'cpu-demo','reviewer-C',[])
    assert consumer_predict(check_root,1.,check_cache)==4.
    pointer_before=(check_root/'active.json').read_bytes(); cache_before=dict(check_cache)
    try: activate(check_root,'C',bundles['B'],approvals['B'],151,'cpu-demo','operator-x',[])
    except ValueError: pass
    else: raise AssertionError('bundle B activated as version C')
    assert (check_root/'active.json').read_bytes()==pointer_before and check_cache==cache_before
    artifact_path=check_root/'artifacts'/'C.json'; original=artifact_path.read_bytes()
    write_json(artifact_path,{'format':'linear-v1','slope':99.,'intercept':1.})
    try: consumer_predict(check_root,1.,check_cache,force_reload=True)
    except ValueError: pass
    else: raise AssertionError('changed C artifact served')
    assert (check_root/'active.json').read_bytes()==pointer_before and check_cache==cache_before
    artifact_path.write_bytes(original)
    c_key=('C',digest(bundles['C']),bundles['C']['artifact_hash'])
    check_cache[c_key]={'format':'linear-v1','slope':99.,'intercept':1.}
    corrupted_cache_before=dict(check_cache)
    try: consumer_predict(check_root,1.,check_cache)
    except ValueError as error: assert 'cached artifact' in str(error)
    else: raise AssertionError('changed cached C artifact served')
    assert (check_root/'active.json').read_bytes()==pointer_before
    assert check_cache==corrupted_cache_before
    check_cache[c_key]=artifacts['C']

print('Version mismatch, changed disk bytes and corrupted cache all rejected without pointer/cache mutation')

# Reference practice: Exercise approval expiry and retirement at a lifecycle boundary
# A distinct lifecycle transfer: exact expiry and active/fallback retirement guards.
with tempfile.TemporaryDirectory(prefix='modelops-lifecycle-transfer-') as folder:
    lifecycle_root=Path(folder); (lifecycle_root/'artifacts').mkdir()
    write_json(lifecycle_root/'tombstones.json',[])
    for name,value in artifacts.items():write_json(lifecycle_root/'artifacts'/f'{name}.json',value)
    activate(lifecycle_root,'A',bundles['A'],approvals['A'],150,'cpu-demo','learner',[])
    lifecycle_cache={}
    assert consumer_predict(lifecycle_root,1.,lifecycle_cache)==3.
    saved_pointer=(lifecycle_root/'active.json').read_bytes()
    try:activate(lifecycle_root,'B',bundles['B'],approvals['B'],500,'cpu-demo','learner',[])
    except ValueError as error:assert 'expired' in str(error)
    else:raise AssertionError('approval accepted at its exclusive expiry boundary')
    assert (lifecycle_root/'active.json').read_bytes()==saved_pointer
    assert consumer_predict(lifecycle_root,1.,lifecycle_cache)==3.
    activate(lifecycle_root,'B',bundles['B'],approvals['B'],160,'cpu-demo','learner',[])
    for protected in ('A','B'):
        saved_pointer=(lifecycle_root/'active.json').read_bytes()
        try:retire(lifecycle_root,protected,'learner',161,[],bundles)
        except ValueError as error:assert 'active or fallback' in str(error)
        else:raise AssertionError('protected version retired')
        assert (lifecycle_root/'active.json').read_bytes()==saved_pointer
    clear_fallback(lifecycle_root,'learner',162)
    retired=retire(lifecycle_root,'A','learner',163,[],bundles)
    saved_pointer=(lifecycle_root/'active.json').read_bytes()
    try:activate(lifecycle_root,'A',bundles['A'],approvals['A'],170,'cpu-demo','learner',retired)
    except ValueError as error:assert 'retired target' in str(error)
    else:raise AssertionError('retired A reactivated')
    assert (lifecycle_root/'active.json').read_bytes()==saved_pointer
    assert consumer_predict(lifecycle_root,1.,lifecycle_cache)==3.4
print('Exclusive expiry, protected retirement and tombstone checks passed; eligible B still serves 3.4')

print("PASS: operations-08")
