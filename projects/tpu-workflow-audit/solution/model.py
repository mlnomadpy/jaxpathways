"""Reference implementation: Validate a staged multi-worker run plan and gate publication on every worker receipt."""


def workflow_report(worker_ids, completed_ids, checkpoint_steps):
    workers=list(worker_ids); done=list(completed_ids); steps=list(checkpoint_steps)
    if not workers or len(workers)!=len(set(workers)): raise ValueError("worker ids must be nonempty and unique")
    if len(done)!=len(set(done)): raise ValueError("completed worker receipts must be unique")
    if set(done)-set(workers): raise ValueError("completed receipt names an unknown worker")
    if len(steps)!=len(workers): raise ValueError("one checkpoint step per worker is required")
    if any(not isinstance(step,int) or isinstance(step,bool) or step<0 for step in steps): raise ValueError("checkpoint steps must be nonnegative integers")
    missing=sorted(set(workers)-set(done)); extras=sorted(set(done)-set(workers))
    return {"workers":workers,"missing":missing,"unexpected":extras,"publishable":not missing and not extras and len(set(steps))==1,"common_step":steps[0] if len(set(steps))==1 else None}
