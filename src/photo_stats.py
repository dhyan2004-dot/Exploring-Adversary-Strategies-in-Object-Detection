"""Photo-level statistics: cluster bootstrap over source photos and a paired sign-flip test.
Instances of one photo are not independent, so whole photos are resampled / sign-flipped."""

import numpy as np

def _per_group_sums(groups,series):
    """groups: list of group ids (one per instance); series: {name: 0/1 array}."""
    ids=sorted(set(groups))
    index={g:i for i,g in enumerate(ids)}
    names=list(series)
    n=np.zeros(len(ids))
    sums=np.zeros((len(ids),len(names)))
    gi=np.array([index[g] for g in groups])
    np.add.at(n,gi,1.0)
    for j,name in enumerate(names):
        np.add.at(sums[:,j],gi,np.asarray(series[name],dtype=float))
    return ids,n,sums,names

def cluster_bootstrap(groups,series,n_boot=4000,seed=0,weighting="instance"):
    """Resample photos with replacement."""
    ids,n,sums,names=_per_group_sums(groups,series)
    G=len(ids)
    if G==0:
        raise ValueError("no instances to analyse")
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,G,size=(n_boot,G))
    if weighting=="photo":
        per=sums/n[:,None]
        rate=per.mean(axis=0)
        boot=per[idx].mean(axis=1) # [B, M]
    else:
        rate=sums.sum(axis=0)/n.sum()
        boot=sums[idx].sum(axis=1)/n[idx].sum(axis=1,keepdims=True) # [B, M]
    lo,hi=np.percentile(boot,[2.5,97.5],axis=0)
    out={"n_photos":G,"n_instances":int(n.sum()),"weighting":weighting,
           "rate":{k:float(rate[j]) for j,k in enumerate(names)},
           "ci95":{k:(float(lo[j]),float(hi[j])) for j,k in enumerate(names)}}
    if "adversarial" in names and len(names)>1:
        a=names.index("adversarial")
        ctrl=[j for j in range(len(names)) if j!=a]
        margin_boot=boot[:,a]-boot[:,ctrl].max(axis=1)
        margin=float(rate[a]-rate[ctrl].max())
        out["margin_over_best_control"]=margin
        out["margin_ci95"]=tuple(float(x) for x in np.percentile(margin_boot,[2.5,97.5]))
    return out

def paired_photo_test(groups,gone_a,gone_b,n_boot=4000,n_perm=20000,seed=0):
    """Paired comparison of two methods scored on the same instances."""
    ids,n,sums,_=_per_group_sums(groups,{"a":gone_a,"b":gone_b})
    d=sums[:,1]-sums[:,0] # per-photo summed difference
    G,N=len(ids),n.sum()
    obs=d.sum()/N
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,G,size=(n_boot,G))
    boot=d[idx].sum(axis=1)/n[idx].sum(axis=1)
    lo,hi=np.percentile(boot,[2.5,97.5])
    nz=d[d!=0]
    if len(nz)==0:
        p=1.0
    elif len(nz)<=16: # exact enumeration
        signs=np.array(np.meshgrid(*[[-1,1]]*len(nz))).reshape(len(nz),-1).T
        p=float((np.abs(signs@nz)/N>=abs(obs)-1e-12).mean())
    else:
        signs=rng.choice([-1.0,1.0],size=(n_perm,len(nz)))
        p=float((1+(np.abs(signs@nz)/N>=abs(obs)-1e-12).sum())/(n_perm+1))
    return {"n_photos":G,"diff":float(obs),"ci95":(float(lo),float(hi)),"p":p,
            "rate_a":float(sums[:,0].sum()/N),"rate_b":float(sums[:,1].sum()/N),
            "n_photos_differ":int(len(nz)),
            "min_possible_p":2.0/2**min(len(nz),30) if len(nz) else 1.0}
