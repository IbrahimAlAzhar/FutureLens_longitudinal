import numpy as np

def bootstrap(values,seed=42,reps=1000):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not len(a): return (np.nan,np.nan)
    rng=np.random.default_rng(seed)
    means=[rng.choice(a,len(a),replace=True).mean() for _ in range(reps)]
    return tuple(np.quantile(means,[.025,.975]))
