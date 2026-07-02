"""CANONICAL §3 kinematic descriptors — documented, reproducible recipe.
Data: similarity_exemplars/{action}_similarity_labels_exemplars_dict_local.pickle (T x 112 = 28 joints x 4 quat).
Recipe (fixed + documented):
 - Per action, per effort-class clip.
 - Per-joint per-frame quaternion (4 comps). Across-frame std computed per component, then the
   per-joint temporal activity = L2 norm of the 4 per-comp stds (a single scalar per joint).
 - ACTIVE JOINT = temporal activity > 0.01 (raw quaternion units; no standardization -- quaternions
   are already unit-normalized, so raw units are the natural, interpretable scale).
 - RANGE OF MOTION per joint = L2 norm over the 4 comps of (max_t - min_t); mean over 28 joints.
 - Report mean over the 57 effort classes per action.
"""
import pickle, numpy as np
SRC="/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/exemplars_dir/similarity_exemplars"
def stats(action, thr=0.01):
    d=pickle.load(open(f"{SRC}/{action}_similarity_labels_exemplars_dict_local.pickle","rb"))
    A,Ts,R=[],[],[]
    for k,v in d.items():
        v=v[0] if isinstance(v,list) else v; v=np.asarray(v,float)
        T=v.shape[0]; vj=v.reshape(T,28,4)
        jact=np.linalg.norm(vj.std(axis=0),axis=-1)      # per-joint temporal activity (28,)
        A.append(int((jact>thr).sum()))
        Ts.append(float(jact.mean()))
        R.append(float(np.linalg.norm(vj.max(0)-vj.min(0),axis=-1).mean()))
    return len(d), np.mean(A), np.mean(Ts), np.mean(R)
print("action  | n_classes | active(of28) | frac  | temporal-activity | ROM")
for a in ["walking","pointing","picking"]:
    n,act,ts,rom=stats(a)
    print(f"{a:8s}|    {n}     |   {act:.1f}       | {act/28:.2f}  |      {ts:.3f}       | {rom:.3f}")
