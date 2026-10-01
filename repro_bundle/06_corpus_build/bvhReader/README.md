# bvh-python
Python utilities for reading, writing, and visualizing BVH files 

## Getting Started

Runs with Python 3 and requires the dependencies [PyGLM](https://pypi.org/project/PyGLM/), [matplotlib](https://matplotlib.org/), and [NumPy](https://numpy.org/).

```
python3 -m pip install --upgrade pip
python3 -m pip install PyGLM
python3 -m pip install numpy
python3 -m pip install matplotlib
```

To visualize a BVH file, use the following code

```
animation = BVH()
animation.load(bvhfilename)
BVHAnimator(animation)
```

See `example.py` for an example. 

![samba](https://github.com/alinen/bvh-python/assets/259657/62856bdc-c08b-420f-ab06-a8af99ffd61e)

---

## Project-specific notes

Adopted into this repo from [alinen/bvh-python](https://github.com/alinen/bvh-python)
and extended for our retargeting + VAE workflow.

### Files

| File | Purpose |
|---|---|
| `bvh.py` | `BVH` and `Joint` classes. Loads / saves BVH, exposes per-joint local/global rotations + positions, knows Euler ↔ quaternion ↔ matrix ↔ 6D conversions. `setLocalRot6D` and `setLocalPos` are the inverse-direction setters used by `autoencoders/features_to_bvh.py`. |
| `retarget.py` | Motion retargeting between skeletons. Used during AMASS/Fit3D/H36M → CMU-33 conversion. |
| `rotation_conversions.py` | Pure-tensor rotation utilities, torch-friendly. |
| `bvh2csv.py`, `bvhConverterToPerform.py`, `bvhvisualize.py`, `example.py` | Legacy / one-off utilities. |

### Retargeting `rest_delta` gotcha

`align_limbs` in `retarget.py` computes a per-joint `rest_delta` rotation —
the structural offset between the source skeleton's T-pose and the target
skeleton's T-pose. An earlier version read this from frame 0 of the
*animation* (not the rest pose), which baked the source clip's frame-0
posture into the target output forever (visible as "arms wide open" for KIT
clips). The current implementation uses `find_rotation_tpose` over the
HIERARCHY offsets — purely structural.

If you regenerate the retargets, expect a per-subset rerun decision; see
`amass/qc_nonkit_samples.py` for the QC script we used to choose which
AMASS subsets to rerun on the fixed pipeline.

