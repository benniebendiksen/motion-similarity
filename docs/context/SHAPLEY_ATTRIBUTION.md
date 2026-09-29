# Per-Part Shapley Attribution — the tool's math and its justification

The Phase-4 web tool colors each body part by **how much it contributes to the validated
whole-body perceptual distance** between the user's motion and a reference — NOT by a per-part
perceptual distance. This doc gives the exact math, why it's a Shapley value, and why the A1
finding (`MECHANISM_FINDINGS.md`) *validates* this framing rather than threatening it.

---

## 1. The reframing (what we claim, and what we do NOT)
- ✅ CLAIM: part P's color = its **share** of the whole-body distance D — "how much would changing
  this part reduce your dissimilarity to the reference."
- ❌ NOT CLAIMED: that P has its own monotone per-part *perceptual* distance, or that P is
  intrinsically "the perceptual part." (A1 shows no part is — joint signal is distributed.)

The whole-body distance D is the perceptually *validated* quantity (ρ vs human `d_perc`). The
per-part decomposition is an *attribution* of that single validated number, exact by construction.

## 2. The additive decomposition (why mean-pool makes this exact)
MAMP embeds a clip as a mean over patch tokens `feat[t,v]` (t=30 time, v=34 joints):
```
e = (1/1020) Σ_{t,v} feat[t,v] = (1/34) Σ_v z_v ,   z_v = (1/30) Σ_t feat[t,v]   (per-joint sub-emb)
```
Group joints into body parts P (arm, leg, torso, …):  `e = Σ_P w_P z_P`, `w_P = |P|/34`.
So the embedding is an **exact additive sum of per-part sub-embeddings** — the mean-pool is what
grants this (no such clean split for latent-query pooling; MAMP mean-pool is the enabling choice).

## 3. Squared distance → exact contribution shares
Let `u_P = w_P (z_P^user − z_P^ref)` be part P's contribution to the difference vector, and
`U = Σ_P u_P = e^user − e^ref`. Using squared distance `D² = ‖U‖²`:
```
D² = ⟨U, U⟩ = Σ_P ⟨u_P, U⟩   ⇒   φ_P := ⟨u_P, e^user − e^ref⟩
```
- **Efficiency (exact):** `Σ_P φ_P = ⟨U,U⟩ = D²`. The colors sum to the whole distance, no residual.
- **Closed form, no sampling:** φ_P is a single dot product per part.
- **Pairwise coupling:** `⟨u_P, u_Q⟩` decomposes how parts P and Q's deviations interact
  (align → reinforce; oppose → cancel).

## 4. Why φ_P IS the Shapley value (not just a convenient split)
For the coalitional game `v(S) = ‖Σ_{P∈S} u_P‖²` (distance achievable using only parts in S), the
Shapley value of P is:
```
Shap_P = ‖u_P‖² + ⟨u_P, Σ_{Q≠P} u_Q⟩ = ⟨u_P, U⟩ = φ_P
```
(marginal of P to a random-order predecessor set has E[predecessors] = ½Σ_{Q≠P}u_Q; the ½ and the
2 from the quadratic cancel). So for the squared-distance game the Shapley value has this exact
closed form — we get Shapley's fairness axioms (efficiency, symmetry, null-player, linearity) for
free, with no Monte-Carlo estimation.

## 5. Why A1 VALIDATES this (the key link)
A1 found **no intrinsic per-joint perceptual specialization** (flat, action-invariant joint
saliency; joint selectivity ~0/negative). If we had claimed "part P has its own perceptual
distance," A1 would contradict us. But φ_P is **per-comparison**: it depends on `u_P` = *this*
user-vs-reference pair's deviation. So even with zero intrinsic specialization, φ_P varies
comparison-to-comparison — for a given pair, whichever parts deviate most *along the embedding's
aligned directions* carry the most contribution, and they always sum to D. A1 therefore **kills the
wrong framing and confirms the right one.**

## 6. Honest boundaries (carry into any writeup / reviewer response)
- **Entanglement:** z_P is arm-*located*, not arm-*exclusive* — global attention means P's
  sub-embedding is influenced by the whole body. φ_P is valid for ATTRIBUTION/coloring ("modify
  here to reduce D"), NOT for isolation-search ("this sub-embedding is a pure arm signal"). Do not
  build stimuli by searching z_P; construct from RAW per-part kinematics (non-circular), let the
  model validate.
- **Squared vs raw distance:** the exact decomposition is for D² (squared). Report/color on D²
  shares, or note the monotone map to D. ρ is validated on the distance; the tool attributes it.
- **Sign:** φ_P ≥ 0 is not guaranteed per-part for arbitrary geometry, but Σφ_P = D² ≥ 0. Coupling
  terms can be negative (parts cancelling) — surface this rather than clip it.

## 7. Phase-3 study's role
The 20-clip triplet study ENRICHES/validates the whole-body D (broader coverage or one part deep,
~190 pairs), strengthening the single validated number the shares decompose. It does NOT need to
prove per-part perception — see `PIPELINE_TRACKER.md` Phase 3.
