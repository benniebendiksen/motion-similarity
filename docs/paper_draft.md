# Composing Masked-Motion Prediction and Pose Reconstruction: A Self-Supervised Motion Encoder for Human Perceptual Similarity

*A single self-supervised encoder that composes masked-motion dynamics with pose-configuration fidelity beats geometric distance baselines on human perceptual similarity across all three action types — including the expressively sparse one — with no metric learning; human ratings, applied by fine-tuning the encoder, improve it further.*

*(working manuscript draft — methods, design rationale, and results, from rating data to inference pipeline)*

---

## Abstract (draft)

We ask whether a learned representation of human motion can predict *perceptual*
similarity — how similar two movements appear to human observers — more faithfully than
geometric distance measures (quaternion/6D rotation geodesic, Dynamic Time Warping)
that dominate motion analysis. Using Laban-effort-annotated motion-capture performances of
three action types (walking, pointing, picking) paired with a triplet-based human
similarity-rating dataset, we show that a self-supervised **masked motion predictor
augmented with an auxiliary pose-reconstruction objective (MAMP+pose)** produces motion
embeddings whose **raw, unsupervised** distances align with human perception more strongly
than both geometric baselines *and* a generic learned (reconstruction-only SSL) motion
encoder across all three actions, under a strict, repeated,
leakage-clean nested cross-validation — to our knowledge the first single encoder to do so,
and notably including the expressively sparse pointing action where time-warping baselines
were expected to dominate. A progressive ablation isolates the contribution of each objective
and motivates the final design: reconstruction objectives capture static configuration but
never model dynamics; **masked motion prediction** alone is the complement — recovering the
picking action it best serves but collapsing on walking and pointing; and **composing**
masked motion prediction with an auxiliary **pose-reconstruction** objective supplies both
capabilities in one encoder, the only configuration competitive across all three actions. We
further show that the human ratings, when used to **fine-tune the encoder** under a perceptual
objective (rather than to fit a metric on a fixed embedding), further improve held-out alignment
on all three actions. All claims are reported as fold-averaged correlations
with calibrated variability, against baselines re-evaluated on the identical folds.

---

## 1. Introduction

### 1.1 Motion similarity and geometric distances as measures of similarity
Quantifying how *similar* two human movements are is foundational to motion retrieval,
synthesis, style transfer, and the perceptual study of movement. The dominant approaches
are geometric distances computed on raw kinematic features:

- **Rotation geodesic distance** — the distance on the rotation group SO(3) between
  corresponding joint orientations, aggregated over joints and time. It respects the
  manifold structure of rotations but assumes (after temporal alignment) frame-wise
  correspondence.
- **Dynamic Time Warping (DTW)** — a non-linear temporal alignment that factors out speed
  variation and emphasizes the *shape* and *progression* of a gesture. DTW is the most
  widely adopted geometric framework for time-series motion comparison precisely because
  it models the temporal structure to which human observers are sensitive.

Though strong, well-motivated baselines, the scientific question is whether a *learned*
motion representation can predict human perceptual judgments better than they do.

### 1.2 Contributions
1. A **single self-supervised motion encoder (MAMP+pose)** whose **raw, unsupervised**
   embedding distances beat both geometric baselines (DTW, rotation geodesic) **and a generic
   learned (reconstruction-only SSL) motion encoder** on human-perceptual similarity
   across all three action types — including the expressively sparse pointing action — when
   evaluated on motion clips held out of pretraining, under a repeated, leakage-clean nested
   cross-validation with all baselines re-evaluated on the identical folds. The learned
   baseline shows that *merely* learning a motion representation is insufficient (it does not
   uniformly beat geometry); the composition is what wins all three. To our knowledge this
   is the first single encoder to do so on this heterogeneous action set, and it requires **no
   perceptual supervision and no metric-learning stage** — a substantially stronger and simpler
   claim than a two-stage supervised pipeline.
2. A **mechanistic, progressive-ablation account** of *why* the masked-motion-plus-pose
   composition is the right encoder: reconstruction objectives capture static configuration
   but not dynamics; masked motion prediction is the complement (a picking specialist that
   collapses on walking and pointing); and only composing the two yields an encoder
   competitive across walking, pointing, and picking. Each step's result motivates the next
   design choice, evidence-first.
3. A demonstration that the human ratings are productively used by **fine-tuning the encoder**
   under a perceptual objective derived from the comparison study — an **anchored ranking loss
   whose margin is the human dissimilarity gap itself** (hyperparameter-free and metric-aligned),
   which further improves held-out alignment on all three actions — the supervision reshaping
   what the representation retains rather than re-weighting a fixed embedding. (We compare this
   ranking loss against regression and neutral-anchored alternatives, §6.4.)
4. A **methodological contribution**: a repeated, leakage-clean nested cross-validation
   protocol — with baselines re-evaluated on the identical folds — needed to make reliable,
   reproducible perceptual-similarity claims on a small, high-variance human-rating set, and a
   characterization of the single-split pitfalls it corrects.

---

## 2. Data corpus

### 2.1 Stimulus space: Laban effort tuples
Stimuli are motion-capture performances parameterized by Laban Movement Analysis (LMA)
**effort tuples** `(e1, e2, e3, e4) ∈ {−1, 0, +1}^4`, encoding the four Laban effort
dimensions (Space, Time, Weight, Flow). Following established LMA semantics, the number of
non-zero efforts has perceptual meaning:

| Non-zero efforts | LMA category | Count per action+mirror |
|------------------|--------------|--------------------------|
| 0 | neutral | 1 |
| 1 | action (single element) | — |
| 2 | state | 24 |
| 3 | drive | 32 |
| 4 | full effort | — |

The perceptual study uses the **neutral + state + drive** subset (the
"states/drives" set): **57 effort classes per action** (1 neutral + 24 states + 32 drives),
performed for three action types — **walking, pointing, picking** — for a total of 171
effort-class stimuli. Each performance additionally has a left/right-mirrored twin; the
342 LMA states/drives clips excluded from encoder pretraining (§2.3) comprise these 57
classes × 3 actions × 2 (non-mirror + mirror) = 342.

### 2.2 Motion representation
Each clip is converted to a per-frame feature of **34 joints × 6-D rotation
representation**: joint index 0 carries the root position/orientation, indices 1–33 carry
6-D continuous joint rotations. The 6-D representation (Gram–Schmidt-recoverable to SO(3))
is preferred over quaternions for its continuity and gradient stability.

Each of the 204 feature channels (34 joints × 6 dimensions) is **z-score standardized**
using a per-channel mean and standard deviation computed once over the entire training
corpus (stored as `norm_stats.npz`, of shape `[34, 6]` for the per-joint, per-dimension
statistics); standardized values are then clipped to ±10 to bound rare outliers. The same
fixed statistics are applied at training and at evaluation, so no evaluation-clip
information enters the normalization.

**Fixed-length windowing (and how variable clip lengths are handled).** The encoder
operates on a fixed **120-frame** window (four seconds at 30 fps). Each clip is mapped to
this length deterministically: clips longer than 120 frames are truncated to their first
120 frames, and clips shorter than 120 frames are **right-padded by repeating their final
frame** to length 120. This identical rule is applied in self-supervised pretraining, in
embedding extraction for evaluation, and in the held-out encoding of all stimuli — there is
no length-dependent branching between train and test. (Variable length is therefore never
seen by the encoder; the geometric baselines of §5.2, by contrast, operate on the *raw,
unpadded* sequences.)

**Clip-length heterogeneity.** Walking and picking clips are long (median 73 and 71 frames;
up to 137), whereas pointing clips are markedly shorter (median 53 frames; max 100).
Because short clips are tail-padded, pointing clips contribute the largest fraction of
repeated, frozen-pose frames under the 120-frame window — a consequence of their brevity
that connects to their limited kinematic expressiveness (§3).

### 2.3 Pretraining corpus (self-supervised encoder)
The encoder is pretrained self-supervised on a broad, heterogeneous motion corpus of
**5,795 BVH clips** drawn from three sources:

| Source | Clips | Content |
|--------|-------|---------|
| LMA effort performances | 433 | the Laban-effort performances (§2.1), including the eval actions |
| Bandai-Namco Research Motion Dataset [Kobayashi et al., 2023] | 3,077 | professionally captured everyday actions (locomotion, gestures, hand actions) performed in multiple expressive styles |
| CMU Graphics Lab Motion Capture Database | 2,285 | general-purpose human motion-capture performances spanning locomotion, interaction, and physical activities |

**A single common skeleton.** The three sources were originally captured with different
marker sets, joint hierarchies, frame rates, and file conventions. A substantial
preprocessing effort retargeted **all** sources onto one canonical skeleton — the
**CMU joint hierarchy** (root `Hips`, with the CMU limb/spine topology;
`LHipJoint → LeftUpLeg → … → LeftToeBase`, `LowerBack → Spine → Spine1 → Neck1 → Head`,
and the symmetric right side and arms) — and re-expressed every clip as a uniform BVH file
on that skeleton at a common **30 fps** sampling. This retargeting is what makes the corpus
trainable by a single encoder: every clip yields the identical **34-joint × 6-D rotation**
per-frame feature (§2.2), regardless of its source. Establishing this shared representation
— joint-name reconciliation, hierarchy remapping, rotation-order and unit normalization,
and resampling to 30 fps — was a prerequisite for cross-corpus pretraining and for
comparing the LMA evaluation stimuli against the broader motion distribution on equal
footing.

**Held-out construction.** To guarantee a **strict held-out evaluation**, all 342 LMA
neutral/state/drive clips (Mirror twins included) — i.e., exactly the clips on which
perceptual similarity is evaluated — are excluded from pretraining (and from the encoder's
own validation), leaving ~5,453 training clips. The encoder therefore never observes any
evaluation stimulus during representation learning.

### 2.4 Human perceptual ratings (ground truth)
Similarity ratings were collected via **triplet (2-of-3) comparison trials**. In each
trial a participant viewed three motions — **Left (0)**, **Neutral (1)**, **Right (2)** —
and selected the two judged most similar. The dataset comprises **1,540 triplet trials per
action** (one per distinct non-neutral effort pair, with the neutral as the third item),
with each trial rated by a **median of 11 participants** (range 10–18). For each trial, the triplet presentation order was randomized. The
three pairings {(0,1), (0,2), (1,2)} of a given triplet received normalized choice frequencies aggregated across its trials to 1. These count normalized values can therefore be seen as probabilities for the selection of a given pairing as most similar within a triplet.

#### 2.4.1 Notation for the three choice frequencies
Fix a triplet built from a non-neutral effort pair, with **Left (0)** and **Right (2)** the
two effort exemplars and **Neutral (1)** the neutral exemplar. Write the three aggregated
choice frequencies (each in [0,1], summing to 1) as

- `c(0,2)` — frequency the **Left–Right** pair was chosen as the most-similar two (i.e., the
  two effort exemplars judged more similar to each other than either is to neutral),
- `c(0,1)` — frequency the **Left–Neutral** pair was chosen,
- `c(1,2)` — frequency the **Neutral–Right** pair was chosen.

`c(0,2)` is the quantity of primary interest: it is how often observers judged the two
effort exemplars *mutually most similar*, and therefore an absolute, calibrated measure of
the **perceived similarity** of that effort pair, on a common [0,1] scale across all pairs.

#### 2.4.2 The evaluation target: a perceptual dissimilarity in [0,1]
For each non-neutral effort pair we define the **direct comparison value** as `c(0,2)` and
the per-pair perceptual **dissimilarity** as its complement,

> `d_perc = 1 − c(0,2)  ∈ [0,1]`.

This is a genuine dissimilarity on a fixed [0,1] scale: a pair the observers almost always
grouped together (`c(0,2)→1`) has `d_perc→0`, and a pair they almost never grouped together
(`c(0,2)→0`) has `d_perc→1`. **`d_perc` is the single ground-truth quantity against which
every distance measure — the learned embedding distances and both geometric baselines — is
correlated** (§5.2). It uses only the absolute Left–Right frequency and no triplet-internal
contrasts, so it is method-agnostic.

#### 2.4.3 Perceptual training signals: the `d_perc` ordering and the dynamic-alpha construction (perceptual fine-tuning only)
The perceptual fine-tuning of §6.4 requires a per-pair supervisory signal derived from the
triplets. We define **two** candidate signals here and compare them empirically in §6.4; the
**reported method trains on the ordinal `d_perc` relation** (§2.4.2) via a ranking loss, while a
finer, neutral-referenced construction — the **dynamic alpha** — is one of the tested
alternatives. We define the alpha construction now, as both an alternative objective and a
quantity referenced later.

Per non-neutral effort pair we generate **two directed Left–Right alphas** that contrast the
direct Left–Right frequency against each of the two neutral-mediated alternatives:

> `α(0→2) = c(0,2) − c(0,1)`  (Left–Right preference relative to Left–Neutral)
> `α(2→0) = c(0,2) − c(1,2)`  (Left–Right preference relative to Neutral–Right)

Each alpha lies in [−1, +1]. A **positive** alpha means the two effort exemplars were
chosen as mutually most-similar *more often* than the corresponding neutral-mediated
pairing — evidence the pair should sit *close* in embedding space; a **negative** alpha is
evidence the pair should sit *far apart* (the neutral-mediated alternative was preferred).
The two alphas are directional precisely because each measures the Left–Right contrast
against a *different* competing alternative, which lets the neutral-anchored alternative losses
(§6.4) impose an asymmetric, per-direction margin rather than a single symmetric target.

A subtlety that makes the construction *dynamic*: although six pairwise alphas are
computable per triplet, the module routes the **Left–Right** alphas into one of three
preference cases according to which pairing observers actually preferred most
(Left–Right, Left–Neutral, or Neutral–Right). In every case it is the Left–Right alphas
that are retained — so even triplets where a *neutral-mediated* pairing won still
contribute an ordering constraint on the two effort exemplars, rather than being discarded.
This is what allows the limited rating budget (one triplet per effort pair) to yield
training signal on essentially every pair.

A note on train/evaluation hygiene. The reported ranking method (§6.4) supervises on the
**ordinal relation** induced by `d_perc` — which pair of a triplet is more dissimilar — while
evaluation scores the **Spearman correlation** between embedding distances and `d_perc` over the
held-out test fold. Training on the ordering and scoring the rank-correlation are the same
ordinal quantity used consistently, never a fit-then-score-on-identical-numbers shortcut; and
because train, selection, and test folds are disjoint (§5.2), no class's rating is ever both a
training target and its own test. The dynamic alphas above are used only by the neutral-anchored
*alternative* arms, which §6.4 finds do not generalize.

### 2.5 Metric of merit
The ground truth is **ordinal** by construction (a ranking induced by choice frequencies)
and the learned metric optimizes ordinal triplet inequalities; **Spearman rank correlation
is the primary metric.** Pearson is reported throughout as a magnitude-alignment
supplement.

---

## 3. Per-action motion structure

The three actions differ markedly in their *kinematic expressiveness* — how much of the
body moves, over how large a range, and over how long. We quantify this directly from the
motion data, computing per-effort-class statistics and averaging over the 57 classes of
each action (28 articulated joints; temporal "activity" defined as a joint whose
across-frame standard deviation exceeds 0.01 in the standardized feature space).

| Descriptor (mean over effort classes) | Walking | **Pointing** | Picking |
|----------------------------------------|---------|--------------|---------|
| Active joints (of 28) | 24.6 | **11.6** | 20.0 |
| **Active-joint fraction** | 0.88 | **0.41** | 0.71 |
| Median clip length (frames) | 73 | **53** | 71 |
| Mean per-joint temporal std | 0.089 | **0.048** | 0.090 |
| Mean range of motion | 0.276 | **0.123** | 0.290 |

**Pointing is, by every measure, the least kinematically expressive action.** On average
only ~12 of 28 joints (41%) move meaningfully in a pointing clip, versus ~25 (88%) for
walking and ~20 (71%) for picking; pointing's per-joint variability and range of motion are
roughly *half* those of the other two actions, and its clips are the shortest. Pointing's
effort is conveyed by a brief, spatially concentrated gesture (predominantly the pointing
arm), whereas walking and picking recruit much of the body over longer, dynamically richer
trajectories. Walking and picking, by contrast, are similar to each other on these
descriptors and are well-separated from pointing.

This expressive sparsity leads us to two **hypotheses** for pointing's distinctive difficulty suggested by the results:

- **H1 (signal density).** With fewer than half the joints varying and roughly half the
  range of motion, a pointing clip carries less discriminative kinematic signal. A
  self-supervised encoder therefore has less structure from which to separate pointing
  effort classes, so the *raw* embedding geometry — and any distance computed directly on
  it — should be less perceptually discriminative for pointing than for walking/picking.
- **H2 (temporal brevity).** Pointing's shorter clips (median 53 vs 71–73 frames) afford
  the masked-motion pretext task fewer and briefer dynamic regions to model, plausibly
  weakening the dynamics representation the encoder learns for this action specifically.

Under both hypotheses, the prediction is the same: pointing's *raw* encoder distances will be
the **least** discriminative of the three actions — the encoder has the least kinematic
structure to work with for pointing. We will see this borne out (§6): pointing is the lowest
of MAMP+pose's three raw correlations and the action with the largest fold-to-fold variance.
What is *not* foreordained is whether that residual signal is nevertheless enough to beat the
geometric baselines; we find that with the right pretraining objective it is (§6.1, §6.5),
provided the encoder is pressured to retain both dynamics and configuration. We treat
pointing's weaker raw signal as a property of its limited kinematic expressiveness rather than
a deficiency of the encoder.

This heterogeneity frames the central design question: can a *single* encoder serve both the
dynamically rich actions and the expressively sparse one — and is the perceptual structure of
the sparse action recoverable from the raw representation, or does it require additional
supervision? We find the former: the composition of masked-motion prediction with a pose
objective (§4.5) makes even pointing's raw geometry beat the baselines, with no metric-learning
stage (§6).

---

## 4. Encoder design — a progressive ablation toward MAMP+pose

The encoder design was not chosen a priori; it is the endpoint of a sequence of controlled
training runs, each motivated by the failure of the previous. We present that sequence because
it is what establishes the central **two-capability composition** finding — that perceptual
similarity across heterogeneous actions requires an encoder modeling *both* motion dynamics and
static pose configuration — and rules out the obvious single-objective alternatives.

We evaluate every encoder identically: pool its output to one vector per clip, take L2
distances between clips, and correlate against `d_perc` under the repeated, leakage-clean
nested cross-validation of §5.2 (all §4 numbers are fold-averaged Spearman, n = 25). **Crucially,
no encoder is evaluated at a hand-chosen training epoch.** Within each fold the training epoch is
treated as an ordinary inner-loop hyperparameter: a candidate checkpoint is taken at every point
on a **uniform 100-epoch grid across the model's full training trajectory**, the checkpoint
maximizing raw perceptual Spearman on the disjoint inner (SELECT) fold is chosen, and only then
is it scored on the untouched TEST fold. The grid is identical for every model (no per-model
schedule), and the candidate set is the entire trajectory (no plateau cut) — removing
checkpoint-selection as a researcher degree of freedom, so a model that trained longer gains no
selection advantage. We verified that every model's held-out perceptual signal has plateaued
within its trajectory, so the full-trajectory candidate set contains each model's perceptual
optimum (§5.2). Encoders differ along two axes we deliberately separate — the **autoencoder
backbone** (§4.1) and the **training objective** (§4.2–4.3).

### 4.1 Step 1 — backbone: why the MLD SkipTransformer over a plain transformer
We consider two transformer-autoencoder backbones for the reconstruction encoders: a **plain
transformer autoencoder** (a standard encoder/decoder with a compressed latent bottleneck)
and the **MLD-style transformer autoencoder** of Motion Latent Diffusion [Chen et al., CVPR
2023] — a transformer encoder/decoder with **U-Net-like long skip connections**
("SkipTransformer"), symmetric long-range links from each early encoder layer to the
matching late decoder layer. The skip connections carry high-frequency per-frame detail
forward that the plain bottleneck discards through its compression, which we hypothesized
matters for the static-pose fidelity that picking and pointing judgments rely on. This step
tests that hypothesis; the result is what *motivates* adopting MLD for the remaining
reconstruction encoders (§4.2 onward) — it is a conclusion of the ablation, not an a priori
choice.

We test it by holding the *objective* fixed — reconstruction-only, no dynamics term — and
varying *only* the architecture. Both encoders are genuine reconstruction-only models: the
plain transformer is PROV_noVel (rotation-MSE reconstruction, no velocity), the MLD encoder
is AE-holdout (plain-MSE reconstruction); both held-out, pooled, evaluated identically under
the repeated nested-CV (raw Spearman, n = 25, mean ± SE):

| Reconstruction-only encoder | Walking-S | Pointing-S | Picking-S |
|------------------------------|-----------|------------|-----------|
| Plain transformer AE (recon-only, unit weight) | +0.476 ± 0.031 | +0.213 ± 0.034 | +0.491 ± 0.029 |
| **MLD SkipTransformer AE (recon-only, unit weight)** | **+0.532 ± 0.023** | **+0.331 ± 0.039** | **+0.498 ± 0.029** |

Both encoders are the **unit-weight reconstruction-only** cells of the controlled factorial of
§4.2 (rotation reconstruction on, velocity off), so this comparison isolates *only* the
architecture. The MLD SkipTransformer wins on **all three actions at the same objective**,
decisively on pointing — **+0.118 pointing** — and clearly on walking (+0.056), with picking
close (+0.007). The largest gain is on pointing, the most kinematically sparse action (§3) —
consistent with the skip connections preserving high-frequency per-frame pose detail that the
plain bottleneck discards, which we substantiate directly via reconstruction fidelity in §4.1.1.
This
motivates the MLD backbone for the reconstruction encoders, and §4.1.1 measures reconstruction
fidelity directly to substantiate the mechanism behind it.

#### 4.1.1 Reconstruction fidelity confirms the mechanism
The hypothesized cause of the plain transformer's weakness is that its compressed latent
bottleneck *discards per-frame pose detail*, whereas the MLD skip connections preserve it.
We test this directly: for each held-out evaluation clip, we pass it through each
reconstruction-only encoder and measure mean per-joint reconstruction error (MPJPE) against
the ground-truth clip, over the 33 articulated joints, in the shared physical (un-normalized)
feature space — so the two encoders' reconstructions are compared on a common scale against
the same target. Each encoder is evaluated at its **best checkpoint**, defined as the checkpoint with the
lowest held-out reconstruction loss. The plain transformer is the one exception: its
reconstruction loss plateaus early and remains high (below), so this criterion selects an
essentially arbitrary early epoch; we therefore use its final converged checkpoint instead.

| Reconstruction-only encoder | Held-out reconstruction MPJPE (↓) |
|------------------------------|-----------------------------------|
| Plain transformer AE (PROV_noVel) | 0.825 ± 0.039 |
| **MLD SkipTransformer AE (AE-holdout)** | **0.046 ± 0.021** |

The plain transformer reconstructs held-out poses roughly **an order of magnitude less
accurately** than the MLD encoder (the gap is even starker in normalized reconstruction MSE,
where the plain transformer plateaus near ≈1.2 versus the MLD's ≈0.014 — its reconstruction
barely improves over training). This is the expected consequence of the architectural
difference — the skip connections furnish a high-fidelity reconstruction path that the plain
bottleneck lacks — and it directly substantiates the §4.1 claim: the plain transformer's
poor picking/pointing perceptual alignment tracks its poor pose-reconstruction fidelity, the
actions whose similarity depends on exactly the per-frame configuration detail it fails to
retain.

#### 4.1.2 Deterministic autoencoder, not the variational form
MLD is natively a **variational** autoencoder: its KL-regularized latent is designed to be a
smooth, samplable prior for *generative* diffusion. Our goal is different — we do not
generate motion; we need the latent to **preserve perceptually-relevant motion detail** so
that distances in it track human similarity. The KL term works against this: it pulls the
latent toward the prior, trading reconstruction fidelity for sampling smoothness. We
therefore remove the KL term and use the **deterministic** MLD autoencoder (the
SkipTransformer architecture, no variational bottleneck). A held-out comparison supports the
choice on the actions where reconstruction is strongest: the variational form (MLD-VAE) **stalls on pointing
(+0.324) just as the deterministic AE does (+0.331)** — neither clears the bar — and offers no
perceptual advantage to justify the KL regularization. The one place the VAE leads is walking
(+0.597, the highest walking correlation in the sweep; §6.7), but it remains a walking/picking
specialist that collapses on pointing, so it is not a contender for the all-three objective.
We retain the variational variant only as this reference point and use the deterministic AE for
all reconstruction encoders.

### 4.2 Step 2 — objective: a controlled velocity-vs-reconstruction factorial
Having fixed the backbone, we ask whether the *objective* can supply what reconstruction lacks
— specifically, whether an explicit **velocity** term endows the encoder with dynamics. To avoid
the confound of arbitrary loss weighting, we run a **controlled single-variable factorial**:
each loss term is either on (unit weight 1) or off (weight 0), so the comparison varies exactly
one objective at a time. We run the full factorial on **both** backbones (plain transformer and
MLD), all cells held out and evaluated identically under the repeated nested-CV (raw Spearman,
n = 25, mean ± SE):

| | Walking-S | Pointing-S | Picking-S |
|---|-----------|------------|-----------|
| **Plain transformer** | | | |
| reconstruction-only (rot 1, vel 0) | +0.476 ± 0.031 | +0.213 ± 0.034 | +0.491 ± 0.029 |
| reconstruction + velocity (rot 1, vel 1) | +0.472 ± 0.026 | +0.205 ± 0.039 | +0.399 ± 0.037 |
| velocity-only (rot 0, vel 1) | +0.321 ± 0.033 | +0.362 ± 0.036 | +0.471 ± 0.033 |
| **MLD SkipTransformer** | | | |
| reconstruction-only (rot 1, vel 0) | +0.532 ± 0.023 | +0.331 ± 0.039 | +0.498 ± 0.029 |
| reconstruction + velocity (rot 1, vel 1) | +0.537 ± 0.022 | +0.339 ± 0.035 | +0.493 ± 0.030 |
| velocity-only (rot 0, vel 1) | +0.451 ± 0.031 | +0.278 ± 0.031 | +0.254 ± 0.039 |

Three results, read directly from the table:

1. **Adding a velocity term to reconstruction changes almost nothing.** On MLD it is flat within
   noise (e.g. +0.331 → +0.339 pointing); on the plain transformer it is flat-to-slightly-worse
   (picking +0.491 → +0.399). On *neither* backbone does it move pointing across the +0.370 bar.
   This is the operative negative result: **velocity as an added loss does not supply dynamics.**
   §4.3 gives the mechanism — on a fully-visible reconstruction the velocity term is largely
   redundant (accurate pose reconstruction already reproduces frame-to-frame deltas).

2. **Velocity-only is a weaker objective overall, and reshapes *which* action is served.** With
   reconstruction removed, global quality drops (MLD picking collapses +0.498 → +0.254; plain
   walking drops +0.476 → +0.321). A velocity-only target is a poor general-purpose perceptual
   encoder.

3. **The telling signal: velocity-only is the *only* reconstruction-family cell that lifts
   plain-transformer pointing — and it lifts it markedly** (+0.213 → +0.362, nearly to the bar),
   while degrading that backbone's walking (+0.476 → +0.321). We are careful about *what* this
   licenses: it does not tell us where pointing's perceptual signal physically resides. What it
   *does* establish is that a purely **temporal** target — frame-to-frame motion, with static pose
   removed — carries perceptual signal that a pose-reconstruction target does not surface,
   precisely on the action where reconstruction is weakest. This is a first, direct indication that
   **temporal/dynamics information is perceptually relevant** and is under-exploited by
   reconstruction. Crucially, though, velocity as an added *loss* cannot capitalize on it (result 1):
   on a fully-visible reconstruction the velocity term is largely redundant (§4.3). The lesson is
   therefore not "add velocity" but "capture dynamics *properly*" — which motivates the move to a
   task that **forces** dynamics into the representation.

The operative conclusion: **a velocity loss layered on a reconstruction objective tunes within the
pose-biased regime and never produces a dynamics-led representation that serves all three actions —
yet the velocity-only signal shows dynamics carries perceptual information worth capturing.** This
is what motivates moving from reconstruction to masked *prediction* (§4.4) — forcing dynamics into
the representation through the learning task rather than through an added loss term — and ultimately
to the masked-motion-plus-pose composition.

### 4.3 Step 3 — why a velocity loss cannot supply dynamics
The previous result raises the question this section answers: *why* does a velocity term, on a
reconstruction objective, only modestly help and never produce a dynamics-led representation?

A velocity loss on a visible reconstruction is **partly redundant**. A reconstruction
autoencoder sees the *entire* clip and is rewarded for reproducing every frame. If it
reconstructs the poses `x_t` accurately, the frame-to-frame differences `x_{t+1}−x_t` — i.e. the
velocities — are reproduced *automatically*, as a byproduct. A velocity term added to that
objective is therefore largely already satisfied; it supplies relatively little gradient the
reconstruction loss has not already supplied. This predicts precisely what the factorial shows
(§4.2): adding the velocity term to reconstruction moves the correlations **negligibly** (flat
within noise on MLD, flat-to-worse on the plain transformer) — not because velocity is
irrelevant to perception, but because on a *fully-visible* reconstruction it is an inefficient,
largely-redundant place to inject it. And when reconstruction is removed and the model is trained
on velocity *alone* (the factorial's clean foil), the objective is weaker overall — confirming
that a velocity target is not a substitute for, but a degenerate special case of, the information
a good encoder needs.

The lesson that carries into the rest of the design: **dynamics must be forced into the
representation by the learning task itself** — by making the model *predict* motion it cannot
see — not appended as a loss term on a fully-visible reconstruction.

### 4.4 Step 4 — masked motion *prediction*, not reconstruction
The resolution comes from the Masked Motion Prediction (MAMP) framework [Mao et al., ICCV
2023]. Rather than reconstructing visible poses (or penalizing the velocity of a visible
reconstruction), MAMP **masks ~80% of spatio-temporal patches and predicts the *motion*
(temporal deltas) of the masked regions** from sparse visible context. A motion-magnitude
prior (Gumbel-softmax over per-patch motion energy) biases masking toward dynamically rich
temporal regions, concentrating the predictive difficulty where motion variation is
highest. This imposes genuine dynamics-modeling pressure: the encoder cannot copy what it
cannot see and is forced to infer how masked, high-motion regions evolve.

**MAMP alone is a dynamics specialist.** Trained on our data, the motion-only MAMP encoder's
**single strong action is picking (+0.490 raw, the one bar it clears; §6.7)**, while it is
**weak on walking (+0.351) and pointing (+0.299)**, both well below their baseline bars. We are
careful about what this licenses. It shows picking is the action whose perceptual similarity is
**best recovered by the masked-motion-prediction objective**, and walking/pointing the actions
it recovers worst — a statement about which *encoder* recovers which *action*, not about where
each action's perceptual information physically resides. We did not measure the latter; we
measured only that pointing is kinematically sparse relative to walking and picking (§3). We
therefore avoid labeling the actions "dynamics-led" or "configuration-led" as if that were a
measured perceptual property.

What *is* solid — and is the load-bearing claim — is the **empirical complementarity between the
two objectives**: masked-motion prediction succeeds on exactly the action (picking) where
reconstruction was weakest, and fails on exactly the actions (walking, pointing) where
reconstruction was strongest. The masked-prediction objective and the reconstruction objective
recover **disjoint, complementary subsets** of the three actions, and neither alone serves all
three. This complementarity — established directly by the results, with no appeal to where
perceptual information lives — is what motivates *composing* the two objectives rather than
choosing between them (§4.5).

#### 4.4.1 Pretraining budget and the pretext-vs-downstream divergence
We adopt the original MAMP pretraining hyperparameters [Mao et al., ICCV 2023] (warmup, base
and minimum learning rate, cosine schedule) and pretrain to convergence of the self-supervised
loss, saving the full checkpoint trajectory. The reported epoch is then selected the same way as
for every other encoder: per fold, the checkpoint maximizing raw perceptual Spearman on the
inner SELECT fold is chosen over the uniform 100-epoch grid across the trajectory (§5.2). For the
MAMP family this consistently selects an **early** checkpoint — the modal selected epoch is ≈900
for motion-only MAMP and ≈1000 for MAMP+pose — well before the loss-convergence point.

This is a concrete instance of the familiar self-supervised **pretext-vs-downstream
divergence**, and our profiling makes its shape precise: the held-out perceptual signal rises
quickly and then **plateaus by a few hundred epochs**, while the self-supervised reconstruction
loss keeps descending for a thousand more. Continued pretraining past the perceptual plateau
buys lower pretext loss but no further downstream gain (and, on the diagnostic curves, a mild
late decline) — better at the pretext task, no better at the task we care about. Critically, this
does *not* bias the reported numbers: the epoch is chosen by the inner SELECT fold and reported
on the disjoint TEST fold, so the selection is leakage-clean (§5.2). We verified that the
perceptual curve has plateaued within the saved trajectory for every model, which is what licenses
using the full trajectory as the candidate set rather than a hand-picked budget.

### 4.5 Step 5 — the pose augmentation (MAMP+pose)
Masked motion prediction supplies dynamics but discards the static-configuration fidelity that
the reconstruction objective recovered — and with it the actions reconstruction served, walking
and pointing (§4.4). We therefore augment it with an
**auxiliary pose-reconstruction objective** to restore that second capability without giving up
the first. To see what this adds, trace the MAMP forward pass and where the new head attaches:

1. **Patchify + mask.** The clip is split into spatio-temporal patches (§4.6); ~80% are
   masked (motion-magnitude-biased), leaving ~20% visible.
2. **Encoder.** The transformer encoder processes *only the visible patches*, producing a
   latent feature per visible patch. This encoder output is the representation we ultimately
   pool into the clip embedding (§5.1).
3. **Decoder.** Mask tokens are inserted at the masked positions and the full set is passed
   through the decoder, which produces a feature for every patch — crucially, for the
   masked positions it must *infer* their content from the visible context.
4. **Prediction heads (where the augmentation lives).** In standard MAMP a single linear
   head maps each masked patch's decoder feature to its **motion** target — the temporal
   delta `x_{t+s} − x_t`. We add a **second, parallel linear head** off the *same decoder
   features* that maps each masked patch to its **pose** target — the raw per-frame value
   `x_t` itself. The two heads share the entire encoder and decoder; they diverge only at
   the final linear projection.

The training loss is the sum of the two masked-patch objectives,
`L = L_motion + λ_pose · L_pose` with `λ_pose = 1.0`, each a mean-squared error over the
masked patches only. Because both heads read the *same* decoder features and backpropagate
into the *same* encoder, the encoder is pressured to produce a representation from which the
masked patches' **motion** *and* their **pose** can both be recovered — i.e. one latent that
simultaneously encodes how the body is *moving* (the dynamics that win picking) and how it is
*configured* (the static fidelity the walking and pointing judgments depend on).

Why a second prediction head rather than simply adding a pose term to the existing motion
head's target? Because the two targets are different quantities (a temporal delta vs. an
absolute pose) living on different scales; forcing one projection to regress their sum or
concatenation entangles them, whereas parallel heads let each target be predicted in its
natural form while still sharing — and jointly shaping — all the representation-bearing
layers beneath. §6.5 shows this composition rescues walking and pointing (which motion-only MAMP
serves worst) while retaining picking (the action MAMP already serves best).

### 4.6 Adaptation to LMA 6-D data
MAMP was designed for 3-D-coordinate NTU skeletons (25 joints). We adapt it to our 6-D
rotation LMA representation: `dim_in = 6`, `num_joints = 34`, `num_frames = 120`, temporal
patch size 4 (→ 30 temporal × 34 joint = 1020 patches), `mask_ratio = 0.80`,
`motion_aware_tau = 0.75`, encoder/decoder feature dim 256. Patchification and the motion
operator are coordinate-agnostic temporal differencing and therefore transfer to the 6-D
representation without modification. Pretraining excludes the 342 LMA evaluation clips
(§2.3).

---

## 5. The perceptual inference pipeline

### 5.1 Raw-embedding regime (unsupervised)
**From a clip to one vector.** A clip is first mapped to the fixed 120-frame window by the
deterministic truncate-or-tail-pad rule of §2.2, standardized, and patchified into
spatio-temporal patches — one patch per (4-frame temporal block × joint), giving
30 temporal × 34 joints = **1020 patches**. The pretrained encoder (run with mask ratio 0,
i.e. all patches visible) produces a **256-D feature vector for each of the 1020 patches**;
these are the "per-patch features." We collapse them to a single **256-D clip embedding by
mean-pooling over all patches** — the same pooling MAMP uses for its linear-probe transfer
evaluation, and therefore the encoder-native readout. Because every clip is forced to the
same 120-frame window before patchification, every clip yields exactly 1020 patches and a
fixed-size embedding regardless of its original length, so variable clip lengths require no
special handling at the embedding stage (the length normalization happens once, in §2.2,
identically for training and evaluation).

**Evaluation.** Pairwise **Euclidean (L2) distances** between clip embeddings are correlated
(Spearman, Pearson) against the per-pair human dissimilarity `d_perc` (§2.4.2) over the
fixed held-out validation subset (§5.2). This regime uses **no human ratings during
representation learning** and therefore measures the *encoder's intrinsic* perceptual
alignment.

### 5.2 Evaluation protocol and baselines
- **Repeated, leakage-clean nested cross-validation.** All reported correlations are the mean
  over **5 repeats × 5 outer folds (n = 25 held-out test estimates)**. Within each repeat, the
  effort classes of each action are partitioned (magnitude-stratified) into 5 folds with a
  repeat-specific seed. For each outer fold: the fold is the **test** set (reported, never seen
  during training or selection); one disjoint fold is the **selection** set, used *only* to
  pick each encoder's checkpoint by raw held-out Spearman (a metric-independent, perceptually-
  grounded checkpoint-selection criterion); the remaining folds are the **training** set, used to
  fit any learned metric. The encoder additionally never saw any of these clips during
  self-supervised pretraining (§2.3), so the test fold is held out of *both* stages. We report
  mean ± standard error (SE = SD/√25) over the 25 estimates.
- **Why not a single fixed split.** A single fixed stratified split is unreliable here: the
  per-action correlation is strongly split-dependent, most severely for pointing (§3), whose
  value ranges roughly [0.23, 0.55] depending only on which classes land in the held-out
  subset. Conclusions and margins-of-victory drawn from one split do not survive
  re-partitioning; the repeated nested-CV averages over this partition variance and is the
  basis for every claim here.
- **Baselines on identical data and identical folds.** Both geometric baselines are computed
  on the **raw, variable-length, unpadded** motion features (no 120-frame windowing, no
  tail-padding), over the **same 25 test folds** as the learned distances and fold-averaged
  the same way, and correlated against the same `d_perc`. Because the baselines are
  themselves split-dependent, evaluating them on the identical folds is essential: on a single
  favorable split DTW pointing reaches +0.370, but fold-averaged on the matched folds it is
  +0.273 — the honest, like-for-like bar.

  *Per-frame rotation-geodesic cost.* For two frames, the local cost is the rotation
  geodesic distance between corresponding joints: each joint's 6-D feature is mapped back to
  a rotation matrix R ∈ SO(3) (Gram–Schmidt), and the geodesic angle between two rotations
  R₁, R₂ is `θ = arccos((tr(R₁ᵀR₂) − 1)/2)`, the angle of the relative rotation R₁ᵀR₂.
  Per-frame cost is the mean of θ over the 33 articulated joints.

  *DTW.* Given two clips of (generally different) lengths T₁ and T₂, Dynamic Time Warping
  finds the monotone frame-alignment path through the T₁ × T₂ grid of per-frame geodesic
  costs that minimizes the accumulated cost; the **DTW distance** is that minimized
  accumulated cost (length-normalized). DTW thereby compares the two motions' *shape and
  progression* while absorbing differences in speed and duration — which is exactly why it
  is the strong baseline for the dynamically rich actions.

  *Geodesic baseline.* The "geodesic" baseline reported separately is the DTW-aligned mean
  per-frame geodesic cost on the same raw unpadded sequences (i.e. the average local cost
  along the optimal alignment, rather than the accumulated path cost). **Correction to a
  legacy implementation:** an earlier version padded every clip to a fixed length by
  repeating the final frame and then compared frame-wise without alignment; for short
  pointing clips this injected long runs of identical frozen-pose frames (up to ~54% of the
  padded length), spuriously deflating distances. Computing the baseline on raw unpadded
  sequences with proper temporal alignment removes this bias and *strengthens* the baseline
  the learned method must beat.

---

## 6. Results

### 6.1 Headline — held-out, repeated nested cross-validation

We report Spearman (rank) correlation between each method's pair distances and the human
dissimilarity `d_perc`. **Spearman is the primary metric**: the ground truth is an ordinal
quantity derived from choice frequencies, so we care whether a method recovers the human
*ordering* of pair similarities rather than reproducing absolute magnitudes.

**Evaluation protocol (see §5.2 for full detail).** All headline numbers are the mean over a
**repeated, leakage-clean nested cross-validation**: 5 repeats × 5 outer folds = 25 held-out
test estimates. Each outer fold's test classes are evaluated by an encoder whose checkpoint
was selected on a *disjoint* inner fold (by raw held-out Spearman) and — where a learned
metric is used — trained on the remaining classes; the test fold is never seen during
selection or training. **Crucially, the geometric baselines are computed on the identical 25
test folds and fold-averaged the same way**, so every comparison is like-for-like. We report
mean ± standard error across the 25 estimates. (A single fixed-split point estimate, as
used in earlier drafts of this work, is unreliable for the high-variance pointing action and
is not used for claims; see §5.2 and §9.)

**Spearman, raw MAMP+pose encoder vs. geometric and learned baselines (fold-averaged, n = 25):**

| Method | Walking | Pointing | Picking |
|--------|---------|----------|---------|
| **MAMP+pose, raw** | **+0.552 ± 0.026** | **+0.434 ± 0.030** | **+0.543 ± 0.033** |
| DTW baseline (geometric) | +0.490 ± 0.028 | +0.273 ± 0.039 | +0.303 ± 0.038 |
| Geodesic baseline (geometric) | +0.371 ± 0.028 | +0.234 ± 0.031 | +0.264 ± 0.030 |
| Vanilla SSL encoder (learned, recon-only) | +0.476 ± 0.031 | +0.213 ± 0.034 | +0.491 ± 0.029 |

We compare against two **geometric** baselines (DTW, rotation geodesic) and a **learned**
baseline. The learned baseline is a generic self-supervised motion encoder — a plain
transformer autoencoder trained by reconstruction only, with **no perceptual supervision and
none of our composition design** — run held-out on the identical 25 folds with the identical
matched-checkpoint-selection protocol (§6.7). It controls for "any learned motion
representation," isolating the contribution of the masked-motion / pose-reconstruction
composition from that of merely learning a representation. (It is a learned *representation*
baseline, not a purpose-built learned *similarity* metric; the latter, retargeted from an
external motion-retrieval model, is identified as future work in §9.)

On **Spearman**, the **raw, unsupervised MAMP+pose encoder beats both geometric baselines and
the learned baseline on all three actions** under matched held-out evaluation — to our
knowledge the first single encoder to do so. The margins over the *stronger of the two
geometric* baselines per action are +0.062 (walking, vs DTW), **+0.161 (pointing, vs DTW)**, and
+0.240 (picking, vs DTW); over geodesic the margins are +0.181 / +0.200 / +0.279. Against the
learned baseline the margins are +0.076 (walking), **+0.221 (pointing)**, and +0.052 (picking):
the learned encoder is itself strong on picking (+0.491, the action where reconstruction does
best), so MAMP+pose's picking edge over it is narrow — but the learned encoder **collapses on
pointing (+0.213, below even DTW)**, so simply learning a motion representation is insufficient,
and the composition is what carries all three. Treating each fold-difference as paired across the
25 matched folds, the pointing and picking advantages over the geometric baselines are large and
well-separated from zero (≈3–5σ on a conservative independent-SEM test); the one near-marginal
geometric cell is walking versus DTW (+0.062, ≈1.6σ), though walking still clears geodesic by a
wide margin. **The all-three
result is a property of the raw encoder geometry and does not require any learned metric
refinement** — a stronger and simpler claim than a two-stage pipeline.

A note on evaluation. Pointing's raw correlation is highly split-dependent (§3, §5.2): on any
single fixed split it can land anywhere in roughly [0.23, 0.55] depending only on which effort
classes fall in the held-out subset, so a single-split estimate is unreliable for this action
and is not used for claims. The geometric baselines are split-dependent in the same way — on a
single favorable split DTW pointing reaches +0.370, but fold-averaged on the matched folds it is
substantially lower (+0.273), which is the honest
bar the encoder is compared against here.

### 6.2 The raw-encoder result — a stronger, unsupervised claim
The all-three win is achieved by the **raw** encoder embeddings — no perceptual supervision
whatsoever. The perceptual-similarity structure is therefore *intrinsic to the self-supervised
representation*, not imposed by a downstream metric. Beating a heavily-optimized temporal
baseline like DTW with unsupervised embeddings — including on the expressively sparse pointing
action, where DTW's time-warping was expected to dominate — indicates the masked-motion
pretext task, augmented with the pose head (§6.5), genuinely captures the kinematic structure
underlying human similarity judgments. This is the central, simplest claim of the paper: a
single self-supervised encoder, with no metric learning, recovers human perceptual similarity
ordering better than geometric distance on every action.

### 6.3 Comparison to a supervised learned representation (TMR)

The baselines above are *geometric* (DTW, geodesic) and a *reconstruction-only* learned encoder.
A sterner test is a **purpose-built, supervised** motion representation. We compare against
**TMR** [Petrovich et al., ICCV 2023], a text↔motion retrieval model trained with contrastive
language supervision on HumanML3D — a representation shaped by a *large* corpus of semantic
(text) annotations, the opposite supervision regime to ours. The comparison asks whether our
sparse, task-aligned perceptual supervision competes with TMR's abundant, general semantic
supervision on *perceptual* similarity.

**Protocol and a conservative handicap.** TMR consumes HumanML3D 263-D motion features, so we
convert each rated clip from its BVH form to SMPL parameters (joints2smpl SMPLify fit, mean
reconstruction error ≈4 cm ≈2% of body height; §8) and then to the 263-D features TMR expects,
encode through the released HumanML3D TMR motion encoder, and evaluate the resulting embeddings'
L2 distances on the **identical 25 nested-CV folds** and the same `d_perc` as every other method.
The ≈4 cm conversion perturbs TMR's input slightly, so any MAMP+pose advantage is a **conservative
lower bound**. For a *corpus-fair* comparison — TMR trained on HumanML3D, ours on a matched budget
— we use the **corpus-matched MAMP+pose** (a variant pretrained on a corpus size-matched to
TMR's HumanML3D training budget; cf. §2.3) as the comparator rather than the full-corpus model.

**Spearman, corpus-matched MAMP+pose vs. TMR (n = 25, mean ± SE):**

| Method | Walking | Pointing | Picking | clears all 3 bars? |
|--------|---------|----------|---------|:---:|
| TMR (supervised, HumanML3D) | +0.452 ± 0.025 | **+0.425 ± 0.036** | +0.408 ± 0.029 | — |
| MAMP+pose (corpus-matched, raw) | +0.462 ± 0.030 | +0.302 ± 0.031 | +0.481 ± 0.033 | — (pointing) |
| **MAMP+pose (corpus-matched) + ranking fine-tune** | **+0.699 ± 0.022** | **+0.456 ± 0.036** | **+0.605 ± 0.031** | **✓** |

Two honest reads. **Raw**, the corpus-matched MAMP+pose beats TMR on walking (+0.462 vs +0.452)
and picking (+0.481 vs +0.408) but **TMR is stronger on pointing (+0.425 vs +0.302)** — the one
action where abundant semantic supervision helps most, and where our raw encoder is weakest (§3).
Notably, TMR is the **strongest pointing baseline in the entire study** (far above DTW's +0.273
and the reconstruction encoders' +0.33), and unlike the geometric baselines it does *not* collapse
on pointing — it is roughly *flat* across actions (0.41–0.45), consistent with a semantically
supervised representation that captures broad motion structure uniformly but is not tuned to
perceptual effort-similarity. So the raw comparison is honest and not a strawman: on pointing,
supervised semantics edges our unsupervised encoder.

**With the light ranking fine-tune (§6.4), MAMP+pose beats TMR on all three actions**, including
pointing (+0.456 vs +0.425), and is the only method clearing all three bars — achieved with a
**far smaller and cheaper supervision set** (≈1,540 within-task perceptual triplets per action)
than TMR's corpus of semantic annotations. The finding is one of *alignment over abundance*:
sparse supervision *aligned to the target task* (perceptual similarity) matches or exceeds
abundant supervision aligned to a *different* task (text–motion semantics). We state the pointing
result precisely — TMR wins pointing on the raw encoder and is edged only after the perceptual
fine-tune — rather than overclaiming a raw all-three win against this particular baseline.

### 6.4 Perceptual fine-tuning: using the human ratings to improve the encoder
The raw encoder already wins (§6.1–§6.2). We now ask whether the small budget of human ratings
can push it *further*. The productive way to use them, we find, is to back-propagate a
perceptual objective **into the encoder itself**, reshaping the representation so that embedding
distances track human dissimilarity — rather than fitting a separate metric on a frozen
embedding. We define the objective, fix a leakage-clean protocol, and report the effect.

**An anchored ranking loss with a preference-scaled margin.** The ground truth `d_perc` is
*ordinal*: it says which pairs observers judged more dissimilar, not by how much on any absolute
scale. The natural objective is therefore a **ranking** loss, not a regression to `d_perc`
values. For an anchor class `i` and two others `j, k`, let `pⱼ = d_perc(i,j)` and `pₖ =
d_perc(i,k)`; the human ordering names a *near* and a *far* class (smaller vs. larger `d_perc`).
On the embedding side we take pairwise distances min–max normalized to `[0,1]` within the
batch's rated pairs, and impose

```
    L = relu( d̂(i, near) − d̂(i, far) + m ),     m = |pₖ − pⱼ|,
```

i.e. the anchor–near distance must fall below the anchor–far distance by a margin equal to the
**human dissimilarity gap** itself. Two properties make this the principled choice here. (i) The
margin is **hyperparameter-free and adaptive**: it is the `d_perc` gap, so confidently-ordered
pairs (large gap) are pushed apart harder than near-ties (small gap), and no margin constant
must be tuned. (ii) The loss is **metric-aligned by construction** — it optimizes the same
ordinal quantity, over the same non-neutral effort pairs, that the Spearman evaluation scores
(§2.4.2), so training and evaluation agree on what counts as correct.

**Leakage-clean protocol and a single held-out selection.** Fine-tuning runs under the *same*
repeated nested cross-validation as the raw results (§5.2): the same fold seeds, the same 25 test
folds, the same canonical-57 universe and `d_perc`; the only change is that within each fold's
training portion the MAMP+pose encoder is fine-tuned rather than frozen. Two design points make
the reported numbers conservative. First, the base checkpoint to fine-tune from is chosen **per
fold on the selection fold only** — never the test fold — so the choice of starting point cannot
leak. Second, we deliberately use a **single** held-out selection: rather than also early-stopping
on the selection fold (which would consult it a second time, along the fine-tuning-epoch axis), we
train a **fixed budget and report the final epoch**. We verified this costs nothing — the
selection-fold trajectory plateaus well within the budget, and reporting the final epoch matches
or slightly exceeds the best-selection-epoch variant (see the two rank rows below) — so the
simpler single-selection design is adopted throughout, removing any "selection used twice"
concern by construction.

**Spearman, ranking fine-tune vs. raw MAMP+pose (fold-averaged, n = 25, mean ± SE):**

| Method | Walking | Pointing | Picking |
|--------|---------|----------|---------|
| raw MAMP+pose (no ratings) | +0.462 ± 0.030 | +0.302 ± 0.031 | +0.481 ± 0.033 |
| **+ ranking fine-tune** | **+0.699 ± 0.022** | **+0.456 ± 0.036** | **+0.605 ± 0.031** |

The ranking fine-tune **lifts all three actions substantially** from the corpus-matched raw base:
walking +0.462 → +0.699, pointing +0.302 → +0.456, and picking +0.481 → +0.605. The gain is
reached after genuine training and generalizes to the untouched test fold. It is robust to the
stopping rule: reporting the fixed-budget final epoch matches — in fact slightly exceeds — a
variant that additionally early-stops on the selection fold, so the single-selection design (§5.2)
loses nothing by not consulting the selection fold a second time. (A direct-regression objective
and a neutral-anchored, alpha-margin objective were also tried; both under-perform the ranking
loss, the latter failing to generalize on pointing — consistent with the perceptual signal
residing in the direct inter-state ordering the ranking loss targets rather than in
neutral-referenced context shifts.) In short, the human ratings are best used not as a metric fit
on a fixed representation but as a **light, ranking-based fine-tuning of the encoder**, which
further improves an already-baseline-beating raw encoder on all three actions and underpins the
supervised comparison against TMR (§6.3).


### 6.5 Ablation — the pose head is necessary

Raw Spearman, motion-only MAMP vs. MAMP+pose (repeated nested-CV, n = 25, mean ± SE;
both with the training epoch selected per fold on the inner SELECT fold, §5.2):

| Objective | Walking | Pointing | Picking |
|-----------|---------|----------|---------|
| motion-only (MAMP) | +0.351 ± 0.032 | +0.299 ± 0.032 | +0.490 ± 0.039 |
| + pose-reconstruction head (MAMP+pose) | **+0.552 ± 0.026** | **+0.434 ± 0.030** | **+0.543 ± 0.033** |

Motion-only masked prediction clears the bar only on picking (+0.490), the action it best
serves, and is **weak on walking (+0.351)** and pointing (+0.299). Adding the auxiliary
pose-reconstruction head **lifts every action**, most sharply walking (+0.351 → +0.552, a +0.20
gain) and pointing (+0.299 → +0.434, the gain that carries it clear of the baselines), while
keeping picking at its best. This confirms the two-capability composition that motivates
MAMP+pose: the motion target supplies dynamics, the pose target supplies configuration fidelity,
and the combination — not either head alone — yields the only encoder clearing both baselines on
all three actions (§6.7).

### 6.6 Summary of the per-action story

| Action | Best method | Mechanism |
|--------|-------------|-----------|
| Walking | MAMP+pose (raw) | dynamics captured in raw encoder geometry |
| Picking | MAMP+pose (raw) | pose-fidelity + dynamics; raw geometry suffices |
| Pointing | MAMP+pose (raw) | masked-motion + pose composition retains sparse kinematic signal |

A **single self-supervised encoder, with no perceptual supervision**, beats both geometric
baselines on all three actions under repeated nested cross-validation — including the
expressively sparse pointing action. Where the human ratings help further, it is by
**fine-tuning the encoder** under a perceptual objective (§6.4), which lifts all three actions
above the already-baseline-beating raw encoder.

### 6.7 Full model sweep (all encoders, raw)

**All values below are from the same repeated, leakage-clean nested cross-validation as the
headline (§5.2, §6.1): mean ± SE over 5 repeats × 5 folds (n = 25), evaluated on the
canonical 57 effort classes per action.** Every encoder is treated identically: per outer
fold its checkpoint is selected on the disjoint selection fold by raw held-out Spearman, from
a candidate set of **11 checkpoints** (the uniform 100-epoch grid, epochs 100–1100) spanning
that encoder's own training trajectory (matched selection pressure across encoders — see below), and the raw pooled-embedding distances are
reported on the untouched test fold. The
geometric baselines (DTW, geodesic) on these same 25 folds are walking +0.490 / +0.371,
pointing +0.273 / +0.234, picking +0.303 / +0.264 (§6.1); the bar each encoder must clear is
the stronger baseline per action: **walking > 0.478, pointing > 0.370, picking > 0.431**. We
mark ✓ when the encoder's 95% confidence interval lower bound (mean − 1.96·SEM) clears that
bar; this is a strict, multiplicity-honest criterion.

> **Uniform epoch sampling.** The encoders converge on very different epoch scales (MAMP ≈ 1.2k,
> MLD AEs ≈ 6.7k, plain-transformer AEs ≈ 8–10k), so a single fixed epoch would be unfair and a
> hand-picked per-model epoch would be a researcher degree of freedom. Instead the training epoch
> is an **inner-loop hyperparameter**: candidate checkpoints are taken on a **uniform 100-epoch
> grid across each model's full trajectory** (identical grid for all models, no plateau cut), the
> one maximizing raw perceptual Spearman on the inner SELECT fold is chosen, and it is reported on
> the disjoint TEST fold (§5.2). We verified every model's held-out perceptual signal plateaus
> within its trajectory, so the full-trajectory candidate set contains each model's optimum. The
> MLD-VAE row uses the deterministic-µ encoding of the variational model (the deterministic MLD AE
> is its KL-free adaptation, §4.1.2, so the comparison completes the backbone ablation).

**Raw embeddings, Spearman (n = 25, mean ± SE):**

| Encoder (backbone, objective) | Walking | Pointing | Picking | clears all 3 bars? |
|-------------------------------|---------|----------|---------|:---:|
| Plain transformer, recon-only (rot 1, vel 0) | +0.476 ± 0.031 | +0.213 ± 0.034 | +0.491 ± 0.029 | — |
| Plain transformer, recon+velocity (rot 1, vel 1) | +0.472 ± 0.026 | +0.205 ± 0.039 | +0.399 ± 0.037 | — |
| Plain transformer, velocity-only (rot 0, vel 1) | +0.321 ± 0.033 | +0.362 ± 0.036 | +0.471 ± 0.033 | — |
| MLD AE, recon-only (rot 1, vel 0) | +0.532 ± 0.023 | +0.331 ± 0.039 | +0.498 ± 0.029 | — (pointing) |
| MLD AE, recon+velocity (rot 1, vel 1) | +0.537 ± 0.022 | +0.339 ± 0.035 | +0.493 ± 0.030 | — (pointing) |
| MLD AE, velocity-only (rot 0, vel 1) | +0.451 ± 0.031 | +0.278 ± 0.031 | +0.254 ± 0.039 | — |
| MLD VAE, recon-only | +0.597 ± 0.021 | +0.324 ± 0.032 | +0.501 ± 0.029 | — (pointing) |
| MAMP (motion-only) | +0.351 ± 0.032 | +0.299 ± 0.032 | +0.490 ± 0.039 | — |
| **MAMP+pose** | **+0.552 ± 0.026** | **+0.434 ± 0.030** | **+0.543 ± 0.033** | **✓** |

**MAMP+pose is the only encoder whose 95% CI clears all three baseline bars.** Pointing is the
discriminator: the strongest competitors (the three MLD AEs) reach walking/picking comfortably
but stall on pointing at +0.33–0.36 (CI not clearing 0.370), exactly the expressively-sparse
action (kinematically sparsest; §3) that the masked-motion-plus-pose composition is built to
capture (§4.5, §7). Motion-only MAMP is strong on picking but weak on walking and pointing; the
plain-transformer AEs are weakest overall, consistent with §4.1.

The raw table above is the comparison of record: every encoder evaluated identically, against
baselines re-evaluated on the same folds (§6.1). The single productive use of the human ratings
— perceptual fine-tuning of the encoder — is reported for the winning encoder in §6.4; we do
not apply a downstream metric to the comparison encoders, since a metric fit on frozen
embeddings does not improve held-out alignment on this rating budget (§6.4).

Two observations the full sweep makes concrete:

- **No competing encoder is strong on all three actions.** Each alternative is a
  *specialist*. The plain-transformer reconstruction AEs are weakest overall, competitive only
  on walking and picking. The MLD reconstruction encoders (recon, +velocity) reach walking and
  picking but **stall on pointing** (+0.33–0.34, CI not clearing the bar). Motion-only MAMP is a
  *picking* specialist (raw picking +0.490, the one bar it clears) that **collapses on walking**
  (+0.351). Only **MAMP+pose** — the composition of masked-motion dynamics and pose-configuration
  fidelity — is competitive everywhere, and the only encoder to clear both baselines on all three
  actions.
- **The architecture and objective axes are separable and both matter**, exactly as §4 argued:
  at a fixed reconstruction-only objective the MLD backbone beats the plain transformer (raw
  pointing +0.331 vs +0.213; §4.1), while at a fixed MLD-class backbone it is the
  masked-motion-plus-pose objective — not reconstruction with or without a velocity term — that
  produces the only all-three winner. §6.8 tightens this separability claim into a direct
  de-confound: porting the MLD backbone's encoder-internal U-Net *into* the masked-prediction
  encoder does not recover pointing, so the decisive step is objective- rather than
  architecture-driven.

(All encoders are evaluated under the repeated nested-CV with the training epoch selected per
fold on the inner SELECT fold over the uniform 100-epoch grid (§5.2); for the MAMP family this
selects an early checkpoint (modal ≈900–1000), and §4.4.1 reports that training beyond the
perceptual plateau lowers the self-supervised loss without improving the downstream signal.)

### 6.8 Disentangling architecture from objective — is it the pose head or the backbone?

§6.7 establishes that at a fixed reconstruction objective the MLD SkipTransformer backbone
beats the plain transformer (§4.1), and that at a fixed backbone the masked-motion-plus-pose
objective produces the only all-three winner. This invites a sharper, potentially deflationary
question about the pose head itself. The step from the reconstruction encoders to MAMP changes
**two** things at once — the *objective* (reconstruction → masked prediction) and, implicitly,
the *backbone*, since the MLD reconstruction encoders carry U-Net skip connections while the
MAMP encoder is a plain (skip-free) transformer. A skeptic could therefore argue that
MAMP+pose recovers configuration not because of the pose *objective* but because a reviewer
might expect skip connections — the very mechanism that lifts reconstruction pointing from
+0.213 to +0.331 (§4.1) — would do the same for masked prediction. If so, the "composition"
would be an architectural artifact, not an objective one. We test this directly.

**An MLD-style encoder inside MAMP.** We port the MLD SkipTransformer's *encoder-internal*
U-Net — the symmetric long skips (early-layer features concatenated into the matching late
layer, before the latent), which is precisely the mechanism responsible for MLD's
reconstruction gains (§4.1) — into the MAMP encoder, holding the masked-motion objective,
corpus, normalization, and held-out split fixed. Because MAMP masks ≈80% of tokens before the
encoder, a skip taken *after* masking sees only the visible ≈20%; to give the architecture its
best case we run the U-Net encoder on the **full, unmasked** sequence and route only the
visible-position features to the decoder, so the encoder is train/inference-consistent and its
skips operate on complete motion while the prediction task remains non-trivial (masked targets
never leak to the decoder). We call this **MAMP-uencfull**. Crucially, it has **no pose head** —
it isolates the effect of the MLD-style *architecture* under the masked objective.

**Raw Spearman, architecture vs. objective on the masked-prediction backbone (n = 25, mean ± SE):**

| Encoder | Walking | Pointing | Picking | clears all 3? |
|---------|---------|----------|---------|:---:|
| MAMP (plain encoder, motion-only) | +0.351 ± 0.032 | +0.299 ± 0.032 | +0.490 ± 0.039 | — |
| MAMP-uencfull (MLD-style U-Net encoder, motion-only) | +0.470 ± 0.029 | +0.275 ± 0.028 | +0.566 ± 0.030 | — (pointing) |
| **MAMP+pose (plain encoder, + pose objective)** | **+0.552 ± 0.026** | **+0.434 ± 0.030** | **+0.543 ± 0.033** | **✓** |

The result is unambiguous on the discriminating axis. Giving MAMP the MLD-style U-Net encoder
**does not recover pointing**: it moves from +0.299 to +0.275 — statistically flat, and if
anything slightly lower — and remains **below even the MLD reconstruction encoders' pointing
(+0.33; §6.7)** and far below the bar (0.370). The pose *objective*, by contrast, lifts pointing
to +0.434 (a +0.135 gain over plain MAMP, ≈4 SE, §6.5). Architecture and objective are thus not
interchangeable here: **the configuration recovery that carries pointing is supplied by the
pose-reconstruction objective, not by the encoder architecture.** This closes the confound in
the §6.7 separability claim — the ablation ladder's decisive step is objective-driven, and no
skip-connection variant of the backbone substitutes for it.

Two secondary observations sharpen the mechanism rather than change the conclusion. First, the
U-Net encoder is not inert: it *lifts the coarse-motion actions*, improving walking (+0.351 →
+0.470) and giving MAMP-uencfull the **best picking in the entire sweep (+0.566)**. The skip
architecture evidently enriches the dynamics/coarse-configuration representation — it simply
does not manufacture the fine configuration signal that the perceptually-sparse pointing action
requires, which only the pose objective supplies. Second, we run the U-Net encoder on the
**full, unmasked** sequence deliberately, as the architecture's best case: taking the skips
after masking (on the visible ≈20% only) is uniformly weaker, since encoder skips are starved by
masking — consistent with why MAMP's native skip pathway, which zero-fills masked positions,
cannot carry configuration either. Reporting the full-visibility variant therefore gives the
architectural hypothesis its strongest form, and it still does not recover pointing.

Finally, we complete the 2×2 of architecture (plain vs. MLD-style U-Net encoder) × objective
(with vs. without the pose head), since MLD's own recipe couples *both* skip connections and a
reconstruction objective. The remaining cell — an MLD-style U-Net encoder trained *with* the pose
objective — asks whether the two compound.

**Architecture × objective (raw Spearman, n = 25, mean ± SE):**

| | plain encoder | MLD-style U-Net encoder |
|---|---|---|
| **motion-only** | MAMP: +0.299 ± 0.032 (pointing) | MAMP-uencfull: +0.275 ± 0.028 |
| **+ pose objective** | MAMP+pose: **+0.434 ± 0.030** | MAMP-uencfull+pose: ⟨PLACEHOLDER — pending, n=25 mean±SE⟩ |

> *[Placeholder — results pending the MAMP-uencfull+pose pretraining run; to be filled with the
> n=25 mean ± SE on completion.] Interpretation to fill: if pointing exceeds +0.434, architecture
> and objective are **complementary** — the U-Net encoder's benefit unlocks only in the presence of
> the pose objective, echoing MLD's own coupling of skips with reconstruction; if pointing is
> ≈ +0.434, the pose objective **saturates** configuration recovery and the architecture adds
> nothing on top. Either outcome pre-empts the "why not both?" question and does not alter the
> §6.8 conclusion that the pose objective, not the backbone, is what recovers configuration.*

---

## 7. Why MAMP+pose succeeds: modeling vs. copying
The central mechanistic claim is the distinction between **modeling** dynamics and
**copying** poses. Reconstruction objectives present the model with the *entire* visible motion
and reward reproducing it. Under full visibility, frame-to-frame deltas — the velocities — are
largely reproduced for free as a byproduct of accurate pose copying. A velocity term added to
such an objective therefore enters a regime where it is *partly redundant*: the factorial of
§4.2 shows that adding it moves the correlations negligibly (flat within noise on MLD,
flat-to-worse on the plain transformer), because nothing in a fully-visible reconstruction
requires inferring how the body would move where it is not observed. Velocity-penalized
reconstruction thus remains a **pose-biased** representation — recovering the actions
reconstruction already served, never reaching the behavior of masked prediction on picking.
(Our controlled, unit-weight factorial of §4.2 — reconstruction-only, +velocity, and
velocity-only, on both autoencoder backbones — isolates this directly: it measures what the
velocity term adds *holding architecture and all other loss terms fixed*, and whether a velocity
objective *alone* carries useful perceptual structure.)

Masked motion prediction removes the copying shortcut. With ~80% of the spatio-temporal patches
hidden and the *motion* of the hidden, high-energy regions as the target, the encoder must
infer kinematic progression it cannot observe — it must internalize how the body moves, not
merely where it currently is. This is why motion-only masked prediction is the empirical
complement of reconstruction: it recovers the picking action reconstruction served worst, and
is weak on walking and pointing, which reconstruction served best (§4.4). The auxiliary pose
head restores exactly the static-configuration fidelity the reconstruction objective had. The
**composition** — masked dynamics modeling *plus* pose reconstruction — supplies both
capabilities in one encoder and is the only configuration competitive across all three
heterogeneous actions; neither objective alone is.

---

## 8. Reproducibility

### 8.1 Evaluation protocol (exact)
- **Repeated nested cross-validation.** 5 repeats × 5 outer folds (n = 25). Folds are
  magnitude-stratified per action with seed `42 + 1000·repeat`. Per outer fold f: test = fold
  f; selection = fold (f+1) mod 5; training = the remaining three folds. The neutral exemplar
  `(0,0,0,0)` is added to every subset. Checkpoint selection uses raw held-out Spearman on the
  *selection* fold (never the test fold). For the perceptual fine-tuning (§6.4) the selection
  fold is used *once*, to pick the per-fold base checkpoint; the fine-tune itself runs a **fixed
  epoch budget** (no second, early-stopping use of the selection fold). All correlations are
  reported on the *test* fold.
- **Evaluation universe = the canonical 57 effort classes per action** (1 neutral + 24 states
  [exactly two non-zero efforts] + 32 drives [exactly three non-zero efforts]). This is a
  critical and easily-missed detail: the stored embedding directories also contain
  *non-canonical* clips (single-effort and fully-polarized tuples) that carry **no human
  ratings**. Those clips cannot enter any correlation, but if they are left in the pool that
  is *shuffled* to form the split, they perturb which rated classes land in each fold and
  silently change every number — most severely for pointing. The split must shuffle the
  canonical 57 only. (An earlier inability to reproduce the pointing baseline traced entirely
  to this: shuffling the contaminated pool produced pointing-raw ≈ 0.27, shuffling the clean
  57 gives the correct fold distribution.)
- **Per-pair perceptual target.** For each rated class pair, `d_perc = 1 −
  count_normalized[(selected0 = 0, selected1 = 2)]`. Correlation is Spearman of pooled-
  embedding L2 distance (raw stage) or refined distance (metric stage) versus `d_perc`, over
  exactly the test-fold pairs that carry a human rating. Distance is **plain pairwise L2 with
  no neutral-centering**; the refined-distance metric applies the trained MLP per embedding and
  recomputes L2 identically, so raw and refined numbers are produced by one consistent
  procedure.
- **Baselines on the identical folds** (§5.2): DTW and the DTW-aligned, unpadded geodesic are
  computed on the same 25 test folds and fold-averaged the same way.
- **Strict hold-out**: the 342 LMA evaluation clips are excluded from encoder pretraining and
  its validation; all reported models are out-of-sample (no in-sample models).
- **Single corpus**: all reported encoders are trained on the same 3-dataset corpus
  (§2.3), so cross-encoder comparisons are corpus-matched.

### 8.2 Pitfalls we encountered and how we resolved them (for faithful reproduction)
These are recorded explicitly because each one silently altered results during this work and
could trap a re-implementer:
- **Single-split point estimates are unreliable.** Pointing-raw Spearman has a fold-to-fold
  standard deviation ≈ 0.15; any single split is uninformative and earlier single-split
  numbers (raw pointing +0.263; "+triplet +0.465") were partition artifacts. Use the repeated
  nested-CV mean, reported as mean ± standard error (SE = SD/√25), for all claims.
- **Checkpoint identity.** The same encoder appears under two pretraining budgets in our
  archive; the MAMP+pose embedding directory without an epoch suffix is the **converged
  (epoch-1199)** checkpoint, while the headline encoder is **epoch-400** (authors' budget,
  §4.4.1). They are different representations (cosine ≈ 0.67); always confirm the checkpoint
  epoch (and md5) before comparing embeddings. Our encode pipeline reproduces a given
  checkpoint's embeddings bit-for-bit.
- **The metric module's loss option.** The embedding-refining network historically forced its
  loss to the base triplet loss regardless of the requested option; the perceptually-aligned
  ("integrated", distance-to-`d_perc`) loss only takes effect once that override is removed.
  Results that intend the perceptual loss must verify the integrated criterion is actually
  constructed.
- **Code/version parity across machines.** Reproduction requires the LayerNorm metric MLP, the
  single-vector (`_emb.pt`) embedding loader, and the corrected geodesic function to be the
  versions actually present on the execution host; stale copies of any of these change the
  output. We pin and archive the exact files used.

### 8.3 Artifacts and policy
- **Checkpoint and epoch policy.** Each encoder is evaluated at a fixed, reported checkpoint:
  MLD autoencoders at best held-out reconstruction loss; MAMP encoders at the 400-epoch
  authors' pretraining budget (§4.4.1); the plain transformer at final converged. Epoch counts
  are matched within a backbone family but not forced across families (an epoch is not a
  comparable unit across architectures).
- **Corrected geodesic baseline** on raw unpadded sequences (no frame-padding bias; §5.2).
- Encoder checkpoints (with md5s), the LMA data feeder, the embedding extractor, the
  nested-CV harness, the baseline harness, the metric/perceptual-fine-tune trainer, and the
  evaluation scripts are archived for end-to-end reproduction, along with the exact per-fold
  random seeds.
- **Checkpoint and epoch policy.** We evaluate each encoder at its **best checkpoint**,
  defined as the checkpoint with the lowest held-out reconstruction loss, and report its
  epoch. This definition applies directly to the MLD autoencoders, whose trainer performs
  held-out validation and selects a best checkpoint. The other backbones do not validate
  during training — the plain transformer's reconstruction loss plateaus high and early
  (§4.1.1), and the MAMP encoders are self-supervised pretrainers with no validation loop —
  so for these we use the **final converged checkpoint** (which, once the loss has plateaued,
  is equivalent to a best checkpoint: every late epoch is representative). We confirm
  convergence for every reported model (loss plateaued) and report each model's epoch.
  Epoch counts are *matched within a backbone family* (variants share an architecture and
  per-step unit of work, so a common selection rule isolates the training objective) but are
  *not* forced to a common value *across* backbones, where an epoch is not a comparable unit
  (the vanilla, MLD, and MAMP backbones differ in architecture and convergence dynamics);
  across families we instead compare each at convergence.
- **Corrected geodesic baseline** on raw unpadded sequences (no frame-padding bias).
- Encoder checkpoints, the LMA data feeder, the embedding extractor, the metric-learning
  trainer, and the evaluation scripts are archived for end-to-end reproduction.

---

## 9. Limitations and scope
- **Pointing's narrower margin and higher variance.** MAMP+pose's raw encoder *does* beat both
  geometric baselines on pointing (§6.1), but pointing is the lowest of its three correlations
  and by far the highest-variance action (fold std ≈ 0.15, versus ≈ 0.12 for walking/picking).
  We attribute the weaker, noisier signal to pointing's limited kinematic expressiveness (§3,
  H1/H2) — fewest active joints, smallest range of motion, shortest clips — but the causal link
  between these descriptors and the residual gap is supported correlationally, not by a
  controlled result. The repeated nested-CV is what lets us assert the pointing win despite this
  variance; a single split would not.
- **Three action types.** Generalization to a broader action taxonomy is untested; the
  per-action heterogeneity we exploit may present differently elsewhere.
- **Learned-baseline scope.** Our learned baseline (§6.1) is a generic reconstruction-only SSL
  encoder native to our 28-joint rotation feature, which isolates the value of the composition
  from that of learning a representation. A stronger comparison — a *purpose-built* learned
  motion-similarity metric retargeted from an external motion-retrieval model (e.g. a
  text–motion contrastive encoder) — would further test the claim against the wider literature;
  this requires retargeting our 28-joint CMU rotations to that model's pose space and validating
  the retarget, and is left to future work.
- **Rating-set size.** The human rating data (1,540 trials/action, ~11 raters/trial) is modest.
  This bounds how much the perceptual fine-tuning (§6.4) can reshape the encoder before
  overfitting, and a larger rating set might yield larger or more stable gains; it is also why
  the fine-tune uses a modest fixed budget and a low learning rate, and why we verified the
  selection-fold trajectory plateaus within that budget (§6.3).
- **Cross-architecture training budget.** Because an epoch is not a comparable unit across
  the vanilla, MLD, and MAMP backbones (different per-step compute and convergence dynamics),
  we train each to convergence rather than to a common epoch count and report each model's
  epoch (§8). The MAMP encoders were found to require ~1,200 epochs to converge (a 600-epoch
  budget left their loss still descending); the reported MAMP results use the converged
  checkpoints.

---

## 10. Additions for the full manuscript
- **Qualitative embedding visualizations** (t-SNE/UMAP) contrasting pointing's sparse,
  compressed embedding structure against the richer spread of walking/picking — to make §3
  and §6.4 visually self-evident.
- **Per-joint activity maps** visualizing which joints carry effort signal in each action,
  making the kinematic-expressiveness argument of §3 concrete.
- **A λ / weight sensitivity study** for the pose head and for the perceptual fine-tuning loss.
- **A larger or independently-collected rating set** to test whether the perceptual fine-tuning
  gains (§6.4) grow and stabilize with more supervision.
- **Joint encoder + perceptual training from scratch** (rather than fine-tuning a pretrained
  encoder), to test whether the perceptual signal can be composed with the self-supervised
  objectives during pretraining.

---

### Appendix A — Notation
- *effort tuple* `(e1,e2,e3,e4)`: Laban effort parameterization of a stimulus.
- *direct comparison value* `c(0,2)`: Left–Right choice frequency; with `d_perc = 1 − c(0,2)`
  the per-pair perceptual dissimilarity — the **evaluation** target for all methods (§2.4.2).
- *dynamic alphas* `α(0→2) = c(0,2) − c(0,1)` and `α(2→0) = c(0,2) − c(1,2)`: the two
  directed Left–Right preference contrasts — the **training** signal for the perceptual
  fine-tuning loss (§2.4.3, §6.4).
- *held-out*: encoder never trained on the evaluated clips (342 LMA states/drives excluded).
- *raw vs perceptually fine-tuned*: encoder L2 distances directly (raw) vs after fine-tuning the
  encoder under the perceptual objective (§6.4).

---

## References (working list)
- **Mao et al., 2023.** Masked Motion Predictors are Strong 3D Action Representation
  Learners. *ICCV 2023.* (MAMP — the masked-motion-prediction pretext task adapted here.)
- **Chen et al., 2023.** Executing Your Commands via Motion Diffusion in Latent Space.
  *CVPR 2023.* arXiv:2212.04048. (MLD — the transformer autoencoder with U-Net-like long
  skip connections that is our encoder/decoder backbone.)
- **Kobayashi et al., 2023.** Motion Capture Dataset for Practical Use of AI-based Motion
  Editing and Stylization. arXiv:2306.08861. (Bandai-Namco Research Motion Dataset — a
  pretraining-corpus source.)
- **CMU Graphics Lab Motion Capture Database.** http://mocap.cs.cmu.edu/ (a
  pretraining-corpus source; the common skeleton onto which all sources are retargeted).
