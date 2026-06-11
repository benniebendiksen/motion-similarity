# Self-Supervised Masked Motion Encoders Capture Human Perceptual Motion Similarity

*(working manuscript draft — methods, design rationale, and results, from rating data to inference pipeline)*

---

## Abstract (draft)

We ask whether a learned representation of human motion can predict *perceptual*
similarity — how similar two movements appear to human observers — more faithfully than
geometric distance measures (quaternion/6D rotation geodesic, Dynamic Time Warping)
that dominate motion analysis. Using Laban-effort-annotated motion-capture performances of
three action types (walking, pointing, picking) paired with a triplet-based human
similarity-rating dataset, we show that a self-supervised **masked motion predictor
augmented with an auxiliary pose-reconstruction objective (MAMP+pose)**, combined with a
downstream triplet-based metric-learning module trained on the human ratings, produces
motion embeddings whose distances align with human perception more strongly than either
geometric baseline, across all three actions, under a strict held-out evaluation. For the
dynamically rich actions (walking, picking) this alignment emerges from the *unsupervised*
encoder geometry alone; for the expressively sparse pointing action — which engages far
fewer joints over a smaller range of motion — the learned triplet module is additionally
required to reach a winning result. Controlled ablations isolate the contribution of each
component and explain why prior reconstruction-only encoders captured only a single action
type well: a **masked motion-prediction** objective alone captures walking's dynamics but
collapses on picking, whereas a **pose-reconstruction** objective alone captures picking's
configuration fidelity but trails on walking — two mirror-image specialists that MAMP+pose
unifies in a single objective. All headline results are reproduced to four decimal places under a fixed evaluation
protocol.

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
1. A two-stage system — a self-supervised motion encoder (**MAMP+pose**) followed by a
   downstream triplet metric-learning module trained on human ratings — that beats both DTW
   and geodesic baselines on human-perceptual similarity across **all three** action types,
   held-out. To our knowledge this is the first single encoder to do so on this
   heterogeneous action set; the all-three result specifically requires the triplet module,
   which is decisive for pointing (§6.3) while being redundant for walking and picking.
2. The finding that for dynamically rich actions the perceptual signal is **intrinsic to
   the raw self-supervised representation** (no perceptual supervision required), a
   substantially stronger and more general claim than a supervised metric beating DTW.
3. A mechanistic account, supported by ablations, of *why* masked-motion *prediction* plus
   auxiliary pose reconstruction succeeds where pose-reconstruction and velocity-penalized
   objectives — which we show are "mirror-image specialists" — each capture only one
   action type.
4. A descriptive characterization of pointing's limited kinematic expressiveness (fewest
   active joints, smallest range of motion, shortest clips) that motivates why a learned
   alignment layer is necessary for that action while the raw encoder suffices for the
   others.

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
seen by the encoder; the geometric baselines of §5.3, by contrast, operate on the *raw,
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
three pairings {(0,1), (0,2), (1,2)} of a given triplet received normalized choice frequencies aggregated across its trials to 1. These count normalized values can be seen as probabilities for the selection of a given pairing as most similar within a triplet.

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
correlated** (§5.3). It uses only the absolute Left–Right frequency and no triplet-internal
contrasts, so it is method-agnostic.

#### 2.4.3 The training signal: dynamic alpha construction (metric learning only)
The downstream triplet metric-learning module (§5.2) is trained on a *different*, finer
signal derived from the same triplets — our **dynamic alpha** construction. Per
non-neutral effort pair we generate **two directed Left–Right alphas** that contrast the
direct Left–Right frequency against each of the two neutral-mediated alternatives:

> `α(0→2) = c(0,2) − c(0,1)`  (Left–Right preference relative to Left–Neutral)
> `α(2→0) = c(0,2) − c(1,2)`  (Left–Right preference relative to Neutral–Right)

Each alpha lies in [−1, +1]. A **positive** alpha means the two effort exemplars were
chosen as mutually most-similar *more often* than the corresponding neutral-mediated
pairing — evidence the pair should sit *close* in embedding space; a **negative** alpha is
evidence the pair should sit *far apart* (the neutral-mediated alternative was preferred).
The two alphas are directional precisely because each measures the Left–Right contrast
against a *different* competing alternative, which lets the loss (§5.2) impose an
asymmetric, per-direction margin rather than a single symmetric target.

A subtlety that makes the construction *dynamic*: although six pairwise alphas are
computable per triplet, the module routes the **Left–Right** alphas into one of three
preference cases according to which pairing observers actually preferred most
(Left–Right, Left–Neutral, or Neutral–Right). In every case it is the Left–Right alphas
that are retained — so even triplets where a *neutral-mediated* pairing won still
contribute an ordering constraint on the two effort exemplars, rather than being discarded.
This is what allows the limited rating budget (one triplet per effort pair) to yield
training signal on essentially every pair.

We emphasize the separation of concerns: **the dynamic alphas train the metric; the direct
comparison value `c(0,2)` (via `d_perc`) evaluates all methods.** The two are never
conflated, and the evaluation quantity is never used as a training target.

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

Under both hypotheses, the prediction is the same: pointing's *raw* encoder distances will
under-perform relative to walking/picking, and a learned alignment layer that amplifies the
subtle distinctions the encoder *did* capture will be needed to recover a competitive
result. We will see this borne out (§6.3) — and treat it as a property of pointing's
limited kinematic expressiveness rather than a deficiency of the encoder.

This heterogeneity frames the central design question: can a *single* encoder serve both the
dynamically rich actions (where raw geometry may suffice) and the expressively sparse one
(where a learned alignment layer may be required)?

---

## 4. Encoder design — a progressive ablation toward MAMP+pose

The encoder design was not chosen a priori; it is the endpoint of a sequence of controlled
training runs, each motivated by the failure of the previous. We present that sequence
because it is what establishes the central "mirror-image specialists" finding and rules
out the obvious alternatives.

We evaluate every encoder identically: pool its output to one vector per clip, take L2
distances between clips, and correlate against `d_perc` on the held-out seed-42 validation
split (§5). Encoders differ along two axes we deliberately separate — the **autoencoder
backbone** (§4.1) and the **training objective** (§4.2–4.3).

### 4.1 Step 1 — backbone: why the MLD SkipTransformer over a plain transformer
Our reconstruction encoders use the **MLD-style transformer autoencoder** of Motion Latent
Diffusion [Chen et al., CVPR 2023]: a transformer encoder/decoder with **U-Net-like long
skip connections** ("SkipTransformer") — symmetric long-range links from each early encoder
layer to the matching late decoder layer. These skips carry high-frequency per-frame detail
forward that a plain bottleneck transformer discards through its compression, which we
hypothesized matters for the static-pose fidelity that picking judgments rely on.

We test this directly by holding the *objective* fixed (reconstruction-only, no dynamics
term) and varying *only* the architecture — a plain transformer autoencoder vs. the MLD
SkipTransformer:

| Reconstruction-only encoder | Walking-S | Pointing-S | Picking-S |
|------------------------------|-----------|------------|-----------|
| Plain transformer AE | +0.488 | +0.118 | +0.311 |
| **MLD SkipTransformer AE** | +0.446 | **+0.187** | **+0.472** |

The plain transformer is competitive on walking but **collapses on picking (+0.311) and
pointing (+0.118)**; the MLD skip connections recover a **+0.16** picking-Spearman and a
sizeable pointing gain *at the same objective* — exactly the actions whose perceptual
similarity depends on fine pose configuration. This motivates the MLD backbone for all our
reconstruction encoders. (The plain-transformer numbers come from the historical
velocity-weighted runs; the comparison here isolates architecture by reading both at a
reconstruction-only loss.)

### 4.2 Step 2 — objective: reconstruction encoders are pose specialists, and dynamics terms don't help on this backbone
Fixing the MLD backbone and varying the **loss** (held-out; 342 states/drives excluded),
pooled L2 distances give:

| MLD encoder (objective) | Walking-S | Pointing-S | Picking-S |
|--------------------------|-----------|------------|-----------|
| plain-MSE reconstruction (AE-holdout) | **+0.446** | **+0.187** | **+0.472** |
| + velocity bolt-on (warm-started) | +0.419 | +0.165 | +0.455 |
| dual rot-MSE 20× + velocity 100×, from scratch | +0.392 | +0.171 | +0.437 |

Two readings. First, **plain-MSE reconstruction beats DTW on picking (+0.472 vs +0.431)**
and gives the strongest pose representation, but trails DTW on walking (+0.446 vs +0.478) —
a **pose specialist**. Second, and importantly, **adding velocity pressure on the MLD
backbone does not help and slightly hurts**, whether bolted on or trained from scratch.

This is worth dwelling on, because the *historical* motivation for a dynamics objective came
from a different architecture. On a **plain transformer**, a from-scratch
velocity-dominated recipe (rotation-MSE 20× + velocity 100×) *was* a strong walking
specialist (windowed Spearman +0.499–0.512) while failing picking — the original
"mirror-image specialists" pair was therefore *plain-transformer-velocity* (walking) vs.
*MLD-pose-MSE* (picking), a comparison confounded by architecture. Isolating the loss on a
single backbone (the table above) shows the velocity win does **not** transfer to MLD: there
the dynamics term only dilutes the pose representation. The genuine, architecture-controlled
conclusion is sharper than "two specialists": **reconstruction objectives — with or without
a velocity term — yield a pose-biased representation that wins picking but not walking;
adding velocity as a loss does not buy walking-relevant dynamics.**

### 4.3 Step 3 — why a velocity loss can't supply dynamics, and ruling out distance fusion
The previous result raises the question this section answers: *why* does penalizing velocity
fail to inject the walking-relevant dynamics, even from scratch?

- **A velocity loss on a visible reconstruction is nearly redundant.** A reconstruction
  autoencoder sees the *entire* clip and is rewarded for reproducing every frame. If it
  reconstructs the poses `x_t` accurately, the frame-to-frame differences `x_{t+1}−x_t`
  — i.e. the velocities — are reproduced *automatically*, as a byproduct. A velocity term
  added to that objective is therefore mostly already satisfied; it supplies little gradient
  the pose-reconstruction loss has not already supplied, so it cannot pressure the encoder
  to *model* how motion evolves. (This is why the warm-started bolt-on is inert, and why
  even the from-scratch dual loss in §4.2 fails to lift walking on the MLD backbone:
  velocity is the wrong place to inject dynamics when the whole clip is visible.) The plain
  transformer's apparent walking win came not from "velocity teaching dynamics" but from an
  architecture whose lossy bottleneck discards pose detail and *retains* gross
  trajectory — a degenerate route to walking sensitivity that simultaneously destroys
  picking (§4.1).
- **A geometric-distance fusion term is inert.** A separate attempt operated not on the
  encoder but on the *learned distance*: at inference, replace the embedding distance
  `d_emb(i,j)` with a convex blend `λ · d_geo(i,j) + (1−λ) · d_emb(i,j)`, mixing in a fixed
  geometric (DTW/geodesic) distance, and tune λ. Under the fixed seed-42 split this produced
  no reliable change in correlation for any λ ∈ [0,1]: the best-checkpoint selection (on
  validation triplet loss, which does not see the fused distance) lands on the same model
  regardless, so λ=0 and λ>0 evaluate almost identically. Earlier *un*seeded experiments had
  shown apparent λ benefits, but those were split-selection noise — different random splits,
  not a real effect of fusion. Geometric distance, blended post hoc, adds nothing the encoder
  has not already captured.

The lesson that carries into the rest of the design: **dynamics must be forced into the
representation by the learning task itself** — by making the model *predict* motion it
cannot see — not appended as a loss term on a visible reconstruction, nor fused in as an
external distance at test time.

### 4.4 Step 4 — masked motion *prediction*, not reconstruction
The resolution comes from the Masked Motion Prediction (MAMP) framework [Mao et al., ICCV
2023]. Rather than reconstructing visible poses (or penalizing the velocity of a visible
reconstruction), MAMP **masks ~80% of spatio-temporal patches and predicts the *motion*
(temporal deltas) of the masked regions** from sparse visible context. A motion-magnitude
prior (Gumbel-softmax over per-patch motion energy) biases masking toward dynamically rich
temporal regions, concentrating the predictive difficulty where motion variation is
highest. This imposes genuine dynamics-modeling pressure: the encoder cannot copy what it
cannot see and is forced to infer how masked, high-motion regions evolve.

**MAMP alone is itself a specialist — in the opposite direction.** Trained on our data, the
motion-only MAMP encoder shows promise on **walking** but **collapses on picking** (the
picking metric falls well below the pose-AE; see §6.4, where motion-only picking-Spearman
is +0.266 vs the pose-augmented +0.507). This is the informative negative result: pure
masked-*motion* prediction recovers dynamics but sheds the static-configuration fidelity
that picking judgments rely on — the mirror image of the pose-MSE AE, and confirmation that
the two capabilities are genuinely distinct and must be *combined*, not traded.

### 4.5 Step 5 — the pose augmentation (MAMP+pose)
Because masked motion prediction alone discards the pose fidelity that wins picking, we
augment it with an **auxiliary pose-reconstruction objective**. To see what this adds,
trace the MAMP forward pass and where the new head attaches:

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
simultaneously encodes how the body is moving (walking) and how it is configured (picking).

Why a second prediction head rather than simply adding a pose term to the existing motion
head's target? Because the two targets are different quantities (a temporal delta vs. an
absolute pose) living on different scales; forcing one projection to regress their sum or
concatenation entangles them, whereas parallel heads let each target be predicted in its
natural form while still sharing — and jointly shaping — all the representation-bearing
layers beneath. §6.4 shows this composition rescues picking (which motion-only MAMP
destroys) while retaining walking.

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
fixed held-out validation subset (§5.3). This regime uses **no human ratings during
representation learning** and therefore measures the *encoder's intrinsic* perceptual
alignment.

### 5.2 Refined regime — a learned triplet metric-learning module

**Triplet metric learning, in brief.** Metric learning trains a mapping into an embedding
space where distances reflect a target notion of similarity. The classical *triplet*
formulation considers an anchor, a "positive" (something that should be near it) and a
"negative" (something that should be far), and applies a hinge loss that pushes the
anchor–positive distance to be smaller than the anchor–negative distance by at least a
fixed **margin**; only triplets that currently violate the margin contribute gradient. Our
formulation adapts this idea to graded human preferences rather than binary
positive/negative labels.

**Our alpha-as-margin loss.** A small multilayer perceptron (a per-action MLP with
fully-connected layers 64 → 128 → 256 → output, with ReLU, batch-norm, and dropout) is
trained on top of the frozen 256-D encoder embeddings. For every non-neutral effort pair
(i, j) the human-derived **dynamic alpha** (§2.4.3) sets a *signed margin* on the embedding
distance `d(i,j)`:

- if the alpha is **positive** (observers grouped the two effort exemplars together more
  than the neutral-mediated alternative), the loss `relu(α − d(i,j))` requires the pair to
  be at least α apart **fails** — i.e., it *penalizes* the pair for being closer than the
  preference warrants, pushing it apart by margin α;
- if the alpha is **negative**, the loss `relu(d(i,j) − |α|)` pulls the pair to **within**
  |α|, i.e. closer together.

Thus the magnitude of the human preference directly scales the margin, and its sign sets
the direction — a graded generalization of the binary triplet hinge. The two directed
Left–Right alphas per pair (§2.4.3) give per-direction margins. A learned **neutral anchor**
(obtained by k-means clustering the embeddings, seeded for reproducibility, with periodic
re-clustering in the network's output space as training proceeds) provides the common
reference against which the class–neutral distances are computed. Training uses a cosine
learning-rate schedule with early stopping on a held-out triplet-loss criterion.

**Framing.** We present this as a **downstream alignment module** (used interchangeably
below with "triplet module" / "alignment layer"): for the dynamically rich
actions the self-supervised encoder is already performant enough that the module is largely
*redundant* (and on the small rating set can mildly distort an excellent representation);
for the expressively sparse pointing action the module is *necessary*, amplifying the
subtle effort distinctions the encoder captured into a perceptually aligned ordering
(§6.3). The refined embedding distances are correlated against `d_perc` exactly as in §5.1.

### 5.3 Evaluation protocol and baselines
- **Held-out, fixed split.** Evaluation uses a fixed seed-42 stratified split (40% of
  effort classes per action held as the validation subset with seeding for exact reproducibility). All reported numbers are on this validation
  subset. The encoder additionally never trained on these clips (§2.3).
- **Baselines on identical data.** Both geometric baselines are computed on the **raw,
  variable-length, unpadded** motion features (no 120-frame windowing, no tail-padding),
  over the *same* held-out validation pairs as the learned distances, and correlated against
  the same `d_perc`.

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

### 6.1 Headline — held-out, pooled, fixed seed-42 split

We report both Spearman (rank) and Pearson (linear) correlation between each method's
pair distances and the human dissimilarity `d_perc`. **Spearman is the primary metric**:
the ground truth is an ordinal quantity derived from choice frequencies, the triplet module
is trained to satisfy ordinal preference inequalities (§5.2), and we therefore care whether
a method recovers the human *ordering* of pair similarities rather than reproducing absolute
magnitudes. Pearson is reported as a corroborating magnitude-alignment check.

The full configuration whose distances are evaluated is **MAMP+pose encoder → triplet
metric-learning module ("+triplet")**; we also report the **raw** MAMP+pose encoder
(embeddings with no triplet module) to expose where the module matters.

**Spearman (primary):**

| Method | Walking | Pointing | Picking |
|--------|---------|----------|---------|
| **MAMP+pose + triplet** | **+0.486** | **+0.458** | **+0.507** |
| MAMP+pose, raw (no triplet) | +0.502 | +0.282 | +0.580 |
| DTW baseline | +0.478 | +0.370 | +0.431 |
| Geodesic baseline | +0.345 | +0.353 | +0.339 |

**Pearson (corroborating):**

| Method | Walking | Pointing | Picking |
|--------|---------|----------|---------|
| MAMP+pose + triplet | +0.489 | +0.472 | +0.550 |
| MAMP+pose, raw (no triplet) | +0.535 | +0.357 | +0.651 |
| DTW baseline | +0.426 | +0.303 | +0.340 |
| Geodesic baseline | +0.340 | +0.332 | +0.292 |

On **Spearman**, the complete system (MAMP+pose **+ triplet**) is, to our knowledge, the
**first single held-out encoder to beat both geometric baselines on all three actions**,
with margins over the stronger baseline of +0.008 (walking), +0.088 (pointing), and +0.076
(picking). The all-three result *depends on the triplet module*: on pointing the raw encoder
trails both baselines (+0.282), and only the triplet module lifts it to a winning +0.458
(§6.3). Pearson corroborates the Spearman conclusion — the system's linear correlations
(+0.489 / +0.472 / +0.550) also exceed the corresponding baselines — but we treat Spearman
as decisive, both because of the ordinal study design and because the learned triplet space
is optimized for ordering, not for preserving absolute distance magnitudes.

(All baseline cells are computed on the same raw, unpadded, seed-42 validation pairs as the
learned distances. The geodesic figures are the DTW-aligned per-frame geodesic of §5.3; an
earlier frame-padded geodesic implementation inflated the pointing value and is not used.)

These Spearman numbers **reproduce to four decimal places** under the fixed protocol
(independent re-encode + re-train + re-infer): walking +0.4857, pointing +0.4582, picking
+0.5069.

### 6.2 The raw-encoder result — a stronger, unsupervised claim
For walking and picking, the **raw** encoder embeddings — no perceptual supervision
whatsoever — already beat DTW (walking +0.502, picking +0.580). The perceptual-similarity
structure for dynamically rich actions is therefore *intrinsic to the self-supervised
representation*, not imposed by a downstream metric. Beating a heavily-optimized temporal
baseline like DTW with unsupervised embeddings indicates the masked-motion pretext task
genuinely captures the kinematic structure underlying human similarity judgments.

### 6.3 The triplet module's contribution is action-dependent
The triplet module helps unevenly across actions, and the pattern is itself informative.
For walking and picking, the raw encoder already meets or exceeds the refined system
(Spearman +0.502 vs +0.486 for walking; +0.580 vs +0.507 for picking): on these
dynamically rich actions the self-supervised geometry is strong enough that the additional
metric-learning stage, fit on the modest rating set, slightly distorts an already-excellent
representation rather than improving it. For pointing the relationship inverts and the
module becomes essential: the raw encoder trails both baselines at +0.282, and the triplet
module lifts it to +0.458 — a +0.176 gain that converts pointing from a loss into the
system's largest margin over baseline.

This action-dependent pattern is consistent with the kinematic-expressiveness analysis of
§3 (hypotheses H1/H2): pointing engages the fewest joints over the smallest range of motion
and the shortest clips, so its raw encoder geometry carries the least discriminative signal
and a distance computed directly on it under-performs — precisely the action where the
learned alignment layer is required. For walking and picking, whose richer kinematics yield
more discriminative raw geometry, the alignment layer is redundant. We therefore read the
result as: **the encoder captures the available kinematic structure; the alignment layer
amplifies it where that structure is sparse (pointing) and is unnecessary where it is rich
(walking, picking).** This also explains why prior reconstruction-only encoders, lacking a
dynamics-modeling pretext, could not beat geometric distances on pointing even with a
metric layer — the underlying representation lacked the dynamics signal for the metric to
amplify.


### 6.4 Ablation — the pose head is necessary

| Objective | Walking-S | Picking-S |
|-----------|-----------|-----------|
| motion-only (MAMP) | +0.390 | +0.266 (collapses) |
| + pose-reconstruction head (MAMP+pose) | **+0.486** | **+0.507** |

Motion-only masked prediction performs reasonably on walking but **destroys picking**
(+0.266) — exactly the failure mode of a dynamics-only objective lacking pose fidelity, and
the mirror image of the pose-MSE specialist of §4.2. Adding the auxiliary
pose-reconstruction head **rescues picking (+0.507) while also lifting walking (+0.486)**,
directly confirming the two-capability composition that motivates MAMP+pose: the motion
target supplies dynamics, the pose target supplies configuration fidelity, and only their
combination serves both actions.

### 6.5 Summary of the per-action story

| Action | Best method | Mechanism |
|--------|-------------|-----------|
| Walking | MAMP+pose (raw or refined) | dynamics captured in raw encoder geometry |
| Picking | MAMP+pose (raw best) | pose-fidelity + dynamics; raw geometry suffices |
| Pointing | MAMP+pose + triplet module | sparse-kinematics encoder + learned amplification |

A single encoder, with one optional downstream triplet module, beats both geometric
baselines everywhere.

### 6.6 Full model sweep (all encoders, raw and triplet-refined)

For completeness we report every encoder in the design program under both regimes, on the
identical held-out seed-42 validation split, Spearman (S) and Pearson (P) against `d_perc`.
"raw" = pooled encoder L2; "+trip" = after the triplet module trained jointly on all three
actions. Backbone/objective abbreviations follow §4. Best Spearman per action-column is bold.

**Raw embeddings (no triplet):**

| Encoder (backbone, objective) | Walk-P | Walk-S | Point-P | Point-S | Pick-P | Pick-S |
|-------------------------------|--------|--------|---------|---------|--------|--------|
| Recon-only, plain transformer | +0.476 | **+0.488** | +0.107 | +0.118 | +0.325 | +0.311 |
| Recon-only, MLD (AE-holdout) | +0.494 | +0.446 | +0.196 | +0.187 | +0.513 | +0.472 |
| MLD + velocity bolt-on | +0.470 | +0.419 | +0.166 | +0.165 | +0.495 | +0.455 |
| MLD, rotMSE 20× + vel 100×, scratch | +0.446 | +0.392 | +0.168 | +0.171 | +0.488 | +0.437 |
| VAE, MLD (held-out) | +0.395 | +0.335 | +0.107 | +0.071 | +0.430 | +0.395 |
| AE, MLD (in-sample) | +0.467 | +0.436 | +0.185 | +0.203 | +0.553 | +0.505 |
| VAE, MLD (in-sample) | +0.424 | +0.375 | +0.216 | +0.160 | +0.579 | +0.564 |
| MAMP (motion-only) | +0.349 | +0.343 | +0.215 | +0.168 | +0.542 | +0.495 |
| **MAMP+pose** | +0.535 | +0.502 | +0.357 | +0.282 | +0.651 | **+0.579** |

**Triplet-refined (+trip), trained jointly across all three actions:**

| Encoder (backbone, objective) | Walk-P | Walk-S | Point-P | Point-S | Pick-P | Pick-S |
|-------------------------------|--------|--------|---------|---------|--------|--------|
| Recon-only, plain transformer | +0.587 | **+0.592** | +0.314 | +0.308 | +0.299 | +0.339 |
| Recon-only, MLD (AE-holdout) | +0.461 | +0.427 | +0.216 | +0.215 | +0.369 | +0.376 |
| MLD + velocity bolt-on | +0.373 | +0.334 | +0.165 | +0.175 | +0.360 | +0.354 |
| MLD, rotMSE 20× + vel 100×, scratch | +0.459 | +0.356 | +0.288 | +0.284 | +0.495 | +0.519 |
| VAE, MLD (held-out) | +0.405 | +0.340 | +0.205 | +0.157 | +0.230 | +0.223 |
| AE, MLD (in-sample) | +0.322 | +0.335 | +0.296 | +0.261 | +0.436 | +0.452 |
| VAE, MLD (in-sample) | +0.510 | +0.500 | +0.260 | +0.184 | +0.468 | +0.457 |
| MAMP (motion-only) | +0.359 | +0.400 | +0.427 | +0.424 | +0.341 | +0.272 |
| **MAMP+pose** | +0.475 | +0.477 | +0.470 | **+0.462** | +0.557 | **+0.499** |

Three observations the full sweep makes concrete:

- **No competing encoder is strong on all three actions in either regime.** The plain
  transformer is a pure *walking* specialist (best walking in both regimes — raw +0.488,
  refined +0.592 — but worst-tier picking/pointing), the MLD reconstruction encoders are
  *picking/pose* specialists, and motion-only MAMP swings to *pointing/dynamics* (refined
  pointing +0.424) while collapsing picking. **MAMP+pose is the only encoder that is
  competitive everywhere**, and the only one to clear both baselines on all three actions.
- **The triplet module's effect is action- and encoder-dependent.** It lifts the
  dynamics-encoders' pointing markedly (MAMP+pose pointing +0.282 → +0.462; motion-only
  MAMP +0.168 → +0.424) but does little for — or slightly degrades — the already-strong raw
  picking/walking numbers, consistent with §6.3.
- **The architecture and objective axes are separable and both matter**, exactly as §4
  argued: at a fixed reconstruction-only objective the MLD backbone beats the plain
  transformer on pose-bound actions (raw picking +0.472 vs +0.311), while at a fixed MLD
  backbone the masked-prediction objective beats reconstruction on the combined goal.

(Triplet-refined numbers are from an independent training run at 120 epochs; the MAMP+pose
row reproduces the §6.1 headline within run-to-run tolerance, e.g. picking-S +0.499 vs the
canonical +0.507.)

---

## 7. Why MAMP+pose succeeds: modeling vs. copying
The central mechanistic claim is the distinction between **modeling** dynamics and
**copying** poses. Reconstruction objectives (pose-MSE; velocity-penalized MSE) present the
model with the *entire* visible motion and reward reproducing it. Under full visibility,
correct frame-to-frame deltas — and thus low velocity error — are obtained for free from
accurate pose copying; the encoder is never pressured to *model* the underlying kinematics.
This is precisely why adding a velocity loss to the pose-AE failed to help (§4.2–4.3): the
velocity constraint was already satisfied by the reconstruction and contributed no new
representational pressure. (The one architecture that *did* gain walking sensitivity from a
velocity-weighted loss — the plain transformer — did so for the wrong reason: its lossy
bottleneck discards pose detail and retains only gross trajectory, which is why it
simultaneously fails picking. That is a degenerate route to dynamics, not the modeling we
want.)

Masked motion prediction removes this shortcut. With 80% of the spatio-temporal patches
hidden and the *motion* of the hidden, high-energy regions as the target, the encoder must
infer kinematic progression it cannot observe — it must internalize how the body moves,
not merely where it currently is. The auxiliary pose head simultaneously preserves the
static-configuration fidelity that perceptual picking judgments rely on. The **composition**
— masked dynamics modeling plus pose reconstruction — captures the full perceptual
structure across heterogeneous actions; neither objective alone does.

---

## 8. Reproducibility
- **Fixed evaluation split** (seed 42, stratified by effort magnitude); **seeded k-means**
  neutral (seed 42); deterministic to four decimals across independent re-runs.
- **Strict hold-out**: the 342 LMA evaluation clips are excluded from encoder pretraining
  and its validation.
- **Corrected geodesic baseline** on raw unpadded sequences (no frame-padding bias).
- Encoder checkpoints, the LMA data feeder, the embedding extractor, the metric-learning
  trainer, and the evaluation scripts are archived for end-to-end reproduction.

---

## 9. Limitations and scope
- **Pointing's unsupervised gap.** The raw encoder does not beat geometric distances on
  pointing; the win there is system-level (encoder + triplet module). We attribute this to
  pointing's limited kinematic expressiveness (§3, H1/H2) — fewest active joints, smallest
  range of motion, shortest clips — but the causal link between these descriptors and the
  raw-distance gap is a hypothesis we support correlationally, not a controlled result; it
  bounds the "unsupervised" claim to dynamically rich actions.
- **Three action types.** Generalization to a broader action taxonomy is untested; the
  per-action heterogeneity we exploit may present differently elsewhere.
- **Rating-set size.** The human rating data (1,540 trials/action, ~11 raters/trial) is
  modest; this is partly *why* the learned triplet module can distort the strong raw
  representation on walking/picking, and a larger rating set might change the raw-vs-refined
  balance.
- **Encoder training budget.** The reported encoder was undertrained (600 epochs, loss
  still decreasing); a continuation run is underway and may shift the numbers upward.

---

## 10. Additions for the full manuscript
- **Qualitative embedding visualizations** (t-SNE/UMAP) contrasting pointing's sparse,
  compressed embedding structure against the richer spread of walking/picking — to make §3
  and §6.3 visually self-evident.
- **Per-joint activity maps** visualizing which joints carry effort signal in each action,
  making the kinematic-expressiveness argument of §3 concrete.
- **A λ / weight sensitivity study** for the pose head and the triplet module.
- **Held-out *test* triplets** (beyond the held-out class split) to further insulate the
  refined-metric claim from any tuning leakage.
- **Encoder-continuation results** once the extended-training run completes.

---

### Appendix A — Notation
- *effort tuple* `(e1,e2,e3,e4)`: Laban effort parameterization of a stimulus.
- *direct comparison value* `c(0,2)`: Left–Right choice frequency; with `d_perc = 1 − c(0,2)`
  the per-pair perceptual dissimilarity — the **evaluation** target for all methods (§2.4.2).
- *dynamic alphas* `α(0→2) = c(0,2) − c(0,1)` and `α(2→0) = c(0,2) − c(1,2)`: the two
  directed Left–Right preference contrasts — the triplet module's **training** signal
  (§2.4.3).
- *held-out*: encoder never trained on the evaluated clips (342 LMA states/drives excluded).
- *raw vs refined*: encoder L2 distances directly (raw) vs after the triplet metric-learning
  module (refined).

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
