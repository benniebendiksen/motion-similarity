# AAAI-27 Reviews — verbatim (primary source)

Paper: *Motion Encoding for Human Perceptual Similarity.* Decision: **Reject** (Phase-1, did not advance).
Preserved verbatim as the source of record for the TMLR revision. Analysis in `01_REVIEWER_MAP.md`.

---

## Reviewer R2e3 — Rating 3 (Clear rejection), Confidence 3

**Paper Summary:** The paper proposes a single self-supervised encoder, MAMP+pose, which predicts human
motion-similarity judgments. Experimental validation shows that the model performs better than geometric
baselines and a reconstruction encoder across three varied actions — picking, pointing, and walking. Masked
prediction and pose reconstruction prove complementary, and light triplet-based supervision improves alignment
further. However, perceptually relevant structure is not captured by reconstruction fidelity, by linearly
accessible kinematics, or by large-scale semantic supervision alone. The paper also presents a
perceptual-similarity dataset and evaluation protocol in supplementary document to explain the findings.

**Strengths:**
- The paper presents rigorous experimental validation of the proposed framework.
- The ablation study shows effectiveness of masked-motion prediction and pose reconstruction in perceptual alignment.
- The supplementary material provides additional information to support the claims.

**Weaknesses:**
- Overall, I suspect heavy use of AI in generating the content which has affected readability of the paper.
  Abbreviations such as MAMP and TMR should be defined the first time they are used but the definitions appear
  much later in the document.
- The proposed work uses existing models such as the MAMP, CLIP and TMR which reduces the technical contribution.
- Although experimental results are provided for various configuration, very little explanation is provided to
  justify selection of the models and configurations.
- Specifically, the 'Effort' aspect of the dataset is unclear in what it means and how it influences the pose
  detection. Datasets, overview of the approach, and the experimental validation lack in providing justification
  of chosen approaches for framework design and validation.

**Justification of Recommendation:** The paper is difficult to read and lacks in justification of design choices
and the structure of the dataset. Discussion falls short to highlight the research contributions. A combination
of existing approaches is used with new data to address the research problem. The paper presents rigorous
validation of various configurations without satisfactorily justifying the design choices in light of the
existing work.

**Specific Points for Rebuttal:** CLIP has higher compute time and although the ablation study shows the effect
of various encoder decoder configurations, I would like to see how the model performs with and without CLIP.
Also, refer to the other comments under weaknesses.

---

## Reviewer FV7E — Rating 6 (Marginally above acceptance), Confidence 3

*"A careful perceptual-similarity benchmark and encoder study, with remaining concerns about stimulus diversity
and statistical independence."*

**Paper Summary:** Asks whether learned motion representations align with human judgments of motion similarity
better than standard kinematic distances. Authors generate 56 LMA Effort variations plus a neutral motion for
each of three base actions — walking, pointing, picking — and collect MTurk triplet judgments. The frequency
with which the two non-neutral motions are selected as most similar defines an ordinal dissimilarity target. The
proposed encoder MAMP+pose combines masked motion-delta prediction with an auxiliary pose-reconstruction head;
mean-pooled clip embeddings are evaluated by Spearman correlation with human-derived rankings. Compares against
DTW, rotation-geodesic, reconstruction AEs, and TMR; architecture/objective ablations; perceptual triplet
fine-tuning; and whether alignment is explained by reconstruction fidelity or pose decodability. Repeated nested
CV over the 57 Effort classes per action.

**Strengths:**
- The human-judgment dataset and protocol are potentially valuable: 1,540 unique triplets per action, median 11
  raters, de-identified raw responses, deterministic aggregation code, explicit train/select/test folds.
- Evaluation substantially more rigorous than a single split: 5×5 nested CV, held-out checkpoint selection,
  paired tests, Holm correction.
- Strong conceptual controls: the 35M-param reconstruction model improves reconstruction while degrading
  alignment (dissociates capacity/fidelity from perceptual agreement); TMR comparison + identical fine-tuning
  separate semantic from perceptual supervision.
- Claims generally qualified (decodability association only pointing, non-causal, TMR still better there).
- Paper and supplement clear and reproducible at the analysis level (bundle includes responses, aggregates,
  embeddings, fold-level results, significance code; large checkpoints omitted).

**Weaknesses:**
- **Mirrored clips do not add independent perceptual evidence (major/moderate).** Every stimulus mirrored and
  inherits the original judgment, doubling 171→342 clips and 4,620→9,240 triplet records without new ratings. If
  original and mirrored can enter different train/eval partitions, this is dependence or leakage; even if grouped
  by Effort key, calling the doubled set a larger corpus overstates effective sample size. Should state grouping
  and report results without mirroring.
- **Baseline coverage could be stronger (moderate).** DTW and joint-wise geodesics reasonable, but stronger
  learned motion representations and perceptual metric-learning baselines are missing. TMR is converted through
  BVH-to-SMPL with ≈4cm error → not a clean matched comparison. Raw MAMP+pose in the matched-corpus TMR
  experiment (0.462/0.302/0.481) is materially lower than the headline (0.552/0.434/0.543), making pretraining
  corpus/domain important.
- **"Limited perceptual supervision rivaling large-scale semantic supervision" is somewhat overstated
  (moderate).** After identical fine-tuning, MAMP+pose beats TMR only on walking before Holm; adjusted result is
  a trend; ties on picking; loses clearly on pointing. Interesting mixed result, not broad parity.
- **Scope of generalization is narrow (moderate).** All rated stimuli synthetic, single skeleton, single
  presentation protocol, only three actions. Claims should be limited to this benchmark until validated on
  natural motion and additional content instances.

**Justification:** Contributes a useful dataset, a careful nested evaluation pipeline, and a well-controlled
study showing masked motion prediction + pose reconstruction aligns better than tested geometric and
reconstruction baselines. Strengths narrowly outweigh limitations. Score capped because effective perceptual
diversity is small, the target uses only one aspect of each triplet, and significance appears to treat
repeated-CV folds as more independent than they are. Would rate higher with grouped/no-mirror analyses,
reliability statistics, hierarchical uncertainty over Effort classes/raters, and validation on multiple base
motions or natural motion clips.

**Specific Points for Rebuttal:**
1. Are an original clip and its mirror always assigned to the same outer fold via a shared Effort key? Please
   report the headline using only the 171 unmirrored clips if available.
2. Why are the 25 repeated-CV fold scores treated as independent for SE and Wilcoxon? Do conclusions hold under
   a bootstrap over the 56 Effort classes or averaging folds within each repeat (n=5)?
3. Please explain the gap between headline MAMP+pose results and the corpus-matched TMR comparison. How much is
   attributable to the different pretraining corpus versus representation conversion?

---

## AI Reviewer

**Synopsis:** Studies whether learned motion embeddings match human similarity judgments better than geometric
and reconstruction-based distances. Introduces a perceptual corpus of controlled style variations across
walking, pointing, picking; proposes MAMP+pose (masked motion prediction + auxiliary pose reconstruction). Also
examines perceptual ranking fine-tuning, semantic supervision, reconstruction fidelity, and pose decodability.

**Strengths:**
- Stimulus construction systematically covers controlled within-action style variation (all pairs among 56
  non-neutral Effort configs vs a common neutral; repeated judgments; three actions).
- Ablations reveal meaningful architecture–objective interactions (Tables 3,5,7; pose head benefits plain masked
  encoder but not U-Net variant).
- Pretraining comparisons address contamination and corpus scale (Sec 3.3 excludes eval clips; AMASS subset
  matched to HumanML3D for Table 8).
- Capacity experiment tests whether reconstruction accuracy explains alignment (Sec 5.5: scaling lowers recon
  loss 0.00563→0.00305 while reducing alignment on walking/picking).

**Weaknesses:**
- **Central target not established as context-independent pairwise dissimilarity.** Sec 3.1 defines the target
  via choice among Left/Neutral/Right; this probability depends on both motions' relations to neutral, not only
  their mutual similarity. Sec 3.4 calls it ordinal whereas Sec 5.4 uses numerical gaps as cardinal margins — so
  evidence supports neutral-conditioned choice ranking, not a general pairwise metric.
- **Novelty insufficiently distinguished** from MotionCritic (Wang et al. 2025), MotionBERT (Zhu et al. 2023),
  SkeletonMAE (Yan et al. 2023), absent from Sec 2/refs.
- **Headline reconstruction comparison omits stronger tested configs.** Table 4 compares only with the plain
  transformer; Table 6 reports MLD VAE exceeds MAMP+pose on walking (.597 vs .552) and excludes it as "context."
- **Reported significance does not account for dependence among repeated-CV scores** (25 from 5×5; signed-rank
  and SE across overlapping folds as if independent; Holm addresses multiplicity, not dependence).
- **Absolute alignment not calibrated against human reliability.** Best correlations .43–.55; margins over
  strongest baseline .16 on pointing but only .06/.05 on walking/picking; targets from 10–18 judgments with no
  main-text reliability or noise ceiling.
- **Benchmark evaluates a narrow form of generalization** (one base motion/action via PERFORM; inherited mirror
  labels; class-wise CV tests new styles from the same base execution and generator; no transfer to new
  performers, executions, body shapes, capture systems, or natural motion).

**Suggestions:** (1) Factor-level error analysis (Space/Weight/Time/Flow, State vs Drive, active region,
padding, agreement). (2) Body-aware or direction-sensitive representation (localized pooling / explicit
directional features) for the pointing gap to TMR. (3) Move complete model/losses/fine-tuning before design-
search results (Secs 4.1–4.2 present tables before MAMP+pose fully specified; protocol/objective appear later).

**Minor:** Sec 3.2 feature-layout inconsistency (channels + 3-D root translation while retaining 204). Sec 4.3
offset in delta undefined; masked-vs-all patches for pose/motion losses unspecified. Table 8 should mark
MAMP+pose rows AMASS-pretrained. Sec 5.1 should state whether quaternion Euclidean DTW is sign-invariant /
path-length normalized.

**References cited:** Wang et al. (2025) *Aligning human motion generation with human perceptions*, ICLR. Yan et
al. (2023) *SkeletonMAE*, ICCV 5606–5618. Zhu et al. (2023) *MotionBERT*, ICCV 15085–15099.

---

## Program Chairs — Decision: Reject
Did not advance to Phase 2. Reviewed by ≥2 human reviewers and ≥1 senior PC member in Phase 1; papers with ≥1
positive review examined by a program chair to ensure rejection justified. Decision final.
