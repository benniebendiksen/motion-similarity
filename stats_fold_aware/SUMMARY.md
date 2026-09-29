# TMLR stats fix — claim survival (S1)

Metric = raw Spearman ρ (embedding distance vs d_perc), mean over the 25 nested-CV test folds (the paper's number). **Bold = higher mean ρ** in each pair. Old = 25 folds as independent, Wilcoxon, Holm. Tier A = 5 repeat-means, paired t (df=4), Holm. **Tier B (primary) = paired cluster bootstrap over the 56 Effort classes (B=10,000), Holm** — gate reproduces all 27 model×action published fold-ρ sets to ≤2.2e-16; bootstrap SE cross-checked by delete-one-class jackknife (SE ratio median 1.01). Holm families = one per paper table.

## F1 headline vs baselines (tab:baselines)

| model A | ρ A | model B | ρ B | action | Δ (A−B) | 95% CI (class boot) | old p_Holm | tier-A p_Holm | **tier-B p_Holm** | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| MAMP+pose | **0.552** | DTW | 0.490 | walking | +0.063 | [-0.059, +0.182] | 0.021 | 0.068 | **0.2** | **LOST** |
| MAMP+pose | **0.434** | DTW | 0.273 | pointing | +0.162 | [+0.010, +0.305] | 0.0056 | 0.0025 | **0.069** | **LOST** |
| MAMP+pose | **0.543** | DTW | 0.303 | picking | +0.240 | [+0.077, +0.377] | 1.6e-05 | 0.0018 | **0.012** | SURVIVES |
| MAMP+pose | **0.552** | Geodesic | 0.371 | walking | +0.181 | [+0.034, +0.333] | 3.1e-05 | 0.01 | **0.04** | SURVIVES |
| MAMP+pose | **0.434** | Geodesic | 0.234 | pointing | +0.200 | [+0.065, +0.312] | 2.4e-06 | 0.0025 | **0.013** | SURVIVES |
| MAMP+pose | **0.543** | Geodesic | 0.264 | picking | +0.279 | [+0.115, +0.415] | 6.9e-06 | 0.0013 | **0.0072** | SURVIVES |
| MAMP+pose | **0.552** | Plain(recon) | 0.476 | walking | +0.076 | [+0.002, +0.164] | 0.0092 | 0.068 | **0.069** | **LOST** |
| MAMP+pose | **0.434** | Plain(recon) | 0.213 | pointing | +0.222 | [+0.097, +0.301] | 1.3e-06 | 0.0004 | **0.0027** | SURVIVES |
| MAMP+pose | **0.543** | Plain(recon) | 0.491 | picking | +0.052 | [-0.026, +0.114] | 0.0092 | 0.068 | **0.2** | **LOST** |

## F2 pose-head x U-Net 2x2 (tab:posehead, tab:archObj)

| model A | ρ A | model B | ρ B | action | Δ (A−B) | 95% CI (class boot) | old p_Holm | tier-A p_Holm | **tier-B p_Holm** | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| MAMP+pose | **0.552** | MAMP | 0.351 | walking | +0.201 | [+0.112, +0.282] | 5e-05 | 0.011 | **0.0024** | SURVIVES |
| MAMP+pose | **0.434** | MAMP | 0.299 | pointing | +0.135 | [+0.017, +0.234] | 0.0057 | 0.02 | **0.21** | **LOST** |
| MAMP+pose | **0.543** | MAMP | 0.490 | picking | +0.053 | [-0.014, +0.122] | 0.069 | 0.099 | **0.63** | null (was null) |
| MAMP-UNet+pose | **0.523** | MAMP-UNet | 0.470 | walking | +0.054 | [-0.035, +0.154] | 0.11 | 0.083 | **0.93** | null (was null) |
| MAMP-UNet+pose | 0.260 | MAMP-UNet | **0.275** | pointing | -0.016 | [-0.113, +0.117] | 0.82 | 0.85 | **1** | null (was null) |
| MAMP-UNet+pose | 0.485 | MAMP-UNet | **0.566** | picking | -0.081 | [-0.123, -0.011] | 0.001 | 0.00028 | **0.19** | **LOST** |
| MAMP-UNet | **0.470** | MAMP | 0.351 | walking | +0.118 | [+0.009, +0.206] | 0.01 | 0.15 | **0.25** | **LOST** |
| MAMP-UNet | 0.275 | MAMP | **0.299** | pointing | -0.024 | [-0.139, +0.076] | 0.82 | 0.85 | **1** | null (was null) |
| MAMP-UNet | **0.566** | MAMP | 0.490 | picking | +0.076 | [+0.005, +0.143] | 0.0092 | 0.026 | **0.25** | **LOST** |
| MAMP-UNet+pose | 0.523 | MAMP+pose | **0.552** | walking | -0.029 | [-0.102, +0.042] | 0.27 | 0.26 | **1** | null (was null) |
| MAMP-UNet+pose | 0.260 | MAMP+pose | **0.434** | pointing | -0.175 | [-0.239, -0.067] | 0.00011 | 0.0018 | **0.011** | SURVIVES |
| MAMP-UNet+pose | 0.485 | MAMP+pose | **0.543** | picking | -0.059 | [-0.102, +0.009] | 0.087 | 0.016 | **0.54** | null (was null) |

## F3b backbone under reconstruction (tab:backbone)

| model A | ρ A | model B | ρ B | action | Δ (A−B) | 95% CI (class boot) | old p_Holm | tier-A p_Holm | **tier-B p_Holm** | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| MLD-AE(recon) | **0.532** | Plain(recon) | 0.476 | walking | +0.056 | [-0.023, +0.138] | 0.43 | 0.52 | **0.34** | null (was null) |
| MLD-AE(recon) | **0.331** | Plain(recon) | 0.213 | pointing | +0.119 | [+0.036, +0.192] | 1.3e-05 | 0.017 | **0.017** | SURVIVES |
| MLD-AE(recon) | **0.498** | Plain(recon) | 0.491 | picking | +0.007 | [-0.054, +0.073] | 1 | 1 | **0.75** | null (was null) |

## F4 masking vs reconstruction refs (tab:sweep; MLD-VAE honesty)

| model A | ρ A | model B | ρ B | action | Δ (A−B) | 95% CI (class boot) | old p_Holm | tier-A p_Holm | **tier-B p_Holm** | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| MAMP+pose | **0.552** | MLD-AE(recon) | 0.532 | walking | +0.021 | [-0.040, +0.090] | 0.26 | 0.15 | **1** | null (was null) |
| MAMP+pose | **0.434** | MLD-AE(recon) | 0.331 | pointing | +0.103 | [-0.012, +0.175] | 0.013 | 0.0056 | **0.42** | **LOST** |
| MAMP+pose | **0.543** | MLD-AE(recon) | 0.498 | picking | +0.045 | [-0.022, +0.096] | 0.015 | 0.02 | **0.96** | **LOST** |
| MAMP+pose | 0.552 | MLD-VAE(recon) | **0.597** | walking | -0.044 | [-0.114, +0.039] | 0.037 | 0.022 | **1** | **LOST** |
| MAMP+pose | **0.434** | MLD-VAE(recon) | 0.324 | pointing | +0.110 | [+0.008, +0.181] | 0.00084 | 0.0051 | **0.19** | **LOST** |
| MAMP+pose | **0.543** | MLD-VAE(recon) | 0.501 | picking | +0.042 | [-0.034, +0.099] | 0.037 | 0.0015 | **1** | **LOST** |

## Per-model mean ρ with class-bootstrap 95% CI

| model | walking | pointing | picking |
|---|---|---|---|
| DTW | 0.490 [0.375, 0.589] | 0.273 [0.139, 0.396] | 0.303 [0.169, 0.419] |
| Geodesic | 0.371 [0.235, 0.485] | 0.234 [0.117, 0.351] | 0.264 [0.137, 0.374] |
| MAMP | 0.351 [0.247, 0.452] | 0.299 [0.191, 0.411] | 0.490 [0.350, 0.583] |
| MAMP+pose | 0.552 [0.447, 0.638] | 0.434 [0.311, 0.534] | 0.543 [0.404, 0.629] |
| MAMP-UNet | 0.470 [0.347, 0.558] | 0.275 [0.155, 0.386] | 0.566 [0.429, 0.639] |
| MAMP-UNet+pose | 0.523 [0.415, 0.612] | 0.260 [0.146, 0.392] | 0.485 [0.362, 0.585] |
| MLD-AE(recon) | 0.532 [0.423, 0.612] | 0.331 [0.213, 0.468] | 0.498 [0.371, 0.596] |
| MLD-VAE(recon) | 0.597 [0.486, 0.669] | 0.324 [0.212, 0.445] | 0.501 [0.376, 0.597] |
| Plain(recon) | 0.476 [0.359, 0.567] | 0.213 [0.112, 0.340] | 0.491 [0.360, 0.588] |
