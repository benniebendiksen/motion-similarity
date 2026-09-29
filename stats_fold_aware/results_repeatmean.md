## Per-model: mean | old SE (25 folds) | fold-aware SE (5 repeat-means)

| model | walking | pointing | picking |
|---|---|---|---|
| MAMP+pose | 0.552 (0.026 → 0.013) | 0.434 (0.030 → 0.014) | 0.543 (0.033 → 0.006) |
| MAMP | 0.351 (0.032 → 0.030) | 0.299 (0.032 → 0.033) | 0.490 (0.039 → 0.015) |
| MAMP-UNet | 0.470 (0.029 → 0.017) | 0.275 (0.028 → 0.021) | 0.566 (0.030 → 0.006) |
| MAMP-UNet+pose | 0.523 (0.028 → 0.017) | 0.260 (0.036 → 0.025) | 0.485 (0.029 → 0.008) |
| DTW | 0.490 (0.028 → 0.022) | 0.273 (0.039 → 0.024) | 0.303 (0.038 → 0.022) |
| Geodesic | 0.371 (0.028 → 0.020) | 0.234 (0.031 → 0.033) | 0.264 (0.030 → 0.026) |
| Plain(recon) | 0.476 (0.031 → 0.022) | 0.213 (0.034 → 0.024) | 0.491 (0.029 → 0.021) |
| Plain(recon+vel) | 0.472 (0.026 → 0.017) | 0.205 (0.039 → 0.025) | 0.399 (0.037 → 0.021) |
| Plain(vel) | 0.321 (0.033 → 0.009) | 0.362 (0.036 → 0.021) | 0.471 (0.033 → 0.013) |
| Plain+skips(recon) | 0.471 (0.024 → 0.016) | 0.221 (0.033 → 0.026) | 0.456 (0.032 → 0.016) |
| Plain-35M(recon) | 0.349 (0.037 → 0.037) | 0.221 (0.037 → 0.020) | 0.301 (0.032 → 0.033) |
| MLD-AE(recon) | 0.532 (0.023 → 0.019) | 0.331 (0.039 → 0.020) | 0.498 (0.029 → 0.013) |
| MLD-AE_full(recon) | 0.532 (0.023 → 0.019) | 0.331 (0.039 → 0.020) | 0.498 (0.029 → 0.013) |
| MLD-AE-noskip | 0.570 (0.022 → 0.021) | 0.330 (0.039 → 0.021) | 0.513 (0.030 → 0.009) |
| MLD-AE(recon+vel) | 0.537 (0.022 → 0.015) | 0.339 (0.035 → 0.021) | 0.493 (0.030 → 0.009) |
| MLD-AE(vel) | 0.451 (0.031 → 0.018) | 0.278 (0.031 → 0.033) | 0.254 (0.039 → 0.024) |
| MLD-VAE(recon) | 0.597 (0.021 → 0.010) | 0.324 (0.032 → 0.020) | 0.501 (0.029 → 0.007) |
| cut1 MAMP+pose | 0.462 (0.027 → 0.017) | 0.302 (0.031 → 0.013) | 0.481 (0.031 → 0.013) |
| cut1 TMR | 0.452 (0.025 → 0.016) | 0.425 (0.036 → 0.027) | 0.408 (0.029 → 0.020) |
| cut1 MAMP+pose+FT | 0.699 (0.022 → 0.011) | 0.463 (0.035 → 0.027) | 0.605 (0.031 → 0.013) |
| cut1 TMR+FT | 0.649 (0.025 → 0.013) | 0.573 (0.036 → 0.017) | 0.611 (0.034 → 0.017) |

## F1 headline vs baselines (tab:baselines)  (Holm m=9)

| x vs y | action | Δ | 95% CI (t4) | repeats same sign | old p (W25) → Holm | repeat-mean p → Holm | verdict |
|---|---|---|---|---|---|---|---|
| MAMP+pose vs DTW (1s) | walking | +0.063 | [-0.012, +0.137] | 4/5 | 0.021 → 0.021 | 0.04 → 0.068 | LOST |
| MAMP+pose vs DTW (1s) | pointing | +0.162 | [+0.112, +0.211] | 5/5 | 0.0014 → 0.0056 | 0.00042 → 0.0025 | SURVIVES |
| MAMP+pose vs DTW (1s) | picking | +0.240 | [+0.175, +0.306] | 5/5 | 2.6e-06 → 1.6e-05 | 0.00026 → 0.0018 | SURVIVES |
| MAMP+pose vs Geodesic (1s) | walking | +0.181 | [+0.091, +0.272] | 5/5 | 6.2e-06 → 3.1e-05 | 0.0025 → 0.01 | SURVIVES |
| MAMP+pose vs Geodesic (1s) | pointing | +0.200 | [+0.137, +0.263] | 5/5 | 3e-07 → 2.4e-06 | 0.00045 → 0.0025 | SURVIVES |
| MAMP+pose vs Geodesic (1s) | picking | +0.279 | [+0.212, +0.346] | 5/5 | 9.8e-07 → 6.9e-06 | 0.00016 → 0.0013 | SURVIVES |
| MAMP+pose vs Plain(recon) (1s) | walking | +0.076 | [+0.002, +0.150] | 5/5 | 0.0031 → 0.0092 | 0.023 → 0.068 | LOST |
| MAMP+pose vs Plain(recon) (1s) | pointing | +0.222 | [+0.183, +0.260] | 5/5 | 1.5e-07 → 1.3e-06 | 4.5e-05 → 0.0004 | SURVIVES |
| MAMP+pose vs Plain(recon) (1s) | picking | +0.052 | [+0.002, +0.103] | 5/5 | 0.0044 → 0.0092 | 0.023 → 0.068 | LOST |

## F2 pose-head x U-Net 2x2 (tab:posehead, tab:archObj)  (Holm m=12)

| x vs y | action | Δ | 95% CI (t4) | repeats same sign | old p (W25) → Holm | repeat-mean p → Holm | verdict |
|---|---|---|---|---|---|---|---|
| MAMP+pose vs MAMP (2s) | walking | +0.201 | [+0.135, +0.267] | 5/5 | 4.2e-06 → 5e-05 | 0.0011 → 0.011 | SURVIVES |
| MAMP+pose vs MAMP (2s) | pointing | +0.135 | [+0.080, +0.190] | 5/5 | 0.00063 → 0.0057 | 0.0025 → 0.02 | SURVIVES |
| MAMP+pose vs MAMP (2s) | picking | +0.053 | [+0.014, +0.093] | 5/5 | 0.011 → 0.069 | 0.02 → 0.099 | null (both) |
| MAMP-UNet+pose vs MAMP-UNet (2s) | walking | +0.054 | [+0.018, +0.090] | 5/5 | 0.027 → 0.11 | 0.014 → 0.083 | null (both) |
| MAMP-UNet+pose vs MAMP-UNet (2s) | pointing | -0.016 | [-0.069, +0.038] | 3/5 | 0.41 → 0.82 | 0.46 → 0.85 | null (both) |
| MAMP-UNet+pose vs MAMP-UNet (2s) | picking | -0.081 | [-0.091, -0.071] | 5/5 | 0.0001 → 0.001 | 2.4e-05 → 0.00028 | SURVIVES |
| MAMP-UNet vs MAMP (2s) | walking | +0.118 | [+0.012, +0.225] | 4/5 | 0.0015 → 0.01 | 0.037 → 0.15 | LOST |
| MAMP-UNet vs MAMP (2s) | pointing | -0.024 | [-0.101, +0.052] | 3/5 | 0.56 → 0.82 | 0.43 → 0.85 | null (both) |
| MAMP-UNet vs MAMP (2s) | picking | +0.076 | [+0.041, +0.111] | 5/5 | 0.0012 → 0.0092 | 0.0038 → 0.026 | SURVIVES |
| MAMP-UNet+pose vs MAMP+pose (2s) | walking | -0.029 | [-0.064, +0.007] | 4/5 | 0.09 → 0.27 | 0.088 → 0.26 | null (both) |
| MAMP-UNet+pose vs MAMP+pose (2s) | pointing | -0.175 | [-0.210, -0.139] | 5/5 | 1e-05 → 0.00011 | 0.00016 → 0.0018 | SURVIVES |
| MAMP-UNet+pose vs MAMP+pose (2s) | picking | -0.059 | [-0.081, -0.037] | 5/5 | 0.017 → 0.087 | 0.0018 → 0.016 | GAINED |

## F3 backbone + skips under reconstruction (tab:backbone, 4-corner)  (Holm m=9)

| x vs y | action | Δ | 95% CI (t4) | repeats same sign | old p (W25) → Holm | repeat-mean p → Holm | verdict |
|---|---|---|---|---|---|---|---|
| MLD-AE(recon) vs Plain(recon) (2s) | walking | +0.056 | [-0.018, +0.130] | 4/5 | 0.071 → 0.43 | 0.1 → 0.52 | null (both) |
| MLD-AE(recon) vs Plain(recon) (2s) | pointing | +0.119 | [+0.074, +0.164] | 5/5 | 1.5e-06 → 1.3e-05 | 0.0018 → 0.017 | SURVIVES |
| MLD-AE(recon) vs Plain(recon) (2s) | picking | +0.007 | [-0.041, +0.056] | 3/5 | 0.73 → 1 | 0.7 → 1 | null (both) |
| Plain+skips(recon) vs Plain(recon) (2s) | walking | -0.005 | [-0.070, +0.060] | 3/5 | 0.12 → 0.48 | 0.85 → 1 | null (both) |
| Plain+skips(recon) vs Plain(recon) (2s) | pointing | +0.008 | [-0.020, +0.037] | 3/5 | 0.62 → 1 | 0.46 → 1 | null (both) |
| Plain+skips(recon) vs Plain(recon) (2s) | picking | -0.035 | [-0.065, -0.005] | 5/5 | 0.032 → 0.22 | 0.032 → 0.26 | null (both) |
| MLD-AE-noskip vs MLD-AE_full(recon) (2s) | walking | +0.038 | [+0.001, +0.075] | 5/5 | 0.0038 → 0.03 | 0.047 → 0.28 | LOST |
| MLD-AE-noskip vs MLD-AE_full(recon) (2s) | pointing | -0.000 | [-0.007, +0.006] | 3/5 | 0.75 → 1 | 0.89 → 1 | null (both) |
| MLD-AE-noskip vs MLD-AE_full(recon) (2s) | picking | +0.015 | [+0.002, +0.028] | 5/5 | 0.085 → 0.43 | 0.033 → 0.26 | null (both) |

## F4 masking vs reconstruction refs (tab:sweep; MLD-VAE honesty)  (Holm m=6)

| x vs y | action | Δ | 95% CI (t4) | repeats same sign | old p (W25) → Holm | repeat-mean p → Holm | verdict |
|---|---|---|---|---|---|---|---|
| MAMP+pose vs MLD-AE(recon) (2s) | walking | +0.021 | [-0.011, +0.052] | 3/5 | 0.26 → 0.26 | 0.15 → 0.15 | null (both) |
| MAMP+pose vs MLD-AE(recon) (2s) | pointing | +0.103 | [+0.067, +0.139] | 5/5 | 0.0025 → 0.013 | 0.0014 → 0.0056 | SURVIVES |
| MAMP+pose vs MLD-AE(recon) (2s) | picking | +0.045 | [+0.021, +0.069] | 5/5 | 0.0038 → 0.015 | 0.0065 → 0.02 | SURVIVES |
| MAMP+pose vs MLD-VAE(recon) (2s) | walking | -0.044 | [-0.072, -0.017] | 5/5 | 0.019 → 0.037 | 0.011 → 0.022 | SURVIVES |
| MAMP+pose vs MLD-VAE(recon) (2s) | pointing | +0.110 | [+0.074, +0.146] | 5/5 | 0.00014 → 0.00084 | 0.001 → 0.0051 | SURVIVES |
| MAMP+pose vs MLD-VAE(recon) (2s) | picking | +0.042 | [+0.033, +0.052] | 5/5 | 0.012 → 0.037 | 0.00025 → 0.0015 | SURVIVES |

## F5 velocity objectives (tab:velLoss)  (Holm m=12)

| x vs y | action | Δ | 95% CI (t4) | repeats same sign | old p (W25) → Holm | repeat-mean p → Holm | verdict |
|---|---|---|---|---|---|---|---|
| Plain(recon+vel) vs Plain(recon) (2s) | walking | -0.003 | [-0.037, +0.031] | 3/5 | 0.6 → 1 | 0.8 → 1 | null (both) |
| Plain(recon+vel) vs Plain(recon) (2s) | pointing | -0.008 | [-0.044, +0.029] | 3/5 | 0.49 → 1 | 0.59 → 1 | null (both) |
| Plain(recon+vel) vs Plain(recon) (2s) | picking | -0.092 | [-0.137, -0.046] | 5/5 | 0.00056 → 0.0056 | 0.0049 → 0.049 | SURVIVES |
| Plain(vel) vs Plain(recon) (2s) | walking | -0.155 | [-0.211, -0.098] | 5/5 | 2.6e-06 → 2.8e-05 | 0.0016 → 0.018 | SURVIVES |
| Plain(vel) vs Plain(recon) (2s) | pointing | +0.150 | [+0.073, +0.226] | 5/5 | 0.011 → 0.092 | 0.0056 → 0.05 | GAINED |
| Plain(vel) vs Plain(recon) (2s) | picking | -0.020 | [-0.094, +0.054] | 3/5 | 0.4 → 1 | 0.5 → 1 | null (both) |
| MLD-AE(recon+vel) vs MLD-AE(recon) (2s) | walking | +0.005 | [-0.018, +0.028] | 3/5 | 0.65 → 1 | 0.59 → 1 | null (both) |
| MLD-AE(recon+vel) vs MLD-AE(recon) (2s) | pointing | +0.007 | [-0.002, +0.017] | 4/5 | 0.31 → 1 | 0.09 → 0.63 | null (both) |
| MLD-AE(recon+vel) vs MLD-AE(recon) (2s) | picking | -0.005 | [-0.029, +0.019] | 4/5 | 0.31 → 1 | 0.58 → 1 | null (both) |
| MLD-AE(vel) vs MLD-AE(recon) (2s) | walking | -0.081 | [-0.134, -0.027] | 5/5 | 0.0081 → 0.073 | 0.014 → 0.11 | null (both) |
| MLD-AE(vel) vs MLD-AE(recon) (2s) | pointing | -0.053 | [-0.129, +0.023] | 4/5 | 0.25 → 1 | 0.12 → 0.74 | null (both) |
| MLD-AE(vel) vs MLD-AE(recon) (2s) | picking | -0.244 | [-0.320, -0.168] | 5/5 | 8.3e-07 → 1e-05 | 0.00087 → 0.01 | SURVIVES |

## F6 capacity control (supp tab:capacity)  (Holm m=3)

| x vs y | action | Δ | 95% CI (t4) | repeats same sign | old p (W25) → Holm | repeat-mean p → Holm | verdict |
|---|---|---|---|---|---|---|---|
| Plain-35M(recon) vs Plain(recon) (2s) | walking | -0.127 | [-0.200, -0.053] | 5/5 | 3.8e-05 → 7.6e-05 | 0.0088 → 0.018 | SURVIVES |
| Plain-35M(recon) vs Plain(recon) (2s) | pointing | +0.008 | [-0.054, +0.070] | 3/5 | 0.71 → 0.71 | 0.74 → 0.74 | null (both) |
| Plain-35M(recon) vs Plain(recon) (2s) | picking | -0.190 | [-0.274, -0.105] | 5/5 | 3e-07 → 8.9e-07 | 0.0034 → 0.01 | SURVIVES |

## F7 TMR corpus-matched (tab:tmr)  (Holm m=6)

| x vs y | action | Δ | 95% CI (t4) | repeats same sign | old p (W25) → Holm | repeat-mean p → Holm | verdict |
|---|---|---|---|---|---|---|---|
| cut1 MAMP+pose vs cut1 TMR (2s) | walking | +0.010 | [-0.030, +0.051] | 4/5 | 0.85 → 1 | 0.52 → 1 | null (both) |
| cut1 MAMP+pose vs cut1 TMR (2s) | pointing | -0.123 | [-0.187, -0.060] | 5/5 | 0.0056 → 0.028 | 0.0058 → 0.035 | SURVIVES |
| cut1 MAMP+pose vs cut1 TMR (2s) | picking | +0.073 | [+0.023, +0.123] | 5/5 | 0.0088 → 0.035 | 0.015 → 0.057 | LOST |
| cut1 MAMP+pose+FT vs cut1 TMR+FT (2s) | walking | +0.050 | [+0.022, +0.078] | 5/5 | 0.024 → 0.071 | 0.008 → 0.04 | GAINED |
| cut1 MAMP+pose+FT vs cut1 TMR+FT (2s) | pointing | -0.110 | [-0.183, -0.036] | 5/5 | 0.0016 → 0.0098 | 0.014 → 0.057 | LOST |
| cut1 MAMP+pose+FT vs cut1 TMR+FT (2s) | picking | -0.006 | [-0.029, +0.017] | 3/5 | 0.98 → 1 | 0.51 → 1 | null (both) |

## F8 perceptual fine-tune lift (cut1)  (Holm m=3)

| x vs y | action | Δ | 95% CI (t4) | repeats same sign | old p (W25) → Holm | repeat-mean p → Holm | verdict |
|---|---|---|---|---|---|---|---|
| cut1 MAMP+pose+FT vs cut1 MAMP+pose (1s) | walking | +0.237 | [+0.173, +0.300] | 5/5 | 3e-08 → 8.9e-08 | 0.00024 → 0.00073 | SURVIVES |
| cut1 MAMP+pose+FT vs cut1 MAMP+pose (1s) | pointing | +0.161 | [+0.067, +0.255] | 5/5 | 3.7e-05 → 3.7e-05 | 0.0045 → 0.0045 | SURVIVES |
| cut1 MAMP+pose+FT vs cut1 MAMP+pose (1s) | picking | +0.124 | [+0.066, +0.182] | 5/5 | 4.1e-06 → 8.2e-06 | 0.002 → 0.004 | SURVIVES |
