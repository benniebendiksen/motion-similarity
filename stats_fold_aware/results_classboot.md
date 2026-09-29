# Class-bootstrap (B=10000, 56 Effort classes/action, paired)

## Per-model mean with class-bootstrap 95% CI

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

## F1 headline vs baselines (tab:baselines)  (Holm m=9)

| x vs y | action | Δ | 95% CI (class boot) | p | p Holm | survives |
|---|---|---|---|---|---|---|
| MAMP+pose vs DTW (1s) | walking | +0.063 | [-0.059, +0.182] | 0.17 | 0.2 | no |
| MAMP+pose vs DTW (1s) | pointing | +0.162 | [+0.010, +0.305] | 0.017 | 0.069 | no |
| MAMP+pose vs DTW (1s) | picking | +0.240 | [+0.077, +0.377] | 0.0017 | 0.012 | YES |
| MAMP+pose vs Geodesic (1s) | walking | +0.181 | [+0.034, +0.333] | 0.0081 | 0.04 | YES |
| MAMP+pose vs Geodesic (1s) | pointing | +0.200 | [+0.065, +0.312] | 0.0021 | 0.013 | YES |
| MAMP+pose vs Geodesic (1s) | picking | +0.279 | [+0.115, +0.415] | 0.0009 | 0.0072 | YES |
| MAMP+pose vs Plain(recon) (1s) | walking | +0.076 | [+0.002, +0.164] | 0.022 | 0.069 | no |
| MAMP+pose vs Plain(recon) (1s) | pointing | +0.222 | [+0.097, +0.301] | 0.0003 | 0.0027 | YES |
| MAMP+pose vs Plain(recon) (1s) | picking | +0.052 | [-0.026, +0.114] | 0.1 | 0.2 | no |

## F2 pose-head x U-Net 2x2 (tab:posehead, tab:archObj)  (Holm m=12)

| x vs y | action | Δ | 95% CI (class boot) | p | p Holm | survives |
|---|---|---|---|---|---|---|
| MAMP+pose vs MAMP (2s) | walking | +0.201 | [+0.112, +0.282] | 0.0002 | 0.0024 | YES |
| MAMP+pose vs MAMP (2s) | pointing | +0.135 | [+0.017, +0.234] | 0.023 | 0.21 | no |
| MAMP+pose vs MAMP (2s) | picking | +0.053 | [-0.014, +0.122] | 0.13 | 0.63 | no |
| MAMP-UNet+pose vs MAMP-UNet (2s) | walking | +0.054 | [-0.035, +0.154] | 0.23 | 0.93 | no |
| MAMP-UNet+pose vs MAMP-UNet (2s) | pointing | -0.016 | [-0.113, +0.117] | 0.97 | 1 | no |
| MAMP-UNet+pose vs MAMP-UNet (2s) | picking | -0.081 | [-0.123, -0.011] | 0.019 | 0.19 | no |
| MAMP-UNet vs MAMP (2s) | walking | +0.118 | [+0.009, +0.206] | 0.034 | 0.25 | no |
| MAMP-UNet vs MAMP (2s) | pointing | -0.024 | [-0.139, +0.076] | 0.56 | 1 | no |
| MAMP-UNet vs MAMP (2s) | picking | +0.076 | [+0.005, +0.143] | 0.032 | 0.25 | no |
| MAMP-UNet+pose vs MAMP+pose (2s) | walking | -0.029 | [-0.102, +0.042] | 0.39 | 1 | no |
| MAMP-UNet+pose vs MAMP+pose (2s) | pointing | -0.175 | [-0.239, -0.067] | 0.001 | 0.011 | YES |
| MAMP-UNet+pose vs MAMP+pose (2s) | picking | -0.059 | [-0.102, +0.009] | 0.09 | 0.54 | no |

## F3b backbone under reconstruction (tab:backbone)  (Holm m=3)

| x vs y | action | Δ | 95% CI (class boot) | p | p Holm | survives |
|---|---|---|---|---|---|---|
| MLD-AE(recon) vs Plain(recon) (2s) | walking | +0.056 | [-0.023, +0.138] | 0.17 | 0.34 | no |
| MLD-AE(recon) vs Plain(recon) (2s) | pointing | +0.119 | [+0.036, +0.192] | 0.0056 | 0.017 | YES |
| MLD-AE(recon) vs Plain(recon) (2s) | picking | +0.007 | [-0.054, +0.073] | 0.75 | 0.75 | no |

## F4 masking vs reconstruction refs (tab:sweep; MLD-VAE honesty)  (Holm m=6)

| x vs y | action | Δ | 95% CI (class boot) | p | p Holm | survives |
|---|---|---|---|---|---|---|
| MAMP+pose vs MLD-AE(recon) (2s) | walking | +0.021 | [-0.040, +0.090] | 0.44 | 1 | no |
| MAMP+pose vs MLD-AE(recon) (2s) | pointing | +0.103 | [-0.012, +0.175] | 0.085 | 0.42 | no |
| MAMP+pose vs MLD-AE(recon) (2s) | picking | +0.045 | [-0.022, +0.096] | 0.24 | 0.96 | no |
| MAMP+pose vs MLD-VAE(recon) (2s) | walking | -0.044 | [-0.114, +0.039] | 0.36 | 1 | no |
| MAMP+pose vs MLD-VAE(recon) (2s) | pointing | +0.110 | [+0.008, +0.181] | 0.032 | 0.19 | no |
| MAMP+pose vs MLD-VAE(recon) (2s) | picking | +0.042 | [-0.034, +0.099] | 0.33 | 1 | no |
