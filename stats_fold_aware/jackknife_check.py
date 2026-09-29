"""Cross-check of the class-bootstrap SE: delete-one-Effort-class jackknife (no duplicated stimuli).
SE_jack = sqrt((n-1)/n * sum (theta_(i) - theta_bar)^2), paired on the difference."""
import json, sys, numpy as np
sys.path.insert(0, '.')
import fold_aware_classboot as fb
man = json.load(open('pairdist/manifest.json'))
boot = np.load('classboot_draws.npz')
acts = {a: fb.Action(a, man) for a in fb.A}
COMPS = [("MAMP+pose", "Plain(recon)"), ("MAMP+pose", "MAMP"), ("MAMP-UNet+pose", "MAMP+pose"),
         ("MAMP-UNet", "MAMP"), ("MAMP+pose", "MLD-VAE(recon)"), ("MAMP+pose", "MLD-AE(recon)"),
         ("MLD-AE(recon)", "Plain(recon)")]
print(f"{'comparison':34} {'action':9} {'Δ':>7} {'SE_boot':>8} {'SE_jack':>8} {'ratio':>6} {'z_jack':>7}")
ratios = []
for x, y in COMPS:
    for a in fb.A:
        jx, jy = [], []
        for i in range(56):
            c = np.ones(56); c[i] = 0
            jx.append(acts[a].stat(x, c)); jy.append(acts[a].stat(y, c))
        d = np.array(jx) - np.array(jy); n = 56
        se_j = np.sqrt((n - 1) / n * ((d - d.mean()) ** 2).sum())
        se_b = (boot[f"{a}|{x}"] - boot[f"{a}|{y}"]).std(ddof=1)
        delta = acts[a].stat(x, np.ones(56)) - acts[a].stat(y, np.ones(56))
        ratios.append(se_b / se_j)
        print(f"{x+' vs '+y:34} {a:9} {delta:+.3f} {se_b:8.3f} {se_j:8.3f} {se_b/se_j:6.2f} {delta/se_j:7.2f}")
print(f"\nSE_boot / SE_jack: median {np.median(ratios):.2f}, range [{min(ratios):.2f}, {max(ratios):.2f}]")
