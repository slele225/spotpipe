# Loss function + model outputs — presentation brief

Source of truth: `src/spotpipe/losses/{detection,localization,intensity}.py`,
`src/spotpipe/models/heads.py`, `src/spotpipe/training/targets.py`. All exact as of 2026-07-15.

## Model outputs (dense, full-resolution [B, C, H, W])
The HRNet backbone emits a feature map; four heads read it at full 256×256 resolution:

| head | shape | meaning |
|---|---|---|
| heatmap | [B,1,H,W] | raw logits; `sigmoid` → P(spot center at this pixel) |
| offset | [B,2,H,W] | sub-pixel correction `(frac_x, frac_y)` |
| logI₁, logI₂ | [B,1,H,W] each | per-channel log integrated intensity (photon-proportional, natural log) |
| logvar₁, logvar₂ | [B,1,H,W] each | per-channel predicted log-variance (uncertainty) |

**Inference:** peak-find the heatmap (local-max NMS + threshold) → at each surviving peak pixel read
offset / logI / logvar → emit one schema row per spot. Heads are dense; only peak pixels are emitted.

## Total loss
`L = λ_hm · heatmap + λ_off · offset + λ_int · (NLL₁ + NLL₂)`, all λ = 1.0.

## 1. Heatmap loss — CenterNet penalty-reduced focal loss
**Target is a Gaussian BLOB, not one-hot.** A unit-peak Gaussian (σ = `heatmap_sigma` = 1.5) is
rendered at each spot's integer center pixel; overlapping blobs combine by element-wise max. A blob
(not a single hot pixel) gives graded "you're near a center" gradients so training is stable.

With `p = sigmoid(logits)` and Gaussian target `y`:
* **center pixels (positives):** `−(1−p)^α · log(p)` — focal up-weighting of under-confident centers.
* **elsewhere (negatives):** `−(1−y)^β · p^α · log(1−p)` — the `(1−y)^β` factor *reduces* the penalty
  for pixels near a center (where y≈1), so the loss doesn't fight the blob's spread.

Exponents: **α = 2** (focal focusing — down-weight easy pixels), **β = 4** (how fast the near-center
penalty relaxes). Normalized by number of centers.

## 2. Offset loss — masked smooth-L1
Target = the **fractional part of the true position**: a spot at (10.3, 5.7) → integer pixel (10,5),
target (0.3, 0.7) stored there. Smooth-L1, **masked to center pixels only** (`center_mask`) — the
target is 0 off-center and the loss has NO gradient off-center. Recovers the sub-pixel position lost
when a spot is snapped to the integer grid; at inference, detected pixel + offset = sub-pixel center.
("centers" = spot centers, not the image center.)

## 3. Intensity loss — heteroscedastic Gaussian NLL (per channel, masked to centers)
`NLL = ½ · [ exp(−s)·(logÎ − logI)² + s ]`, with `s = logvar` clamped to [−10, 6].

* `exp(−s)·(error)²` — precision-weighted squared error. Confident (small variance) → errors hurt more.
* `+ s` — penalty for claiming large variance (stops the model inflating variance to zero the first term).

Optimum: predicted variance ≈ actual squared error → **calibrated** uncertainty. Overconfidence
punished by term 1, cowardice by term 2.

### logvar
The model's prediction of **how uncertain it is about its own intensity estimate**, as a separate
head. Variance `σ² = exp(logvar)`; the model emits *log*-variance so the output is unconstrained and
numerically stable (variance must be positive). On hard spots (dim, overlapped) it learns to report a
larger logvar. At inference `uncertainty = exp(0.5·logvar)` = predicted SD of the log-intensity → the
schema `uncertainty1/2` columns → the calibrated per-spot error bar that feeds α downstream.

## Note on the SHIPPED model (delta head)
The headline model (`headfix40k-DELTA`) predicts `logI₁` and `Δ = logI₂ − logI₁` directly (the NLL
is on logI₁ and Δ); `logI₂ = logI₁ + Δ` and `logvar₂ = logaddexp(logvar₁, logvar_Δ)` are DERIVED.
Same NLL form, reparameterized so the ratio is the model's own estimand — this is what removed the
protein-channel-under-crowding ratio bias. See `docs/head_fix_results.md`.

## Targets recap (training/targets.py) — what each supervises
| target | [shape] | value |
|---|---|---|
| heatmap | [1,H,W] | Gaussian blob, peak 1.0 at centers, else decaying to 0 |
| center_mask | [1,H,W] | 1.0 at integer center pixels, else 0 (the mask for offset + intensity) |
| offset | [2,H,W] | (frac_x, frac_y) at centers, else 0 |
| logI₁, logI₂ | [1,H,W] | true log-intensity at centers, else 0 |
| delta | [1,H,W] | logI₂ − logI₁ at centers, else 0 (delta head only) |
