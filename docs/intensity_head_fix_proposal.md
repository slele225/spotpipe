# The intensity-head fix — diagnosis and proposal

> ## ✅ BUILT 2026-07-14 — approved and implemented; awaiting the GPU bake-off
> Shiv approved (a) modifying the frozen files and (b) the `(logI1, Δ)` design, and (implicitly,
> by approving the design) the §4 reading that a per-spot Δ target is outside Durable Rule 3.
> **Implemented:**
> * `models/heads.py` — `head_parameterisation` ∈ {`independent` (default), `delta`}. Delta mode
>   predicts `(logI1, Δ)`, derives `logI2 = logI1 + Δ` and `logvar2 = logaddexp(logvar1,
>   logvar_delta)`, and exposes native `delta`/`logvar_delta` for the loss. **State-dict identical
>   across modes** → every existing checkpoint still loads; mode travels in the config, not the weights.
> * `losses/intensity.py` — `intensity_nll` auto-routes to `(logI1, Δ)` when the head emits `delta`,
>   so the frozen loss *combiner* is untouched.
> * `training/targets.py` — adds the per-spot `delta = logI2 − logI1` target (always present, cheap).
> * `predict_spots` and the **schema are unchanged** — `logI2` is derived; `log_ratio == Δ`.
> * Golden test `tests/test_delta_head.py` (perfect derivations, state-dict compat, loss routing,
>   schema round-trip). Bake-off configs via `scripts/make_delta_configs.py`; runner
>   `scripts/run_headfix_bakeoff.sh`.
> **Verified in the no-GPU sandbox:** all files parse; forward-dict key wiring correct in both modes;
> config single-knob difference asserted. **NOT yet verified (needs real torch — run on the box):**
> the golden test's numerical claims. STEP 0 of the runner runs it before any GPU spend.
> **To run:** `git pull` on the A100, then `REPO=~/spotpipe_new bash scripts/run_headfix_bakeoff.sh`.



**Date:** 2026-07-14 · **Status:** PROPOSAL. Requires an explicit decision (see §6 — it touches
FROZEN modules and brushes against Durable Rule 3).

Reading order: `coverage_probe_findings.md` (coverage is dead) → `rarity_probe_findings.md`
(rarity is dead) → this.

---

> ## ⚠️ AMENDED 2026-07-14 (after the probe ran) — the diagnosis is LOCALISED
> The CPU probe was run. The literal §2 prediction ("`s < 1` everywhere") **FAILED** — only
> **dense protein (ch2)** shrinks (slope 0.700); sparse and lipid (ch1) are unbiased (0.98–1.06).
> The mechanism below is CONFIRMED in a **narrower** form: conditional-mean regression appears
> only where identifiability is lost (wide-PSF ch2 under overlap), not "toward a global brightness
> prior everywhere". Two pre-registered directional predictions (dense<sparse, ch2<ch1) both
> passed, and `s₁ − s₂ = +0.283` reproduces the measured ratio bias. The `logvar`-saturation
> sub-hypothesis is DEAD (0% at clamps). **Full, honest scorecard: `docs/shrinkage_probe_findings.md`.**
> The proposed fix (§3) is unchanged and now better-motivated. Read §1 with "ambiguous evidence"
> meaning specifically "wide PSF + overlap", not "far from the prior".

## 1. The diagnosis: conditional-mean shrinkage (where evidence is lost)

The intensity head is trained with a heteroscedastic Gaussian NLL:

```
NLL_k = ½ · [ exp(−s_k) · (logÎ_k − logI_k)² + s_k ]        (losses/intensity.py)
```

**The minimiser of that loss for the mean is the conditional mean `E[logI | patch]`.** A
conditional mean *shrinks toward the prior* whenever the image evidence is ambiguous. Write it as

```
logÎ  ≈  s · logI  +  (1 − s) · logI_prior          with shrinkage slope  s ∈ (0, 1]
```

`s → 1` when the spot is unambiguous; `s → 0` when the patch cannot identify it. **This is not a
bug in the data. It is the Bayes-optimal thing to do under this loss** — which is exactly why the
defect *grows with training* (40k CURRENT: −1.01; 8k arm A: −0.52).

### It explains every observation, with no free parameters

| observation | shrinkage explains it how |
|---|---|
| bright spots break, dim spots don't | bias = `(1−s)·(logI_prior − logI)`; the training prior median is **~29 photons**, so a 1,365-photon spot is ~3.9 log-units away — the bias scales with that distance |
| dense breaks, sparse doesn't (8× worse at the same brightness) | overlap weakens the likelihood → `s` falls |
| **ch2 breaks worse than ch1** (−1.85 vs −1.18) | σ₂ = 1.68 > σ₁ = 1.4 → flatter peak → worse identifiability → `s₂ < s₁` |
| the defect grows with training | the head converges *toward* the conditional mean, not away from it |
| coverage probe found nothing | shrinkage needs no coverage gap — it happens on data the model has seen plenty of |
| **arm B improved both channels and made the RATIO worse** | see below — this is the tell |

### The tell: why the ratio behaves the way it does

The quantity that matters is the ratio. Under shrinkage:

```
log-ratio bias  =  (s₁ − s₂) · (logI − logI_fixed_point)
```

The ratio's bias is driven by the **DIFFERENCE in the two channels' shrinkage**, *not* by either
channel's accuracy. When `s₁ ≈ s₂`, the two errors **cancel in the difference** and the ratio is
fine even though both channels are individually biased.

That is precisely what the rarity probe found and could not explain:

| arm | logI1_bias | logI2_bias | **log-ratio bias** |
|---|---|---|---|
| A (dim_bias 1.6) | −1.184 | −1.854 | **−0.669** |
| B (dim_bias 1.0) | **−0.382** ✅ | **−1.282** ✅ | **−0.900** ❌ |

Arm B moved the prior brighter, which shrank *both* per-channel biases — but it moved `s₁` more
than `s₂`, **decorrelating** the errors, and the ratio got worse. Under the shrinkage model this is
not a paradox; it is the prediction.

> **Consequence for the whole project: per-channel intensity accuracy is NOT the objective.**
> The model does not need *smaller* channel errors — it needs *equal* ones. Any fix tuned on
> `logI1_mae` / `logI2_mae` will keep walking into this. Optimise and report `log_ratio_bias`,
> stratified by density.

This also qualifies PROJECT_STATE §2's *"the ratio comes for free — if logI1 and logI2 are each
unbiased, log(A₂/A₁) = logI2 − logI1"*. True **at** the optimum. Off the optimum, the ratio's bias
depends on how the two channels' errors **co-vary**, and nothing in the current loss makes them
co-vary.

### ⚠️ And it silently threatens α

`s` depends on brightness and crowding, and **√A₁ is the size ruler**. A shrinkage that varies with
A₁ is *precisely* a "bias that changes with size" — the exact thing PROJECT_STATE §1 names as the
mechanism that tilts α. Today α MAE = 0.036 only because family 2 runs at **low density**, where
`s ≈ 1`. **Push the benchmark into the crowded regime — which v3 deliberately does — and this
becomes a live α risk, not just an intensity-metric wart.** That is the real reason to fix it.

## 2. VERIFY FIRST (CPU, ~5 min, no GPU)

Do not build on this until it is measured. `scripts/shrinkage_probe.py` regresses predicted logI on
true logI across a wide intensity sweep and checks four predictions:

1. slope `s < 1` everywhere,
2. `s(dense) < s(sparse)`,
3. `s(ch2) < s(ch1)`,
4. the fixed point sits near the training prior (~29 photons).

```bash
python scripts/shrinkage_probe.py --checkpoint hrnet_large_measured
```

**If prediction 1 fails (`s ≈ 1`), this entire diagnosis is wrong — stop.** It also reports whether
`logvar` is pinned at its `[−10, 6]` clamp in the crowded corner, which would mean the NLL's
self-calibration is switched off exactly where the defect lives.

## 3. The proposed fix: predict the RATIO, not two independent channels

**Reparameterise the intensity head** from `(logI1, logI2)` to `(logI1, Δ)` where
**`Δ ≡ logI2 − logI1`**, and put the NLL on `Δ` directly:

```
loss = NLL(logÎ1, logI1) + NLL(Δ̂, Δ)          # logI2 is DERIVED: logÎ2 = logÎ1 + Δ̂
```

**Why this is the right fix, and not a hack:**

* It makes the estimator's error structure match the estimand. Today the ratio is a *difference of
  two independently-shrunk estimates*, so its bias is `(s₁ − s₂)·(…)` — an uncontrolled residual of
  two nuisance quantities. With `Δ` predicted directly, the ratio's error is **its own** error, and
  the `s₁ − s₂` term **does not exist**.
* `Δ` is intrinsically far easier to estimate than either channel: it is bounded (the ratio law has
  `scatter_std ∈ [0.03, 0.25]`), and it is **largely invariant to the very things that destroy
  per-channel identifiability** — the total photon budget, the per-image gains, and (to first order)
  the local overlap, which attenuates both channels together.
* Shrinkage does not disappear — `Δ̂` will shrink toward the *prior mean of Δ*. But that prior is
  narrow (a fraction of a log unit) instead of ~4 log units wide, so the residual bias is smaller by
  roughly the ratio of those spreads. **Shrinking toward a tight prior is cheap; shrinking toward a
  4-decade prior is what is killing us.**
* The `logvar` head comes along for free and now expresses uncertainty **on the ratio** — which is
  what the downstream α error bar actually wants (PROJECT_STATE §3 head-group 4).

**Cheaper alternatives, and why they are worse:**

| option | verdict |
|---|---|
| Tie the two channels' variances / add a channel-error-correlation penalty | Attacks `(s₁ − s₂)` directly and is a smaller change, but it *equalises* shrinkage rather than *removing* it — the ratio becomes unbiased while both channels stay wrong, and the residual is fragile to anything that perturbs one channel. Worth testing as arm 2. |
| Flatten / re-weight the intensity prior | **Already tested and refuted** (rarity probe). It moves the prior, it does not remove the shrinkage. |
| Bigger model / more capacity | May raise `s` a little, but the shrinkage is a property of the *loss*, not the capacity. Expensive, and unfalsifiable. |
| Train at GT centres, infer at decoded centres (current) | A real train/infer mismatch (see `losses/intensity.py` docstring) but it hits BOTH channels at the same decoded pixel, so it largely cancels in the ratio. Not the culprit here. |

## 4. Does this violate Durable Rule 3 ("NO slope/alpha/beta loss")?

**No — but the distinction must be made explicitly, because it is close to the line.**

Rule 3 and `losses/ratio.py` forbid an **in-batch SLOPE**: regressing `log(A₂/A₁)` against
`log(√A₁)` *across spots* and training on that slope. The stated reasons are (a) an in-batch slope
is an attenuated/biased estimator because the regressor carries error, and (b) size-correlated
weighting could **manufacture** curvature.

`Δ` is a **per-spot supervised target**, identical in kind to `logI1` and `logI2`. It involves:

* **no regression across spots**, **no batch statistic**, **no slope**, **no α**;
* no size-dependent weighting — every spot contributes equally, as now.

The model still **never sees α**, and cannot: α is a property of a *population* of spots, and
nothing in this loss aggregates across spots. Indeed, this change makes the *manufactured-curvature*
risk **smaller**, because today's size-dependent shrinkage is itself a size-correlated bias — the
very thing Rule 3 exists to prevent — and it is currently arriving through the back door.

⚠️ **The one new risk to watch:** `Δ̂` shrinks toward the prior mean of `Δ`. If that shrinkage is
**size-dependent** (i.e. `s_Δ` varies with A₁), it *flattens* the ratio-vs-size relation and would
bias α **toward zero** — a *conservative* failure (it destroys signal, it does not fabricate it),
but it must be measured, not assumed. **The α=0 null control and the known-α sets are the gate**,
and they must be run on any new checkpoint before it is believed.

## 5. Validation plan (cheap → expensive; each gates the next)

1. **CPU, 5 min:** `shrinkage_probe.py` on `hrnet_large_measured`. Confirm or kill the diagnosis.
2. **CPU, 10 min:** same probe on the two 8k arm checkpoints. The shrinkage model predicts
   `s₁ − s₂` is *larger* in arm B than arm A. If it is not, the model is wrong even though the
   diagnosis "explains" the bias — that is the strongest available falsification, and it costs nothing.
3. **GPU, ~1 h:** a 3-arm 8k probe, one knob each, same harness as the rarity probe:
   * **arm C** — `(logI1, Δ)` reparameterisation (this proposal)
   * **arm D** — channel-error-correlation penalty (the cheaper alternative)
   * **arm A** — the shipped head, as the control (reuse the existing run)
   Read out with `bright_dense_probe.py`. **Gate: `log_ratio_bias` in bright+dense, AND no
   regression in dim+dense.** A fix that wins bright and loses dim is a downgrade (rarity probe).
4. **GPU, 40k:** only the winner. Then: threshold retune (off-benchmark), `bench-gen` v3,
   `infer`, `evaluate` — with **Gate A (known-α recovery) and the α=0 null control FIRST**.

## 6. 🛑 DECISION REQUIRED BEFORE ANY CODE IS WRITTEN

This proposal modifies **FROZEN** modules — `losses/intensity.py` and `models/heads.py` — which
CLAUDE.md Durable Rule 1 forbids touching *"without an explicit instruction naming the file"*, and
Rule 8 says a change touching a frozen module means **STOP and ask**.

It also changes the **checkpoint interface**: a `(logI1, Δ)` head is not state-dict-compatible with
the existing checkpoints, so `hrnet_large_measured` and the legacy models would need the old head
kept alongside (a `head_parameterisation` flag in the model config, defaulting to the current
behaviour). The frozen **schema** does NOT change — `predict_spots` still emits `logI1`, `logI2`,
`log_ratio` exactly as now; `logI2` simply becomes a derived output. **No downstream code, and no
part of the evaluator, changes.**

**What I need from you:**

1. Run step 1 (the CPU probe) and paste the table — the diagnosis is unconfirmed until then.
2. Explicit go-ahead to modify `src/spotpipe/losses/intensity.py` and `src/spotpipe/models/heads.py`.
3. A ruling on §4: I read a per-spot `Δ` target as **outside** Rule 3's prohibition. You own that rule.
