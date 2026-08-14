# NSMjl — Neural Species Mediator in Julia

A from-scratch Julia rewrite of the NSM model (Thompson et al. 2025, Venturelli lab),
built to support **both** a physics-embedded RHS **and** a loss-residual
(PINN-style) variant, with **time-dependent perturbation inputs** `u(t)` as a
first-class feature. The Bayesian pipeline is matched — component-by-component —
to the reference `nsm/` Python/JAX package and to the paper's Methods equations.

---

## Results — paper reproduction

Two held-out experiments reproduce the paper's headline figures using the exact
protocol from the reference repo (fold structure, `fit_posterior_EM`, scaling,
hyperparameters) run through the Julia implementation.

### 1. FP-DP — leave-one-co-culture-out (paper Fig 3d)

`scripts/run_nsm.jl` → `figures/fig_nsm_fpdp_loo.pdf`

![FP-DP leave-one-co-culture-out](figures/fig_nsm_fpdp_loo.pdf)

Hold out one co-culture (`Pair_*`) per fold (10 folds); monocultures always in
training and never scored; fit each fold with `fit_posterior_EM!`; pool the 10
held-out co-cultures and report per-output Pearson r.

| Output  | r (this repo) | paper Fig 3d (approx) |
|---------|:-------------:|:---------------------:|
| FP      | **0.89**      | ~0.9                  |
| DP      | **0.81**      | ~0.8                  |
| Sulfide | **0.89**      | ~0.94                 |

Negative (unphysical) predictions: 0.0%. Matches the reference to within
run-to-run seed / integrator variance.

### 2. Clark 25-species — 16h interpolation (paper Fig 4c)

`scripts/run_clark.jl` → `figures/fig_clark_16h.pdf`

![Clark 16h interpolation](figures/fig_clark_16h.pdf)

Train on all timepoints except 16h; predict the (0 → 16h) transition; fit with
`fit_posterior_EM!`, `n_hidden=20`. 757 treatments, 852 train / 95 test samples.
Reverse-mode adjoint (`AD = :reverse`) is required at this scale (~1,400 params).

| Metabolite | r        | RMSE     |
|------------|:--------:|:--------:|
| Acetate    | _(fill)_ | _(fill)_ |
| Butyrate   | _(fill)_ | _(fill)_ |
| Lactate    | _(fill)_ | _(fill)_ |
| Succinate  | _(fill)_ | _(fill)_ |

> Fill the Clark row from the run's final console output
> (`=== Clark 16h held-out metabolites (NSM, fit_posterior_EM) ===`).

> **Rendering note:** figures are written as **PDF** (CairoMakie). GitHub
> renders PNG/SVG inline but not PDF; save a PNG alongside if you want the
> images to preview directly in this README.

---

## Reproduction scripts

- `scripts/run_nsm.jl` — FP-DP Fig 3d. Leave-one-co-culture-out over the 10
  `Pair_*` conditions, `fit_posterior_EM!` per fold, per-fold training-target
  scaling, `n_hidden=15`, 4-panel figure (3 posterior-predictive trajectory
  panels + pooled scatter). Includes a **BC026 data-integrity check** that halts
  before the fit if `data/BC026_FPDP_GlLaSu_fmt.csv` isn't the paper's file
  (150 rows, times `0/6.97/10.28/18.66/36.58`).
- `scripts/run_clark.jl` — Clark Fig 4c. 16h interpolation, `fit_posterior_EM!`,
  `n_hidden=20`, per-metabolite r/RMSE scatter. `AD = :reverse` flag for the
  25-species scale.

Both use the paper's exact fit pipeline (below) and score with per-output
Pearson r on the held-out data.

---

## Method fidelity (matched to the reference)

The Bayesian objective is the KL-minimizing variational posterior; ELBO
maximization ≡ KL minimization by `log p(D) = ELBO + KL[q‖p]` (paper
Eq 14→16). Each piece is matched:

| Component | Reference | Match |
|---|---|---|
| RHS `nsm_rhs!`, softplus₁₀, positivity `g` | `nsm_system.py::runODE` | ✅ (JAX float64 to ~1e-10) |
| Prior `Σ½α(θ−z_prior)²`, **z_prior = 0** (paper Eq 13, zero-mean) | Methods Eq 13/16 | ✅ |
| Likelihood `ν²+σ²·clip(ŷ,0)²`, `½` terms | `log_likelihood_lmbda` | ✅ |
| Reparam `μ+e^{log_s}·y`, entropy `Σlog_s` | `T`, `log_abs_det` | ✅ |
| `fit_posterior` — **per-sample Adam**, ELBO-slope convergence (`tol=1e-3`, `patience=5`), keep-best | `nsm.py::fit_posterior` | ✅ |
| α update `= 1/E[θ²] = 1/(μ²+σ²)` (paper Eq 18, zero-mean) | Methods Eq 18 | ✅ |
| ν²/σ² per-output LSQ (Eq 19), variance floor `1e-4` (positivity) | `update_hypers` | ✅ |
| EM loop — evidence recomputed via `approx_evidence`, running-max, best-iterate restore | `fit_posterior_EM` | ✅ |
| Posterior init std (per-block: C,P=1/nₛ; W1=1/√n_x; Wh,Wo=1/√n_h; biases,d_m=1) | `init_params::params_std` | ✅ |
| Per-fold scaling by final-state max | `NSM.__init__` | ✅ |
| Fold = leave-one-co-culture-out; r pooling | `kfold_fpdp.py`, `FPDP_kfold_stats.ipynb` | ✅ |

Remaining non-identities are substrate-level only: the ODE integrator
(JAX `odeint` vs `Tsit5`) and the RNG stream (NumPy vs Julia `Random`), so
results match to seed/integrator variance, not to the digit.

**Autodiff.** `fit_posterior!`/`fit_rmse!` take an `ad` switch:
`:forward` (ForwardDiff-through-solve — fast, exact, validated; best for small
models like FP-DP) or `:reverse` (Zygote + SciMLSensitivity
`InterpolatingAdjoint(ReverseDiffVJP(true))` — cost ~independent of parameter
count; required for Clark's 25 species). Reverse-mode *hurts* tiny models
(per-solve adjoint overhead) but is a ~100× win at ~1,000+ parameters.

---

## Status

- **Step 1 — dynamical system ✅**
- **Step 2 — training (loss + Adam fit) ✅**
- **Step 3a — Bayesian objective (nlp + MAP) ✅**
- **Step 3b — variational posterior + EM + evidence ✅**
- **Step 4 — prediction with uncertainty ✅**
- **Step 5 — gLV residual-penalty (PINN) variant ✅**
- **Paper reproduction — FP-DP Fig 3d ✅ ; Clark Fig 4c 🔄 (run in progress)**

### Step 1 — the RHS

- `src/system.jl` — activations, positivity transform `g`, `NSMConfig`,
  `init_params`, and `nsm_rhs!` (faithful port of `nsm_system.py::system`). The
  one deliberate change vs. the reference: `inputs` is a **function `u(t)`**
  evaluated inside the RHS at every solver step, instead of a constant vector —
  the perturbation-aware extension.
- `src/NSMjl.jl` — module + `solve_forward` wrapper.
- `test/smoke_forward.jl` — solves with a time-dependent Gaussian perturbation
  and checks the trajectory against a verified reference (`max abs error` ≈ 1e-9).

### Step 2 — the training layer

- `src/train.jl` — `Sample`, `predict_endpoint` (forwards `sensealg`),
  `sample_rmse`, `rmse`, `fit_rmse!` (per-sample Adam, `ad` switch, clip 1000).
  Each `Sample` is an (initial condition → one later observation) pair,
  as `utilities.py::process_df` builds them.
- `test/train_fit.jl` — RMSE value, autodiff-through-solver gradients vs exact
  JAX float64 for 8 parameters, and that `fit_rmse!` reduces the loss.

### Step 3a — the Bayesian objective

- `src/bayes.jl` — `prior_params` (**zero-mean** Gaussian prior for all
  parameters, paper Eq 13), `log_prior`, `nll_sample` (heteroscedastic Gaussian,
  var = ν² + σ²·max(pred,0)²), `nlp`, `grad_nlp`, `fit_map!`.
- `test/bayes_fit.jl` — `log_prior`/`nlp` vs JAX, `grad_nlp` vs exact JAX
  gradients, and that `fit_map!` reduces nlp and RMSE.

### Step 3b — variational posterior + EM + evidence

- `src/vi.jl` — `VarPosterior` (mean-field diagonal Gaussian; `VarPosterior(p,
  cfg)` seeds the per-parameter init std), `neg_elbo_flat`, `approx_evidence`
  (`Σlog_s − nlp(mu)`), `fit_posterior!` (**per-sample Adam** to ELBO-slope
  convergence, keep-best), `update_hypers!` (Eq 18/19 EM), `fit_posterior_EM!`
  (evidence-recompute, running-max, best-iterate restore).
- `test/vi_fit.jl` — evidence and reparameterised neg-ELBO vs JAX, ELBO gradient
  vs exact JAX, the α update vs its closed form, and that `fit_posterior!` raises
  the evidence.

### Step 4 — prediction with uncertainty

- `src/predict.jl` — `predict_point`, `predict_sample`, `credible_bands`. Port of
  `nsm.py::predict_point/predict_sample`. `inputs` is a function `u(t)`.

### Step 5 — gLV residual-penalty (PINN) variant

- `src/residual.jl` — black-box neural-ODE RHS `dx/dt = NN([x; u(t)])` with a gLV
  residual folded into the loss: `loss = (1-λ)·data_rmse + λ·‖NN − f_gLV‖²`. The
  head-to-head against the embedded NSM.

### Real data — Cdiff experiment

- `src/dataio.jl` — `load_nsm_csv` (port of `utilities.py::process_df`).
- `scripts/experiment_realdata.jl` — fits embedded NSM and gLV-residual PINN on a
  training fold, scores both held-out (RMSE, relative RMSE, per-output Pearson r,
  % unphysical). Writes per-output panels (`fig5`) and per-output r (`fig6`).
- `scripts/experiment_datafraction.jl` — the paper's **Fig 4b**: held-out mean
  Pearson r vs training-data fraction for both models. Writes
  `fig7_datafraction.pdf`.

---

## Run it

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first time (CairoMakie is large)

# component validation (vs JAX float64)
julia --project=. test/smoke_forward.jl               # step 1
julia --project=. test/train_fit.jl                   # step 2
julia --project=. test/bayes_fit.jl                   # step 3a
julia --project=. test/vi_fit.jl                      # step 3b
julia --project=. test/predict_fit.jl                 # step 4
julia --project=. test/residual_fit.jl                # step 5

# paper reproduction
julia --project=. scripts/run_nsm.jl                  # FP-DP Fig 3d  -> fig_nsm_fpdp_loo.pdf
julia --project=. scripts/run_clark.jl                # Clark Fig 4c  -> fig_clark_16h.pdf  (set AD=:reverse)
```

> **Timing.** FP-DP `run_nsm.jl` is a multi-hour run (10 folds × EM to
> convergence, `:forward`). Clark `run_clark.jl` **requires `AD = :reverse`** —
> `:forward` at 25 species is days-to-weeks; reverse-mode brings it to
> hours/overnight. There is no mid-run checkpoint — let it finish before
> interrupting or unmounting the drive.

---

## The model (what `nsm_rhs!` computes)

State `u = [s; m]` (species stacked on mediators). With `[o_s; o_m] = NN([s; m; u(t)])`:

```
dsdt = s .* o_s .* (1 - s/s_cap)                                  # eNODE (keeps s ≥ 0)
dmdt = softplus(o_m) .* ( (P·relu(dsdt)).*(1 - m/m_cap)           # consumer-resource
                          - m .* (C·s + d_m) )                    #  (NN multiplies it)
```

`C, P, d_m` are stored unconstrained and passed through `g(·)` inside the RHS to
stay positive.

---

## Source map (what to port from the Python repo)

Only the `nsm/` package matters (ignore all notebooks / experiment folders).

| Python | Julia | status |
|---|---|---|
| `nsm_system.py :: system` | `src/system.jl :: nsm_rhs!` | ✅ step 1 |
| `nsm_system.py :: transform,g,g_inv` | `src/system.jl :: g,g_inv` | ✅ step 1 |
| `nsm_system.py :: runODE(_teval)` | `src/NSMjl.jl :: solve_forward` | ✅ step 1 |
| `nsm_system.py :: root_mean_squared_error` | `src/train.jl :: sample_rmse` | ✅ step 2 |
| `nsm.py :: rmse, fit_rmse` | `src/train.jl :: rmse, fit_rmse!` | ✅ step 2 |
| `utilities.py :: process_df` | `src/train.jl :: Sample` / `src/dataio.jl :: load_nsm_csv` | ✅ |
| `nsm_system.py :: log_prior_z, log_likelihood_z` | `src/bayes.jl :: log_prior, nll_sample` | ✅ step 3a |
| `nsm.py :: nlp, grad_nlp` | `src/bayes.jl :: nlp, grad_nlp` | ✅ step 3a |
| `nsm.py :: fit_posterior, fit_posterior_EM` | `src/vi.jl :: fit_posterior!, fit_posterior_EM!` | ✅ step 3b |
| `nsm.py :: update_hypers, approx_evidence` | `src/vi.jl :: update_hypers!, approx_evidence` | ✅ step 3b |
| `nsm.py :: predict(_point/_sample)` | `src/predict.jl :: predict_point, predict_sample` | ✅ step 4 |
| gLV residual PINN ("pNODE" formulation) | `src/residual.jl` | ✅ step 5 |
