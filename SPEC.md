# Quorum — Internal Implementation Spec (authoritative)

This file is the **single source of truth** for the math and the module interfaces.
Every module MUST implement exactly the signatures below. Do not invent alternative
names. If something here is ambiguous, prefer the formula over the prose.

Numeric stack: `numpy`, `scipy` only in the core. `pandas`/`matplotlib` only in
`benchmarks/` and `app.py`. No `torch`, no `pymc`, no network calls in the core.

---

## 0. The one-paragraph thesis

The decision we care about is a **ranking of models**, not a set of absolute scores.
Quorum makes the ranking the estimand, and then does three things a sequential-test
wrapper cannot do: (b) it **chooses which rollout to buy next** by expected reduction in
rank uncertainty per dollar, (c) it runs a **nested** sequential procedure — an outer
anytime-valid confidence sequence on rank contrasts, and an inner per-rollout
truncation rule — with an explicit correction that keeps the outer estimator unbiased,
and (d) it uses **known-mean control variates** to cut estimator variance.
Inference is **design-based and model-assisted**: validity comes from the sampling
design (Horvitz–Thompson + martingale), the Bayesian cell model only steers allocation.
A wrong model costs efficiency, never validity. That separation is the point.

---

## 1. Vocabulary and index conventions

- `n_tasks`  = T, task index `t`
- `n_seeds`  = K, seed index `k` (a seed is a *problem instance variant*, not RNG-only)
- `n_scaffolds` = S, scaffold index `s`
- `n_models` = M, model index `m`

- **Cell** = `(t, k, s, m)`. Has a true success probability `p[t,k,s,m] ∈ (0,1)`,
  a cost, and generates Bernoulli rollouts.
- **Block** `b = (t, k, s)`. There are `B = T*K*S` blocks. A block is the unit that is
  *shared* between two models being compared, which is what makes pairing possible.
- **Target weights** `w[b] ≥ 0`, `sum(w) == 1`. Default uniform `1/B`. FIXED for the
  whole run — the estimand is defined by `w`, so it must never adapt.

Block flattening is **row-major over (t, k, s)**: `b = (t*K + k)*S + s`.
Use `quorum.types.block_index(t,k,s,K,S)` and `unblock_index`. Everything must agree.

---

## 2. Estimands (what the tests are allowed to claim)

### 2.1 Pairwise contrast (the outer estimand)

For an ordered model pair `(a, c)`:

```
Delta[a,c] = sum_b w[b] * ( p[b, a] - p[b, c] )
```

This is a **paired, scaffold-marginalised** contrast. Note two properties we will
exploit and test:

- It is invariant to anything that shifts both models equally within a block.
- Under the simulator's `rank_distortion = 0` regime, `sign(Delta[a,c])` is the same
  for every scaffold subset, i.e. **scaffolds move absolute scores but not the rank**.

### 2.2 The decision

`RankDecision` = the total order of models induced by `Delta`. The procedure stops when
every **adjacent** pair in the current point-estimate order has a confidence-sequence
interval excluding 0. That certifies the full order by transitivity (state this in the
docstring: adjacent separation + transitivity of `Delta` ⇒ full order, at family-wise
level `alpha` after the correction in §5.4).

`Delta` is transitive by construction: `Delta[a,c] = theta[a] - theta[c]` where
`theta[m] = sum_b w[b] p[b,m]`. Say this explicitly — it is why adjacent pairs suffice.

---

## 3. `quorum/types.py`

```python
@dataclass(frozen=True)
class GridSpec:
    n_tasks: int; n_seeds: int; n_scaffolds: int; n_models: int
    @property
    def n_blocks(self) -> int  # T*K*S

@dataclass
class Rollout:
    task: int; seed: int; scaffold: int; model: int
    success: int            # 0/1, the *observed* outcome (may be a truncation guess)
    completed: bool         # False if inner rule truncated it
    n_steps: int            # steps actually executed (drives compute cost)
    n_steps_full: int       # steps it would have taken if never truncated
    cost_usd: float; cost_sec: float
    trajectory: np.ndarray  # shape (n_steps,), per-step scalar "health" features
    audited: bool = False   # True if the audit coin selected it for completion
    truncated_at: int = -1  # step at which the inner rule fired, -1 if never
    true_success: int = -1  # ground truth if known (simulator only; never used by
                            # any estimator — for tests/diagnostics only)

@dataclass
class RankDecision:
    order: list[int]                 # model indices, best first
    resolved: bool
    n_rollouts: int; cost_usd: float; cost_sec: float
    intervals: dict[tuple[int,int], tuple[float,float]]   # (a,c) -> (lo, hi) on Delta
    undecided_pairs: list[tuple[int,int]]
```

`block_index(t,k,s,K,S) -> int`, `unblock_index(b,K,S) -> (t,k,s)`.

---

## 4. `quorum/simulator.py`

### 4.1 Truth generator

```python
@dataclass
class GridTruth:
    spec: GridSpec
    p: np.ndarray            # (T,K,S,M) success probabilities
    theta: np.ndarray        # (M,) = sum_b w[b] p[b,m]  under uniform w
    true_order: np.ndarray   # (M,) model indices, best first, by theta
    ability: np.ndarray      # (M,) beta_m
    difficulty: np.ndarray   # (T,) alpha_t
    scaffold_effect: np.ndarray  # (S,) gamma_s
    cost_usd: np.ndarray     # (T,K,S,M) mean $ per rollout
    cost_sec: np.ndarray
    proxy_p: np.ndarray      # (T,K,S) cheap proxy-model success prob -> CUPED covariate
```

Latent linear predictor:

```
logit p[t,k,s,m] = mu
                 + alpha[t]                # task difficulty      ~ N(0, sd_task^2)
                 + beta[m]                 # model ability        (given / spaced)
                 + gamma[s]                # scaffold main effect ~ N(0, sd_scaffold^2)
                 + eta[t,k]                # seed jitter          ~ N(0, sd_seed^2)
                 + zeta[t,m]               # task x model         ~ N(0, sd_task_model^2)
                 + delta[s,m]              # scaffold x model     ~ N(0, rank_distortion^2)
```

- `delta` is the **only** term that can reorder models across scaffolds. With
  `rank_distortion=0.0` the rank is scaffold-invariant by construction (monotone in
  `beta[m]` at fixed block) — this is the empirical regularity we are reproducing.
- `ability` defaults to evenly spaced values with a controllable minimum gap
  `ability_gap`; that gap is the hardness knob for the whole benchmark.

`proxy_p[t,k,s] = sigmoid(mu + alpha[t] + gamma[s] + eta[t,k] + beta_proxy)` — i.e. a
cheap model on the *same* block, correlated with the block's difficulty but carrying no
information about which of the compared models is better. This is exactly the covariate
CUPED wants: **strongly correlated with the outcome, independent of the treatment
contrast**, and its `w`-weighted mean is computable in closed form over the whole grid,
so it is a *known-mean* control variate.

```python
def make_truth(spec, *, seed=0, mu=0.0, sd_task=1.0, sd_seed=0.35,
               sd_scaffold=0.8, sd_task_model=0.4, rank_distortion=0.0,
               ability_gap=0.35, ability=None,
               cost_usd_base=0.02, cost_sec_base=8.0, cost_dispersion=0.6,
               beta_proxy=-0.6) -> GridTruth
```

Costs must be **non-uniform across cells** (that is what makes the cost-aware allocator
non-trivial): `cost_usd[t,k,s,m] = cost_usd_base * exp(N(0, cost_dispersion^2))
* scaffold_cost_mult[s] * model_cost_mult[m]`, with slower/pricier models roughly the
stronger ones (a realistic and adversarial correlation: the informative cells are the
expensive ones).

### 4.2 Scaffold shift

```python
def apply_scaffold_shift(truth, *, shift, scaffolds=None, rank_distortion=None, seed=0) -> GridTruth
```
Adds `shift` to `gamma[s]` for the selected scaffolds (a large absolute-score move) and
optionally injects `rank_distortion`. Used to show (B) breaking while (D) holds.

### 4.3 Rollout simulator

```python
class RolloutSimulator:
    def __init__(self, truth: GridTruth, *, seed=0, max_steps=24,
                 signal_strength=2.2, cost_noise=0.25)
    n_rollouts: int; total_usd: float; total_sec: float; total_steps: int
    def rollout(self, t, k, s, m, *, truncator=None, rng=None) -> Rollout
    def reset_counters(self) -> None
```

Trajectory model (this is what the inner rule reads):

1. Draw `success ~ Bernoulli(p[t,k,s,m])` **first** (ground truth for this rollout).
2. Draw `n_steps_full ~ 6 + Poisson(max_steps/3)`, clipped to `[3, max_steps]`.
3. Trajectory `x[j]` for `j=0..n_steps_full-1` is a random walk whose drift depends on
   the eventual outcome:
   `x[j] = x[j-1] + N(drift, 1.0)`, `x[0]=N(0,1)`,
   `drift = +signal_strength/n_steps_full` if success else `-signal_strength/n_steps_full`.
   So the *running mean* of `x` separates success from failure progressively — early
   steps are ambiguous, late steps are informative. This reproduces "failure becomes
   visible partway through".
4. Cost accrues **per step**: a rollout truncated at step `j` costs
   `cost_* * (j+1)/n_steps_full * lognormal noise`. Truncation therefore genuinely
   saves money and seconds, which metric (4) measures.

If `truncator` is given, it is consulted after each step (see §6.2); when it fires the
rollout stops early **unless** the audit coin says to continue.

`Rollout.success` for a truncated, non-audited rollout is set to `0` (the rule's guess)
and `Rollout.completed=False`. **Estimators must not use `Rollout.success` naively for
truncated rollouts — they must go through `nested_seq.corrected_outcome`.** A test
asserts that naive use is biased and the corrected use is not.

### 4.4 Real-data loader — `quorum/loaders.py`

```python
def load_rollouts_jsonl(path) -> tuple[list[Rollout], GridSpec, dict]
def load_rollouts_csv(path) -> ...
def load_hf_dataset(repo_id, split="train", token=None) -> ...   # lazy import datasets
def to_frozen_grid(rollouts, spec) -> "FrozenGridSimulator"
```
`FrozenGridSimulator` exposes the same `.rollout()` API as `RolloutSimulator` but
samples with replacement from the observed rollouts of that cell — so every Quorum
method runs unchanged on real logged data. Required columns:
`task, seed, scaffold, model, success[, n_steps, cost_usd, cost_sec, trajectory]`.

---

## 5. `quorum/nested_seq.py` — the core

### 5.1 Empirical-Bernstein confidence sequence

Implement the **predictable plug-in empirical-Bernstein** CS (Waudby-Smith & Ramdas
style). For observations `X_i` with conditional mean `mu`, known bounds `[L,U]`:

Rescale `Y_i = (X_i - L)/(U - L) ∈ [0,1]`, target `m = (mu - L)/(U - L)`.
Predictable plug-ins, using only data up to `i-1`:

```
mu_hat[i]    = (1/2 + sum_{j<i} Y_j) / (1 + (i-1))
sigma2_hat[i]= (1/4 + sum_{j<i} (Y_j - mu_hat[j])^2) / (1 + (i-1))
lambda[i]    = min( c_lambda,  sqrt( 2*log(2/alpha) / (sigma2_hat[i] * i * log(1+i)) ) )
               with c_lambda = 0.5  (must satisfy lambda in (0,1))
v[i]         = 4 * (Y_i - mu_hat[i])^2
psi_E(l)     = (-log(1-l) - l) / 4
```

The CS for `m` is the set of `m` with
`| sum_i lambda[i]*(Y_i - m) | <= log(2/alpha) + sum_i v[i]*psi_E(lambda[i])`,
which solves in closed form:

```
center = sum_i lambda[i]*Y_i / sum_i lambda[i]
radius = ( log(2/alpha) + sum_i v[i]*psi_E(lambda[i]) ) / sum_i lambda[i]
m in [center - radius, center + radius]
```
Map back: `mu in [L + (U-L)*(center-radius), L + (U-L)*(center+radius)]`, intersected
with `[L,U]`, and **intersected with the previous interval** (running intersection is
still valid and is what makes it a *sequence*).

```python
class EmpiricalBernsteinCS:
    def __init__(self, alpha, lower, upper, c_lambda=0.5)
    def update(self, x: float | np.ndarray) -> tuple[float,float]
    @property
    def interval(self) -> tuple[float,float]
    @property
    def n(self) -> int
    def excludes(self, value=0.0) -> bool
```

**Validity requirement (tested):** `X_i` need only be a bounded sequence with
`E[X_i | F_{i-1}] = mu`. It does **not** need to be i.i.d. This is precisely why
adaptive allocation is allowed. Put this sentence in the class docstring.

### 5.2 Horvitz–Thompson pair channel

For pair `(a,c)`, at step `i` the allocator produces a **predictable** distribution
`pi_i` over blocks with `pi_i[b] >= pi_min > 0`. Sample `b_i ~ pi_i`, run one rollout of
each model at `b_i`, obtain corrected outcomes `Ytil_a, Ytil_c` (§6.3), then

```
Z_i = (w[b_i] / pi_i[b_i]) * (Ytil_a - Ytil_c)
```

`E[Z_i | F_{i-1}] = sum_b w[b] (p[b,a] - p[b,c]) = Delta[a,c]`   ← unbiased for ANY
predictable `pi_i`. This identity is the whole reason adaptive allocation is legal;
state it in the docstring of `PairChannel`.

Bounds: `|w[b]/pi_i[b]| <= wmax/pi_min =: G` and `|Ytil| <= R` (§6.3), so
`Z_i ∈ [-G*2R, +G*2R]` — pass these to the CS as `lower/upper`. Enforce `pi_min` in the
allocator, never trust it silently: `PairChannel` asserts it.

```python
class PairChannel:
    def __init__(self, pair, w, alpha, *, pi_min, outcome_range, cuped=None)
    def observe(self, block, pi_b, ytil_a, ytil_c, cov=None) -> tuple[float,float]
    @property interval, n, z_history, excludes_zero
```

### 5.3 Inner truncation

```python
class InnerTruncator:
    """Sequentially decides, from a partial trajectory, that this rollout will fail."""
    def __init__(self, *, threshold=-1.0, min_steps=3, audit_rate=0.25,
                 q_hat=0.0, calibrated=None, rng=None)
    def should_stop(self, traj_prefix: np.ndarray, step: int) -> bool
    def audit(self, rng) -> bool          # independent Bernoulli(audit_rate)
    def q(self, traj_prefix, step) -> float   # predicted P(success | stopped here)
```

Rule: fire when `step >= min_steps` and `mean(traj_prefix) <= threshold` — i.e. the
running-mean statistic of §4.3. `q_hat` is the calibrated `P(success | fired)`; it may
be a constant, or a `CalibratedQ` fitted on audited rollouts.

**The audit coin must be drawn independently of the outcome**, at the moment the rule
fires. If audited, the rollout runs to completion (no compute saved, true outcome seen).

### 5.4 The correction — why nesting stays valid

```python
def corrected_outcome(r: Rollout, q: float, audit_rate: float) -> float
```

```
if r.completed:                       # rule never fired
    Ytil = r.success                                   in [0,1]
else:                                 # rule fired
    Ytil = q + (1{r.audited} / rho) * (r.success - q)  in [-1/rho, 1 + 1/rho]
```

Conditional unbiasedness, for any `q` whatsoever (correct or not):

```
E[Ytil | fired, features] = q + (rho/rho) * (E[X|features] - q) = E[X|features]
```

So **the truncation rule may be arbitrarily miscalibrated and the outer estimator stays
unbiased**; miscalibration costs variance, not validity. The variance penalty is
`(1-rho)/rho * E[(X-q)^2 | fired]`, which is near zero for a confident, well-calibrated
rule — which is why the compute saving is nearly free. Both halves of this claim are
tested (`tests/test_truncation_bias.py`).

Outcome range: `R = max(1, 1 + 1/rho)`. Report it via `outcome_range(audit_rate, any_truncation: bool)`.

### 5.5 Family-wise control across pairs

`M-1` adjacent pairs are tested. Allocate `alpha` across the `M*(M-1)/2` **candidate**
pairs by Bonferroni (`alpha_pair = alpha / n_candidate_pairs`) — not just the adjacent
ones, because which pairs end up adjacent is data-dependent. Document that this is
conservative and is what makes the family-wise claim honest. `alpha_split ∈
{"bonferroni","none"}`, default `"bonferroni"`.

### 5.6 The driver

```python
class NestedSequentialTest:
    def __init__(self, spec, simulator, *, alpha=0.05, w=None,
                 allocator=None, truncator=None, cuped=None,
                 pi_min_frac=0.10, max_rollouts=200_000, budget_usd=None,
                 budget_sec=None, alpha_split="bonferroni", rng=None,
                 check_every=1, candidate_pairs=None)
    def run(self) -> RankDecision
    def step(self) -> None
    def current_order(self) -> list[int]
```

Loop: (1) rank models by current channel estimates → adjacent pairs; (2) ask the
allocator for the pair to fund and the block distribution `pi`; (3) sample the block,
run two rollouts through the truncator; (4) push corrected, CUPED-adjusted outcomes into
the channel; (5) stop when every adjacent pair's interval excludes 0, or budget is
exhausted. `pi_min = pi_min_frac / B`.

---

## 6. `quorum/cell_model.py`

Hierarchical Bayesian partial-pooling model over cells, fit by **MAP + Laplace**, no MCMC.

```
logit p = mu + alpha[t] + beta[m] + gamma[s] + zeta[t,m] + delta[s,m]
priors: alpha ~ N(0, tau_task^2), beta ~ N(0, tau_model^2), gamma ~ N(0, tau_scaffold^2),
        zeta ~ N(0, tau_tm^2), delta ~ N(0, tau_sm^2), mu ~ N(0, 5^2)
```

- Design matrix is sparse one-hot; fit by Newton / IRLS on the penalised Bernoulli
  log-likelihood over **aggregated counts per cell** (successes, trials) — that is
  `O(n_cells)` per iteration and is fast.
- Posterior covariance `Sigma = H^{-1}` at the mode (Laplace).
- Variance components `tau_*` estimated by a few EB fixed-point updates
  (`tau^2 <- (||coef||^2 + trace(Sigma_block)) / n_block`), clipped to `[0.05, 5]`.

```python
class HierarchicalCellModel:
    def __init__(self, spec, *, tau_init=..., empirical_bayes=True, max_iter=50, ridge=1e-6)
    def update(self, rollouts_or_counts) -> None       # incremental; accepts Rollout list
    def add(self, t,k,s,m, success:int) -> None
    def fit(self) -> None
    def predict_p(self) -> np.ndarray                  # (T,K,S,M) posterior-mean p
    def ability_posterior(self, n_draws=512, rng=None) -> np.ndarray   # (n_draws, M)
    def rank_posterior(self, n_draws=512, rng=None) -> np.ndarray      # (M,M) P(a > c)
    def contrast_variance(self, a, c) -> float          # v^T Sigma v, v = e_a - e_c
    def info_gain_for_cell(self, t,k,s,m, a, c) -> float  # closed form, §7.1
```

Note the seed dimension is pooled through `alpha[t] + eta` — the model deliberately does
**not** give each seed its own free parameter (that is the partial pooling that makes
unvisited blocks predictable). Document this.

**The cell model never enters the validity argument.** Say so in the module docstring.

---

## 7. `quorum/allocator.py`

### 7.1 Closed-form decision-targeted value of information

For a Laplace-Gaussian posterior with covariance `Sigma` over coefficients, adding one
Bernoulli observation at cell `x` (its one-hot design row) with Fisher weight
`u = p(1-p)` gives the exact rank-1 posterior update

```
Sigma' = Sigma - (u * Sigma x x^T Sigma) / (1 + u * x^T Sigma x)
```

so the variance of the decision contrast `v = e_a - e_c` drops by exactly

```
gain(cell; a,c) = u * (v^T Sigma x)^2 / (1 + u * x^T Sigma x)      >= 0
```

This is a **c-optimal (decision-targeted) design criterion**: it scores a cell by how
much it shrinks the uncertainty of *the contrast that decides the ranking*, not by how
much it shrinks uncertainty overall. Implement it exactly; it is O(1) per cell given
`Sigma x`.

Cost-awareness: score `= gain / (cost_usd + lambda_sec * cost_sec + eps)`, i.e.
**information per dollar**. Cells that are informative but ruinously expensive lose to
cells that are slightly less informative and 10x cheaper — that is metric (1).

Pair weighting: sum `gain` over candidate pairs weighted by how close each pair is to
flipping, `pair_weight[a,c] = P(sign flip)` estimated from the rank posterior, so
already-resolved pairs stop attracting budget.

### 7.2 Interface

```python
class DecisionTargetedAllocator:
    def __init__(self, spec, cell_model, costs_usd, costs_sec, *, w=None,
                 pi_min_frac=0.10, strategy="info_cost", lambda_sec=0.0,
                 refit_every=25, n_draws=256, rng=None, temperature=1.0)
    def select(self, pairs, channels) -> tuple[tuple[int,int], np.ndarray, int, float]
        # -> (pair, pi over blocks (len B, sums to 1, >= pi_min), sampled block, pi[b])
    def observe(self, rollout) -> None
```

`strategy ∈ {"info_cost", "thompson", "uniform", "neyman"}`:
- `"info_cost"` — §7.1, the default.
- `"thompson"` — draw two ability vectors from the posterior, pick the pair whose sign
  disagrees most often, then sample blocks ∝ predicted contrast variance / cost.
- `"uniform"` — the (C) baseline: `pi = w`, pair = round robin over adjacent pairs.
- `"neyman"` — `pi ∝ w * sd_b / sqrt(cost_b)`, the classical optimum, as an ablation.

**Mixing to the floor is mandatory:** the returned `pi` must be
`(1 - eps) * pi_raw + eps * w` with `eps = pi_min_frac`, guaranteeing
`pi[b] >= pi_min_frac * w[b] > 0` and hence the bounded HT weight of §5.2.
This is the exploration floor that keeps the confidence sequence finite-width; explain
that in the docstring.

---

## 8. `quorum/cuped.py`

```python
class CupedAdjuster:
    """Known-mean control variates. Y~ = Y - phi^T (C - E[C]); E[C] must be KNOWN."""
    def __init__(self, cov_means: np.ndarray, *, warmup=30, ridge=1e-8,
                 clip=10.0, per_pair=True)
    def adjust(self, pair, z, cov_vec) -> float
    def observe(self, pair, z, cov_vec) -> None
    def theta(self, pair) -> np.ndarray
    def variance_ratio(self, pair) -> float     # Var(Y~)/Var(Y) on collected history
```

- `phi` is refit from history **strictly before** the current observation
  (`phi_i` is `F_{i-1}`-measurable) — this is what preserves
  `E[Y~_i | F_{i-1}] = Delta`. A test asserts non-anticipativity.
- Covariates used, all with **exactly known** `w`-weighted means (this is the design
  contribution — an estimated mean would silently bias the CS):
  1. proxy-model success at the block, HT-weighted: `(w_b/pi_b) * proxy_p[b]`
     → mean `sum_b w_b proxy_p[b]`, computable in closed form on the whole grid.
  2. historical success rate at the block from a previous run, same weighting.
  3. block difficulty from a **frozen pilot fit** of the cell model (frozen ⇒ a fixed
     function of `b` ⇒ known mean).
  The HT weight appears in the covariate too, which is important: it lets the control
  variate absorb the weight noise `w_b/pi_b`, not just the outcome noise.
- Optional `allow_estimated_mean=False` (default). Trajectory-prefix features of the
  *same* rollout have no known mean; they are handled in the `q`-model of §5.3 instead.
  Document this refusal — it is a correctness decision, not an omission.

`variance_ratio` must be measured and reported by the benchmark; the pytest asserts a
significant reduction.

---

## 9. `benchmarks/methods.py` — the four conditions

All four run on the **same** `GridTruth` and are scored on whether the returned order
equals `truth.true_order`. All return `RankDecision`.

- **A `naive_uniform`** — `n_per_cell` rollouts in *every* cell, rank by grand mean per
  model, no sequential stopping. Swept over `n_per_cell` to find the smallest budget at
  which the error rate over replications is `<= alpha`.
- **B `subset_irt`** — tinyBenchmarks-style. Pilot on all blocks with few rollouts; fit
  a 2PL IRT (`p = sigmoid(disc_t * (beta_m - diff_t))`) by MAP; pick `n_anchor` items
  maximising Fisher information at the current ability estimates (plus a
  cluster-diversity term); estimate abilities from the anchors only; rank.
  **Crucially, B estimates absolute ability from a fixed anchor set and has no scaffold
  term** — so a scaffold shift that changes which items are hard invalidates the anchors
  and flips the order. That is the point of metric (3).
- **C `sequential_only`** — `NestedSequentialTest` with `allocator=uniform`,
  `truncator=None`, `cuped=None`. This is "an existing anytime-valid method".
- **D `quorum`** — `NestedSequentialTest` with the info-cost allocator, the inner
  truncator, and CUPED. All three on.

`benchmarks/run.py` runs `n_reps` replications per condition over a seed sweep and
writes `results/*.csv` plus the figures. It must record, per rep: rollouts used, `$`,
seconds, steps executed vs. steps that would have been executed, whether the order was
correct, and per-pair interval widths.

Metrics, in the order the brief asks for them:
1. rollouts and cost to a *correct* order (D lowest),
2. error rate ≤ nominal alpha (coverage),
3. accuracy under scaffold shift (B collapses, D holds),
4. compute saved by inner truncation = `1 - steps_executed / steps_full`.

---

## 10. Figures (`benchmarks/figures.py` → `figures/*.png`)

1. `fig1_cost_to_decision.png` — grouped bars, rollouts and `$` for A/B/C/D, log y,
   with error bars. The headline.
2. `fig2_scaffold_shift.png` — accuracy vs shift magnitude, one line per method; B falls
   off a cliff, D flat.
3. `fig3_coverage.png` — empirical error rate vs nominal alpha, with the `y=x` line.
4. `fig4_variance_reduction.png` — CUPED variance ratio, and CS width vs rollouts for
   C vs D.
5. `fig5_truncation.png` — compute saved vs audit rate, with the bias (≈0) overlaid.

Matplotlib only, no seaborn, no custom colour cycles beyond a small explicit palette,
`dpi=150`, tight bbox.

---

## 11. CLI (`quorum/cli.py`)

```
quorum compare MODEL_A MODEL_B --benchmark NAME [--budget $50] [--alpha 0.05]
                               [--scaffold-shift 0.0] [--seed 0] [--json out.json]
quorum rank --models m0,m1,m2,m3 --benchmark NAME --budget $100
quorum bench [--reps 30] [--out results/] [--quick]
quorum simulate --out data/rollouts.jsonl --n 5000
quorum figures [--results results/] [--out figures/]
```

`--budget` accepts `$50`, `50`, `50usd`, `120s`, `1h`. Parser: `parse_budget(str) ->
(usd|None, sec|None)`. Exit code 0 if resolved, 2 if budget exhausted while undecided.
Output must state, when undecided: *"indistinguishable at this budget"* — never assert
one model is better. Include the projected extra rollouts needed (§12).

---

## 12. Extra-rollouts projection (used by CLI and the Space)

For a pair with current CS half-width `h_n` after `n` observations and a target of
excluding 0 at `|Delta_hat|`, the EB width shrinks like `h_n ≈ c / sqrt(n)`; fit `c`
from the realised `(n, h_n)` and solve for `n*` with `h_{n*} = |Delta_hat| * safety`,
i.e. `n* = n * (h_n / (|Delta_hat| * safety))^2`. Return `max(0, ceil(n*) - n)`.
`safety=0.95` default. Function: `quorum.nested_seq.rollouts_to_resolve(n, half_width,
delta_hat, safety=0.95) -> int`. This is explicitly a **projection, not a guarantee** —
label it that way in every surface that prints it.

---

## 13. HF surfaces

- `app.py` — Gradio Blocks. Loads a leaderboard CSV (rollout-level or aggregate),
  recomputes confidence sequences, renders a table where **pairs that are not
  statistically distinguishable are highlighted red**, and shows the projected extra
  rollouts per unresolved pair. Must run with zero network access using a bundled demo
  CSV. Never emit a claim that a named model is worse than another; the copy says
  "not distinguishable at alpha=..." and the README repeats the norm.
- `scripts/push_rollouts.py` — build a `datasets.Dataset` from rollout-level JSONL/CSV
  and `push_to_hub`. `--dry-run` must work with no token and no network.
- `quorum/adapters/lighteval_adapter.py` — a `QuorumSequentialSampler` that plugs into a
  lighteval-style loop: given a task list and a callable `run_fn(task, seed, scaffold,
  model) -> Rollout`, it drives the same `NestedSequentialTest`, so lighteval users get
  early stopping for free. Must degrade gracefully (`lighteval` not installed ⇒ the
  adapter still works on plain callables).

---

## 14. Tests (`tests/`)

- `test_coverage.py` — over ≥300 replications with a **null pair** (equal ability), the
  fraction of runs where the CS ever excludes 0 is ≤ `alpha` (+ MC slack). Also test the
  CS under adaptive allocation, which is the non-obvious case.
- `test_efficiency.py` — D reaches the correct order with strictly fewer rollouts than A
  and C, on matched seeds, with a paired test.
- `test_truncation_bias.py` — corrected mean ≈ true mean (CI covers), while the naive
  truncated mean is detectably biased low. Include a deliberately **miscalibrated** `q`
  to prove unbiasedness does not depend on the rule being right.
- `test_cuped.py` — variance ratio significantly < 1; and `phi` non-anticipativity.
- `test_simulator.py` — rank preserved across scaffolds when `rank_distortion=0`;
  broken when it is large; costs non-uniform.
- `test_allocator.py` — `pi >= pi_min` always, sums to 1; info-gain matches a brute-force
  recomputation of `v^T Sigma' v`.
- `test_cli.py` — `compare` runs end to end and never prints a superiority claim when
  undecided.

Tests must be fast: whole suite under ~3 minutes. Use small grids and `pytest.mark.slow`
for anything heavier.

---

## 15. House rules

- Every public function's docstring ends with a line:
  `Claim: <low-cost | coverage | shift-robust | unbiased-truncation | variance-reduction | infrastructure>`
  naming which thesis claim it substantiates. This is a hard requirement from the brief.
- No `print` in library code; use `logging`.
- Type hints everywhere, `from __future__ import annotations`.
- Deterministic given a seed. Every stochastic function takes an explicit `rng`.
- No claim about real named models anywhere in output text.
