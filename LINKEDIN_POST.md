# LinkedIn post

Primary version below. Two shorter cuts follow for X / Bluesky and for a comment reply.

---

## Main post (English, ~330 words)

A leaderboard column of numbers invites you to read a ranking off it. Very often the data
does not support that ranking — and nothing in the table tells you which gaps are real.

I built **Quorum** to make that visible, and then to make it cheap.

**Try the demo (loads instantly, no sign-in):**
https://huggingface.co/spaces/NagaYu/quorum

Every score in that table is different. At α = 0.05, **half the pairs are statistically
indistinguishable**. Move the alpha slider and watch pairs cross the boundary. That
boundary — not the score column — is what the data actually supports.

**The idea.** Make the *ranking decision* the estimand, not the absolute scores. Estimate a
paired contrast blocked on (task, seed, scaffold). Two things fall out: it decomposes as
θ_a − θ_c, so it is transitive and separating adjacent pairs certifies the whole order; and
it is invariant to anything that moves both models equally within a block, so a scaffold
change does not flip it.

**Measured over 12 independent worlds, family-wise α = 0.05:**

• **34% cheaper** than a sequential test on a uniform design, to the same certified ranking
• Error rate at or below nominal across 300 null-pair replications — including under
adaptive allocation, which is the case the Horvitz–Thompson construction exists for
• Under a scaffold shift, a tinyBenchmarks-style method reusing a published anchor set
degrades from 0.67 to 0.33 accuracy. Quorum holds at 1.00 throughout

**I also shipped everything that didn't work.**

→ CUPED reduces variance by ~0% under a paired design. That is structural, not a tuning
failure: a covariate symmetric in the two models cannot correlate with their difference to
first order. It earns its keep where pairing is impossible — logged data, public
leaderboards — and there it removes 26%.

→ Inner truncation saves 19% of trajectory steps, but the audit correction that keeps it
unbiased widens the confidence sequence enough that total cost goes *up*. Off by default.

→ A covariate centred on the wrong constant gave me a 100% false-positive rate during
development. It is a regression test now.

Deleting the negative results just means the next person rediscovers them.

**Code, benchmarks, figures, full test suite:** https://github.com/NagaYu/quorum

Apache-2.0. Feedback and disagreement welcome — especially on the nesting correction.

#LLMEvaluation #ExperimentalDesign #Statistics #MLOps #OpenSource #HuggingFace

---

## Short version (X / Bluesky)

A leaderboard invites you to rank the numbers. The data often doesn't support the ranking.

Quorum makes the *ranking* the estimand: 34% cheaper to the same certified order, coverage
holds under adaptive allocation, robust to scaffold shift.

Demo (half the pairs are indistinguishable):
https://huggingface.co/spaces/NagaYu/quorum
Code: https://github.com/NagaYu/quorum

---

## Comment reply, if someone asks "why is it cheaper?"

Two reasons, and only one of them is the sequential test.

The estimand is a paired contrast, so the shared instance difficulty cancels before any
inference happens. Then the allocator scores each candidate cell by the exact rank-1 drop
in the variance of the contrast that decides the ranking — `u·(vᵀΣx)²/(1+u·xᵀΣx)` — divided
by that cell's dollar cost. It buys information about the decision, not about the grid.

Adaptivity would normally break the confidence sequence. It doesn't here because the
allocator's sampling distribution is predictable and the estimator reweights by w/π, so
E[Z | F_{i−1}] = Δ for *any* π the allocator picks. The exploration floor π ≥ ε·w is what
bounds the importance weight — it's a requirement, not a heuristic.

---

## Notes for posting

- The Space link is the hook: it loads in under a second and the red/green matrix makes
  the point faster than any paragraph. Lead with it.
- LinkedIn strips most formatting. The bullets above use `•` and `→` so they survive.
- If you want a single image for the post, use `figures/fig2_scaffold_shift.png` — the
  divergence between the flat lines and the collapsing one reads clearly at thumbnail size.
