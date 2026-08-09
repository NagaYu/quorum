---
title: Quorum
emoji: "⚖️"
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
tags:
  - llm-evaluation
  - sequential-testing
  - anytime-valid
  - experimental-design
---

# Quorum &mdash; which leaderboard gaps are real?

A leaderboard column of numbers invites you to read a ranking off it. Often the data does
not support that ranking.

This Space recomputes **anytime-valid confidence sequences** from rollout-level results
and highlights, in red, every pair of models the data does **not** separate &mdash; with a
projection of how many more rollouts would settle each one.

On the bundled synthetic demo, **half the pairs are statistically indistinguishable** at
alpha = 0.05 even though every score in the table is different. Move the alpha control and
watch pairs cross over: that boundary, not the score column, is what the data actually
supports.

## What this is not

"Not distinguishable" means *this data cannot tell these two apart*. It does not mean the
models are equal, and it is not a criticism of either one. The demo data is synthetic and
its model labels are placeholders &mdash; nothing here measures, ranks or disparages a
real system.

## Method

The estimand is a **paired, scaffold-marginalised contrast**
`Delta[a,c] = sum_b w[b] (p[b,a] - p[b,c])` over blocks `b = (task, seed, scaffold)`.
It decomposes as `theta_a - theta_c`, so it is transitive &mdash; separating adjacent
pairs certifies the whole order &mdash; and it is invariant to anything that shifts both
models equally within a block, which is why a scaffold change does not flip it.

The page is static: the analysis was run at build time by the same code the test suite
exercises, and the alpha control switches between precomputed results. To run it on your
own data, use the CLI:

```bash
pip install -e .
quorum compare modelA modelB --benchmark synthetic-easy --budget $50
```

Full method, benchmarks, figures and test suite:
**[github.com/NagaYu/quorum](https://github.com/NagaYu/quorum)**
