# X / Twitter posts
Character counts verified against X's rules (URLs count as 23 regardless of length).

## Single tweet

```
A leaderboard invites you to rank the numbers. Often the data doesn't support that ranking.

Quorum makes the ranking itself the estimand.

In this demo every score differs — yet half the pairs are statistically indistinguishable:

https://huggingface.co/spaces/NagaYu/quorum
```

255/280 characters. Attach `figures/fig2_scaffold_shift.png` — it reads at thumbnail size.

## Thread

**1/9** (273/280)

```
A leaderboard invites you to rank the numbers. Often the data doesn't support that ranking — and nothing tells you which gaps are real.

So I built Quorum. Demo loads instantly, no sign-in. Every score differs; half the pairs are indistinguishable:

https://huggingface.co/spaces/NagaYu/quorum
```

**2/9** (239/280)

```
The idea: make the ranking decision the estimand, not the absolute scores.

Estimate a paired contrast blocked on (task, seed, scaffold). It decomposes as θ_a − θ_c, so it's transitive — separating adjacent pairs certifies the whole order.
```

**3/9** (217/280)

```
It's also invariant to anything that moves both models equally within a block.

So a scaffold change shifts every absolute score and doesn't touch the ranking. That's the property the whole design is built to exploit.
```

**4/9** (242/280)

```
Measured over 12 independent worlds at family-wise α=0.05:

• 34% cheaper than a sequential test on a uniform design, to the same certified ranking
• Error rate at or below nominal over 300 null-pair reps — including under adaptive allocation
```

**5/9** (265/280)

```
Adaptive allocation normally breaks a confidence sequence. Not here: the allocator's sampling distribution is predictable and the estimator reweights by w/π, so E[Z | F_{i−1}] = Δ for ANY π it picks.

The exploration floor π ≥ ε·w is a requirement, not a heuristic.
```

**6/9** (250/280)

```
Under a scaffold shift, a tinyBenchmarks-style method reusing a published anchor set drops from 0.67 to 0.33 accuracy. Quorum holds at 1.00 throughout.

Its estimand is absolute ability read off a frozen anchor set. The shift invalidates the anchors.
```

**7/9** (267/280)

```
I also shipped everything that didn't work.

CUPED cuts variance by ~0% under a paired design. Structural, not a tuning failure: a covariate symmetric in the two models can't correlate with their difference to first order.

It removes 26% where pairing is impossible.
```

**8/9** (268/280)

```
Inner truncation saves 19% of trajectory steps — but the audit correction that keeps it unbiased widens the interval enough that total cost goes UP. Off by default.

And a covariate centred on the wrong constant gave me a 100% false-positive rate. Regression test now.
```

**9/9** (217/280)

```
Deleting the negative results just means the next person rediscovers them.

Code, benchmarks, figures, full test suite (Apache-2.0):
https://github.com/NagaYu/quorum

Disagreement welcome, especially on the nesting correction.
```

## Notes

- Lead with the Space link in 1/9. It loads in under a second and the red/green matrix
  makes the argument faster than any sentence — that is the tweet people quote.
- Attach an image to 1/9 (`figures/fig2_scaffold_shift.png`) and to 4/9
  (`figures/fig1_cost_to_decision.png`) if you want the numbers to travel.
- No hashtags. On X they cost reach on technical posts; the terms people search for
  (anytime-valid, confidence sequence, LLM eval) are already in the text.
- 5/9 and 7/9 are the two most quotable tweets. If the thread underperforms, either
  works as a standalone post later.
