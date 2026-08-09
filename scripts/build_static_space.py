"""Generate a free, browser-only Hugging Face **static** Space.

A Gradio Space needs a paid runtime; a static Space is free. This build therefore runs the
**real** analysis at build time -- the same ``app.analyse`` the server app calls, over the
same estimator in ``quorum.nested_seq`` -- and bakes the results into a self-contained
page. Switching alpha in the browser is a lookup into precomputed results, not a
re-computation, so the page loads instantly and has no runtime dependencies at all.

Why not run Python in the browser? Gradio Lite (Pyodide) was the first choice and was
abandoned after testing: every published ``@gradio/lite`` build we tried fails to resolve
its own ``huggingface_hub`` dependency chain under Pyodide (missing pure-Python wheels,
then a missing ``httpcore``, then a failing submodule import). A page that spends three
minutes loading and then shows a traceback is worse than no page. The trade is that
visitors cannot upload their own CSV here; the README points them at the CLI for that,
which is honest and takes one command.

Usage::

    python scripts/build_static_space.py --out space/

Claim: infrastructure
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


ALPHAS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)

INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Quorum &mdash; which leaderboard gaps are real?</title>
<style>
  :root {{ color-scheme: light dark; --fg:#1b1b1b; --muted:#5a5a5a; --line:#e2e2e2;
           --bg:#ffffff; --card:#fafafa; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e9e9e9; --muted:#a5a5a5; --line:#333; --bg:#141414; --card:#1c1c1c; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); line-height:1.6;
          font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
  main {{ max-width: 60rem; margin: 0 auto; padding: 2rem 1.2rem 4rem; }}
  h1 {{ font-size: 1.9rem; margin: 0 0 .2rem; letter-spacing: -.02em; }}
  .lede {{ font-size: 1.12rem; color: var(--muted); margin: 0 0 1.6rem; }}
  .note {{ background: var(--card); border:1px solid var(--line); border-left:3px solid #c0392b;
           border-radius:6px; padding: .9rem 1.1rem; margin: 1.4rem 0; font-size:.95rem; }}
  .controls {{ display:flex; align-items:center; gap:1rem; flex-wrap:wrap;
               background:var(--card); border:1px solid var(--line);
               border-radius:8px; padding:1rem 1.2rem; margin:1.6rem 0; }}
  .controls label {{ font-weight:600; font-size:.95rem; }}
  input[type=range] {{ flex:1; min-width:14rem; }}
  #alphaval {{ font-variant-numeric: tabular-nums; font-weight:700; min-width:4.5rem; }}
  table {{ border-collapse: collapse; font-size: .88rem; width:100%; }}
  th, td {{ padding:.45rem .6rem; border:1px solid var(--line); text-align:center; }}
  th {{ background: var(--card); }}
  td.name, th.name {{ text-align:left; white-space:nowrap; }}
  .scroll {{ overflow-x:auto; margin: .8rem 0 1.6rem; }}
  .sep  {{ background:#e8f5ec; color:#1d6b45; }}
  .nosep {{ background:#fdeaea; color:#a32222; font-weight:600; }}
  @media (prefers-color-scheme: dark) {{
    .sep {{ background:#14301f; color:#7fd7a6; }}
    .nosep {{ background:#3a1717; color:#f09a9a; }}
  }}
  small {{ color: var(--muted); }}
  a {{ color:#2e6fb5; }}
  footer {{ margin-top:2.5rem; padding-top:1.2rem; border-top:1px solid var(--line);
            font-size:.9rem; color:var(--muted); }}
</style>
</head>
<body>
<main>
  <h1>Quorum</h1>
  <p class="lede">Which leaderboard gaps does the data actually support?</p>

  <p>The table below is a synthetic leaderboard. The scores differ. Read down the
  <b>observed score</b> column and you will start ranking the models &mdash; that is what a
  column of numbers does to a reader.</p>

  <p>Now look at the pairwise matrix. Every pair recomputed with an <b>anytime-valid
  confidence sequence</b> on the paired, scaffold-marginalised contrast. Red means the
  interval contains zero: <b>this data cannot tell those two apart.</b></p>

  <div class="controls">
    <label for="alpha">Family-wise alpha</label>
    <input type="range" id="alpha" min="0" max="{n_alpha_max}" step="1" value="{default_idx}" />
    <span id="alphaval"></span>
  </div>

  <div class="note" id="summary"></div>

  <h2>Models</h2>
  <div class="scroll" id="models"></div>

  <h2>Pairwise separation</h2>
  <div class="scroll" id="matrix"></div>

  <p><small>Intervals are anytime-valid: they may be inspected after every observation
  with no correction for peeking. Alpha is split across all candidate pairs by Bonferroni.
  The &ldquo;additional rollouts&rdquo; figure extrapolates the observed shrinkage rate
  &mdash; it is a projection, not a guarantee.</small></p>

  <h2>How to read &ldquo;not distinguishable&rdquo;</h2>
  <p>It means <i>this data cannot separate these two</i>. It does not mean the models are
  equal, and it is not a criticism of either one. The demo data here is synthetic and the
  labels are placeholders &mdash; nothing on this page measures, ranks or disparages a
  real system.</p>

  <h2>Run it on your own data</h2>
  <p>This page is static, so it cannot take an upload. The same analysis runs locally in
  one command:</p>
  <pre style="background:var(--card);border:1px solid var(--line);border-radius:6px;padding:1rem;overflow-x:auto"><code>pip install -e .
quorum compare modelA modelB --benchmark synthetic-easy --budget $50</code></pre>

  <footer>
    Generated by <code>scripts/build_static_space.py</code> from the same estimator the
    test suite exercises &mdash;
    <a href="https://github.com/NagaYu/quorum">github.com/NagaYu/quorum</a>.
    Apache-2.0.
  </footer>
</main>

<script id="results" type="application/json">{payload}</script>
<script>
  const DATA = JSON.parse(document.getElementById("results").textContent);
  const slider = document.getElementById("alpha");
  function render() {{
    const r = DATA[slider.value];
    document.getElementById("alphaval").textContent = "alpha = " + r.alpha;
    document.getElementById("summary").innerHTML = r.summary;
    document.getElementById("models").innerHTML = r.models;
    document.getElementById("matrix").innerHTML = r.matrix;
  }}
  slider.addEventListener("input", render);
  render();
</script>
</body>
</html>
"""

SPACE_CARD = """\
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
"""


def _compute_results() -> list[dict[str, str]]:
    """Run the real ``app.analyse`` at each alpha and capture its rendered output.

    Importing ``app`` here is the point: the numbers on the published page come from the
    same function the server app calls, so the Space cannot drift away from the estimator
    it claims to demonstrate.

    Claim: coverage
    """
    import sys

    sys.path.insert(0, str(ROOT))
    import app as server_app  # noqa: PLC0415 - needs the path set first

    server_app._ensure_demo()
    out: list[dict[str, str]] = []
    for alpha in ALPHAS:
        table, note, matrix = server_app.analyse(None, alpha)
        rows = "".join(
            "<tr><td class='name'>{}</td><td>{:.4f}</td><td>{:,}</td><td>{}</td></tr>".format(
                r["model"], r["observed score"], int(r["rollouts"]),
                int(r["pairs not separated"]))
            for _, r in table.iterrows()
        )
        models_html = (
            "<table><thead><tr><th class='name'>model</th><th>observed score</th>"
            "<th>rollouts</th><th>pairs not separated</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
        summary = note.replace("**", "").replace("\n", " ")
        matrix = (matrix
                  .replace("background:#e8f5ec;color:#1d6b45", "")
                  .replace("background:#fdeaea;color:#a32222", "")
                  .replace("style='padding:6px;text-align:center;'>separated",
                           "class='sep'>separated")
                  .replace("style='padding:6px;text-align:center;'><b>not distinguishable</b>",
                           "class='nosep'><b>not distinguishable</b>"))
        out.append({"alpha": f"{alpha:g}", "summary": html.escape(summary),
                    "models": models_html, "matrix": matrix})
        logger.info("computed alpha=%g", alpha)
    return out


def build(out: Path) -> None:
    """Write ``index.html`` and the Space card into ``out``.

    Claim: infrastructure
    """
    out.mkdir(parents=True, exist_ok=True)
    results = _compute_results()
    default_idx = ALPHAS.index(0.05) if 0.05 in ALPHAS else 0
    index = INDEX_TEMPLATE.format(
        payload=json.dumps(results),
        n_alpha_max=len(ALPHAS) - 1,
        default_idx=default_idx,
    )
    (out / "index.html").write_text(index)
    (out / "README.md").write_text(SPACE_CARD)
    logger.info("wrote %s (%.0f KB) and %s", out / "index.html",
                (out / "index.html").stat().st_size / 1024, out / "README.md")


def main(argv: Sequence[str] | None = None) -> int:
    """Build the static Space bundle.

    Claim: infrastructure
    """
    p = argparse.ArgumentParser(description="Build the browser-only static Space")
    p.add_argument("--out", default="space")
    args = p.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build(Path(args.out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
