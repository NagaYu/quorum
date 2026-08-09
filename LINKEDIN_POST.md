# LinkedIn 投稿文案

---

## 日本語版（推奨・約400字）

リーダーボードの数字は「順位」を読ませようとしますが、その順位をデータが支えているとは限りません。

そこで **Quorum** を作りました。LLM評価において「順位という判断」そのものを推定対象に据えた、入れ子の逐次実験計画のプロトタイプです。

考え方はシンプルです。絶対スコアではなく、ブロック（タスク×シード×scaffold）でペア化した差分を推定対象にする。すると (1) 推移律が効くので隣接ペアだけ確定させれば全順位が確定し、(2) scaffoldが両モデルを同じだけ動かしても符号が変わらない。

実測（12世界、α=0.05 family-wise）:
・同じ「確定した順位」に到達するコストが逐次検定のみの手法比で **34%減**
・帰無ペア300反復で誤判定率は名目以下（適応的割当の下でも）
・scaffoldシフト下で、公開済みアンカー集合を使う tinyBenchmarks風の手法は正解率0.67→0.33に劣化。Quorumは全域1.00

そして**上手くいかなかったことも全部書きました**。

・CUPEDはペア化した設計では分散を1%も下げません。対称な共変量は差と一次で相関しないという構造的な理由です。効くのはペア化できない領域（ログデータ、リーダーボード）で、そこでは26%減。
・内側の早期打ち切りはステップを19%削減しますが、無バイアス性を保つ監査補正が信頼列を広げ、総コストは増えます。既定でオフにしました。
・開発中、共変量の中心化を誤って誤判定率100%のバグを作りました。今は回帰テストです。

負の結果を消すと、読む人は「どこで効かないか」を自分で踏み直すことになります。

コード・図・全テスト: https://github.com/NagaYu/quorum

#LLM評価 #実験計画 #統計 #MLOps #オープンソース

---

## English version (~300 words)

A leaderboard column of numbers invites you to read a ranking off it. The data often doesn't support that ranking.

So I built **Quorum**: a research prototype that makes the *ranking decision itself* the estimand in LLM evaluation, using nested sequential experimental design.

The core move is simple. Instead of estimating absolute scores, estimate a **paired contrast** blocked on (task, seed, scaffold). Two things fall out: it decomposes as θ_a − θ_c, so it's transitive — separating adjacent pairs certifies the whole order; and it's invariant to anything that moves both models equally within a block, so a scaffold change doesn't flip it.

Measured over 12 independent worlds at family-wise α = 0.05:

• **34% cheaper** than a sequential test on a uniform design, to the same certified ranking
• Error rate at or below nominal across 300 null-pair replications — including under adaptive allocation, which is the case the Horvitz–Thompson construction exists for
• Under a scaffold shift, a tinyBenchmarks-style method using a published anchor set degrades from 0.67 to 0.33 accuracy. Quorum stays at 1.00 throughout

**I also shipped everything that didn't work.**

→ CUPED reduces variance by ~0% under a paired design. That's structural, not a tuning failure: a covariate symmetric in the two models can't correlate with their difference to first order. It earns its keep where pairing is impossible — logged data, public leaderboards — and there it removes 26%.

→ Inner truncation saves 19% of trajectory steps, but the audit correction that keeps it unbiased widens the confidence sequence enough that total cost goes *up*. Off by default.

→ A covariate centred on the wrong constant gave me a 100% false-positive rate during development. It's a regression test now.

Deleting the negative results just means the next person rediscovers them.

Code, figures, full test suite: https://github.com/NagaYu/quorum

#LLMEvaluation #ExperimentalDesign #Statistics #MLOps #OpenSource

---

## 短縮版（X / Bluesky 用、日本語）

リーダーボードの数字は順位を読ませようとするが、データがその順位を支えているとは限らない。

順位そのものを推定対象にした逐次実験計画「Quorum」を公開しました。同じ結論に34%安く到達、被覆率は名目以下、scaffoldシフトにも頑健。

効かなかった手法（CUPED、早期打ち切り）も測定値ごと全部載せています。

https://github.com/NagaYu/quorum
