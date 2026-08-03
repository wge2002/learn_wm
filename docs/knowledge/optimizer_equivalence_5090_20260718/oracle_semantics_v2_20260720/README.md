# OE oracle-semantics audit v2 — 2026-07-20

## Status and version boundary

This is the valid Stage-0 semantics audit for Search/Operator-Aligned LeWM.
It asks whether the frozen K3 LeWM distance between a **real executed terminal
observation** and the goal observation can replace privileged PushT physical
cost as the oracle that supervises a CEM update.

The earlier `v1` outputs left on the 5090 are invalid preflight artifacts.
Directly changing Pymunk body state without reindexing its shapes can leave the
rendered framebuffer stale. This made some candidate populations appear to
have constant visual cost. Version 2 calls
`space.reindex_shapes_for_body(...)` after exact terminal-state restoration.
Every v2 population has nontrivial visual-cost support. No v1 number should be
used in analysis or writing.

## Locked protocol

```text
checkpoint / generator / scorer = pd_d192_k3_eval
cells                           = H5/off40, H8/off60
states per cell                 = 12 paired dataset starts
saved CEM steps                 = 4, 9, 19, 29
candidates per population       = 300 complete candidates
elite count                     = 30
visual oracle                   =
  || frozen_K3(render(true terminal)) - frozen_K3(render(goal)) ||²
physical oracle                 = PushT pose cost
bootstrap                       = 20,000 paired state resamples
```

For each identical candidate population, the audit compares the hard top-30
CEM mean update induced by:

1. the stored learned rollout cost;
2. frozen-LeWM visual cost evaluated on the true terminal observation;
3. privileged simulator physical cost.

The predeclared visual-oracle gate required both cells to reach:

```text
visual-to-physical update cosine >= 0.70
visual/physical elite overlap    >= 0.50
physical oracle-gap recovery     >= 0.50
```

Gap recovery is computed from the physical cost of each selected elite set:

```text
(learned-selected cost - visual-selected cost)
------------------------------------------------
(learned-selected cost - physical-selected cost)
```

## Integrity checks

Both cells pass all reconstruction checks:

| check | H5/off40 | H8/off60 |
| --- | ---: | ---: |
| initial-state max abs mismatch | 0 | 0 |
| goal-state max abs mismatch | 0 | 0 |
| recomputed physical-cost max abs mismatch | 0 | 0 |
| success-label disagreements | 0 | 0 |
| render-visible terminal-state max abs mismatch | 2.84e-14 | 5.68e-14 |

## Main result

| cell | learned cosine | visual cosine | delta | learned overlap | visual overlap | visual/physical Spearman | true-cost gain | gap recovery | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H5/off40 | .121 | **.426** | **+.305** [.212, .406] | .147 | **.374** | .478 | **+4.29** [1.84, 6.70] | .235 [.092, .444] | MISS |
| H8/off60 | .266 | **.541** | **+.276** [.124, .431] | .260 | **.499** | .613 | **+8.68** [.16, 16.83] | .385 [.009, .645] | MISS |

The frozen visual true-terminal oracle is therefore **causally informative but
not sufficient**:

- It improves the CEM update direction over learned rollout cost in both
  cells, with paired-state bootstrap intervals strictly above zero.
- It selects physically better elite sets in both cells.
- It nevertheless misses the strict substitution gate in both cells. It
  recovers only about `24%` of the H5 physical-oracle gap and `39%` of the H8
  gap on average.

The signal is strongest in several middle/late CEM rounds:

| cell / step | visual cosine | visual overlap | gap recovery |
| --- | ---: | ---: | ---: |
| H5 / 19 | .542 | .467 | .466 |
| H5 / 29 | .471 | .475 | .446 |
| H8 / 9 | .517 | .497 | .456 |
| H8 / 19 | **.650** | **.581** | .468 |

This prevents an overstrong negative conclusion. Raw frozen LeWM geometry
does contain decision-relevant terminal information, especially where CEM has
already concentrated. It is simply not accurate enough to be the sole general
teacher.

## Decision

1. **Do not use raw frozen-K3 terminal distance as the flagship oracle.**
   The strict reward-free substitution hypothesis misses its gate.
2. **Keep the visual oracle as an auxiliary/control.** Its paired improvements
   rule out the claim that true-terminal observation geometry is useless.
3. **Use physical cost for the first end-to-end mechanism proof.** This is
   explicitly task-aware/simulator-supervised and tests whether full LeWM
   training can realize the already established operator-update ceiling.
4. **The general method should combine operator alignment with a learned
   reachability/task metric** (for example an RC-aux/TRM-style metric teacher).
   The metric method supplies a better terminal geometry; operator alignment
   makes predicted rollouts preserve the adaptive search update under that
   geometry. These are complementary roles rather than a literature
   collision.

The next controlled training screen remains:

```text
Query-Data:
  LeWM prediction + SIGReg on matched planner-query data

Query-Rank:
  Query-Data + physical candidate ranking

Query-Operator:
  Query-Rank + physical-oracle CEM update matching
```

All arms must start from the same K3 checkpoint, see the same query bundles
and optimizer steps, and train the actual LeWM modules. The visual-oracle arm
is retained as a secondary control, not the primary teacher.

## Reproduction and artifacts

Implementation:

```text
scripts/plan/oe_oracle_semantics_audit.py
scripts/plan/run_5090_oe_oracle_semantics_20260720.sh
```

Remote root:

```text
/mnt/data/wge/learn_wm/outputs/week1/
  oe_oracle_semantics_5090_20260720/
```

Compact v2 artifacts copied here:

| artifact | SHA-256 |
| --- | --- |
| `h5_off40_visual_semantics_v2.npz` | `2068095629dad6edb2a3942dcc52b0f0ebab254861487e2d105d71892e73af85` |
| `h8_off60_visual_semantics_v2.npz` | `386fbb5b88e86af5e8f58ff00656bb146d561216529fff24588828281fd415b7` |

The corresponding JSON files contain the state-blocked bootstrap summaries;
the generated Markdown reports retain all per-step aggregates.
