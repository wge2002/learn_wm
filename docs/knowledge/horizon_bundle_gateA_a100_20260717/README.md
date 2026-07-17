# A100 Gate-A end-to-end audit artifacts

This directory records the complete 2026-07-17 PushT Gate-A matrix:

```text
5 K_train × 5 H_plan × 3 goal offsets × 2 compute protocols
= 150 runs × 50 paired evaluation episodes
```

Pairing is within each goal-offset stratum: all K/H/protocol runs at one
offset use the same ordered start rows. Different offsets use different
physical rows and are resampled as separate strata.

Remote source:

```text
A100:/225010117/stablewm/checkpoints/gateA_*.txt
driver log: /225010117/logs/week1_gateA_driver.log
```

The driver ran from `2026-07-17 16:55:58` to `23:11:31` Asia/Shanghai and
reported `150 DONE / 0 FAILED`.

Files:

- `runs.csv`: all run-level settings, success rates, wall times, and ordered
  50-bit episode success vectors;
- `horizon_summary.csv`: success averaged over the three goal offsets;
- `cell_best_vs_runner.csv`: descriptive paired bootstrap comparisons in each
  `(protocol, H, offset)` cell;
- `horizon_best_vs_runner.csv`: descriptive comparisons after averaging the
  three offsets and paired-resampling each offset as a separate stratum;
- `summary.json`: compact winner credits, global means, horizon-matching audit,
  and the repeated-configuration determinism check.

Regenerate with:

```bash
python scripts/plan/summarize_gate_a_end_to_end.py \
  /path/to/copied/results \
  --out-dir docs/knowledge/horizon_bundle_gateA_a100_20260717 \
  --bootstrap 20000 \
  --seed 20260717 \
  --source-label 'A100:/225010117/stablewm/checkpoints/gateA_*.txt'
```

The best-vs-runner intervals are explicitly descriptive: winner and runner-up
are selected from the same outcomes and no multiple-comparison correction is
applied. They are gate diagnostics, not confirmatory p-values.
