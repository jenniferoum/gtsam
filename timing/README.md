# Timing Benchmarks

This directory contains timing executables and helper scripts for GTSAM.

## Bayes-Tree Covariance Results

The Bayes-tree covariance paper uses generated benchmark output rather than
checked-in CSV files. The files under `timing/results/` can be regenerated from
the commands below and do not need to be committed.

### Build

From the build directory:

```bash
make -j6 timeBayesTreeCovariance
make -j6 exportBayesTreeCovarianceVisuals
```

### Generate benchmark CSVs

Run from `build/`:

```bash
./timing/timeBayesTreeCovariance \
  --datasets w100.graph,w10000.graph,w20000.txt \
  --repeats 10 \
  --output-dir ../timing/results/bayes_tree_covariance
```

This writes:

- `timing/results/bayes_tree_covariance/raw.csv`
- `timing/results/bayes_tree_covariance/per_query.csv`
- `timing/results/bayes_tree_covariance/summary.csv`

The generated CSVs now include the small-query families used to benchmark the
legacy common cases:

- `single_pose` for `Q = 1`
- `pair_pose` for `Q = 2`

These are timed with the same benchmark executable and appear alongside the
larger `local_window`, `wide_separated`, `overlap_window`, and
`selected_cross` workloads.

### Export `w100` visual data

Run from `build/`:

```bash
./timing/exportBayesTreeCovarianceVisuals \
  --dataset w100.graph \
  --output-dir ../timing/results/bayes_tree_covariance/visuals
```

This writes the CSV files used for the `w100` query and covariance figures.

### Generate figures

Run from the repository root:

```bash
python3 timing/plot_bayes_tree_covariance.py \
  --input timing/results/bayes_tree_covariance/summary.csv \
  --output-dir ../BayesTreeCovariance/figures/generated \
  --copy-csv-dir ../BayesTreeCovariance/data \
  --visual-data-dir timing/results/bayes_tree_covariance/visuals
```

This generates:

- `results-smallq.pdf`
- `results-ablation.pdf`
- `results-ordering.pdf`
- `results-structure.pdf`
- `results-cross.pdf`
- `results-w100-queries.pdf`
- `results-w100-covariance.pdf`

## Notes

- The benchmark timings measure covariance-query work after optimization and
  linearization.
- Each distinct query is run once as an untimed warmup and then `--repeats`
  times as measured repetitions.
- `raw.csv` stores one row per measured repetition.
- `per_query.csv` stores one row per distinct query, aggregating repeated
  timings by per-query means.
- `summary.csv` stores one row per query family bucket, aggregating the
  per-query means and structural statistics by medians across the sampled
  queries.
- The `results-smallq.pdf` figure summarizes the `Q = 1` and `Q = 2` query
  families; the larger performance figures focus on `Q > 2`, where Steiner
  localization changes the asymptotic behavior.
- The benchmark compares four variants:
  - `legacy_dense`
  - `steiner_dense`
  - `legacy_solve`
  - `steiner_solve`
- If the generated results become stale, it is safe to delete
  `timing/results/bayes_tree_covariance/` and regenerate it.
