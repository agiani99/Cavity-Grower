# De Novo Sweep Workflow

This workspace now has two scripts:

- `sweep_de_novo_cavity_growth.py`: runs the parameter sweep and writes one summary CSV for all runs.
- `analyze_sweep_results.py`: post-processes that summary into Pareto-front outputs, charts, and a ranked shortlist.

## What The Analysis Produces

The analysis step implements the four outputs requested for rational parameter selection:

1. Pareto-front table.
2. Bubble scatter for diversity vs docking quality.
3. Pairwise heatmaps over the main parameter pairs.
4. Final ranked shortlist of parameter sets.

The analysis treats parameter selection as a multi-objective tradeoff across:

- diversity, using `cluster_count / accepted_molecules`
- docking quality, using `vina_mean` by default
- runtime, using `duration_sec`
- molecular length/size, using the median heavy-atom count derived from each run SDF

## Why Heavy-Atom Count Is Used For Length

Raw SMILES length is not a chemically stable proxy for molecular size. The analysis instead reads each run SDF and computes the median heavy-atom count and median molecular weight of the accepted molecules.

If you do not provide a preferred target length, the analysis automatically uses the median heavy-atom count across eligible runs. That makes the length score favor balanced runs instead of extreme small or extreme large products.

## Composite Score

The Pareto front is generated first. The final shortlist is then ranked with a weighted composite score:

$$
\text{score} = 0.35 \cdot \text{diversity score}
+ 0.35 \cdot \text{Vina score}
+ 0.20 \cdot \text{length score}
+ 0.10 \cdot \text{runtime score}
$$

Interpretation:

- higher diversity score is better
- lower Vina is better
- lower runtime is better
- length score is highest when the run median heavy-atom count is close to the target

## Typical Workflow

Run the sweep:

```powershell
& "c:\Users\andrea.DESKTOP-26V6UN4\Documents\PDB_LIGAND\GLUERS\Claude\.venv\Scripts\python.exe" \
  ".\sweep_de_novo_cavity_growth.py" \
  --pdb-code 5IGO \
  --du-index 0 \
  --max-output 25 \
  --output-root ".\parameter_sweeps_max25_final"
```

Analyze the resulting summary:

```powershell
& "c:\Users\andrea.DESKTOP-26V6UN4\Documents\PDB_LIGAND\GLUERS\Claude\.venv\Scripts\python.exe" \
  ".\analyze_sweep_results.py" \
  ".\parameter_sweeps_max25_final\5IGO_DU0\5IGO_DU0_sweep_summary.csv"
```

If you want to favor a specific molecular size, set a target heavy-atom count explicitly:

```powershell
& "c:\Users\andrea.DESKTOP-26V6UN4\Documents\PDB_LIGAND\GLUERS\Claude\.venv\Scripts\python.exe" \
  ".\analyze_sweep_results.py" \
  ".\parameter_sweeps_max25_final\5IGO_DU0\5IGO_DU0_sweep_summary.csv" \
  --target-heavy-atoms 28 \
  --length-tolerance 5 \
  --shortlist-size 12
```

## Analysis Outputs

The analysis script writes a folder next to the input summary CSV named like:

```text
5IGO_DU0_sweep_summary_analysis
```

Inside it, you will find:

- `enriched_summary.csv`: all sweep rows plus derived metrics
- `pareto_front.csv`: non-dominated parameter sets
- `pareto_front.md`: readable Pareto table
- `bubble_tradeoff.png`: diversity vs Vina scatter
- `heatmap_vina_mean.png`: docking quality heatmap
- `heatmap_diversity_ratio.png`: diversity heatmap
- `heatmap_duration_sec.png`: runtime heatmap
- `heatmap_composite_score.png`: combined ranking heatmap
- `ranked_shortlist.csv`: top-ranked parameter sets
- `ranked_shortlist.md`: readable shortlist
- `analysis_summary.md`: compact report tying everything together

## How To Read The Charts

### 1. Pareto Front

This is the most important output. A parameter set is on the Pareto front if no other set is simultaneously better in diversity, docking, runtime, and target-length closeness.

Use this table first to discard dominated choices.

### 2. Bubble Scatter

- x-axis: diversity ratio
- y-axis: Vina mean score, with the axis inverted so better docking appears higher
- bubble size: accepted molecules
- bubble color: median heavy-atom count
- red outlines: Pareto-front runs

Good regions are toward high diversity and strong docking, without drifting into extreme size.

### 3. Pairwise Heatmaps

Each heatmap averages one response over two parameter pairs:

- `n_steps` vs `beam_width`
- `max_attach` vs `max_frags`

These figures help identify whether performance is driven mainly by search depth and beam size, or by attachment/fragment growth settings.

### 4. Ranked Shortlist

The shortlist is intended for decision-making after the Pareto filter. In practice:

1. Inspect the Pareto front.
2. Look at the bubble plot to see the global tradeoff.
3. Use the heatmaps to understand which axes are helping or hurting.
4. Use the shortlist when you need one final parameter set to test first.

## Notes

- The analysis expects real sweep results, not a pure `--dry-run` summary.
- Runs without accepted molecules, without a readable SDF, or without a Vina value are excluded from the ranking.
- The current sweep grid in this workspace is:
  - `n_steps = 10, 20, 30`
  - `n_embed_attempts = 15`
  - `beam_width = 30, 50`
  - `max_attach = 6, 8, 10`
  - `max_frags = 10, 15, 20, 25, 30`
  - total runs: `90`
