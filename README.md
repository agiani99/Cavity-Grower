# DU-anchored de novo cavity growth (RDKit)

This folder contains a small pipeline for **de novo fragment-growing** of molecules in protein cavities, where the cavity center is provided as one or more **`DU` marker atoms** inside a PDB file (e.g. produced by a DoGSite/DU-marker pre-step).

At a high level:

1. **Generate** molecules anchored to a DU coordinate using beam-search fragment growth.
2. **Batch-run** across many PDBs and many DU markers per PDB.
3. **Postprocess** SDF outputs: deduplicate, compute descriptors, filter (Lipinski/Veber/PAINS/Brenk), cluster for diversity, optionally annotate known PubChem molecules.
4. **Annotate** postprocessed summary CSVs with **PDB→UniProt** mapping and (best-effort) **ChEMBL binding activities** (type B, confidence 9) via pChEMBL.

`Script_description.txt` contains free-form notes and a narrative description of the approach; the README below formalizes it into a runnable workflow.

---

## Requirements

- Python 3
- RDKit
- NumPy
- Optional:
  - SciPy (speeds up clash checking in the grower via KDTree)
  - Pillow (only needed if you want grid images from postprocessing)

If you use conda, RDKit is usually easiest via:

```bash
conda install -c conda-forge rdkit numpy
```

## Windows ADFR receptor preparation

The grower supports two receptor-preparation backends for Vina:

- `meeko` (default)
- `adfr`

On this workspace, native Windows ADFR works with:

- `C:\Program Files (x86)\ADFRsuite-1.1dev\bin\prepare_receptor.bat`
- `C:\Program Files (x86)\ADFRsuite-1.1dev\bin\reduce.bat`

Direct grower example:

```bash
python de_novo_cavity_growth.py \
  --pdb path/to/1ABC_with_DU.pdb \
  --vina-enable \
  --vina-receptor-backend adfr \
  --vina-prepare-receptor-exe "C:\Program Files (x86)\ADFRsuite-1.1dev\bin\prepare_receptor.bat" \
  --vina-reduce-exe "C:\Program Files (x86)\ADFRsuite-1.1dev\bin\reduce.bat"
```

Note: `reduce.bat` can return a non-zero exit code for chain-break warnings while still producing a valid hydrogenated receptor. The grower now accepts that case and continues with ADFR preparation.

## Windows standalone package build

You can package the grower as a Windows `.exe` with PyInstaller. A Vina-capable distribution is supported by building an onedir app and copying the local docking tools into a `tools` folder beside the executable.

### Release-friendly repository layout

Keep the Git repository source-first and publish packaged Windows binaries as GitHub Release assets rather than committing the full `dist/` tree.

Recommended tracked layout:

```text
.
|-- README.md
|-- .gitignore
|-- de_novo_cavity_growth.py
|-- analyze_cavity_residues.py
|-- batch_run_pdb_dus.py
|-- batch_run_pdb_dus_v1.py
|-- postprocess_designs.py
|-- postprocess_pareto_vina.py
|-- annotate_summaries_chembl.py
|-- build_cavity_grower.ps1
|-- build_pdb_du_preparer.ps1
|-- build_release_packages.ps1
|-- cavity_grower.spec
|-- pdb_du_preparer.spec
|-- example_fixed_fragments.smi
|-- example_ring_fragments.smi
|-- ring_blocks_unique.csv
|-- ring_secramine.csv
|-- run_all_pdb_dus.bat
`-- Script_description.txt
```

Keep these paths out of Git and generate them locally when needed:

- `dist/` for PyInstaller output
- `build/` for PyInstaller working files
- `release/` for local zip staging before uploading to GitHub Releases
- `sdf_out*/`, `out_sdf/`, and `postprocessed*/` for generated design outputs

Recommended release flow:

1. Build the two onedir apps locally into `dist/cavity-grower/` and `dist/pdb-du-preparer/`.
2. Zip each application folder, for example `release/cavity-grower-windows-x64.zip` and `release/pdb-du-preparer-windows-x64.zip`.
3. Attach those zip files to a GitHub Release.
4. Tell users to download the release asset instead of cloning `dist/` from the repository.

Included repo files:

- `cavity_grower.spec` - PyInstaller spec for the CLI app
- `build_cavity_grower.ps1` - Windows build script that:
  - installs PyInstaller into the selected Python environment
  - builds `dist/cavity-grower/cavity-grower.exe`
  - copies `vina.exe` into `dist/cavity-grower/tools/vina/`
  - copies the local ADFR Suite install into `dist/cavity-grower/tools/adfr/`
- `build_release_packages.ps1` - Windows release-packaging script that zips `dist/cavity-grower/` and `dist/pdb-du-preparer/` into `release/`

Typical build from the repo root:

```powershell
.\build_cavity_grower.ps1
```

Custom paths if needed:

```powershell
.\build_cavity_grower.ps1 \
  -PythonExe .\.venv\Scripts\python.exe \
  -VinaExe "C:\Program Files (x86)\Vina\vina.exe" \
  -AdfrSuiteDir "C:\Program Files (x86)\ADFRsuite-1.1dev"
```

After building both packaged apps, create the release zip files with:

```powershell
.\build_release_packages.ps1
```

Optional versioned archive names:

```powershell
.\build_release_packages.ps1 -Version v1.0.0
```

You can also package just one app while testing the release flow:

```powershell
.\build_release_packages.ps1 -Version v1.0.0 -PackageNames pdb-du-preparer
```

This writes files such as:

- `release/cavity-grower-v1.0.0-windows-x64.zip`
- `release/pdb-du-preparer-v1.0.0-windows-x64.zip`

The packaged executable can then be run like:

```powershell
dist\cavity-grower\cavity-grower.exe --help
```

The packaged executable also supports custom fragment libraries:

```powershell
dist\cavity-grower\cavity-grower.exe \
  --pdb path\to\1ABC_with_DU.pdb \
  --du-index 0 \
  --fixed-smiles-file .\example_fixed_fragments.smi \
  --ring-smiles-file .\ring_blocks_unique.csv \
  --out-sdf dist\custom_fragments.sdf \
  --out-csv dist\custom_fragments.csv
```

Headered CSV fragment libraries are also supported in the packaged app, for example:

```csv
smiles,reagent_id
S=C=Nc1ccccc1,3
O=Cc1ccccn1,15
```

Notes:

- The build uses `--onedir`, not `--onefile`, because RDKit and the docking toolchain are much more reliable as a folder-based distribution.
- Vina mode still depends on the copied sidecar tools in `dist/cavity-grower/tools/`.
- If you omit the docking tools, the packaged app still works for non-Vina runs.

---

## End-to-end workflow

### 0) Inputs: PDBs with DU markers

The grower expects PDB files containing one or more DU “cavity center” markers, typically as `HETATM` records where either:

- residue name is `DU`, **or**
- atom name is `DU`

Each DU marker is treated as one cavity anchor; pick which one to target with `--du-index`.

Optional: characterize pocket chemistry around DU markers

If you want to quantify whether your cavities are mainly polar vs hydrophobic (and the
pos/neg or aromatic fractions), you can run:

```bash
python analyze_cavity_residues.py --pdb path/to/1ABC_with_DU.pdb --radius 8 --out-csv cavity_env.csv
```

This produces one row per DU marker with residue counts and ratios within the radius.

Optional: split the same analysis by chain (and annotate partner proteins)

If the DU cavity is formed at an interface (two chains), you can compute the same pocket
summary **separately per chain** (within the same DU-centered sphere) and write both
chain summaries into a single output row:

```bash
python analyze_cavity_residues.py --pdb path/to/1ABC_with_DU.pdb --radius 8 --split-by-chain --out-csv cavity_env_split_by_chain.csv
```

This adds columns like `chain1_id`, `chain1_*` and `chain2_id`, `chain2_*` (top 2 chains,
ranked by number of nearby residues).

If you also want gene symbols for the two partner chains (best-effort PDB→UniProt→gene
lookup; uses HGNC gene symbols when available for human proteins):

```bash
python analyze_cavity_residues.py --pdb path/to/1ABC_with_DU.pdb --radius 8 --split-by-chain --annotate-genes --out-csv cavity_env_split_by_chain_genes.csv
```

Gene lookup uses a JSON cache (default: `.cache/pdb_chain_gene_cache.json`) so reruns are
fast and robust.

### 1) Generate molecules for one cavity

Run the grower on a single PDB + selected DU index:

```bash
python de_novo_cavity_growth.py \
  --pdb path/to/1ABC_with_DU.pdb \
  --du-index 0 \
  --target-min 16 --target-max 20 \
  --beam-width 80 --n-steps 30 \
  --max-attach 3 --max-frags 8 \
  --qed-min 0.30 --sa-max 4.5 \
  --cavity-radius 14.0 --clash-radius 1.5 \
  --out-sdf sdf_out/1ABC_DU0_default.sdf \
  --out-csv sdf_out/1ABC_DU0_default.csv
```

Custom fragment files are optional. If you do not pass them, the grower falls back to the built-in `FIXED_SMILES_FRAGS` and `RING_FRAGS` defined in the script.

Example with user-supplied fragment files:

```bash
python de_novo_cavity_growth.py \
  --pdb path/to/1ABC_with_DU.pdb \
  --du-index 0 \
  --fixed-smiles-file example_fixed_fragments.smi \
  --ring-smiles-file example_ring_fragments.smi \
  --out-sdf sdf_out/1ABC_DU0_custom_frags.sdf \
  --out-csv sdf_out/1ABC_DU0_custom_frags.csv
```

Supported fragment file formats:

- Fixed fragments: `SMILES`, `SMILES name [attach_map_num]` (`.smi` style), `name,SMILES`, `name,SMILES,attach_map_num`, or headered CSV with a `smiles` column plus optional `name`/`id` and `attach_map_num` columns
- Ring fragments: `SMILES`, `SMILES name` (`.smi` style), `name,SMILES`, or headered CSV with a `smiles` column plus optional `name`/`id` columns

Example files included in this repo:

- `example_fixed_fragments.smi`
- `example_ring_fragments.smi`

Generated CSVs like `ring_blocks_unique.csv` are valid input as long as they include a `smiles` column. If no explicit `name` column is present, the loader will fall back to an `id` column such as `reagent_id`.

If you enable Vina rescoring during growth, beam selection now uses a stricter rule:

- if any deduplicated candidates at a step have a negative Vina score, only those candidates are allowed to move forward
- if no candidate has a negative Vina score, the grower falls back to the 10 best compounds ranked by ascending Vina score

This replaces the older `--vina-beam-top-n` beam-stage shortcut. The flag is still accepted for compatibility, but it is ignored.

### 2) Batch-run across many PDBs and DU markers

Windows wrapper:

```bat
run_all_pdb_dus.bat ..\dogsite_results_450\pdb_with_du --beam-width 80 --qed-min 0.30 --sa-max 4.5
```

Pure Python batch runner:

```bash
python batch_run_pdb_dus.py --pdb-dir ..\dogsite_results_450\pdb_with_du --out-dir sdf_out --suffix default -- \
  --beam-width 80 --qed-min 0.30 --sa-max 4.5
```

Batch-run with auto-detected installed ADFR Suite on Windows:

```bash
python batch_run_pdb_dus.py --pdb-dir ..\dogsite_results_450\pdb_with_du --out-dir sdf_out --suffix default --use-installed-adfr -- \
  --beam-width 80 --qed-min 0.30 --sa-max 4.5
```

Example with Vina-enabled beam filtering using the current behavior:

```bash
python batch_run_pdb_dus.py --pdb-dir ..\dogsite_results_700\pdb_with_du --out-dir .\sdf_out_700_clash_vina20_ph405 --use-installed-adfr -- \
  --vina-enable --beam-width 80 --qed-min 0.30 --sa-max 4.5
```

With `--vina-enable`, there is no need to pass `--vina-beam-top-n`: the grower now scores all deduplicated beam candidates, forwards only negative-Vina compounds when available, and otherwise keeps the 10 best Vina-scored compounds.

Windows wrapper with auto-detected ADFR Suite:

```bat
run_all_pdb_dus.bat ..\dogsite_results_450\pdb_with_du --use-installed-adfr --beam-width 80 --qed-min 0.30 --sa-max 4.5
```

### 3) Postprocess SDF outputs (batch mode)

Postprocess each SDF in a folder and produce per-input outputs prefixed by the SDF stem:

```bash
python postprocess_designs.py \
  --sdf-dir sdf_out \
  --out-dir postprocessed \
  --lipinski --veber --pains \
  --qed-min 0.30 --sa-max 4.5 \
  --cluster-radius 0.60 \
  --grid-n 48 \
  --pubchem --pubchem-vendors
```

This will create (for each input SDF) files like:

- `{stem}_all_filtered.sdf`
- `{stem}_diverse_picks.sdf`
- `{stem}_summary.csv`
- `{stem}_grid_top{N}.png` (best-effort)

### 4) Annotate summaries with UniProt + ChEMBL pChEMBL values

Annotate all postprocessed summary CSVs:

```bash
python annotate_summaries_chembl.py --csv-dir postprocessed --out-dir postprocessed_chembl --only-pubchem-rows
```

This appends columns such as `uniprot_ids`, `chembl_target_ids`, and pChEMBL summary fields.

Note: the public ChEMBL API can intermittently return HTTP 5xx for some filtered queries. The annotator is “best effort” and uses a JSON cache; rerunning later can fill in values when the endpoint is stable.

---

## What the grower is doing (summary)

The core generator in `de_novo_cavity_growth.py` is a **beam search** over molecules.

- **Seed**: a single carbon atom placed exactly at the chosen DU coordinate.
- **Growth actions** (sampled per step):
  - single atoms (C/N/O/S)
  - multi-atom functional groups (e.g. `C=O`, `C=N`, `C#N`, `S(=O)=O`)
  - internal alkene growth with explicit E/Z stereo (`C=C(E)`, `C=C(Z)`)
  - ring fragments (aromatic, heteroaromatic, saturated, and selected bicyclics)
- **Embedding**: ETKDGv3 + optional MMFF, then translate so atom 0 stays on the anchor. Retries also apply random rotations to explore orientations.
- **Geometry filters**: discard if too many atoms clash into protein atoms or escape the cavity sphere.
- **Scoring**: combines QED, SA, clash fraction, and outside-cavity fraction into a composite.
- **Optional Vina-guided beam filtering**: when `--vina-enable` is set, each beam step scores deduplicated candidates with Vina, forwards only negative-Vina candidates if any exist, and otherwise falls back to the 10 best compounds by ascending Vina score.

---

## Script reference

### `de_novo_cavity_growth.py`

Beam-search fragment grower anchored to a selected DU marker.

Inputs
- `--pdb`: PDB file containing DU marker atoms.
- `--du-index`: which DU cavity marker to use.

Outputs
- `--out-sdf`: SDF of accepted molecules (with properties written as SDF props).
- `--out-csv`: CSV summary (rank/id/smiles/QED/SA/clash/out/heavy_atoms/...)

Common knobs
- Search size: `--beam-width`, `--n-steps`, `--max-attach`, `--max-frags`
- Acceptance window: `--target-min`, `--target-max`
- Quality gates: `--qed-min`, `--sa-max`
- Geometry: `--clash-radius`, `--cavity-radius`
- Rings: `--no-rings`, `--no-ring-attach-rotate`, `--ring-attach-max`
- Vina receptor prep: `--vina-receptor-backend`, `--vina-prepare-receptor-exe`, `--vina-reduce-exe`, `--vina-strict`
- Vina beam behavior: `--vina-enable` turns on beam-stage filtering that prefers negative Vina scores; `--vina-beam-top-n` is deprecated and ignored

### `batch_run_pdb_dus.py`

Batch runner for `de_novo_cavity_growth.py`.

- Scans `--pdb-dir` for PDBs (pattern `--pattern`, default `*.pdb`).
- Counts DU markers and runs all DU indices per PDB.
- Forwards any args after `--` directly to the grower.
- Can auto-detect a Windows ADFR install with `--use-installed-adfr`.

Outputs
- `{out_dir}/{PDB}_DU{idx}_{suffix}.sdf`
- `{out_dir}/{PDB}_DU{idx}_{suffix}.csv`

### `run_all_pdb_dus.bat`

Windows convenience wrapper around `batch_run_pdb_dus.py`.

- If no folder is provided, uses the default:
  - `..\dogsite_results_450\pdb_with_du`
- Supports `--dry-run` (prints commands but does not execute).
- Supports `--use-installed-adfr` to auto-detect ADFR Suite and enable the grower ADFR backend.
- Forwards the rest of arguments to the grower.

### `postprocess_designs.py`

Postprocessing for SDF outputs.

Core steps
- Load SDF(s)
- Deduplicate by InChIKey
- Compute descriptors (MW/LogP/TPSA/QED/SA/etc.)
- Optional hard filters: Lipinski/Veber/Lead/PAINS/Brenk
- Cluster (Butina on Morgan fingerprints) and pick diverse representatives
- Optional PubChem annotation:
  - `pubchem_cid`, `pubchem_vendors`, `is_pubchem_hit`
  - can “rescue” PubChem hits through filters (default behavior)

Modes
- Single combined job: `--sdf file1.sdf file2.sdf ...`
- Batch-per-file job: `--sdf-dir sdf_out/` (adds filename prefixes)

Outputs (per job)
- `*_all_filtered.sdf`
- `*_diverse_picks.sdf`
- `*_summary.csv`
- `*_grid_top{N}.png` (best-effort)
- `pubchem_cache.json` (if PubChem enabled)

### `annotate_summaries_chembl.py`

Annotate postprocessed `*_summary.csv` files with protein and bioactivity metadata.

- Extracts `pdb_id` from the first 4 chars of the filename by default.
- PDBe mapping: `pdb_id → UniProt accessions`
- ChEMBL lookup (best effort):
  - map `InChIKey → ChEMBL molecule`
  - query type `B` activities, confidence 9, and pull `pchembl_value`

Outputs
- Writes annotated CSVs to `--out-dir` (or overwrites with `--in-place`).
- Appends columns: `pdb_id`, `uniprot_ids`, `chembl_target_ids`, `chembl_molecule_id`, `chembl_max_pchembl`, `chembl_n_pchembl`, `chembl_best_target_id`.

Useful flags
- `--only-pubchem-rows` (faster; default behavior is also PubChem-focused)
- `--annotate-all-rows` (slower; tries any row with an InChIKey)
- `--http-timeout`, `--http-retries` (helps with flaky public endpoints)

### `analyze_cavity_residues.py`

Summarize residue chemistry around DU cavity markers in PDB files.

Core behavior

- Finds all DU marker atoms (resname `DU` or atom name `DU`).
- For each DU, counts nearby residues within `--radius` (default 8 Å).
- A residue is counted if *any heavy atom* is within the radius.
- Waters and hydrogens are excluded.

Inputs

- `--pdb file1.pdb file2.pdb ...` OR `--pdb-dir <folder> --pattern "*.pdb"`

Outputs

- `--out-csv`: one row per DU marker.
- Aggregate columns (all chains combined): `n_residues`, `n_pos`, `n_neg`, `pos_over_posneg`, `n_polar`, `n_apolar`, `polar_over_polarapolar`, `n_aromatic`, `aromatic_fraction`, `residue_hist`, etc.

Optional per-chain split

- `--split-by-chain`: appends `chain1_*` and `chain2_*` columns for the top 2 chains by nearby-residue count.

Optional partner annotation

- `--annotate-genes`: adds `chain{i}_uniprot` and `chain{i}_gene_symbol` columns using a best-effort online lookup.
- `--gene-cache-json`: path for the lookup cache (default: `.cache/pdb_chain_gene_cache.json`).

---

## Files and folders you will see

- `sdf_out/`, `sdf_out_expanded/`: per-PDB/DU grower outputs (`*.sdf`, `*.csv`)
- `postprocessed/`: postprocessing outputs (filtered SDFs, diverse picks, summary CSVs, grids)
- `pubchem_cache.json`: created under the `postprocessed/` output folder when PubChem is enabled
- `chembl_annotation_cache*.json`: JSON caches for ChEMBL/PDBe queries (safe to keep; speeds up reruns)
- `First_two_runs_config.txt`, `Expanded_second_runs_config.txt`: notes of typical parameter sets used in past runs

---

## Troubleshooting / tips

- **Empty or broken SDF files**: `postprocess_designs.py` skips missing/0-byte/unreadable SDFs in batch mode.
- **RDKit warnings during embedding (UFFTYPER, etc.)**: the grower blocks RDKit log spam unless `--verbose`.
- **Ring attachment sanitization/kekulization issues**: keep ring-attachment rotation enabled (default) and tune `--ring-attach-max` if needed.
- **ChEMBL 5xx responses**: retry later and/or reduce aggressiveness:
  - lower `--http-timeout`, set small `--http-retries`, rerun to fill cache.

---

## One-command “typical run” (generator → postprocess)

```bash
python batch_run_pdb_dus.py --pdb-dir ..\dogsite_results_450\pdb_with_du --out-dir sdf_out --suffix default -- \
  --beam-width 80 --n-steps 30 --max-attach 3 --max-frags 8 --target-min 16 --target-max 20 --qed-min 0.30 --sa-max 4.5

python postprocess_designs.py --sdf-dir sdf_out --out-dir postprocessed --lipinski --veber --pains --qed-min 0.30 --sa-max 4.5 --cluster-radius 0.60 --pubchem --pubchem-vendors
```
