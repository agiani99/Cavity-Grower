#!/usr/bin/env python3
"""batch_run_pdb_dus.py

Batch runner for de_novo_cavity_growth.py.

What it does
------------
- Scans a folder for PDB files.
- For each PDB, detects how many DU markers are present.
- If no DU markers are present, the PDB is skipped.
- For each DU index, runs de_novo_cavity_growth.py with auto-named outputs:
    .\\out_sdf\\{PDB}_DU{index}_default.sdf
    .\\out_sdf\\{PDB}_DU{index}_default.csv

Pass-through args
-----------------
Any args after a standalone "--" are forwarded to de_novo_cavity_growth.py.
Example:
  python batch_run_pdb_dus.py --pdb-dir pdbs -- --beam-width 80 --n-steps 40
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


def _count_du_markers(pdb_text: str) -> int:
    """Count DU markers using the same heuristic as de_novo_cavity_growth.py."""
    count = 0
    for ln in pdb_text.splitlines():
        if len(ln) < 54:
            continue
        if not (ln.startswith("HETATM") or ln.startswith("ATOM")):
            continue
        resname = ln[17:20].strip()
        atom_name = ln[12:16].strip()
        if resname == "DU" or atom_name == "DU":
            count += 1
    return count


def _iter_pdbs(pdb_dir: Path, pattern: str) -> Iterable[Path]:
    # Use glob for flexible patterns, but keep it deterministic.
    yield from sorted(pdb_dir.glob(pattern))


@dataclass
class RunResult:
    pdb_path: Path
    du_index: int
    returncode: int


def _run_one(
    de_novo_script: Path,
    pdb_path: Path,
    du_index: int,
    out_sdf: Path,
    out_csv: Path,
    extra_args: List[str],
    dry_run: bool,
) -> RunResult:
    cmd = [
        sys.executable,
        str(de_novo_script),
        "--pdb",
        str(pdb_path),
        "--du-index",
        str(du_index),
        "--out-sdf",
        str(out_sdf),
        "--out-csv",
        str(out_csv),
        *extra_args,
    ]

    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return RunResult(pdb_path=pdb_path, du_index=du_index, returncode=0)

    completed = subprocess.run(cmd)
    return RunResult(pdb_path=pdb_path, du_index=du_index, returncode=completed.returncode)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Batch-run de_novo_cavity_growth.py over all PDBs in a folder and "
            "all DU indices in each PDB, with automatic out_sdf/{PDB}_DU{idx}_default naming."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument(
        "--pdb-dir",
        type=Path,
        default=Path("."),
        help="Folder containing PDB files",
    )
    ap.add_argument(
        "--pattern",
        default="*.pdb",
        help="Glob pattern for PDB files within --pdb-dir",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out_sdf"),
        help="Output directory for per-DU SDF/CSV files",
    )
    ap.add_argument(
        "--suffix",
        default="default",
        help="Suffix in output filenames: {PDB}_DU{index}_{suffix}.(sdf|csv)",
    )
    ap.add_argument(
        "--de-novo-script",
        type=Path,
        default=Path(__file__).with_name("de_novo_cavity_growth.py"),
        help="Path to de_novo_cavity_growth.py",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands but do not execute them",
    )

    # Everything after -- is passed through.
    ap.add_argument(
        "de_novo_args",
        nargs=argparse.REMAINDER,
        help="Extra args to forward to de_novo_cavity_growth.py (prepend with --)",
    )

    args = ap.parse_args(argv)

    # Drop an initial "--" from argparse.REMAINDER if present.
    extra_args = list(args.de_novo_args)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]

    pdb_dir: Path = args.pdb_dir
    out_dir: Path = args.out_dir
    de_novo_script: Path = args.de_novo_script

    if not pdb_dir.exists():
        print(f"ERROR: --pdb-dir does not exist: {pdb_dir}", file=sys.stderr)
        return 2
    if not de_novo_script.exists():
        print(f"ERROR: --de-novo-script not found: {de_novo_script}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    pdbs = list(_iter_pdbs(pdb_dir, args.pattern))
    if not pdbs:
        print(f"No PDBs matched {args.pattern} in {pdb_dir}")
        return 0

    print(f"Found {len(pdbs)} PDB file(s) in {pdb_dir} matching {args.pattern}")
    if extra_args:
        print("Forwarding extra args to de_novo_cavity_growth.py:", " ".join(extra_args))

    results: List[RunResult] = []
    skipped = 0

    for pdb_path in pdbs:
        try:
            pdb_text = pdb_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"SKIP: could not read {pdb_path.name}: {exc}")
            skipped += 1
            continue

        du_count = _count_du_markers(pdb_text)
        if du_count == 0:
            print(f"SKIP: {pdb_path.name} (no DU markers found)")
            skipped += 1
            continue

        pdb_id = pdb_path.stem
        print(f"RUN: {pdb_path.name} ({du_count} DU marker(s))")

        for du_index in range(du_count):
            out_sdf = out_dir / f"{pdb_id}_DU{du_index}_{args.suffix}.sdf"
            out_csv = out_dir / f"{pdb_id}_DU{du_index}_{args.suffix}.csv"
            print(f"  DU[{du_index}] -> {out_sdf.name}")
            rr = _run_one(
                de_novo_script=de_novo_script,
                pdb_path=pdb_path,
                du_index=du_index,
                out_sdf=out_sdf,
                out_csv=out_csv,
                extra_args=extra_args,
                dry_run=args.dry_run,
            )
            results.append(rr)

    failed = [r for r in results if r.returncode != 0]

    print("\nSummary")
    print("-------")
    print(f"PDBs scanned     : {len(pdbs)}")
    print(f"Jobs attempted   : {len(results)}")
    print(f"Jobs failed      : {len(failed)}")
    print(f"PDBs skipped     : {skipped}")

    if failed:
        print("\nFailed jobs:")
        for r in failed:
            print(f"  {r.pdb_path.name} DU[{r.du_index}] -> exit {r.returncode}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
