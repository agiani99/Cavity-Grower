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

Windows ADFR convenience
------------------------
Use --use-installed-adfr to auto-detect a local ADFR Suite install and append
the required grower flags for receptor preparation. Use --adfr-suite-dir to
point at a non-default ADFR install.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


_COMMON_ADFR_DIRS: Tuple[Path, ...] = (
    Path(r"C:\Program Files (x86)\ADFRsuite-1.1dev"),
    Path(r"C:\Program Files\ADFRsuite-1.1dev"),
)


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


def _resolve_adfr_suite_dir(adfr_suite_dir: Path | None) -> Path | None:
    if adfr_suite_dir is not None:
        return adfr_suite_dir
    for candidate in _COMMON_ADFR_DIRS:
        if candidate.exists():
            return candidate
    return None


def _append_if_missing(extra_args: List[str], option: str, value: str | None = None) -> None:
    if option in extra_args:
        return
    extra_args.append(option)
    if value is not None:
        extra_args.append(value)


def _configure_adfr_args(extra_args: List[str], adfr_suite_dir: Path) -> List[str]:
    bin_dir = adfr_suite_dir / "bin"
    prepare_receptor = bin_dir / "prepare_receptor.bat"
    reduce = bin_dir / "reduce.bat"

    if not prepare_receptor.exists():
        raise FileNotFoundError(f"ADFR prepare_receptor launcher not found: {prepare_receptor}")
    if not reduce.exists():
        raise FileNotFoundError(f"ADFR reduce launcher not found: {reduce}")

    configured_args = list(extra_args)
    _append_if_missing(configured_args, "--vina-enable")
    _append_if_missing(configured_args, "--vina-receptor-backend", str("adfr"))
    _append_if_missing(configured_args, "--vina-prepare-receptor-exe", str(prepare_receptor))
    _append_if_missing(configured_args, "--vina-reduce-exe", str(reduce))
    return configured_args


def _extract_misplaced_batch_args(extra_args: List[str]) -> tuple[List[str], bool, Path | None]:
    """Recover batch-only flags accidentally placed after the standalone `--`."""
    cleaned_args: List[str] = []
    use_installed_adfr = False
    adfr_suite_dir: Path | None = None

    idx = 0
    while idx < len(extra_args):
        arg = extra_args[idx]
        if arg == "--use-installed-adfr":
            use_installed_adfr = True
            idx += 1
            continue
        if arg == "--adfr-suite-dir":
            if idx + 1 >= len(extra_args):
                raise ValueError("--adfr-suite-dir requires a following path")
            adfr_suite_dir = Path(extra_args[idx + 1])
            idx += 2
            continue
        cleaned_args.append(arg)
        idx += 1

    return cleaned_args, use_installed_adfr, adfr_suite_dir


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
    ap.add_argument(
        "--use-installed-adfr",
        action="store_true",
        help="Auto-detect a local Windows ADFR Suite install and forward the matching grower flags",
    )
    ap.add_argument(
        "--adfr-suite-dir",
        type=Path,
        default=None,
        help="Path to an ADFR Suite installation root; implies ADFR grower flags",
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

    try:
        extra_args, misplaced_use_installed_adfr, misplaced_adfr_suite_dir = _extract_misplaced_batch_args(extra_args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    use_installed_adfr = bool(args.use_installed_adfr or misplaced_use_installed_adfr)
    adfr_suite_dir_arg = args.adfr_suite_dir if args.adfr_suite_dir is not None else misplaced_adfr_suite_dir

    if use_installed_adfr or adfr_suite_dir_arg is not None:
        adfr_suite_dir = _resolve_adfr_suite_dir(adfr_suite_dir_arg)
        if adfr_suite_dir is None:
            print(
                "ERROR: requested ADFR integration, but no ADFR Suite install was found. "
                "Pass --adfr-suite-dir explicitly or install ADFRsuite-1.1dev.",
                file=sys.stderr,
            )
            return 2
        try:
            extra_args = _configure_adfr_args(extra_args, adfr_suite_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

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
