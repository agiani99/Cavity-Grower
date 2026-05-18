#!/usr/bin/env python3
"""Parameter sweep runner for de_novo_cavity_growth.py.

Runs one PDB/DU cavity through a grid of growth parameters while keeping all
other grower settings at their existing defaults unless extra grower arguments
are forwarded after a standalone "--".

Example
-------
python sweep_de_novo_cavity_growth.py --pdb-code 1BUH --du-index 0

If Vina scoring should be enabled for every run, either use the direct flags
below or forward the existing grower flags after "--", for example:

python sweep_de_novo_cavity_growth.py --pdb-code 1BUH -- --vina-enable

Windows ADFR convenience:

python sweep_de_novo_cavity_growth.py --pdb-code 1BUH --vina-enable --use-installed-adfr
"""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina


DEFAULT_CAVITY_DIR = Path(
    r"C:\Users\andrea.DESKTOP-26V6UN4\Documents\PDB_LIGAND\GLUERS\dogsite_results_450\pdb_with_du"
)
DEFAULT_PYTHON_EXE = Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "parameter_sweeps"
DEFAULT_GROWER_SCRIPT = Path(__file__).with_name("de_novo_cavity_growth.py")
COMMON_ADFR_DIRS = (
    Path(r"C:\Program Files (x86)\ADFRsuite-1.1dev"),
    Path(r"C:\Program Files\ADFRsuite-1.1dev"),
)
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

N_STEPS_VALUES = tuple(range(10, 31, 10))
N_EMBED_ATTEMPTS_VALUES = (15,)
BEAM_WIDTH_VALUES = (30, 50)
MAX_ATTACH_VALUES = tuple(range(6, 11, 2))
MAX_FRAGS_VALUES = tuple(range(10, 31, 5))

SUMMARY_FIELDNAMES = [
    "run_index",
    "pdb_code",
    "pdb_path",
    "du_index",
    "n_steps",
    "n_embed_attempts",
    "beam_width",
    "max_attach",
    "max_frags",
    "status",
    "returncode",
    "duration_sec",
    "accepted_molecules",
    "unique_smiles",
    "cluster_count",
    "qed_min",
    "qed_median",
    "qed_mean",
    "qed_max",
    "sa_min",
    "sa_median",
    "sa_mean",
    "sa_max",
    "vina_min",
    "vina_median",
    "vina_mean",
    "vina_max",
    "output_dir",
    "csv_path",
    "sdf_path",
    "log_path",
    "error",
]


@dataclass(frozen=True)
class RunSpec:
    n_steps: int
    n_embed_attempts: int
    beam_width: int
    max_attach: int
    max_frags: int


def _append_if_missing(extra_args: list[str], option: str, value: str | None = None) -> None:
    if option in extra_args:
        return
    extra_args.append(option)
    if value is not None:
        extra_args.append(value)


def _resolve_adfr_suite_dir(adfr_suite_dir: Path | None) -> Path | None:
    if adfr_suite_dir is not None:
        return adfr_suite_dir
    for candidate in COMMON_ADFR_DIRS:
        if candidate.exists():
            return candidate
    return None


def _configure_adfr_args(extra_args: list[str], adfr_suite_dir: Path) -> list[str]:
    bin_dir = adfr_suite_dir / "bin"
    prepare_receptor = bin_dir / "prepare_receptor.bat"
    reduce = bin_dir / "reduce.bat"

    if not prepare_receptor.exists():
        raise FileNotFoundError(f"ADFR prepare_receptor launcher not found: {prepare_receptor}")
    if not reduce.exists():
        raise FileNotFoundError(f"ADFR reduce launcher not found: {reduce}")

    configured_args = list(extra_args)
    _append_if_missing(configured_args, "--vina-enable")
    _append_if_missing(configured_args, "--vina-receptor-backend", "adfr")
    _append_if_missing(configured_args, "--vina-prepare-receptor-exe", str(prepare_receptor))
    _append_if_missing(configured_args, "--vina-reduce-exe", str(reduce))
    return configured_args


def _build_vina_grower_args(args: argparse.Namespace) -> list[str]:
    grower_args: list[str] = []

    if args.vina_enable:
        grower_args.append("--vina-enable")
    if args.vina_strict:
        grower_args.append("--vina-strict")
    if args.vina_receptor_backend is not None:
        grower_args.extend(["--vina-receptor-backend", args.vina_receptor_backend])
    if args.vina_prepare_receptor_exe is not None:
        grower_args.extend(["--vina-prepare-receptor-exe", str(args.vina_prepare_receptor_exe)])
    if args.vina_reduce_exe is not None:
        grower_args.extend(["--vina-reduce-exe", str(args.vina_reduce_exe)])
    if args.vina_beam_top_n is not None:
        grower_args.extend(["--vina-beam-top-n", str(args.vina_beam_top_n)])

    if args.use_installed_adfr or args.adfr_suite_dir is not None:
        adfr_suite_dir = _resolve_adfr_suite_dir(args.adfr_suite_dir)
        if adfr_suite_dir is None:
            raise FileNotFoundError(
                "requested ADFR integration, but no ADFR Suite install was found. "
                "Pass --adfr-suite-dir explicitly or install ADFRsuite-1.1dev."
            )
        grower_args = _configure_adfr_args(grower_args, adfr_suite_dir)

    return grower_args


def _iter_specs() -> Iterable[RunSpec]:
    for values in itertools.product(
        N_STEPS_VALUES,
        N_EMBED_ATTEMPTS_VALUES,
        BEAM_WIDTH_VALUES,
        MAX_ATTACH_VALUES,
        MAX_FRAGS_VALUES,
    ):
        yield RunSpec(*values)


def _format_stat(values: Sequence[float], reducer) -> float | str:
    if not values:
        return ""
    return round(float(reducer(values)), 4)


def _summarize_metric(values: Sequence[float], prefix: str) -> dict[str, float | str]:
    return {
        f"{prefix}_min": _format_stat(values, min),
        f"{prefix}_median": _format_stat(values, statistics.median),
        f"{prefix}_mean": _format_stat(values, statistics.mean),
        f"{prefix}_max": _format_stat(values, max),
    }


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _resolve_pdb_path(pdb_code: str, cavity_dir: Path) -> Path:
    normalized = pdb_code.strip().upper()
    if not normalized:
        raise FileNotFoundError("Empty PDB code provided.")

    direct_match = cavity_dir / f"{normalized}_DU.pdb"
    if direct_match.exists():
        return direct_match

    matches = [
        path
        for path in cavity_dir.glob("*_DU.pdb")
        if path.stem.upper() == f"{normalized}_DU"
    ]
    if not matches:
        raise FileNotFoundError(
            f"No cavity PDB named {normalized}_DU.pdb found in {cavity_dir}."
        )
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple cavity PDBs matched {normalized}: {', '.join(str(path) for path in matches)}"
        )
    return matches[0]


def _read_design_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _count_clusters(smiles_list: Sequence[str], similarity_cutoff: float) -> int:
    molecules = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecules.append(mol)

    if not molecules:
        return 0
    if len(molecules) == 1:
        return 1

    fingerprints = [MORGAN_GENERATOR.GetFingerprint(mol) for mol in molecules]
    distances: list[float] = []
    for idx in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[idx], fingerprints[:idx])
        distances.extend(1.0 - similarity for similarity in similarities)

    distance_cutoff = 1.0 - similarity_cutoff
    clusters = Butina.ClusterData(distances, len(fingerprints), distance_cutoff, isDistData=True)
    return len(clusters)


def _build_summary_row(
    run_index: int,
    pdb_code: str,
    pdb_path: Path,
    spec: RunSpec,
    status: str,
    returncode: int,
    duration_sec: float,
    rows: Sequence[dict[str, str]],
    output_dir: Path,
    out_csv: Path,
    out_sdf: Path,
    log_path: Path,
    similarity_cutoff: float,
    error_text: str,
) -> dict[str, int | float | str]:
    smiles = [row.get("smiles", "").strip() for row in rows if row.get("smiles")]
    qed_values = [value for row in rows if (value := _parse_float(row.get("QED"))) is not None]
    sa_values = [value for row in rows if (value := _parse_float(row.get("SA_score"))) is not None]
    vina_values = [value for row in rows if (value := _parse_float(row.get("vina_score"))) is not None]

    summary = {
        "run_index": run_index,
        "pdb_code": pdb_code,
        "pdb_path": str(pdb_path),
        "du_index": 0,
        "n_steps": spec.n_steps,
        "n_embed_attempts": spec.n_embed_attempts,
        "beam_width": spec.beam_width,
        "max_attach": spec.max_attach,
        "max_frags": spec.max_frags,
        "status": status,
        "returncode": returncode,
        "duration_sec": round(duration_sec, 2),
        "accepted_molecules": len(rows),
        "unique_smiles": len(set(smiles)),
        "cluster_count": _count_clusters(smiles, similarity_cutoff),
        "output_dir": str(output_dir),
        "csv_path": str(out_csv),
        "sdf_path": str(out_sdf),
        "log_path": str(log_path),
        "error": error_text,
    }
    summary.update(_summarize_metric(qed_values, "qed"))
    summary.update(_summarize_metric(sa_values, "sa"))
    summary.update(_summarize_metric(vina_values, "vina"))
    return summary


def _build_command(
    python_exe: Path,
    grower_script: Path,
    pdb_path: Path,
    du_index: int,
    max_output: int,
    spec: RunSpec,
    out_sdf: Path,
    out_csv: Path,
    extra_grower_args: Sequence[str],
) -> list[str]:
    return [
        str(python_exe),
        str(grower_script),
        "--pdb",
        str(pdb_path),
        "--du-index",
        str(du_index),
        "--n-steps",
        str(spec.n_steps),
        "--n-embed",
        str(spec.n_embed_attempts),
        "--beam-width",
        str(spec.beam_width),
        "--max-attach",
        str(spec.max_attach),
        "--max-frags",
        str(spec.max_frags),
        "--max-output",
        str(max_output),
        "--out-sdf",
        str(out_sdf),
        "--out-csv",
        str(out_csv),
        *extra_grower_args,
    ]


def _write_log(log_path: Path, command: Sequence[str], stdout: str, stderr: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_text = "\n".join(
        [
            f"COMMAND: {' '.join(command)}",
            "",
            "STDOUT:",
            stdout.strip(),
            "",
            "STDERR:",
            stderr.strip(),
            "",
        ]
    )
    log_path.write_text(log_text, encoding="utf-8")


def _run_one(
    run_index: int,
    total_runs: int,
    pdb_code: str,
    pdb_path: Path,
    du_index: int,
    python_exe: Path,
    grower_script: Path,
    output_root: Path,
    max_output: int,
    spec: RunSpec,
    extra_grower_args: Sequence[str],
    similarity_cutoff: float,
    dry_run: bool,
    skip_existing: bool,
) -> dict[str, int | float | str]:
    run_name = (
        f"r{run_index:04d}_ns{spec.n_steps}_ne{spec.n_embed_attempts}_"
        f"bw{spec.beam_width}_ma{spec.max_attach}_mf{spec.max_frags}"
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    out_sdf = output_dir / "designs.sdf"
    out_csv = output_dir / "designs.csv"
    log_path = output_dir / "run.log"
    command = _build_command(
        python_exe=python_exe,
        grower_script=grower_script,
        pdb_path=pdb_path,
        du_index=du_index,
        max_output=max_output,
        spec=spec,
        out_sdf=out_sdf,
        out_csv=out_csv,
        extra_grower_args=extra_grower_args,
    )

    print(
        f"[{run_index:03d}/{total_runs:03d}] ns={spec.n_steps:2d} "
        f"ne={spec.n_embed_attempts:2d} bw={spec.beam_width:2d} "
        f"ma={spec.max_attach:2d} mf={spec.max_frags:2d}"
    )

    if skip_existing and out_csv.exists():
        rows = _read_design_rows(out_csv)
        return _build_summary_row(
            run_index=run_index,
            pdb_code=pdb_code,
            pdb_path=pdb_path,
            spec=spec,
            status="reused",
            returncode=0,
            duration_sec=0.0,
            rows=rows,
            output_dir=output_dir,
            out_csv=out_csv,
            out_sdf=out_sdf,
            log_path=log_path,
            similarity_cutoff=similarity_cutoff,
            error_text="",
        )

    if dry_run:
        print("  DRY-RUN", " ".join(command))
        return _build_summary_row(
            run_index=run_index,
            pdb_code=pdb_code,
            pdb_path=pdb_path,
            spec=spec,
            status="dry-run",
            returncode=0,
            duration_sec=0.0,
            rows=[],
            output_dir=output_dir,
            out_csv=out_csv,
            out_sdf=out_sdf,
            log_path=log_path,
            similarity_cutoff=similarity_cutoff,
            error_text="",
        )

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(grower_script.parent),
    )
    duration_sec = time.perf_counter() - started
    _write_log(log_path, command, completed.stdout, completed.stderr)

    rows = _read_design_rows(out_csv) if completed.returncode == 0 else []
    error_text = ""
    if completed.returncode != 0:
        error_text = completed.stderr.strip() or completed.stdout.strip() or "Run failed"

    return _build_summary_row(
        run_index=run_index,
        pdb_code=pdb_code,
        pdb_path=pdb_path,
        spec=spec,
        status="ok" if completed.returncode == 0 else "failed",
        returncode=completed.returncode,
        duration_sec=duration_sec,
        rows=rows,
        output_dir=output_dir,
        out_csv=out_csv,
        out_sdf=out_sdf,
        log_path=log_path,
        similarity_cutoff=similarity_cutoff,
        error_text=error_text,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a parameter sweep for one cavity PDB code through "
            "de_novo_cavity_growth.py and summarize QED, SA, Vina, and diversity."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pdb-code", required=True, help="PDB code, for example 1BUH")
    parser.add_argument("--du-index", type=int, default=0, help="DU cavity index to target")
    parser.add_argument("--cavity-dir", type=Path, default=DEFAULT_CAVITY_DIR, help="Folder containing *_DU.pdb files")
    parser.add_argument(
        "--python-exe",
        type=Path,
        default=DEFAULT_PYTHON_EXE,
        help="Python interpreter used to launch every growth run",
    )
    parser.add_argument(
        "--grower-script",
        type=Path,
        default=DEFAULT_GROWER_SCRIPT,
        help="Path to de_novo_cavity_growth.py",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder for all run outputs and the sweep summary",
    )
    parser.add_argument("--max-output", type=int, default=50, help="Forwarded to de_novo_cavity_growth.py")
    parser.add_argument(
        "--cluster-similarity-cutoff",
        type=float,
        default=0.65,
        help="Tanimoto similarity cutoff used for Butina clustering",
    )
    parser.add_argument("--vina-enable", action="store_true", help="Enable Vina scoring in every growth run")
    parser.add_argument("--vina-strict", action="store_true", help="Fail each run if Vina preparation or scoring fails")
    parser.add_argument(
        "--vina-receptor-backend",
        choices=("meeko", "adfr"),
        default=None,
        help="Receptor preparation backend forwarded to de_novo_cavity_growth.py",
    )
    parser.add_argument(
        "--vina-prepare-receptor-exe",
        type=Path,
        default=None,
        help="Path to the ADFR prepare_receptor executable forwarded to the grower",
    )
    parser.add_argument(
        "--vina-reduce-exe",
        type=Path,
        default=None,
        help="Path to the REDUCE executable forwarded to the grower",
    )
    parser.add_argument(
        "--vina-beam-top-n",
        type=int,
        default=None,
        help="Forwarded Vina top-N candidate limit; omit to leave the grower default unchanged",
    )
    parser.add_argument(
        "--use-installed-adfr",
        action="store_true",
        help="Auto-detect a local Windows ADFR Suite install and forward the matching grower flags",
    )
    parser.add_argument(
        "--adfr-suite-dir",
        type=Path,
        default=None,
        help="Path to an ADFR Suite installation root; implies ADFR grower flags",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without executing the grower")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse metrics from an existing per-run CSV instead of rerunning that combination",
    )
    parser.add_argument(
        "grower_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded unchanged to de_novo_cavity_growth.py after a standalone --",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not (0.0 < args.cluster_similarity_cutoff <= 1.0):
        print("ERROR: --cluster-similarity-cutoff must be in the interval (0, 1].", file=sys.stderr)
        return 2

    cavity_dir = args.cavity_dir.resolve()
    grower_script = args.grower_script.resolve()
    python_exe = args.python_exe.resolve()

    if not cavity_dir.exists():
        print(f"ERROR: cavity directory does not exist: {cavity_dir}", file=sys.stderr)
        return 2
    if not grower_script.exists():
        print(f"ERROR: grower script does not exist: {grower_script}", file=sys.stderr)
        return 2
    if not python_exe.exists():
        print(f"ERROR: python executable does not exist: {python_exe}", file=sys.stderr)
        return 2

    try:
        pdb_path = _resolve_pdb_path(args.pdb_code, cavity_dir)
    except (FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    extra_grower_args = list(args.grower_args)
    if extra_grower_args[:1] == ["--"]:
        extra_grower_args = extra_grower_args[1:]
    try:
        extra_grower_args = [*_build_vina_grower_args(args), *extra_grower_args]
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    pdb_code = args.pdb_code.strip().upper()
    sweep_root = args.output_root.resolve() / f"{pdb_code}_DU{args.du_index}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    summary_csv = sweep_root / f"{pdb_code}_DU{args.du_index}_sweep_summary.csv"

    specs = list(_iter_specs())
    print(f"Using cavity file: {pdb_path}")
    print(f"Grower script   : {grower_script}")
    print(f"Python exe      : {python_exe}")
    print(f"DU index        : {args.du_index}")
    print(f"Max output/run  : {args.max_output}")
    print(f"Run count       : {len(specs)}")
    if len(specs) != 180:
        print(
            "WARNING: the requested parameter ranges expand to "
            f"{len(specs)} runs, not 180."
        )
    if extra_grower_args:
        print(f"Extra grower args: {' '.join(extra_grower_args)}")

    ok_runs = 0
    failed_runs = 0
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()

        for run_index, spec in enumerate(specs, start=1):
            summary_row = _run_one(
                run_index=run_index,
                total_runs=len(specs),
                pdb_code=pdb_code,
                pdb_path=pdb_path,
                du_index=args.du_index,
                python_exe=python_exe,
                grower_script=grower_script,
                output_root=sweep_root,
                max_output=args.max_output,
                spec=spec,
                extra_grower_args=extra_grower_args,
                similarity_cutoff=args.cluster_similarity_cutoff,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )
            summary_row["du_index"] = args.du_index
            writer.writerow(summary_row)
            handle.flush()

            if summary_row["status"] in {"ok", "reused"}:
                ok_runs += 1
            elif summary_row["status"] == "failed":
                failed_runs += 1

    print(f"\nSweep summary written to: {summary_csv}")
    print(f"Successful or reused runs: {ok_runs}")
    print(f"Failed runs              : {failed_runs}")
    return 0 if failed_runs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())