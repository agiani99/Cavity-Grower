#!/usr/bin/env python3
"""
postprocess_pareto_vina.py
==========================
Standalone Pareto/Vina post-processing for summary CSV files produced by
de_novo_cavity_growth.py or postprocess_designs.py.

What it does
------------
1. Reads one or many *_summary.csv files.
2. Detects the composite and Vina columns.
3. Computes Pareto fronts in 2D:
     maximize composite
     minimize vina_score
4. Writes per-cavity outputs:
     - pareto_annotated.csv
     - pareto_front1.csv
     - pareto_selected.csv
     - pareto_front1.sdf      (if matching *_all_filtered.sdf exists)
     - pareto_selected.sdf    (if matching *_all_filtered.sdf exists)
5. Writes a combined cavity_report.csv.

Representative selections
-------------------------
- best_composite : highest composite
- best_vina      : minimum Vina (most negative = best)
- best_balanced  : highest normalized average of composite and Vina goodness
- front_pick     : additional molecules from the first N Pareto fronts

Usage
-----
  python postprocess_pareto_vina.py --summary-dir postprocessed_ph405

  python postprocess_pareto_vina.py \
      --summary postprocessed_ph405/1C4Z_DU_DU1_default_summary.csv \
      --out-dir pareto_postprocessed
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem


@dataclass
class Candidate:
    row: Dict[str, str]
    cavity_id: str
    smiles: str
    canonical_smiles: str
    composite: float
    vina_score: Optional[float]
    rank: Optional[int]
    source_id: str
    pareto_front: Optional[int] = None
    pareto_rank: Optional[int] = None
    balanced_score: float = 0.0
    selection_reason: str = ""


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_int(value: object) -> Optional[int]:
    fv = _coerce_float(value)
    if fv is None:
        return None
    try:
        return int(fv)
    except (TypeError, ValueError):
        return None


def _first_present(row: Dict[str, str], names: Sequence[str]) -> str:
    for name in names:
        if name in row and str(row[name]).strip():
            return str(row[name]).strip()
    return ""


def _canonicalize_smiles(smiles: str) -> str:
    text = (smiles or "").strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return text
    return Chem.MolToSmiles(mol)


def _normalize_high(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 1.0
    return (value - min_value) / (max_value - min_value)


def _normalize_low(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 1.0
    return (max_value - value) / (max_value - min_value)


def _vina_value(score: Optional[float]) -> float:
    return float("inf") if score is None else float(score)


def _dominates(a: Candidate, b: Candidate) -> bool:
    a_vina = _vina_value(a.vina_score)
    b_vina = _vina_value(b.vina_score)
    better_or_equal = a.composite >= b.composite and a_vina <= b_vina
    strictly_better = a.composite > b.composite or a_vina < b_vina
    return better_or_equal and strictly_better


def _candidate_sort_key(candidate: Candidate) -> Tuple[int, float, float, float, str]:
    front = candidate.pareto_front if candidate.pareto_front is not None else 10**9
    return (
        front,
        -candidate.composite,
        _vina_value(candidate.vina_score),
        -candidate.balanced_score,
        candidate.canonical_smiles,
    )


def _find_summary_paths(summary_paths: Sequence[str], summary_dir: Optional[str], pattern: str) -> List[Path]:
    found: List[Path] = []
    for item in summary_paths:
        path = Path(item)
        if path.exists() and path.is_file():
            found.append(path)

    if summary_dir:
        directory = Path(summary_dir)
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Summary directory not found: {summary_dir}")
        found.extend(sorted(directory.glob(pattern)))

    unique: List[Path] = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _load_candidates(summary_path: Path) -> List[Candidate]:
    cavity_id = summary_path.name[:-12] if summary_path.name.endswith("_summary.csv") else summary_path.stem
    candidates: List[Candidate] = []
    with open(summary_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            smiles = _first_present(row, ["smiles", "SMILES"])
            composite = _coerce_float(_first_present(row, ["composite", "composite_grower"]))
            if not smiles or composite is None:
                continue

            vina_score = _coerce_float(_first_present(row, ["vina_score", "vina", "VINA", "docking_score", "binding_energy", "affinity"]))
            rank = _coerce_int(row.get("rank"))
            source_id = _first_present(row, ["id", "source_id", "_Name"]) or f"row_{len(candidates)+1:04d}"
            canonical_smiles = _canonicalize_smiles(smiles)
            candidates.append(
                Candidate(
                    row=dict(row),
                    cavity_id=cavity_id,
                    smiles=smiles,
                    canonical_smiles=canonical_smiles,
                    composite=float(composite),
                    vina_score=vina_score,
                    rank=rank,
                    source_id=source_id,
                )
            )
    return candidates


def _assign_balanced_scores(candidates: List[Candidate]) -> None:
    if not candidates:
        return

    composite_values = [candidate.composite for candidate in candidates]
    comp_min = min(composite_values)
    comp_max = max(composite_values)

    vina_values = [candidate.vina_score for candidate in candidates if candidate.vina_score is not None]
    vina_min = min(vina_values) if vina_values else None
    vina_max = max(vina_values) if vina_values else None

    for candidate in candidates:
        comp_norm = _normalize_high(candidate.composite, comp_min, comp_max)
        if vina_min is None or vina_max is None:
            candidate.balanced_score = comp_norm
            continue
        if candidate.vina_score is None:
            candidate.balanced_score = 0.5 * comp_norm
            continue
        vina_norm = _normalize_low(candidate.vina_score, vina_min, vina_max)
        candidate.balanced_score = 0.5 * (comp_norm + vina_norm)


def _assign_pareto_fronts(candidates: List[Candidate]) -> List[Candidate]:
    remaining = list(candidates)
    ranked: List[Candidate] = []
    front_index = 1
    while remaining:
        front: List[Candidate] = []
        for candidate in remaining:
            if not any(_dominates(other, candidate) for other in remaining if other is not candidate):
                front.append(candidate)

        front_sorted = sorted(
            front,
            key=lambda candidate: (-candidate.composite, _vina_value(candidate.vina_score), -candidate.balanced_score, candidate.canonical_smiles),
        )
        for rank_in_front, candidate in enumerate(front_sorted, start=1):
            candidate.pareto_front = front_index
            candidate.pareto_rank = rank_in_front
        ranked.extend(front_sorted)
        remaining = [candidate for candidate in remaining if candidate not in front]
        front_index += 1
    return ranked


def _append_reason(candidate: Candidate, reason: str) -> None:
    current = [item for item in candidate.selection_reason.split(";") if item]
    if reason not in current:
        current.append(reason)
    candidate.selection_reason = ";".join(current)


def _select_representatives(candidates: List[Candidate], keep_fronts: int, picks_per_cavity: int) -> List[Candidate]:
    selected: List[Candidate] = []
    seen_smiles = set()

    def add(candidate: Optional[Candidate], reason: str) -> None:
        if candidate is None:
            return
        key = candidate.canonical_smiles or candidate.smiles
        if key in seen_smiles:
            _append_reason(candidate, reason)
            return
        seen_smiles.add(key)
        _append_reason(candidate, reason)
        selected.append(candidate)

    if candidates:
        add(max(candidates, key=lambda candidate: candidate.composite), "best_composite")

    vina_candidates = [candidate for candidate in candidates if candidate.vina_score is not None]
    if vina_candidates:
        add(min(vina_candidates, key=lambda candidate: candidate.vina_score if candidate.vina_score is not None else float("inf")), "best_vina")

    if candidates:
        add(max(candidates, key=lambda candidate: candidate.balanced_score), "best_balanced")

    front_pool = [candidate for candidate in candidates if (candidate.pareto_front or 10**9) <= keep_fronts]
    for candidate in sorted(front_pool, key=_candidate_sort_key):
        if len(selected) >= picks_per_cavity:
            break
        add(candidate, "front_pick")
    return selected


def _candidate_to_row(candidate: Candidate) -> Dict[str, object]:
    row = dict(candidate.row)
    row["cavity_id"] = candidate.cavity_id
    row["source_id"] = candidate.source_id
    row["canonical_smiles"] = candidate.canonical_smiles
    row["composite_value"] = round(candidate.composite, 4)
    row["vina_value"] = round(candidate.vina_score, 4) if candidate.vina_score is not None else ""
    row["pareto_front"] = candidate.pareto_front if candidate.pareto_front is not None else ""
    row["pareto_rank"] = candidate.pareto_rank if candidate.pareto_rank is not None else ""
    row["balanced_score"] = round(candidate.balanced_score, 4)
    row["selection_reason"] = candidate.selection_reason
    return row


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows):5d} rows         -> {path}")


def _find_matching_sdf(summary_path: Path) -> Optional[Path]:
    if summary_path.name.endswith("_summary.csv"):
        candidate = summary_path.with_name(summary_path.name.replace("_summary.csv", "_all_filtered.sdf"))
        if candidate.exists():
            return candidate
    return None


def _load_sdf_index(sdf_path: Path) -> Dict[str, Tuple[Chem.Mol, Dict[str, str]]]:
    index: Dict[str, Tuple[Chem.Mol, Dict[str, str]]] = {}
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    for mol in supplier:
        if mol is None:
            continue
        props = {name: mol.GetProp(name) for name in mol.GetPropNames()}
        smiles = props.get("SMILES") or Chem.MolToSmiles(mol)
        canonical_smiles = _canonicalize_smiles(smiles)
        if canonical_smiles and canonical_smiles not in index:
            index[canonical_smiles] = (Chem.Mol(mol), props)
    return index


def _write_selected_sdf(candidates: List[Candidate], sdf_index: Dict[str, Tuple[Chem.Mol, Dict[str, str]]], out_path: Path) -> None:
    if not candidates or not sdf_index:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))
    n_written = 0
    for candidate in candidates:
        hit = sdf_index.get(candidate.canonical_smiles)
        if hit is None:
            continue
        mol, props = hit
        out = Chem.Mol(mol)
        for key, value in props.items():
            out.SetProp(str(key), str(value))
        row = _candidate_to_row(candidate)
        for key, value in row.items():
            if value == "":
                continue
            out.SetProp(str(key), str(value))
        writer.write(out)
        n_written += 1
    writer.close()
    print(f"  Wrote {n_written:5d} molecules    -> {out_path}")


def _process_one(summary_path: Path, out_dir: Path, keep_fronts: int, picks_per_cavity: int) -> Dict[str, object]:
    print(f"\nProcessing {summary_path.name}")
    candidates = _load_candidates(summary_path)
    if not candidates:
        print("  No usable rows found; skipping.")
        return {
            "cavity_id": summary_path.stem,
            "summary_path": str(summary_path),
            "n_input": 0,
            "n_front1": 0,
            "n_selected": 0,
            "best_composite": "",
            "best_vina": "",
        }

    _assign_balanced_scores(candidates)
    ranked = _assign_pareto_fronts(candidates)
    front1 = [candidate for candidate in ranked if candidate.pareto_front == 1]
    selected = _select_representatives(ranked, keep_fronts=keep_fronts, picks_per_cavity=picks_per_cavity)

    cavity_dir = out_dir / ranked[0].cavity_id
    annotated_rows = [_candidate_to_row(candidate) for candidate in ranked]
    front1_rows = [_candidate_to_row(candidate) for candidate in front1]
    selected_rows = [_candidate_to_row(candidate) for candidate in selected]

    _write_csv(annotated_rows, cavity_dir / f"{ranked[0].cavity_id}_pareto_annotated.csv")
    _write_csv(front1_rows, cavity_dir / f"{ranked[0].cavity_id}_pareto_front1.csv")
    _write_csv(selected_rows, cavity_dir / f"{ranked[0].cavity_id}_pareto_selected.csv")

    sdf_path = _find_matching_sdf(summary_path)
    if sdf_path is not None:
        sdf_index = _load_sdf_index(sdf_path)
        _write_selected_sdf(front1, sdf_index, cavity_dir / f"{ranked[0].cavity_id}_pareto_front1.sdf")
        _write_selected_sdf(selected, sdf_index, cavity_dir / f"{ranked[0].cavity_id}_pareto_selected.sdf")
    else:
        print("  Matching *_all_filtered.sdf not found; skipping SDF subset outputs.")

    valid_vina = [candidate.vina_score for candidate in ranked if candidate.vina_score is not None]
    report_row = {
        "cavity_id": ranked[0].cavity_id,
        "summary_path": str(summary_path),
        "n_input": len(ranked),
        "n_front1": len(front1),
        "n_selected": len(selected),
        "best_composite": round(max(candidate.composite for candidate in ranked), 4),
        "best_vina": round(min(valid_vina), 4) if valid_vina else "",
        "top_smiles": ranked[0].smiles,
        "top_pareto_front": ranked[0].pareto_front,
    }
    return report_row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone Pareto/Vina post-processing for cavity summary CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--summary",
        nargs="*",
        default=[],
        help="One or more summary CSV files to process",
    )
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="Directory containing summary CSV files",
    )
    parser.add_argument(
        "--pattern",
        default="*_summary.csv",
        help="Glob used inside --summary-dir",
    )
    parser.add_argument(
        "--out-dir",
        default="pareto_postprocessed",
        help="Output directory for Pareto/Vina reports",
    )
    parser.add_argument(
        "--keep-fronts",
        type=int,
        default=1,
        help="Number of Pareto fronts eligible for front_pick selection",
    )
    parser.add_argument(
        "--picks-per-cavity",
        type=int,
        default=12,
        help="Maximum number of representative molecules exported per cavity",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    summary_paths = _find_summary_paths(args.summary, args.summary_dir, args.pattern)
    if not summary_paths:
        print("ERROR: no summary CSV files found.")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(summary_paths)} summary CSV file(s)")
    report_rows: List[Dict[str, object]] = []
    for summary_path in summary_paths:
        report_rows.append(
            _process_one(
                summary_path=summary_path,
                out_dir=out_dir,
                keep_fronts=max(1, int(args.keep_fronts)),
                picks_per_cavity=max(1, int(args.picks_per_cavity)),
            )
        )

    _write_csv(report_rows, out_dir / "cavity_report.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())