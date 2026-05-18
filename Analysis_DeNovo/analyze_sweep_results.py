#!/usr/bin/env python3
"""Post-process sweep summaries into charts and ranked candidate sets."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


HEATMAP_PAIRS = (
    ("n_steps", "beam_width"),
    ("max_attach", "max_frags"),
)
SHORTLIST_COLUMNS = [
    "run_index",
    "n_steps",
    "n_embed_attempts",
    "beam_width",
    "max_attach",
    "max_frags",
    "accepted_molecules",
    "cluster_count",
    "diversity_ratio",
    "vina_mean",
    "duration_sec",
    "median_heavy_atoms",
    "median_mol_wt",
    "length_gap",
    "length_score",
    "composite_score",
    "pareto_front",
    "output_dir",
]


@dataclass(frozen=True)
class SizeMetrics:
    median_heavy_atoms: float | None
    median_mol_wt: float | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a sweep summary CSV and produce Pareto, chart, and shortlist outputs."
        )
    )
    parser.add_argument("summary_csv", type=Path, help="Sweep summary CSV produced by sweep_de_novo_cavity_growth.py")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where analysis outputs are written (defaults next to the summary CSV)",
    )
    parser.add_argument(
        "--vina-column",
        default="vina_mean",
        help="Summary-column used as the docking quality metric (default: vina_mean)",
    )
    parser.add_argument(
        "--min-accepted",
        type=int,
        default=3,
        help="Minimum accepted molecules required for a run to be considered in ranking (default: 3)",
    )
    parser.add_argument(
        "--target-heavy-atoms",
        type=float,
        default=None,
        help=(
            "Preferred median heavy-atom count. If omitted, the script uses the median across eligible runs."
        ),
    )
    parser.add_argument(
        "--length-tolerance",
        type=float,
        default=6.0,
        help="Heavy-atom deviation tolerated before the length score drops to zero (default: 6)",
    )
    parser.add_argument(
        "--shortlist-size",
        type=int,
        default=10,
        help="Number of ranked runs written to the shortlist outputs (default: 10)",
    )
    parser.add_argument(
        "--annotate-top",
        type=int,
        default=8,
        help="Number of top composite-score points to annotate in the bubble plot (default: 8)",
    )
    return parser


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _round_or_blank(value: float | None, digits: int = 4) -> float | str:
    if value is None:
        return ""
    return round(value, digits)


def _load_size_metrics(sdf_path: Path) -> SizeMetrics:
    heavy_atoms: list[float] = []
    mol_weights: list[float] = []
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        heavy_atoms.append(float(mol.GetNumHeavyAtoms()))
        mol_weights.append(float(Descriptors.MolWt(mol)))

    return SizeMetrics(
        median_heavy_atoms=_median_or_none(heavy_atoms),
        median_mol_wt=_median_or_none(mol_weights),
    )


def _minmax_score(value: float, minimum: float, maximum: float, higher_is_better: bool) -> float:
    if math.isclose(maximum, minimum):
        return 1.0
    scaled = (value - minimum) / (maximum - minimum)
    return scaled if higher_is_better else 1.0 - scaled


def _length_score(median_heavy_atoms: float | None, target: float, tolerance: float) -> tuple[float | None, float | None]:
    if median_heavy_atoms is None:
        return None, None
    gap = abs(median_heavy_atoms - target)
    score = max(0.0, 1.0 - gap / tolerance)
    return gap, score


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _label_for_row(row: dict[str, object]) -> str:
    return (
        f"ns{int(row['n_steps'])}/ne{int(row['n_embed_attempts'])}/bw{int(row['beam_width'])}/"
        f"ma{int(row['max_attach'])}/mf{int(row['max_frags'])}"
    )


def _dominates(left: dict[str, object], right: dict[str, object], vina_column: str) -> bool:
    comparisons = (
        float(left["diversity_ratio"]) >= float(right["diversity_ratio"]),
        float(left[vina_column]) <= float(right[vina_column]),
        float(left["duration_sec"]) <= float(right["duration_sec"]),
        float(left["length_gap"]) <= float(right["length_gap"]),
    )
    strictly_better = (
        float(left["diversity_ratio"]) > float(right["diversity_ratio"]),
        float(left[vina_column]) < float(right[vina_column]),
        float(left["duration_sec"]) < float(right["duration_sec"]),
        float(left["length_gap"]) < float(right["length_gap"]),
    )
    return all(comparisons) and any(strictly_better)


def _compute_pareto_front(rows: Sequence[dict[str, object]], vina_column: str) -> set[int]:
    pareto_indices: set[int] = set()
    for index, row in enumerate(rows):
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            if _dominates(other, row, vina_column):
                dominated = True
                break
        if not dominated:
            pareto_indices.add(index)
    return pareto_indices


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _markdown_table(rows: Sequence[dict[str, object]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _write_markdown_summary(
    path: Path,
    summary_csv: Path,
    output_dir: Path,
    total_rows: int,
    eligible_rows: Sequence[dict[str, object]],
    pareto_rows: Sequence[dict[str, object]],
    shortlist_rows: Sequence[dict[str, object]],
    target_heavy_atoms: float,
    vina_column: str,
) -> None:
    text = f"""# Sweep Analysis Summary

## Inputs

- Summary CSV: {summary_csv}
- Output directory: {output_dir}
- Docking metric: {vina_column}
- Total runs in summary: {total_rows}
- Eligible runs ranked: {len(eligible_rows)}
- Pareto-front runs: {len(pareto_rows)}
- Heavy-atom target used for the length score: {target_heavy_atoms:.2f}

## Outputs

- `enriched_summary.csv`: original summary rows plus derived metrics and scores
- `pareto_front.csv`: non-dominated runs across diversity, Vina, runtime, and size target gap
- `pareto_front.md`: readable markdown version of the Pareto front
- `bubble_tradeoff.png`: diversity vs docking, with point size showing accepted molecules and color showing median heavy atoms
- `heatmap_vina_mean.png`: pairwise parameter heatmaps for docking quality
- `heatmap_diversity_ratio.png`: pairwise parameter heatmaps for diversity
- `heatmap_duration_sec.png`: pairwise parameter heatmaps for runtime
- `heatmap_composite_score.png`: pairwise parameter heatmaps for the final weighted score
- `ranked_shortlist.csv`: top-ranked candidate parameter sets
- `ranked_shortlist.md`: readable markdown version of the shortlist

## Pareto Front

{_markdown_table(pareto_rows[:15], SHORTLIST_COLUMNS[:-1]) if pareto_rows else 'No Pareto-front rows were found.'}

## Ranked Shortlist

{_markdown_table(shortlist_rows, SHORTLIST_COLUMNS[:-1]) if shortlist_rows else 'No shortlist rows were produced.'}
"""
    path.write_text(text, encoding="utf-8")


def _annotate_points(ax: plt.Axes, rows: Sequence[dict[str, object]], top_n: int) -> None:
    rows_to_label = sorted(rows, key=lambda row: float(row["composite_score"]), reverse=True)[:top_n]
    for row in rows_to_label:
        ax.annotate(
            _label_for_row(row),
            (float(row["diversity_ratio"]), float(row["vina_mean"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )


def _build_bubble_plot(rows: Sequence[dict[str, object]], output_path: Path, annotate_top: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    x_values = np.array([float(row["diversity_ratio"]) for row in rows], dtype=float)
    y_values = np.array([float(row["vina_mean"]) for row in rows], dtype=float)
    sizes = np.array([max(40.0, float(row["accepted_molecules"]) * 20.0) for row in rows], dtype=float)
    colors = np.array([float(row["median_heavy_atoms"]) for row in rows], dtype=float)
    pareto_mask = np.array([bool(row["pareto_front"]) for row in rows], dtype=bool)

    scatter = ax.scatter(
        x_values,
        y_values,
        s=sizes,
        c=colors,
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.scatter(
        x_values[pareto_mask],
        y_values[pareto_mask],
        s=sizes[pareto_mask] * 1.2,
        facecolors="none",
        edgecolors="crimson",
        linewidths=1.5,
        label="Pareto front",
    )
    _annotate_points(ax, rows, annotate_top)
    ax.set_xlabel("Diversity ratio (cluster_count / accepted_molecules)")
    ax.set_ylabel("Vina mean score")
    ax.set_title("Parameter-set tradeoff: diversity vs docking quality")
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Median heavy atoms")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _mean_by_pair(rows: Sequence[dict[str, object]], metric: str, x_key: str, y_key: str) -> tuple[list[int], list[int], np.ndarray]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        x_value = int(row[x_key])
        y_value = int(row[y_key])
        grouped[(x_value, y_value)].append(float(row[metric]))

    x_values = sorted({pair[0] for pair in grouped})
    y_values = sorted({pair[1] for pair in grouped})
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for (x_value, y_value), values in grouped.items():
        row_index = y_values.index(y_value)
        col_index = x_values.index(x_value)
        matrix[row_index, col_index] = float(statistics.mean(values))
    return x_values, y_values, matrix


def _render_heatmaps(rows: Sequence[dict[str, object]], metric: str, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    cmap = "viridis_r" if metric in {"vina_mean", "duration_sec"} else "viridis"

    for ax, (x_key, y_key) in zip(axes, HEATMAP_PAIRS):
        x_values, y_values, matrix = _mean_by_pair(rows, metric, x_key, y_key)
        image = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(x_values)), labels=x_values)
        ax.set_yticks(range(len(y_values)), labels=y_values)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(f"{x_key} vs {y_key}")
        for y_index in range(len(y_values)):
            for x_index in range(len(x_values)):
                value = matrix[y_index, x_index]
                if not math.isnan(value):
                    ax.text(x_index, y_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.85)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _format_output_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    formatted_rows: list[dict[str, object]] = []
    for row in rows:
        formatted_row = dict(row)
        for key in (
            "duration_sec",
            "diversity_ratio",
            "median_heavy_atoms",
            "median_mol_wt",
            "length_gap",
            "length_score",
            "composite_score",
            "vina_mean",
        ):
            formatted_row[key] = _round_or_blank(_parse_float(str(formatted_row.get(key, ""))))
        formatted_row["pareto_front"] = "yes" if row.get("pareto_front") else "no"
        formatted_rows.append(formatted_row)
    return formatted_rows


def _build_enriched_rows(
    raw_rows: Sequence[dict[str, str]],
    vina_column: str,
    min_accepted: int,
    target_heavy_atoms: float | None,
    length_tolerance: float,
) -> tuple[list[dict[str, object]], float]:
    size_cache: dict[Path, SizeMetrics] = {}
    enriched_rows: list[dict[str, object]] = []

    for raw_row in raw_rows:
        row = dict(raw_row)
        accepted_molecules = _parse_int(raw_row.get("accepted_molecules")) or 0
        cluster_count = _parse_int(raw_row.get("cluster_count")) or 0
        duration_sec = _parse_float(raw_row.get("duration_sec")) or 0.0
        vina_value = _parse_float(raw_row.get(vina_column))
        sdf_path_text = raw_row.get("sdf_path", "").strip()
        median_heavy_atoms = None
        median_mol_wt = None

        if sdf_path_text:
            sdf_path = Path(sdf_path_text)
            if sdf_path.exists():
                metrics = size_cache.setdefault(sdf_path, _load_size_metrics(sdf_path))
                median_heavy_atoms = metrics.median_heavy_atoms
                median_mol_wt = metrics.median_mol_wt

        row.update(
            {
                "accepted_molecules": accepted_molecules,
                "cluster_count": cluster_count,
                "duration_sec": duration_sec,
                "diversity_ratio": _safe_ratio(cluster_count, accepted_molecules),
                "median_heavy_atoms": median_heavy_atoms,
                "median_mol_wt": median_mol_wt,
                "eligible": False,
                "pareto_front": False,
                "composite_score": None,
            }
        )
        row[vina_column] = vina_value
        enriched_rows.append(row)

    if target_heavy_atoms is None:
        heavy_atom_values = [
            float(row["median_heavy_atoms"])
            for row in enriched_rows
            if row["median_heavy_atoms"] is not None
        ]
        target_heavy_atoms = float(statistics.median(heavy_atom_values)) if heavy_atom_values else 25.0

    for row in enriched_rows:
        length_gap, length_score = _length_score(
            row["median_heavy_atoms"],
            target_heavy_atoms,
            length_tolerance,
        )
        row["length_gap"] = length_gap
        row["length_score"] = length_score
        row["eligible"] = bool(
            row.get("status") in {"ok", "reused"}
            and int(row["accepted_molecules"]) >= min_accepted
            and row.get(vina_column) is not None
            and length_gap is not None
        )

    eligible_rows = [row for row in enriched_rows if row["eligible"]]
    if eligible_rows:
        diversity_values = [float(row["diversity_ratio"]) for row in eligible_rows]
        vina_values = [float(row[vina_column]) for row in eligible_rows]
        runtime_values = [float(row["duration_sec"]) for row in eligible_rows]

        diversity_min, diversity_max = min(diversity_values), max(diversity_values)
        vina_min, vina_max = min(vina_values), max(vina_values)
        runtime_min, runtime_max = min(runtime_values), max(runtime_values)

        for row in eligible_rows:
            diversity_score = _minmax_score(float(row["diversity_ratio"]), diversity_min, diversity_max, True)
            vina_score = _minmax_score(float(row[vina_column]), vina_min, vina_max, False)
            runtime_score = _minmax_score(float(row["duration_sec"]), runtime_min, runtime_max, False)
            length_score_value = float(row["length_score"])
            row["composite_score"] = (
                0.35 * diversity_score
                + 0.35 * vina_score
                + 0.20 * length_score_value
                + 0.10 * runtime_score
            )
    else:
        for row in enriched_rows:
            row["composite_score"] = 0.0

    return enriched_rows, target_heavy_atoms


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    summary_csv = args.summary_csv.resolve()
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV does not exist: {summary_csv}")

    output_dir = args.output_dir.resolve() if args.output_dir else summary_csv.parent / f"{summary_csv.stem}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _read_csv_rows(summary_csv)
    if not raw_rows:
        raise ValueError(f"Summary CSV is empty: {summary_csv}")
    if args.vina_column not in raw_rows[0]:
        raise KeyError(f"Column '{args.vina_column}' is missing from {summary_csv}")

    enriched_rows, target_heavy_atoms = _build_enriched_rows(
        raw_rows=raw_rows,
        vina_column=args.vina_column,
        min_accepted=args.min_accepted,
        target_heavy_atoms=args.target_heavy_atoms,
        length_tolerance=args.length_tolerance,
    )
    eligible_rows = [row for row in enriched_rows if row["eligible"]]
    pareto_indices = _compute_pareto_front(eligible_rows, args.vina_column) if eligible_rows else set()
    for index, row in enumerate(eligible_rows):
        row["pareto_front"] = index in pareto_indices

    pareto_rows = [row for row in eligible_rows if row["pareto_front"]]
    pareto_rows.sort(key=lambda row: float(row["composite_score"]), reverse=True)

    shortlist_source = pareto_rows if pareto_rows else eligible_rows
    shortlist_rows = sorted(
        shortlist_source,
        key=lambda row: float(row["composite_score"]),
        reverse=True,
    )[: args.shortlist_size]

    enriched_csv = output_dir / "enriched_summary.csv"
    enriched_columns = list(raw_rows[0].keys()) + [
        "diversity_ratio",
        "median_heavy_atoms",
        "median_mol_wt",
        "length_gap",
        "length_score",
        "eligible",
        "pareto_front",
        "composite_score",
    ]
    _write_csv(enriched_csv, _format_output_rows(enriched_rows), enriched_columns)

    formatted_pareto_rows = _format_output_rows(pareto_rows)
    formatted_shortlist_rows = _format_output_rows(shortlist_rows)
    _write_csv(output_dir / "pareto_front.csv", formatted_pareto_rows, SHORTLIST_COLUMNS)
    _write_csv(output_dir / "ranked_shortlist.csv", formatted_shortlist_rows, SHORTLIST_COLUMNS)
    (output_dir / "pareto_front.md").write_text(
        _markdown_table(formatted_pareto_rows, SHORTLIST_COLUMNS[:-1]) if formatted_pareto_rows else "No Pareto-front rows were found.\n",
        encoding="utf-8",
    )
    (output_dir / "ranked_shortlist.md").write_text(
        _markdown_table(formatted_shortlist_rows, SHORTLIST_COLUMNS[:-1]) if formatted_shortlist_rows else "No shortlist rows were produced.\n",
        encoding="utf-8",
    )

    if eligible_rows:
        _build_bubble_plot(eligible_rows, output_dir / "bubble_tradeoff.png", args.annotate_top)
        _render_heatmaps(eligible_rows, args.vina_column, output_dir / f"heatmap_{args.vina_column}.png", "Docking quality by parameter pair")
        _render_heatmaps(eligible_rows, "diversity_ratio", output_dir / "heatmap_diversity_ratio.png", "Diversity by parameter pair")
        _render_heatmaps(eligible_rows, "duration_sec", output_dir / "heatmap_duration_sec.png", "Runtime by parameter pair")
        _render_heatmaps(eligible_rows, "composite_score", output_dir / "heatmap_composite_score.png", "Composite score by parameter pair")

    _write_markdown_summary(
        path=output_dir / "analysis_summary.md",
        summary_csv=summary_csv,
        output_dir=output_dir,
        total_rows=len(raw_rows),
        eligible_rows=eligible_rows,
        pareto_rows=formatted_pareto_rows,
        shortlist_rows=formatted_shortlist_rows,
        target_heavy_atoms=target_heavy_atoms,
        vina_column=args.vina_column,
    )

    print(f"Summary CSV        : {summary_csv}")
    print(f"Analysis directory : {output_dir}")
    print(f"Total rows         : {len(raw_rows)}")
    print(f"Eligible rows      : {len(eligible_rows)}")
    print(f"Pareto-front rows  : {len(pareto_rows)}")
    print(f"Shortlist rows     : {len(shortlist_rows)}")
    print(f"Target heavy atoms : {target_heavy_atoms:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())