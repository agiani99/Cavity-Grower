#!/usr/bin/env python3
"""dogsite_interface_cavities.py

Batch-run DoGSiteScorer (proteins.plus) for PDB entries listed in a CSV and
extract *interface* cavities (pockets) between two chains.

What it does per PDB:
- Creates/polls a DoGSiteScorer job via https://proteins.plus/api/dogsite_rest
  - Uses analysisDetail="0" (pockets only; no subpockets/subcavities)
- Downloads the descriptor table ("*_desc.txt") and extracts pocket volumes
- For pockets with volume >= --min-volume, downloads the pocket residue PDB
  ("*_P_<i>_res.pdb") and parses the "Geometric pocket center" coordinates
- Downloads the original PDB from RCSB and auto-selects the two main chains
  (or uses --chain-a/--chain-b)
- Keeps pockets whose center is within --center-dist-threshold Å of *both*
  chains (a practical "between two chains" heuristic)
- Writes:
  - JSON with the raw job payload + parsed pocket summary and filtering
  - PDB with one DU marker atom per selected pocket center

Notes
- The proteins.plus REST endpoint takes a PDB code (no structure upload).
- The interface test is geometric proximity (center close to both chains),
  because the DoGSite residue PDB files use a dummy chain ID and do not
  preserve original chain IDs.

Python: 3.10+
Dependencies: requests
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

DOGSITE_CREATE_URL = "https://proteins.plus/api/dogsite_rest"
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


@dataclass(frozen=True)
class Pocket:
    pocket_id: str  # e.g. "P_0"
    volume_A3: float
    center: tuple[float, float, float] | None = None
    min_dist_chain_a: float | None = None
    min_dist_chain_b: float | None = None


class DogSiteError(RuntimeError):
    pass


def _safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _job_id_from_location(location: str) -> str:
    # https://proteins.plus/api/dogsite_rest/<job_id>
    return location.rstrip("/").split("/")[-1]


def localize_dogsite_job_assets(
    session: requests.Session,
    pdb_id: str,
    job_location: str,
    job_payload: dict[str, Any],
    out_root: Path,
    *,
    download_pocket_grids: bool,
    embed_text_in_json: bool,
) -> dict[str, Any]:
    """Replace proteins.plus URLs with locally stored artifacts and/or embedded text."""

    job_id = _job_id_from_location(job_location)
    job_dir = out_root / "dogsite_assets" / pdb_id
    desc_path = job_dir / f"{pdb_id.lower()}_desc.txt"

    result_table_url = str(job_payload.get("result_table") or "")
    if not result_table_url:
        raise DogSiteError("DoGSite job payload missing result_table")
    desc_txt = _fetch_text(session, result_table_url)
    _safe_write_text(desc_path, desc_txt)

    # descriptor explanations (static, but store locally anyway)
    descriptor_url = str(job_payload.get("descriptor_explanation") or "")
    descriptor_path: Path | None = None
    descriptor_txt: str | None = None
    if descriptor_url:
        descriptor_path = job_dir / "descriptor_explanation.txt"
        descriptor_txt = _fetch_text(session, descriptor_url)
        _safe_write_text(descriptor_path, descriptor_txt)

    # residues PDBs (text)
    residues: list[str] = [str(u) for u in (job_payload.get("residues") or [])]
    residues_local: dict[str, dict[str, Any]] = {}
    for url in residues:
        m = re.search(r"_P_(\d+)_res\.pdb$", url)
        if not m:
            continue
        pocket_id = f"P_{int(m.group(1))}"
        pdb_text = _fetch_text(session, url)
        local_path = job_dir / "residues" / f"{pdb_id.lower()}_{pocket_id}_res.pdb"
        _safe_write_text(local_path, pdb_text)
        entry: dict[str, Any] = {"path": str(local_path.as_posix())}
        if embed_text_in_json:
            entry["pdb_text"] = pdb_text
        residues_local[pocket_id] = entry

    # pocket grids (binary .ccp4.gz) can be large; optional
    pocket_grids_local: dict[str, dict[str, Any]] = {}
    if download_pocket_grids:
        grids: list[str] = [str(u) for u in (job_payload.get("pockets") or [])]
        for url in grids:
            m = re.search(r"_P_(\d+)_gpsAll\.ccp4\.gz$", url)
            if not m:
                continue
            pocket_id = f"P_{int(m.group(1))}"
            data = _fetch_bytes(session, url)
            local_path = job_dir / "pocket_grids" / f"{pdb_id.lower()}_{pocket_id}_gpsAll.ccp4.gz"
            _safe_write_bytes(local_path, data)
            pocket_grids_local[pocket_id] = {"path": str(local_path.as_posix())}

    offline_job: dict[str, Any] = {
        "proteins_plus_job_id": job_id,
        "parameters": job_payload.get("parameters"),
        "analysis": {
            "desc_path": str(desc_path.as_posix()),
            "desc_txt": desc_txt if embed_text_in_json else None,
        },
        "descriptor_explanation": {
            "path": str(descriptor_path.as_posix()) if descriptor_path else None,
            "txt": descriptor_txt if (descriptor_txt is not None and embed_text_in_json) else None,
        },
        "residues": residues_local,
        "pocket_grids": pocket_grids_local,
    }

    return offline_job


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv_pdb_ids(csv_path: Path, pdb_column: str | None) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header row found in {csv_path}")

        fieldnames = list(reader.fieldnames)
        if pdb_column is None:
            candidates = [
                "PDB_ID",
                "pdb_id",
                "pdb",
                "PDB",
                "pdbid",
                "PDBID",
            ]
            found = next((c for c in candidates if c in fieldnames), None)
            if not found:
                raise ValueError(
                    f"Could not auto-detect PDB id column. Available columns: {fieldnames}. "
                    "Use --pdb-column to specify it."
                )
            pdb_column = found

        pdb_ids: list[str] = []
        for row in reader:
            raw = (row.get(pdb_column) or "").strip()
            if not raw:
                continue
            pdb_ids.append(raw.upper())

    # preserve order, drop duplicates
    seen: set[str] = set()
    out: list[str] = []
    for pid in pdb_ids:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    return s


def dogsite_submit_job(
    session: requests.Session,
    pdb_id: str,
    analysis_detail: str,
    granularity: str,
    chain: str,
    ligand: str,
    *,
    submit_retries: int = 6,
    initial_backoff_s: float = 15.0,
) -> str:
    payload = {
        "dogsite": {
            "pdbCode": pdb_id.lower(),
            "analysisDetail": analysis_detail,
            "bindingSitePredictionGranularity": granularity,
            "ligand": ligand,
            "chain": chain,
        }
    }

    backoff = float(initial_backoff_s)
    last_error: str | None = None
    last_status: int | None = None
    for attempt in range(submit_retries + 1):
        r = session.post(DOGSITE_CREATE_URL, json=payload, timeout=60)
        last_status = r.status_code
        if r.status_code in (200, 202):
            data = r.json()
            loc = data.get("location")
            if not loc:
                raise DogSiteError(f"DoGSite submit returned no location for {pdb_id}: {data}")
            return str(loc)

        if r.status_code == 429:
            last_error = r.text
            # proteins.plus does not reliably return Retry-After; be conservative
            sleep_s = min(180.0, max(backoff, 30.0))
            time.sleep(sleep_s)
            backoff = min(180.0, backoff * 1.6)
            continue

        # transient server hiccups
        if 500 <= r.status_code < 600:
            last_error = r.text
            time.sleep(min(60.0, max(5.0, backoff)))
            backoff = min(60.0, backoff * 1.4)
            continue

        raise DogSiteError(f"DoGSite submit failed for {pdb_id}: HTTP {r.status_code}: {r.text[:300]}")

    raise DogSiteError(
        f"DoGSite submit failed for {pdb_id} after retries (last HTTP {last_status}): {str(last_error)[:300]}"
    )


def dogsite_poll_job(session: requests.Session, location: str, timeout_s: int, poll_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    delay = poll_s

    while True:
        if time.time() > deadline:
            raise DogSiteError(f"Timed out waiting for DoGSite job: {location}")

        r = session.get(location, timeout=60)
        if r.status_code == 429:
            # backoff when throttled
            time.sleep(min(60.0, max(5.0, delay)))
            delay = min(60.0, delay * 1.5)
            continue
        if r.status_code not in (200, 202):
            raise DogSiteError(f"DoGSite poll failed: HTTP {r.status_code}: {r.text[:300]}")

        data = r.json()
        status_code = data.get("status_code")
        if status_code == 200:
            return data
        if status_code == 202:
            time.sleep(delay)
            continue

        # 4xx in payload
        msg = data.get("message") or data.get("error") or str(data)
        raise DogSiteError(f"DoGSite job failed: {msg}")


def _fetch_text(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def _fetch_bytes(session: requests.Session, url: str) -> bytes:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def parse_dogsite_desc_table(desc_txt: str) -> dict[str, Pocket]:
    """Parse the *_desc.txt table.

    Format observed (April 2026): whitespace-separated table with a header line.
    First column is pocket name like "P_0", and one of the columns is "volume".

    Returns pocket_id -> Pocket(volume_A3).
    """
    lines = [ln.strip() for ln in desc_txt.splitlines() if ln.strip()]
    if not lines:
        raise DogSiteError("Empty desc.txt")

    header = re.split(r"\s+", lines[0])
    try:
        name_i = header.index("name")
        volume_i = header.index("volume")
    except ValueError as e:
        raise DogSiteError(f"Unexpected desc.txt header (missing expected columns): {header}") from e

    pockets: dict[str, Pocket] = {}
    for ln in lines[1:]:
        cols = re.split(r"\s+", ln)
        if len(cols) <= max(name_i, volume_i):
            continue
        pid = cols[name_i]
        if not pid.startswith("P_"):
            continue
        try:
            vol = float(cols[volume_i])
        except ValueError:
            continue
        pockets[pid] = Pocket(pocket_id=pid, volume_A3=vol)

    return pockets


_CENTER_RE = re.compile(
    r"Geometric pocket center at\s+"  # prefix
    r"([-+]?\d+(?:\.\d+)?)\s+"  # x
    r"([-+]?\d+(?:\.\d+)?)\s+"  # y
    r"([-+]?\d+(?:\.\d+)?)"  # z
)


def parse_pocket_center_from_residue_pdb(pdb_text: str) -> tuple[float, float, float]:
    for ln in pdb_text.splitlines()[:50]:
        m = _CENTER_RE.search(ln)
        if m:
            return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    raise DogSiteError("Could not find 'Geometric pocket center' in residue PDB header")


def download_rcsb_pdb(session: requests.Session, pdb_id: str, out_path: Path) -> None:
    url = RCSB_PDB_URL.format(pdb_id=pdb_id.upper())
    r = session.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download PDB {pdb_id} from RCSB: HTTP {r.status_code}")
    out_path.write_bytes(r.content)


def iter_chain_atom_coords_from_pdb_lines(lines: Iterable[str]) -> dict[str, list[tuple[float, float, float]]]:
    chains: dict[str, list[tuple[float, float, float]]] = {}
    for ln in lines:
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        if len(ln) < 54:
            continue
        chain_id = ln[21:22]  # column 22 (1-based)
        try:
            x = float(ln[30:38])
            y = float(ln[38:46])
            z = float(ln[46:54])
        except ValueError:
            continue
        chains.setdefault(chain_id, []).append((x, y, z))
    return chains


def iter_atom_coords_from_pdb_text(pdb_text: str) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    for ln in pdb_text.splitlines():
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        if len(ln) < 54:
            continue
        try:
            x = float(ln[30:38])
            y = float(ln[38:46])
            z = float(ln[46:54])
        except ValueError:
            continue
        coords.append((x, y, z))
    return coords


def choose_chain_pair(chains: dict[str, list[tuple[float, float, float]]], chain_a: str | None, chain_b: str | None) -> tuple[str, str]:
    available = sorted(chains.keys())
    if chain_a and chain_b:
        if chain_a not in chains or chain_b not in chains:
            raise ValueError(f"Requested chains {chain_a},{chain_b} not found. Available: {available}")
        if chain_a == chain_b:
            raise ValueError("--chain-a and --chain-b must be different")
        return chain_a, chain_b

    # auto: pick top-2 by atom count
    ranked = sorted(chains.items(), key=lambda kv: len(kv[1]), reverse=True)
    if len(ranked) < 2:
        raise ValueError(f"Need at least 2 chains in structure, found {available}")
    return ranked[0][0], ranked[1][0]


def min_distance(point: tuple[float, float, float], coords: list[tuple[float, float, float]]) -> float:
    px, py, pz = point
    best = float("inf")
    for (x, y, z) in coords:
        dx = x - px
        dy = y - py
        dz = z - pz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best:
            best = d2
    return math.sqrt(best) if best != float("inf") else float("inf")


def min_distance_sq(point: tuple[float, float, float], coords: list[tuple[float, float, float]]) -> float:
    px, py, pz = point
    best = float("inf")
    for (x, y, z) in coords:
        dx = x - px
        dy = y - py
        dz = z - pz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best:
            best = d2
    return best


def count_pocket_atoms_per_chain(
    pocket_atom_coords: list[tuple[float, float, float]],
    chain_a_coords: list[tuple[float, float, float]],
    chain_b_coords: list[tuple[float, float, float]],
    match_tolerance_A: float,
) -> tuple[int, int]:
    """Count how many pocket atoms belong to each chain.

    We assume the DoGSite pocket residue PDB contains coordinates copied from the
    original structure; we therefore map each pocket atom to the closest chain
    if it matches within a small tolerance.

    Returns (n_atoms_chain_a, n_atoms_chain_b).
    """
    tol2 = float(match_tolerance_A) ** 2
    n_a = 0
    n_b = 0
    for p in pocket_atom_coords:
        da2 = min_distance_sq(p, chain_a_coords)
        db2 = min_distance_sq(p, chain_b_coords)
        if da2 <= tol2:
            n_a += 1
        if db2 <= tol2:
            n_b += 1
    return n_a, n_b


def add_du_markers_to_pdb(
    pdb_bytes: bytes,
    centers: list[tuple[float, float, float]],
    out_path: Path,
    du_chain: str = "Z",
    du_resname: str = "DU",
) -> None:
    """Append DU HETATM records before END/ENDMDL.

    Uses a fixed element "DU". PDB serial numbers continue from the max existing.
    """
    text = pdb_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=False)

    max_serial = 0
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 11:
            try:
                serial = int(ln[6:11])
            except ValueError:
                continue
            max_serial = max(max_serial, serial)

    insert_at = len(lines)
    for i, ln in enumerate(lines):
        # Keep DU atoms inside the coordinate section.
        # Typical PDB ordering is: ATOM/HETATM ... MASTER END
        if ln.startswith(("MASTER", "ENDMDL", "END")):
            insert_at = i
            break

    new_lines: list[str] = []
    new_lines.extend(lines[:insert_at])

    serial = max_serial
    res_seq = 1
    for (x, y, z) in centers:
        serial += 1
        # PDB fixed-width fields
        # Columns: 1-6 record, 7-11 serial, 13-16 atom name, 18-20 resname, 22 chain,
        # 23-26 resseq, 31-38 x, 39-46 y, 47-54 z, 55-60 occ, 61-66 temp, 77-78 element
        line = (
            f"HETATM{serial:5d}  DU  {du_resname:>3s} {du_chain:1s}{res_seq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          DU"
        )
        new_lines.append(line)
        res_seq += 1

    new_lines.extend(lines[insert_at:])
    out_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _pocket_index(pocket_id: str) -> int:
    # pocket_id is "P_<int>"
    m = re.match(r"^P_(\d+)$", pocket_id)
    if not m:
        raise ValueError(f"Unexpected pocket id: {pocket_id}")
    return int(m.group(1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run DoGSiteScorer for PDBs in a CSV and extract interface cavities (>350 A^3) between two chains.",
    )
    ap.add_argument("--csv", dest="csv_path", default="complex_no_ligand_pdb.csv", help="Input CSV containing PDB IDs")
    ap.add_argument("--pdb-column", default=None, help="Column name holding PDB IDs (default: auto-detect)")
    ap.add_argument(
        "--pdb-id",
        default=None,
        help="Run a single PDB ID (overrides --csv), e.g. 2FLU",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N PDB IDs (after de-duplication)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Process at most N PDB IDs per run (recommended: 50)",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=None,
        help="1-based batch number; uses --batch-size to select a slice (e.g. batch 2 of size 50 => items 51-100)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the PDB IDs that would be processed and exit",
    )

    ap.add_argument("--min-volume", type=float, default=350.0, help="Minimum pocket volume (A^3)")
    ap.add_argument(
        "--center-dist-threshold",
        type=float,
        default=10.0,
        help="Pocket center must be within this distance (A) to BOTH chains",
    )
    ap.add_argument(
        "--min-pocket-atoms-per-chain",
        type=int,
        default=1,
        help="Require at least this many pocket binding-site atoms from EACH chain",
    )
    ap.add_argument(
        "--pocket-atom-match-tolerance",
        type=float,
        default=0.25,
        help="Atom coordinate tolerance (A) for mapping pocket residue atoms back to chains",
    )

    ap.add_argument("--chain-a", default=None, help="Override chain A (single-character PDB chain ID)")
    ap.add_argument("--chain-b", default=None, help="Override chain B (single-character PDB chain ID)")

    ap.add_argument(
        "--analysis-detail",
        default="0",
        choices=["0", "1"],
        help='DoGSite analysisDetail: "0" pockets only; "1" pockets + subpockets (default: 0)',
    )
    ap.add_argument(
        "--granularity",
        default="0",
        choices=["0", "1"],
        help='DoGSite bindingSitePredictionGranularity: "0" properties; "1" properties + druggability (default: 0)',
    )
    ap.add_argument("--dogsite-chain", default="", help="DoGSite chain parameter (default: all chains)")
    ap.add_argument("--dogsite-ligand", default="", help="DoGSite ligand parameter (default: none)")

    ap.add_argument("--timeout", type=int, default=900, help="Timeout per PDB job (seconds)")
    ap.add_argument("--poll", type=float, default=5.0, help="Polling interval for job status (seconds)")
    ap.add_argument(
        "--delay-between-jobs",
        type=float,
        default=60.0,
        help="Polite delay between DoGSite job submissions (seconds)",
    )
    ap.add_argument(
        "--delay-betweenjobs",
        dest="delay_between_jobs",
        type=float,
        help="Alias for --delay-between-jobs",
    )
    ap.add_argument(
        "--dealy-between-jobs",
        dest="delay_between_jobs",
        type=float,
        help="Common typo alias for --delay-between-jobs",
    )
    ap.add_argument(
        "--download-pocket-grids",
        action="store_true",
        help="Download CCP4 grid files locally (can be large)",
    )
    ap.add_argument(
        "--embed-text-in-json",
        action="store_true",
        help="Embed desc.txt and residue PDB texts directly into JSON (increases JSON size)",
    )
    ap.add_argument(
        "--submit-retries",
        type=int,
        default=6,
        help="Retries for DoGSite job submission when throttled (HTTP 429)",
    )
    ap.add_argument(
        "--submit-backoff",
        type=float,
        default=15.0,
        help="Initial backoff (seconds) used when HTTP 429 is returned",
    )

    ap.add_argument("--out-dir", default="dogsite_out", help="Output directory")
    ap.add_argument("--force", action="store_true", help="Re-run even if outputs exist")

    ap.add_argument("--du-chain", default="Z", help="Chain ID used for DU marker atoms")

    ns = ap.parse_args(argv)

    csv_path = Path(ns.csv_path)
    out_dir = Path(ns.out_dir)
    _ensure_dir(out_dir)

    json_dir = out_dir / "json"
    pdb_dir = out_dir / "pdb_with_du"
    cache_dir = out_dir / "cache"
    _ensure_dir(json_dir)
    _ensure_dir(pdb_dir)
    _ensure_dir(cache_dir)

    if ns.pdb_id:
        pdb_ids = [str(ns.pdb_id).strip().upper()]
    else:
        pdb_ids = _read_csv_pdb_ids(csv_path, ns.pdb_column)

    if ns.limit is not None:
        pdb_ids = pdb_ids[: max(0, int(ns.limit))]

    # Batch slicing (for polite daytime runs)
    if ns.batch is not None:
        if ns.batch_size is None:
            raise SystemExit("--batch requires --batch-size (suggested: 50)")
        if int(ns.batch) < 1:
            raise SystemExit("--batch must be >= 1")
        bs = max(1, int(ns.batch_size))
        start = (int(ns.batch) - 1) * bs
        end = start + bs
        pdb_ids = pdb_ids[start:end]
        print(f"Batch selection: batch={ns.batch} batch_size={bs} (0-based slice {start}:{end})")
    elif ns.batch_size is not None:
        pdb_ids = pdb_ids[: max(0, int(ns.batch_size))]

    if ns.dry_run:
        print("PDB IDs to process:")
        for pid in pdb_ids:
            print(pid)
        return 0
    if not pdb_ids:
        raise SystemExit("No PDB IDs found in CSV")

    session = _requests_session()

    for pdb_id in pdb_ids:
        out_json = json_dir / f"{pdb_id}.json"
        out_pdb_du = pdb_dir / f"{pdb_id}_DU.pdb"
        out_pdb_nodu = pdb_dir / f"{pdb_id}_noDU.pdb"
        pdb_file = cache_dir / f"{pdb_id}.pdb"

        if not ns.force and out_json.exists() and (out_pdb_du.exists() or out_pdb_nodu.exists()):
            print(f"{pdb_id}: skipping (already exists)")
            continue

        print(f"{pdb_id}: submitting DoGSite job...")
        try:
            loc = dogsite_submit_job(
                session=session,
                pdb_id=pdb_id,
                analysis_detail=ns.analysis_detail,
                granularity=ns.granularity,
                chain=ns.dogsite_chain,
                ligand=ns.dogsite_ligand,
                submit_retries=int(ns.submit_retries),
                initial_backoff_s=float(ns.submit_backoff),
            )
            job = dogsite_poll_job(session=session, location=loc, timeout_s=ns.timeout, poll_s=ns.poll)

            # Localize all DoGSite URLs so JSON doesn't depend on expiring links
            offline_job = localize_dogsite_job_assets(
                session=session,
                pdb_id=pdb_id,
                job_location=loc,
                job_payload=job,
                out_root=out_dir,
                download_pocket_grids=bool(ns.download_pocket_grids),
                embed_text_in_json=bool(ns.embed_text_in_json),
            )

            # parse pocket volumes from localized desc
            desc_txt = offline_job["analysis"]["desc_txt"]
            if not desc_txt:
                # not embedded => read from disk
                desc_txt = Path(offline_job["analysis"]["desc_path"]).read_text(encoding="utf-8")
            pockets = parse_dogsite_desc_table(str(desc_txt))

            # download original pdb for chain selection and distances
            if not pdb_file.exists() or ns.force:
                download_rcsb_pdb(session, pdb_id, pdb_file)
            pdb_lines = pdb_file.read_text(encoding="utf-8", errors="replace").splitlines()
            chain_coords = iter_chain_atom_coords_from_pdb_lines(pdb_lines)
            chain_a, chain_b = choose_chain_pair(chain_coords, ns.chain_a, ns.chain_b)

            # filter by volume, then compute centers and distances
            vol_cut = float(ns.min_volume)
            dist_cut = float(ns.center_dist_threshold)
            min_atoms_per_chain = max(1, int(ns.min_pocket_atoms_per_chain))
            match_tol = float(ns.pocket_atom_match_tolerance)

            residues_by_pocket: dict[str, Path] = {}
            for pid, entry in (offline_job.get("residues") or {}).items():
                p = entry.get("path")
                if p:
                    residues_by_pocket[str(pid)] = Path(p)

            selected: list[Pocket] = []
            pocket_summaries: dict[str, Any] = {}

            for pid, p in sorted(pockets.items(), key=lambda kv: _pocket_index(kv[0])):
                pocket_summaries[pid] = {
                    "volume_A3": p.volume_A3,
                    "selected": False,
                    "reasons": [],
                }
                if p.volume_A3 < vol_cut:
                    pocket_summaries[pid]["reasons"].append(f"volume<{vol_cut}")
                    continue

                res_path = residues_by_pocket.get(pid)
                if not res_path:
                    pocket_summaries[pid]["reasons"].append("missing_residue_pdb")
                    continue

                res_pdb_text: str
                if ns.embed_text_in_json and "pdb_text" in (offline_job.get("residues") or {}).get(pid, {}):
                    res_pdb_text = str((offline_job.get("residues") or {}).get(pid, {}).get("pdb_text"))
                else:
                    res_pdb_text = res_path.read_text(encoding="utf-8", errors="replace")
                center = parse_pocket_center_from_residue_pdb(res_pdb_text)

                pocket_atom_coords = iter_atom_coords_from_pdb_text(res_pdb_text)
                n_a, n_b = count_pocket_atoms_per_chain(
                    pocket_atom_coords=pocket_atom_coords,
                    chain_a_coords=chain_coords[chain_a],
                    chain_b_coords=chain_coords[chain_b],
                    match_tolerance_A=match_tol,
                )

                da = min_distance(center, chain_coords[chain_a])
                db = min_distance(center, chain_coords[chain_b])

                pocket = Pocket(pocket_id=pid, volume_A3=p.volume_A3, center=center, min_dist_chain_a=da, min_dist_chain_b=db)

                pocket_summaries[pid].update(
                    {
                        "center": {"x": center[0], "y": center[1], "z": center[2]},
                        "min_dist_chain_a_A": da,
                        "min_dist_chain_b_A": db,
                        "n_pocket_atoms_chain_a": n_a,
                        "n_pocket_atoms_chain_b": n_b,
                    }
                )

                if not (da <= dist_cut and db <= dist_cut):
                    pocket_summaries[pid]["reasons"].append(f"center_not_close_to_both_chains(<= {dist_cut}A)")
                    continue

                if n_a < min_atoms_per_chain or n_b < min_atoms_per_chain:
                    pocket_summaries[pid]["reasons"].append(
                        f"not_all_chains_involved(min_atoms_per_chain={min_atoms_per_chain})"
                    )
                    continue

                pocket_summaries[pid]["selected"] = True
                selected.append(pocket)

            centers = [p.center for p in selected if p.center is not None]
            centers = [c for c in centers if c is not None]

            du_written = bool(centers)
            if centers:
                add_du_markers_to_pdb(
                    pdb_bytes=pdb_file.read_bytes(),
                    centers=centers,
                    out_path=out_pdb_du,
                    du_chain=str(ns.du_chain)[:1],
                )
                out_pdb_path = out_pdb_du
            else:
                if out_pdb_du.exists():
                    out_pdb_du.unlink()
                # Keep a local copy for inspection, but avoid naming it *_DU.pdb
                # so downstream DU-dependent analysis won't pick it up.
                out_pdb_nodu.write_bytes(pdb_file.read_bytes())
                out_pdb_path = out_pdb_nodu

            payload_out = {
                "pdb_id": pdb_id,
                "output_pdb": {
                    "path": str(out_pdb_path.as_posix()),
                    "du_written": int(du_written),
                    "du_count": int(len(centers)),
                },
                "dogsite": offline_job,
                "interface_filter": {
                    "chain_a": chain_a,
                    "chain_b": chain_b,
                    "min_volume_A3": vol_cut,
                    "center_dist_threshold_A": dist_cut,
                    "min_pocket_atoms_per_chain": min_atoms_per_chain,
                    "pocket_atom_match_tolerance_A": match_tol,
                    "selected_pockets": [p.pocket_id for p in selected],
                },
                "pockets": pocket_summaries,
            }
            out_json.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")

            if du_written:
                print(f"{pdb_id}: done. selected={len(selected)} du_markers={len(centers)}")
            else:
                print(f"{pdb_id}: done. selected={len(selected)} (no DU written)")

        except Exception as e:
            # still write an error JSON for bookkeeping
            err = {
                "pdb_id": pdb_id,
                "error": str(e),
                "type": type(e).__name__,
            }
            out_json.write_text(json.dumps(err, indent=2), encoding="utf-8")
            print(f"{pdb_id}: ERROR: {e}")

        # be polite even after errors
        if ns.delay_between_jobs and float(ns.delay_between_jobs) > 0:
            time.sleep(float(ns.delay_between_jobs))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
