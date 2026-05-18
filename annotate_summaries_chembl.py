#!/usr/bin/env python3
"""annotate_summaries_chembl.py

Annotate many postprocessed summary CSV files with protein UniProt IDs (from PDB)
and ChEMBL binding activities (type B, confidence 9) for any PubChem-hit molecules.

Typical usage
-------------
  python annotate_summaries_chembl.py --csv-dir postprocessed --out-dir postprocessed_chembl

Notes
-----
- PDB ID is extracted from the first 4 characters of the CSV filename by default.
  You can override with --pdb-from-column if your CSV has a column containing a
  PDB-like identifier.
- Protein mapping uses PDBe's UniProt mappings endpoint.
- ChEMBL access uses the public EBI ChEMBL REST API.
- This script is best-effort and uses an on-disk JSON cache to avoid re-querying.

Output
------
Adds columns (at end of each row):
- pdb_id
- uniprot_ids
- chembl_target_ids
- chembl_molecule_id
- chembl_max_pchembl
- chembl_n_pchembl
- chembl_best_target_id

By default, only rows with a non-empty pubchem_cid (or is_pubchem_hit==1) are
queried/mapped to ChEMBL, to keep the run practical.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PDBe_UNIPROT_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Tunable at runtime via CLI (see main()).
HTTP_TIMEOUT_S = 20
HTTP_RETRIES = 2


# -------------------------
# Small HTTP / JSON helpers
# -------------------------

def _http_get_json(url: str, timeout_s: Optional[int] = None) -> Any:
    timeout_s = int(timeout_s if timeout_s is not None else HTTP_TIMEOUT_S)
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "pdb-ligand-annotator/1.0 (local script)",
        },
        method="GET",
    )

    last_err: Optional[Exception] = None
    retries = max(0, int(HTTP_RETRIES))
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
        except urllib.error.HTTPError as e:
            last_err = e
            # Treat 5xx as transient; retry.
            if 500 <= int(getattr(e, "code", 0) or 0) < 600 and attempt < retries:
                time.sleep(0.5 * (2**attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.5 * (2**attempt))
                continue
            raise

    # Should be unreachable.
    raise RuntimeError(f"HTTP failed after retries: {url} ({last_err})")


def _chembl_url(resource: str, params: Dict[str, str]) -> str:
    # ChEMBL supports both /resource and /resource.json.
    base = f"{CHEMBL_API_BASE}/{resource}.json"
    return base + "?" + urllib.parse.urlencode(params)


def _chembl_get_all(resource: str, params: Dict[str, str], list_key: str) -> List[Dict[str, Any]]:
    # Paginate using 'offset' if required.
    limit = int(params.get("limit", "1000"))
    offset = int(params.get("offset", "0"))
    out: List[Dict[str, Any]] = []

    while True:
        p = dict(params)
        p["limit"] = str(limit)
        p["offset"] = str(offset)
        url = _chembl_url(resource, p)
        data = _http_get_json(url)
        chunk = data.get(list_key, []) if isinstance(data, dict) else []
        if not chunk:
            break
        out.extend(chunk)

        page_meta = data.get("page_meta") if isinstance(data, dict) else None
        if not page_meta:
            break
        if not page_meta.get("next"):
            break

        # ChEMBL's page_meta may already include next/prev URLs, but offset/limit works.
        offset += limit

    return out


# -------------------------
# Cache
# -------------------------

@dataclass
class Cache:
    path: Path
    data: Dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "Cache":
        if path.exists():
            try:
                return cls(path=path, data=json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                return cls(path=path, data={})
        return cls(path=path, data={})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


# -------------------------
# PDBe mapping: PDB -> UniProt accessions
# -------------------------

def pdb_to_uniprots(pdb_id: str, cache: Cache) -> List[str]:
    pdb_id = pdb_id.lower()
    key = f"pdbe_uniprot:{pdb_id}"
    if key in cache.data:
        return list(cache.data[key])

    try:
        url = PDBe_UNIPROT_URL.format(pdb_id=pdb_id)
        data = _http_get_json(url)
    except Exception:
        cache.data[key] = []
        return []

    # Response shape: { "1abc": { "UniProt": { "P12345": {...}, ... } } }
    uniprots: List[str] = []
    try:
        block = data.get(pdb_id, {})
        up = block.get("UniProt", {})
        uniprots = sorted(up.keys())
    except Exception:
        uniprots = []

    cache.data[key] = uniprots
    return uniprots


# -------------------------
# ChEMBL mapping
# -------------------------

def uniprot_to_chembl_targets(uniprot: str, cache: Cache) -> List[str]:
    uniprot = uniprot.strip()
    # Use the search endpoint because certain target_components filters can
    # trigger 5xx responses on the public API.
    key = f"chembl_targets_for_uniprot_search:{uniprot}"
    if key in cache.data:
        return list(cache.data[key])

    # Important: don't cache failures as empty, because ChEMBL endpoints can be
    # temporarily unstable and return 5xx.
    try:
        targets = _chembl_get_all(
            "target/search",
            params={
                "q": uniprot,
                "limit": "100",
            },
            list_key="targets",
        )
        ids = sorted({t.get("target_chembl_id") for t in targets if t.get("target_chembl_id")})
    except Exception:
        # No caching on failure.
        return []

    cache.data[key] = ids
    return ids


def inchikey_to_chembl_molecule_id(inchikey: str, cache: Cache) -> Optional[str]:
    inchikey = inchikey.strip()
    if not inchikey:
        return None
    key = f"chembl_mol_for_inchikey:{inchikey}"
    if key in cache.data:
        return cache.data[key] or None

    # Important: don't cache failures as None, because ChEMBL endpoints can be
    # temporarily unstable and return 5xx.
    try:
        mols = _chembl_get_all(
            "molecule",
            params={
                "molecule_structures__standard_inchi_key": inchikey,
                "limit": "20",
            },
            list_key="molecules",
        )
    except Exception:
        # No caching on failure.
        return None
    mol_id: Optional[str] = None
    for m in mols:
        cid = m.get("molecule_chembl_id")
        if cid:
            mol_id = str(cid)
            break

    cache.data[key] = mol_id
    return mol_id


def _confidence_is_9(activity: Dict[str, Any]) -> bool:
    # Different ChEMBL API versions expose this under different keys.
    for k in ("target_confidence_score", "confidence_score"):
        v = activity.get(k)
        if v is None:
            continue
        try:
            return int(v) == 9
        except Exception:
            return False
    # If field isn't present, we can't verify; treat as not matching.
    return False


def molecule_vs_targets_pchembl(
    molecule_chembl_id: str,
    target_ids: List[str],
    cache: Cache,
    *,
    assay_type: str = "B",
    require_confidence_9: bool = True,
) -> Tuple[Optional[float], int, Optional[str]]:
    """Return (max_pchembl, n_values, best_target_id)."""
    if not molecule_chembl_id or not target_ids:
        return None, 0, None

    # Cache per molecule+sorted(targets) because this can be expensive.
    target_key = ";".join(sorted(target_ids))
    key = f"chembl_act:{molecule_chembl_id}:{assay_type}:{int(require_confidence_9)}:{target_key}"
    if key in cache.data:
        val = cache.data[key]
        return val.get("max"), val.get("n", 0), val.get("best_target")

    max_val: Optional[float] = None
    best_target: Optional[str] = None
    n_vals = 0
    had_error = False

    # Query per target to keep URLs simple and cacheable.
    for tid in target_ids:
        try:
            activities = _chembl_get_all(
                "activity",
                params={
                    "molecule_chembl_id": molecule_chembl_id,
                    "target_chembl_id": tid,
                    "assay_type": assay_type,
                    "pchembl_value__isnull": "false",
                    "limit": "1000",
                },
                list_key="activities",
            )
        except Exception:
            had_error = True
            continue

        for a in activities:
            if a.get("assay_type") and str(a.get("assay_type")) != assay_type:
                continue
            if require_confidence_9 and not _confidence_is_9(a):
                continue
            pv = a.get("pchembl_value")
            if pv is None or pv == "":
                continue
            try:
                pvf = float(pv)
            except Exception:
                continue
            n_vals += 1
            if max_val is None or pvf > max_val:
                max_val = pvf
                best_target = tid

    # Only cache a fully-successful computation; otherwise allow reruns to fill in.
    if not had_error:
        cache.data[key] = {"max": max_val, "n": n_vals, "best_target": best_target}
    return max_val, n_vals, best_target


# -------------------------
# CSV plumbing
# -------------------------

def _iter_csvs(csv_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(csv_dir.glob(pattern))


def _extract_pdb_from_filename(path: Path) -> Optional[str]:
    stem = path.stem
    if len(stem) < 4:
        return None
    pdb = stem[:4]
    if all(ch.isalnum() for ch in pdb):
        return pdb.upper()
    return None


def annotate_one_csv(
    csv_path: Path,
    out_path: Path,
    cache: Cache,
    *,
    pdb_from_column: Optional[str],
    only_pubchem_rows: bool,
    require_confidence_9: bool,
) -> Tuple[int, int]:
    """Return (rows_written, rows_queried)."""
    pdb_id_default = _extract_pdb_from_filename(csv_path)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    # Determine which column holds PubChem CID / InChIKey.
    has_cid_col = "pubchem_cid" in fieldnames
    has_hit_col = "is_pubchem_hit" in fieldnames
    has_ik_col = "InChIKey" in fieldnames

    # Add new columns (append only).
    new_cols = [
        "pdb_id",
        "uniprot_ids",
        "chembl_target_ids",
        "chembl_molecule_id",
        "chembl_max_pchembl",
        "chembl_n_pchembl",
        "chembl_best_target_id",
    ]
    for c in new_cols:
        if c not in fieldnames:
            fieldnames.append(c)

    # Resolve UniProt list once per file.
    pdb_id = None
    if pdb_from_column and rows and pdb_from_column in rows[0]:
        v = (rows[0].get(pdb_from_column) or "").strip()
        pdb_id = v[:4].upper() if len(v) >= 4 else None
    if not pdb_id:
        pdb_id = pdb_id_default

    uniprots = pdb_to_uniprots(pdb_id or "", cache) if pdb_id else []
    target_ids: List[str] = []
    for up in uniprots:
        target_ids.extend(uniprot_to_chembl_targets(up, cache))
    target_ids = sorted(set([t for t in target_ids if t]))

    rows_queried = 0

    for r in rows:
        r["pdb_id"] = pdb_id or ""
        r["uniprot_ids"] = ";".join(uniprots)
        r["chembl_target_ids"] = ";".join(target_ids)

        # Decide whether to query for this molecule.
        if only_pubchem_rows:
            is_hit = False
            if has_hit_col:
                is_hit = str(r.get("is_pubchem_hit") or "").strip() in ("1", "True", "true", "YES", "yes")
            if has_cid_col and str(r.get("pubchem_cid") or "").strip():
                is_hit = True
            if not is_hit:
                continue

        inchikey = (r.get("InChIKey") or "").strip() if has_ik_col else ""
        if not inchikey:
            continue

        mol_chembl_id = inchikey_to_chembl_molecule_id(inchikey, cache)
        r["chembl_molecule_id"] = mol_chembl_id or ""
        if not mol_chembl_id or not target_ids:
            continue

        rows_queried += 1
        max_pchembl, n_vals, best_target = molecule_vs_targets_pchembl(
            mol_chembl_id,
            target_ids,
            cache,
            assay_type="B",
            require_confidence_9=require_confidence_9,
        )
        r["chembl_max_pchembl"] = "" if max_pchembl is None else f"{max_pchembl:.3f}"
        r["chembl_n_pchembl"] = str(n_vals)
        r["chembl_best_target_id"] = best_target or ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), rows_queried


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Annotate postprocessed *_summary.csv files with PDB->UniProt mappings and "
            "ChEMBL type-B, confidence-9 binding activities (pChEMBL values) for PubChem-hit molecules."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv-dir", type=Path, default=Path("postprocessed_ph405"), help="Folder containing summary CSV files")
    ap.add_argument("--pattern", default="*_summary.csv", help="Glob pattern under --csv-dir")
    ap.add_argument("--out-dir", type=Path, default=Path("postprocessed_chembl"), help="Output folder for annotated CSVs")
    ap.add_argument("--in-place", action="store_true", help="Overwrite the input CSVs (writes to temp then replaces)")
    ap.add_argument("--cache", type=Path, default=Path("chembl_annotation_cache.json"), help="JSON cache path")
    ap.add_argument(
        "--pdb-from-column",
        default=None,
        help=(
            "Optional column name to read a PDB-like identifier from (first 4 chars). "
            "If omitted, uses first 4 characters of the filename."
        ),
    )
    ap.add_argument(
        "--only-pubchem-rows",
        action="store_true",
        help="Only annotate rows that have pubchem_cid or is_pubchem_hit==1 (faster)",
    )
    ap.add_argument(
        "--annotate-all-rows",
        action="store_true",
        help="Annotate all rows with an InChIKey (slower; may increase API calls)",
    )
    ap.add_argument(
        "--no-require-confidence-9",
        action="store_true",
        help=(
            "Do not require confidence==9 when filtering ChEMBL activities. "
            "Use this if your ChEMBL API responses do not expose a confidence field."
        ),
    )
    ap.add_argument("--save-cache-every", type=int, default=50, help="Save cache to disk every N files")
    ap.add_argument("--http-timeout", type=int, default=20, help="HTTP timeout (seconds) for PDBe/ChEMBL requests")
    ap.add_argument("--http-retries", type=int, default=2, help="Retry count for transient HTTP failures")

    args = ap.parse_args(argv)

    global HTTP_TIMEOUT_S, HTTP_RETRIES
    HTTP_TIMEOUT_S = int(args.http_timeout)
    HTTP_RETRIES = int(args.http_retries)

    csv_dir: Path = args.csv_dir
    if not csv_dir.exists():
        print(f"ERROR: --csv-dir not found: {csv_dir}", file=sys.stderr)
        return 2

    cache = Cache.load(args.cache)

    only_pubchem_rows = True
    if args.annotate_all_rows:
        only_pubchem_rows = False
    if args.only_pubchem_rows:
        only_pubchem_rows = True

    require_conf_9 = not args.no_require_confidence_9

    files = list(_iter_csvs(csv_dir, args.pattern))
    if not files:
        print(f"No CSVs matched {args.pattern} in {csv_dir}")
        return 0

    t0 = time.time()
    total_rows = 0
    total_queried = 0

    for i, p in enumerate(files, start=1):
        if args.in_place:
            out_path = p
            tmp_out = p.with_suffix(p.suffix + ".tmp")
        else:
            out_path = args.out_dir / p.name
            tmp_out = out_path

        try:
            n_rows, n_q = annotate_one_csv(
                p,
                tmp_out,
                cache,
                pdb_from_column=args.pdb_from_column,
                only_pubchem_rows=only_pubchem_rows,
                require_confidence_9=require_conf_9,
            )
            total_rows += n_rows
            total_queried += n_q

            if args.in_place and tmp_out != out_path:
                tmp_out.replace(out_path)

            if i % 10 == 0:
                elapsed = time.time() - t0
                print(f"[{i:4d}/{len(files)}] {p.name}: rows={n_rows} queried={n_q}  ({elapsed:.1f}s)")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"WARN: failed to annotate {p.name}: {exc}", file=sys.stderr)

        if i % int(args.save_cache_every) == 0:
            cache.save()

    cache.save()
    elapsed = time.time() - t0
    print(f"\nDone. Files: {len(files)}  Rows: {total_rows}  Molecules queried: {total_queried}  Time: {elapsed:.1f}s")
    print(f"Cache: {args.cache}")

    if not args.in_place:
        print(f"Outputs: {args.out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
