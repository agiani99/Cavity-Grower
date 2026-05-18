#!/usr/bin/env python3
"""
postprocess_designs.py
======================
Post-processing pipeline for SDF files produced by de_novo_cavity_growth.py.

Pipeline
--------
  1. Load one or more SDF files (from different PDBs / DU cavities)
  2. Remove exact duplicates by InChIKey
  3. Compute a full descriptor profile
       MW · LogP · HBD · HBA · TPSA · RotBonds · Rings · ArRings ·
       Fsp3 · HAC · QED · SA · composite (from grower)
  4. Apply optional hard filters (independently combinable):
       --lipinski   MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10
       --veber      RotBonds ≤ 10, TPSA ≤ 140
       --lead       MW ≤ 350, LogP ≤ 3.5, RotBonds ≤ 7  (lead-like space)
       --pains      remove PAINS A/B/C alerts (RDKit FilterCatalog)
       --brenk      remove Brenk structural alerts
  5. Tanimoto sphere-exclusion clustering (Morgan r=2, 2048 bits)
       → one representative per cluster (cluster centroid = best score)
  6. Outputs
       <out>/all_filtered.sdf         all molecules passing hard filters
       <out>/diverse_picks.sdf        one molecule per cluster centre
       <out>/summary.csv              full descriptor + filter + cluster table
       <out>/grid_top{N}.png          2D structure grid (RDKit MolDraw2D)

Dependencies
------------
  pip install rdkit numpy Pillow       # Pillow only needed for --grid

Usage
-----
  # minimal
  python postprocess_designs.py --sdf cavity_designs.sdf

    # batch mode (process every .sdf in a folder; outputs are prefixed per file)
    python postprocess_designs.py --sdf-dir sdf_out/ --out-dir postprocessed/

  # combine outputs from multiple PDB jobs, apply all filters, cluster, draw
  python postprocess_designs.py \\
      --sdf designs_2FLU.sdf designs_4TKL.sdf designs_6XY1.sdf \\
      --lipinski --veber --pains \\
      --qed-min 0.30 --sa-max 4.5 \\
      --cluster-radius 0.60 \\
      --grid-n 64 \\
      --out-dir results/

    # annotate known molecules via PubChem (CID + vendor names) and prioritize them
    python postprocess_designs.py --sdf cavity_designs.sdf --pubchem --pubchem-vendors

  # lead-like space, no rings removed, extra-tight diversity
  python postprocess_designs.py --sdf cavity_designs.sdf \\
      --lead --pains \\
      --cluster-radius 0.50 \\
      --out-dir lead_results/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── RDKit ──────────────────────────────────────────────────────────────────────
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem, Crippen, Descriptors, FilterCatalog,
    QED, rdMolDescriptors, inchi,
)
from rdkit.ML.Cluster import Butina


class PipelineAbort(Exception):
    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = int(code)

# ── SA scorer (same loader as de_novo_cavity_growth.py) ───────────────────────
def _load_sascorer():
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return sascorer
    except ImportError:
        pass
    try:
        from rdkit.Chem import RDConfig
        sys.path.insert(0, os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        return sascorer
    except ImportError:
        return None

_SA = _load_sascorer()


# ──────────────────────────────────────────────────────────────────────────────
# PubChem lookup (optional)
# ──────────────────────────────────────────────────────────────────────────────

def _http_get_json(url: str, timeout_s: float = 20.0) -> Optional[Dict]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "postprocess_designs.py (RDKit)" ,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _load_pubchem_cache(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_pubchem_cache(path: Path, cache: Dict[str, Dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
    except Exception:
        return


def pubchem_cid_from_inchikey(inchikey: str, timeout_s: float = 20.0) -> Optional[int]:
    ik = inchikey.strip()
    if not ik:
        return None
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
        + urllib.parse.quote(ik)
        + "/cids/JSON"
    )
    payload = _http_get_json(url, timeout_s=timeout_s)
    try:
        cids = payload["IdentifierList"]["CID"]
        if cids:
            return int(cids[0])
    except Exception:
        return None
    return None


def pubchem_vendors_from_cid(cid: int, timeout_s: float = 20.0, max_vendors: int = 8) -> List[str]:
    """Best-effort vendor extraction from PUG-View; returns vendor names."""
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/"
        + str(int(cid))
        + "/JSON?heading="
        + urllib.parse.quote("Chemical Vendors")
    )
    payload = _http_get_json(url, timeout_s=timeout_s)
    if not payload:
        return []

    # PUG-View schema is nested; vendor names typically appear as strings in
    # Information -> Value -> StringWithMarkup.
    vendors: List[str] = []
    try:
        record = payload.get("Record", {})
        sections = record.get("Section", [])

        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "String" and isinstance(v, str):
                        vendors.append(v)
                    else:
                        walk(v)
            elif isinstance(obj, list):
                for it in obj:
                    walk(it)

        walk(sections)
    except Exception:
        return []

    # De-dupe while preserving order; trim.
    seen = set()
    uniq: List[str] = []
    for v in vendors:
        vv = v.strip()
        if not vv:
            continue
        if vv in seen:
            continue
        seen.add(vv)
        uniq.append(vv)
        if len(uniq) >= max_vendors:
            break
    return uniq


def pubchem_lookup(
    inchikey: str,
    cache: Dict[str, Dict],
    fetch_vendors: bool,
    timeout_s: float,
    max_vendors: int,
) -> Tuple[Optional[int], List[str]]:
    """Returns (cid, vendors). Uses and updates cache dict."""
    ik = (inchikey or "").strip()
    if not ik or len(ik) < 10:
        return None, []

    hit = cache.get(ik)
    if isinstance(hit, dict) and "cid" in hit:
        cid = hit.get("cid")
        try:
            cid_int = int(cid) if cid else None
        except Exception:
            cid_int = None

        if not fetch_vendors:
            return cid_int, []

        vendors = hit.get("vendors", [])
        if isinstance(vendors, list) and vendors:
            return cid_int, vendors

        # CID is cached but vendors are missing; optionally backfill.
        if cid_int is not None:
            vendors2 = pubchem_vendors_from_cid(cid_int, timeout_s=timeout_s, max_vendors=max_vendors)
            hit["vendors"] = vendors2
            hit["ts"] = int(time.time())
            cache[ik] = hit
            return cid_int, vendors2
        return None, []

    cid = pubchem_cid_from_inchikey(ik, timeout_s=timeout_s)
    vendors: List[str] = []
    if cid is not None and fetch_vendors:
        vendors = pubchem_vendors_from_cid(cid, timeout_s=timeout_s, max_vendors=max_vendors)

    cache[ik] = {
        "cid": int(cid) if cid is not None else None,
        "vendors": vendors,
        "ts": int(time.time()),
    }
    return cid, vendors


# ──────────────────────────────────────────────────────────────────────────────
# Descriptor calculation
# ──────────────────────────────────────────────────────────────────────────────

def compute_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Compute the full descriptor profile for one molecule.
    Returns a flat dict with float / int values.
    """
    mw       = Descriptors.ExactMolWt(mol)
    logp     = Crippen.MolLogP(mol)
    hbd      = rdMolDescriptors.CalcNumHBD(mol)
    hba      = rdMolDescriptors.CalcNumHBA(mol)
    tpsa     = rdMolDescriptors.CalcTPSA(mol)
    rotb     = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings    = rdMolDescriptors.CalcNumRings(mol)
    ar_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    fsp3     = rdMolDescriptors.CalcFractionCSP3(mol)
    hac      = mol.GetNumHeavyAtoms()
    qed      = QED.qed(mol)

    sa: float
    if _SA is not None:
        try:
            sa = float(_SA.calculateScore(mol))
        except Exception:
            sa = float("nan")
    else:
        ri = mol.GetRingInfo()
        n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        sa = min(6.5, 2.0 + 0.25 * ri.NumRings() + 0.6 * n_chiral)

    return {
        "MW":       round(mw,   3),
        "LogP":     round(logp, 3),
        "HBD":      int(hbd),
        "HBA":      int(hba),
        "TPSA":     round(tpsa, 2),
        "RotBonds": int(rotb),
        "Rings":    int(rings),
        "ArRings":  int(ar_rings),
        "Fsp3":     round(fsp3, 3),
        "HAC":      int(hac),
        "QED":      round(qed,  4),
        "SA":       round(sa,   4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Hard filters
# ──────────────────────────────────────────────────────────────────────────────

def lipinski_pass(d: Dict) -> bool:
    """Lipinski Ro5: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10."""
    return d["MW"] <= 500 and d["LogP"] <= 5 and d["HBD"] <= 5 and d["HBA"] <= 10

def veber_pass(d: Dict) -> bool:
    """Veber oral bioavailability: RotBonds ≤ 10, TPSA ≤ 140."""
    return d["RotBonds"] <= 10 and d["TPSA"] <= 140

def lead_pass(d: Dict) -> bool:
    """Lead-likeness: MW ≤ 350, LogP ≤ 3.5, RotBonds ≤ 7."""
    return d["MW"] <= 350 and d["LogP"] <= 3.5 and d["RotBonds"] <= 7

def _build_filter_catalog(use_pains: bool, use_brenk: bool) -> Optional[FilterCatalog.FilterCatalog]:
    if not (use_pains or use_brenk):
        return None
    params = FilterCatalog.FilterCatalogParams()
    if use_pains:
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    if use_brenk:
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


# ──────────────────────────────────────────────────────────────────────────────
# Tanimoto sphere-exclusion clustering (Butina)
# ──────────────────────────────────────────────────────────────────────────────

def morgan_fps(mols: List[Chem.Mol], radius: int = 2, n_bits: int = 2048):
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
        return [gen.GetFingerprint(m) for m in mols]
    except ImportError:
        return [
            AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            for m in mols
        ]


def cluster_by_tanimoto(
    fps,
    cutoff: float = 0.60,
) -> List[Tuple[int, List[int]]]:
    """
    Sphere-exclusion (Butina) clustering on pre-computed Morgan fingerprints.

    *cutoff* is the minimum Tanimoto similarity for two molecules to be
    considered in the same cluster (higher → tighter clusters, more picks).

    Returns a list of (centroid_idx, [member_idx, ...]) tuples, sorted by
    cluster size (largest first).
    """
    n = len(fps)
    if n == 0:
        return []

    # Butina needs upper-triangle distance list (row-major)
    dist_list: List[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dist_list.extend(1.0 - s for s in sims)

    # ClusterData threshold is a *distance* threshold
    raw = Butina.ClusterData(dist_list, n, 1.0 - cutoff, isDistData=True)
    return [(cluster[0], list(cluster)) for cluster in raw]


# ──────────────────────────────────────────────────────────────────────────────
# 2-D grid visualisation
# ──────────────────────────────────────────────────────────────────────────────

def draw_grid(
    mols:       List[Chem.Mol],
    legends:    List[str],
    out_path:   str,
    n_per_row:  int   = 8,
    img_size:   Tuple = (200, 200),
) -> bool:
    """
    Draw 2-D structures to a PNG grid.
    Returns True on success, False if Pillow/Cairo is unavailable.
    """
    try:
        from rdkit.Chem import Draw
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow  = n_per_row,
            subImgSize  = img_size,
            legends     = legends,
            returnPNG   = False,
        )
        img.save(out_path)
        return True
    except Exception as exc:
        print(f"  WARNING: could not save grid image ({exc})")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# SDF I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_sdf_files(paths: List[str]) -> List[Tuple[Chem.Mol, Dict[str, str]]]:
    """
    Load all molecules from a list of SDF paths.
    Returns list of (mol, {prop_name: value_str}) tuples.
    Skips None molecules silently.
    """
    records = []
    n_skipped = 0
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  WARNING: missing SDF file, skipping: {path}")
            n_skipped += 1
            continue
        try:
            if p.stat().st_size == 0:
                print(f"  WARNING: empty SDF file (0 bytes), skipping: {path}")
                n_skipped += 1
                continue
        except OSError:
            print(f"  WARNING: cannot stat SDF file, skipping: {path}")
            n_skipped += 1
            continue

        try:
            sup = Chem.SDMolSupplier(str(p), removeHs=True, sanitize=True)
        except OSError as exc:
            print(f"  WARNING: could not open SDF file, skipping: {path} ({exc})")
            n_skipped += 1
            continue
        for mol in sup:
            if mol is None:
                continue
            props = {k: mol.GetProp(k) for k in mol.GetPropNames()}
            records.append((mol, props))
    msg = f"  Loaded {len(records)} molecules from {len(paths)} SDF file(s)"
    if n_skipped:
        msg += f"  ({n_skipped} file(s) skipped)"
    print(msg)
    return records


def write_sdf(records: List[Tuple[Chem.Mol, Dict]], path: str) -> None:
    w = Chem.SDWriter(path)
    for mol, props in records:
        out = Chem.RWMol(mol)
        for k, v in props.items():
            out.SetProp(k, str(v))
        w.write(out.GetMol())
    w.close()
    print(f"  Wrote {len(records):5d} molecules  →  {path}")


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows):5d} rows         →  {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _gather_sdf_paths_from_dir(sdf_dir: str) -> List[str]:
    p = Path(sdf_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"SDF directory not found: {sdf_dir}")
    sdf_paths = sorted(
        [str(x) for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".sdf"]
    )
    return sdf_paths


def run_one(cfg, sdf_paths: List[str], out_prefix: str = "") -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_prefix.strip()
    prefix = (prefix + "_") if prefix else ""

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1] Loading SDF files …")
    records = load_sdf_files(sdf_paths)
    if not records:
        print("ERROR: no valid molecules loaded.", file=sys.stderr)
        raise PipelineAbort("no valid molecules loaded", code=1)
    n_loaded = len(records)

    # ── 2. Deduplicate by InChIKey ────────────────────────────────────────────
    print("[2] Deduplicating by InChIKey …")
    seen_ikeys: set = set()
    deduped: List[Tuple[Chem.Mol, Dict]] = []
    for mol, props in records:
        try:
            ikey = inchi.MolToInchiKey(mol)
        except Exception:
            ikey = Chem.MolToSmiles(mol)
        if ikey not in seen_ikeys:
            seen_ikeys.add(ikey)
            props["InChIKey"] = ikey
            deduped.append((mol, props))
    print(f"  After dedup: {len(deduped)} unique molecules  "
          f"({len(records) - len(deduped)} removed)")
    records = deduped
    n_dedup = len(records)

    # ── 3. Compute descriptors ────────────────────────────────────────────────
    print("[3] Computing descriptors …")
    desc_list: List[Dict] = []
    valid_records = []
    for mol, props in records:
        try:
            d = compute_descriptors(mol)
            # Prefer composite / QED / SA from SDF if present (grower values)
            if "composite" in props:
                d["composite_grower"] = float(props["composite"])
            else:
                # Re-derive composite from current descriptors
                sa_n = max(0.0, (6.5 - d["SA"]) / 5.5)
                d["composite_grower"] = round(0.40 * d["QED"] + 0.35 * sa_n, 4)
            d["SMILES"]   = props.get("SMILES", Chem.MolToSmiles(mol))
            d["InChIKey"] = props.get("InChIKey", "")
            d["source_id"]= props.get("_Name", "")
            d["pubchem_cid"] = ""
            d["pubchem_vendors"] = ""
            d["is_pubchem_hit"] = 0
            desc_list.append(d)
            valid_records.append((mol, props))
        except Exception as exc:
            print(f"    WARNING: descriptor calculation failed ({exc}); skipping")

    records = valid_records
    print(f"  Descriptors computed for {len(records)} molecules")

    # ── 4. PubChem annotation (optional; before filters so hits can be rescued)
    pubchem_cache: Dict[str, Dict] = {}
    cache_path = Path(cfg.pubchem_cache) if getattr(cfg, "pubchem_cache", None) else (out_dir / "pubchem_cache.json")
    if cfg.pubchem:
        print("[4] PubChem lookup (by InChIKey) …")
        pubchem_cache = _load_pubchem_cache(cache_path)
        n_hit = 0
        for d in desc_list:
            ik = (d.get("InChIKey") or "").strip()
            cid, vendors = pubchem_lookup(
                ik,
                pubchem_cache,
                fetch_vendors=cfg.pubchem_vendors,
                timeout_s=cfg.pubchem_timeout,
                max_vendors=cfg.pubchem_max_vendors,
            )
            if cid is not None:
                n_hit += 1
                d["pubchem_cid"] = str(cid)
                d["is_pubchem_hit"] = 1
                if vendors:
                    d["pubchem_vendors"] = ";".join(vendors)
        _save_pubchem_cache(cache_path, pubchem_cache)
        print(f"  PubChem hits: {n_hit}/{len(desc_list)}")

    # ── 5. Score-based threshold filter (QED / SA from user CLI) ─────────────
    if cfg.qed_min > 0 or cfg.sa_max < 9.9:
        before = len(records)
        filtered = []
        for rec, d in zip(records, desc_list):
            is_hit = int(d.get("is_pubchem_hit", 0)) == 1
            if is_hit and cfg.keep_pubchem_hits:
                filtered.append((rec, d))
                continue
            if d["QED"] >= cfg.qed_min and d["SA"] <= cfg.sa_max:
                filtered.append((rec, d))
        records, desc_list = ([r for r, _ in filtered], [d for _, d in filtered])
        print(f"[5] Score filter  QED≥{cfg.qed_min}  SA≤{cfg.sa_max}: "
              f"{before} → {len(records)}")
    n_after_score = len(records)

    # ── 6. Hard structural filters ────────────────────────────────────────────
    catalog = _build_filter_catalog(cfg.pains, cfg.brenk)
    filter_labels: List[str] = []
    if cfg.lipinski: filter_labels.append("Lipinski")
    if cfg.veber:    filter_labels.append("Veber")
    if cfg.lead:     filter_labels.append("Lead-like")
    if cfg.pains:    filter_labels.append("PAINS")
    if cfg.brenk:    filter_labels.append("Brenk")

    if filter_labels:
        print(f"[6] Hard filters: {', '.join(filter_labels)} …")
        before = len(records)
        kept_recs, kept_desc = [], []
        n_fail_lip = n_fail_veb = n_fail_lead = n_fail_alert = 0
        for (mol, props), d in zip(records, desc_list):
            is_hit = int(d.get("is_pubchem_hit", 0)) == 1
            if is_hit and cfg.keep_pubchem_hits:
                kept_recs.append((mol, props))
                kept_desc.append(d)
                continue
            if cfg.lipinski and not lipinski_pass(d):
                n_fail_lip += 1; continue
            if cfg.veber and not veber_pass(d):
                n_fail_veb += 1; continue
            if cfg.lead and not lead_pass(d):
                n_fail_lead += 1; continue
            if catalog is not None and catalog.HasMatch(mol):
                n_fail_alert += 1; continue
            kept_recs.append((mol, props))
            kept_desc.append(d)

        records, desc_list = kept_recs, kept_desc
        print(f"  {before} → {len(records)} molecules  "
              f"(Ro5:{n_fail_lip} Veber:{n_fail_veb} "
              f"Lead:{n_fail_lead} Alerts:{n_fail_alert} removed)")
    else:
        print("[6] No hard filters applied.")

    n_after_hard = len(records)

    if not records:
        print("  No molecules remain after filtering. "
              "Try relaxing --qed-min / --sa-max or removing filter flags.")
        raise PipelineAbort("no molecules remain after filtering", code=0)

    # ── 7. Sort by priority (PubChem hits first) then composite score ─────────
    order = sorted(
        range(len(records)),
        key=lambda i: (
            int(desc_list[i].get("is_pubchem_hit", 0)),
            float(desc_list[i].get("composite_grower", 0.0)),
        ),
        reverse=True,
    )
    records   = [records[i]   for i in order]
    desc_list = [desc_list[i] for i in order]

    # ── 8. Tanimoto clustering ────────────────────────────────────────────────
    mols = [r[0] for r in records]
    print(f"[7] Clustering {len(mols)} molecules "
          f"(Tanimoto radius = {cfg.cluster_radius}) …")
    fps      = morgan_fps(mols)
    clusters = cluster_by_tanimoto(fps, cutoff=cfg.cluster_radius)
    print(f"  Found {len(clusters)} clusters")

    # Choose one representative per cluster:
    # - Prefer PubChem hits (if any in the cluster)
    # - Otherwise the highest composite (records are already sorted by priority)
    chosen_idxs: List[int] = []
    mol2cluster: Dict[int, int] = {}
    for ci, (_centroid, members) in enumerate(clusters):
        for m in members:
            mol2cluster[m] = ci

        best = None
        for m in members:
            d = desc_list[m]
            key = (
                int(d.get("is_pubchem_hit", 0)),
                float(d.get("composite_grower", 0.0)),
            )
            if best is None or key > best[0]:
                best = (key, m)
        if best is not None:
            chosen_idxs.append(best[1])

    # De-dupe, then sort chosen representatives by the current global priority
    # (after the earlier sort, lower index = higher priority).
    chosen_idxs_sorted = sorted(set(chosen_idxs))
    chosen_idx_set = set(chosen_idxs_sorted)

    for i, d in enumerate(desc_list):
        d["cluster_id"] = mol2cluster.get(i, -1)
        d["is_centroid"] = int(i in chosen_idx_set)

    diverse_records = [records[i] for i in chosen_idxs_sorted]
    diverse_desc = [desc_list[i] for i in chosen_idxs_sorted]

    # ── 9. Write filtered SDF ─────────────────────────────────────────────────
    print("[8] Writing outputs …")
    all_out = []
    for (mol, props), d in zip(records, desc_list):
        merged = {**props, **{k: str(v) for k, v in d.items()}}
        all_out.append((mol, merged))
    write_sdf(all_out, str(out_dir / f"{prefix}all_filtered.sdf"))

    # ── 10. Write diverse picks SDF ───────────────────────────────────────────
    div_out = []
    for (mol, props), d in zip(diverse_records, diverse_desc):
        merged = {**props, **{k: str(v) for k, v in d.items()}}
        div_out.append((mol, merged))
    write_sdf(div_out, str(out_dir / f"{prefix}diverse_picks.sdf"))

    # ── 11. Write summary CSV ─────────────────────────────────────────────────
    col_order = [
        "source_id", "SMILES", "InChIKey",
        "pubchem_cid", "pubchem_vendors", "is_pubchem_hit",
        "composite_grower", "QED", "SA",
        "MW", "LogP", "HBD", "HBA", "TPSA",
        "RotBonds", "Rings", "ArRings", "Fsp3", "HAC",
        "cluster_id", "is_centroid",
    ]
    rows = []
    for i, d in enumerate(desc_list):
        row = {k: d.get(k, "") for k in col_order}
        row["rank"] = i + 1
        rows.append(row)
    write_csv(rows, str(out_dir / f"{prefix}summary.csv"))

    # ── 12. 2-D grid of top-N ─────────────────────────────────────────────────
    grid_n = min(cfg.grid_n, len(diverse_records))
    if grid_n > 0:
        top_mols = [diverse_records[i][0] for i in range(grid_n)]
        top_labs = [
            f"QED={diverse_desc[i]['QED']:.2f}  SA={diverse_desc[i]['SA']:.1f}\n"
            f"MW={diverse_desc[i]['MW']:.0f}  LogP={diverse_desc[i]['LogP']:.1f}"
            for i in range(grid_n)
        ]
        grid_path = str(out_dir / f"{prefix}grid_top{grid_n}.png")
        ok = draw_grid(top_mols, top_labs, grid_path, n_per_row=cfg.grid_cols)
        if ok:
            print(f"  Saved grid ({grid_n} structures)  →  {grid_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Input molecules        : {n_loaded}")
    print(f"  After InChIKey dedup    : {n_dedup}")
    print(f"  After score filter     : {n_after_score}")
    print(f"  After hard filters     : {n_after_hard}")
    print(f"  Clusters found         : {len(clusters)}")
    print(f"  Diverse picks          : {len(diverse_records)}")
    if diverse_desc:
        top = diverse_desc[0]
        print(f"  Top pick  QED={top['QED']:.3f}  SA={top['SA']:.2f}  "
              f"MW={top['MW']:.1f}  LogP={top['LogP']:.2f}")
        print(f"  Top SMILES: {top['SMILES']}")
    print(f"{'='*65}\n")


def run(cfg) -> None:
    # If user passed a folder, run each SDF separately with a filename prefix.
    if cfg.sdf_dir:
        sdf_files = _gather_sdf_paths_from_dir(cfg.sdf_dir)
        if not sdf_files:
            print(f"ERROR: no .sdf files found in {cfg.sdf_dir}", file=sys.stderr)
            raise SystemExit(1)
        print(f"Batch mode: found {len(sdf_files)} SDF files in {cfg.sdf_dir}")
        n_ok = 0
        n_fail = 0
        for sdf_path in sdf_files:
            stem = Path(sdf_path).stem
            print(f"\n{'#'*72}\nProcessing: {sdf_path}\n{'#'*72}")
            try:
                run_one(cfg, [sdf_path], out_prefix=stem)
                n_ok += 1
            except PipelineAbort as exc:
                n_fail += 1
                print(f"  WARNING: skipping {sdf_path} ({exc})")
            except Exception as exc:
                n_fail += 1
                print(f"  WARNING: unexpected error for {sdf_path} ({exc}); skipping")
        print(f"\nBatch summary: {n_ok} succeeded, {n_fail} skipped/failed")
        return

    # Default: process provided --sdf list as one combined job.
    try:
        run_one(cfg, cfg.sdf, out_prefix=cfg.out_prefix or "")
    except PipelineAbort as exc:
        raise SystemExit(exc.code)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Post-process, filter, cluster, and visualise de novo design SDFs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sdf", nargs="+",
                   help="One or more SDF files from de_novo_cavity_growth.py")
    g.add_argument("--sdf-dir", default=None,
                   help="Batch mode: a directory containing .sdf files to process one-by-one")

    p.add_argument("--out-prefix", default="",
                   help="Prefix for output files (ignored in --sdf-dir batch mode)")

    # Score thresholds (applied before structural filters)
    p.add_argument("--qed-min",  type=float, default=0.0,
                   help="Minimum QED (0–1)")
    p.add_argument("--sa-max",   type=float, default=9.9,
                   help="Maximum SA score (1–6.5)")

    # Structural filters (all off by default; user opts in)
    p.add_argument("--lipinski", action="store_true",
                   help="Apply Lipinski Ro5 (MW≤500, LogP≤5, HBD≤5, HBA≤10)")
    p.add_argument("--veber",    action="store_true",
                   help="Apply Veber oral bioavailability (RotBonds≤10, TPSA≤140)")
    p.add_argument("--lead",     action="store_true",
                   help="Apply lead-likeness (MW≤350, LogP≤3.5, RotBonds≤7)")
    p.add_argument("--pains",    action="store_true",
                   help="Remove PAINS A/B/C alerts (RDKit FilterCatalog)")
    p.add_argument("--brenk",    action="store_true",
                   help="Remove Brenk structural alerts")

    # Clustering
    p.add_argument("--cluster-radius", type=float, default=0.60,
                   help="Tanimoto similarity cutoff for sphere-exclusion clustering "
                        "(0–1; higher = tighter clusters = more diverse picks)")

    # Grid
    p.add_argument("--grid-n",    type=int, default=48,
                   help="Number of top diverse picks to draw in the 2D grid")
    p.add_argument("--grid-cols", type=int, default=8,
                   help="Columns in the 2D structure grid")

    # Output
    p.add_argument("--out-dir",   default="postprocessed",
                   help="Output directory for all files")

    # PubChem (optional)
    p.add_argument("--pubchem", action="store_true",
                   help="Annotate molecules with PubChem CID (by InChIKey) and prioritize hits")
    p.add_argument("--pubchem-vendors", action="store_true",
                   help="Also attempt to fetch vendor names from PubChem (best-effort)")
    p.add_argument("--no-keep-pubchem-hits", action="store_false", dest="keep_pubchem_hits",
                   help="Do not rescue PubChem hits that fail filters")
    p.set_defaults(keep_pubchem_hits=True)
    p.add_argument("--pubchem-cache", default="",
                   help="Path to JSON cache for PubChem lookups (default: <out-dir>/pubchem_cache.json)")
    p.add_argument("--pubchem-timeout", type=float, default=20.0,
                   help="Timeout (seconds) for PubChem HTTP requests")
    p.add_argument("--pubchem-max-vendors", type=int, default=8,
                   help="Max number of vendor names to store per hit")

    return p


def main(argv=None) -> int:
    cfg = _build_parser().parse_args(argv)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
