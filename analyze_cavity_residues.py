#!/usr/bin/env python3
"""analyze_cavity_residues.py

Analyze the residue chemistry around DU cavity markers in PDB files.

Goal
----
Given a PDB containing one or more DU markers (DoGSite cavity-centre atoms),
report the residue composition within a radius of each DU anchor.

This is intended to help choose sensible weights for pharmacophore-aware
scoring in de novo growth (e.g. whether a pocket is mainly polar vs hydrophobic).

Key outputs (per DU marker)
--------------------------
- number of nearby residues
- counts of positive/negative residues and ratio
- counts of polar/apolar residues and ratio
- counts of aromatic/non-aromatic residues and fraction
- residue-type histogram (compact string)

Notes
-----
- A residue is considered "near" a DU anchor if ANY heavy atom of the residue is
  within the radius.
- Waters are excluded based on common residue names.
- Hydrogens are excluded.
- This script does NOT require RDKit.

Examples
--------
Analyze one PDB:
  python analyze_cavity_residues.py --pdb path/to/1ABC_with_DU.pdb --radius 8 --out-csv cavity_env.csv

Analyze a directory of PDBs:
  python analyze_cavity_residues.py --pdb-dir path/to/pdbs --pattern "*.pdb" --radius 8 --out-csv cavity_env.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast


_WATER_RESNAMES: frozenset[str] = frozenset(
    {
        "HOH",
        "WAT",
        "H2O",
        "TIP",
        "TIP3",
        "TIP4",
        "SPC",
        "SOL",
        "DOD",
        "TP3",
        "OHH",
    }
)


# Residue classes (simple, tweakable heuristics)
_POS: frozenset[str] = frozenset({"LYS", "ARG", "HIS"})
_NEG: frozenset[str] = frozenset({"ASP", "GLU"})
_AROMATIC: frozenset[str] = frozenset({"PHE", "TYR", "TRP", "HIS"})

# For polar/apolar, we assign each residue to one of the two when possible.
_POLAR: frozenset[str] = frozenset(
    {
        "SER",
        "THR",
        "ASN",
        "GLN",
        "TYR",
        "CYS",
        "HIS",
        "ASP",
        "GLU",
        "LYS",
        "ARG",
    }
)
_APOLAR: frozenset[str] = frozenset(
    {
        "ALA",
        "VAL",
        "LEU",
        "ILE",
        "MET",
        "PRO",
        "GLY",
        "PHE",
        "TRP",
    }
)


@dataclass(frozen=True)
class DUEntry:
    x: float
    y: float
    z: float
    label: str  # chain/resSeq (best effort)


@dataclass(frozen=True)
class AtomRec:
    x: float
    y: float
    z: float
    resname: str
    chain: str
    resseq: str
    icode: str
    atom_name: str
    elem: str

    @property
    def resid_key(self) -> str:
        # stable residue key within a PDB
        ic = self.icode.strip()
        ic_part = ic if ic else ""
        return f"{self.chain}:{self.resseq}{ic_part}:{self.resname}"


def parse_du_atoms(pdb_text: str) -> List[DUEntry]:
    entries: List[DUEntry] = []
    for ln in pdb_text.splitlines():
        if len(ln) < 54:
            continue
        if not (ln.startswith("HETATM") or ln.startswith("ATOM")):
            continue

        resname = ln[17:20].strip()
        atom_name = ln[12:16].strip()
        if resname != "DU" and atom_name != "DU":
            continue

        try:
            x = float(ln[30:38])
            y = float(ln[38:46])
            z = float(ln[46:54])
        except ValueError:
            continue

        chain = (ln[21:22].strip() or "?") if len(ln) >= 22 else "?"
        resseq = (ln[22:26].strip() or "?") if len(ln) >= 26 else "?"
        label = f"{chain}/{resseq}"
        entries.append(DUEntry(x=x, y=y, z=z, label=label))
    return entries


def _elem_from_line(ln: str, atom_name: str) -> str:
    elem = ln[76:78].strip() if len(ln) >= 78 else ""
    if elem:
        return elem
    # fallback: first letter of atom name (PDB-style)
    an = atom_name.strip()
    if not an:
        return ""
    return an[0]


def parse_protein_atoms(pdb_text: str, include_hetatm: bool = True) -> List[AtomRec]:
    atoms: List[AtomRec] = []

    for ln in pdb_text.splitlines():
        if len(ln) < 54:
            continue
        record = ln[:6]
        if record not in ("ATOM  ", "HETATM"):
            continue
        if record == "HETATM" and not include_hetatm:
            continue

        resname = ln[17:20].strip().upper()
        atom_name = ln[12:16].strip()

        # Skip DU markers and waters
        if resname == "DU" or atom_name.strip().upper() == "DU":
            continue
        if resname in _WATER_RESNAMES:
            continue

        chain = (ln[21:22].strip() or "?") if len(ln) >= 22 else "?"
        resseq = (ln[22:26].strip() or "?") if len(ln) >= 26 else "?"
        icode = (ln[26:27].strip() or "") if len(ln) >= 27 else ""

        elem = _elem_from_line(ln, atom_name).strip().upper()

        # Skip hydrogens
        if elem == "H" or (not elem and atom_name.startswith("H")):
            continue

        try:
            x = float(ln[30:38])
            y = float(ln[38:46])
            z = float(ln[46:54])
        except ValueError:
            continue

        atoms.append(
            AtomRec(
                x=x,
                y=y,
                z=z,
                resname=resname,
                chain=chain,
                resseq=resseq,
                icode=icode,
                atom_name=atom_name,
                elem=elem,
            )
        )

    return atoms


def _safe_ratio(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def _metrics_from_res_counts(res_counts: Counter[str]) -> Dict[str, object]:
    n_res = int(sum(res_counts.values()))
    n_pos = sum(res_counts[r] for r in res_counts if _classify_residue(r)["pos"])
    n_neg = sum(res_counts[r] for r in res_counts if _classify_residue(r)["neg"])
    n_arom = sum(res_counts[r] for r in res_counts if _classify_residue(r)["aromatic"])
    n_polar = sum(res_counts[r] for r in res_counts if _classify_residue(r)["polar"])
    n_apolar = sum(res_counts[r] for r in res_counts if _classify_residue(r)["apolar"])

    pos_neg_ratio = _safe_ratio(n_pos, n_pos + n_neg)
    polar_apolar_ratio = _safe_ratio(n_polar, n_polar + n_apolar)
    aromatic_fraction = _safe_ratio(n_arom, n_res)

    hist = ";".join(
        f"{r}:{c}"
        for r, c in sorted(res_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    return {
        "n_residues": n_res,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "pos_over_posneg": None if pos_neg_ratio is None else round(pos_neg_ratio, 4),
        "n_polar": int(n_polar),
        "n_apolar": int(n_apolar),
        "polar_over_polarapolar": None
        if polar_apolar_ratio is None
        else round(polar_apolar_ratio, 4),
        "n_aromatic": int(n_arom),
        "n_nonaromatic": int(n_res - n_arom),
        "aromatic_fraction": None
        if aromatic_fraction is None
        else round(aromatic_fraction, 4),
        "residue_hist": hist,
    }


def _classify_residue(resname: str) -> Dict[str, bool]:
    r = resname.upper()
    return {
        "pos": r in _POS,
        "neg": r in _NEG,
        "aromatic": r in _AROMATIC,
        "polar": r in _POLAR,
        "apolar": r in _APOLAR,
    }


def analyze_anchor(
    anchor: DUEntry,
    atoms: Sequence[AtomRec],
    radius: float,
) -> Dict[str, object]:
    r2 = float(radius) * float(radius)
    ax, ay, az = anchor.x, anchor.y, anchor.z

    # residue_key -> min_dist2
    min_d2: Dict[str, float] = {}
    resname_by_key: Dict[str, str] = {}
    chain_by_key: Dict[str, str] = {}

    for a in atoms:
        dx = a.x - ax
        dy = a.y - ay
        dz = a.z - az
        d2 = dx * dx + dy * dy + dz * dz
        if d2 > r2:
            continue

        k = a.resid_key
        if k not in min_d2 or d2 < min_d2[k]:
            min_d2[k] = d2
            resname_by_key[k] = a.resname
            chain_by_key[k] = a.chain

    near_keys = list(min_d2.keys())
    residue_names = [resname_by_key[k] for k in near_keys]

    res_counts = Counter(residue_names)
    metrics = _metrics_from_res_counts(res_counts)

    # Per-chain residue counts (each residue counted once if any heavy atom is within radius)
    res_counts_by_chain: Dict[str, Counter[str]] = defaultdict(Counter)
    for k in near_keys:
        ch = chain_by_key.get(k, "?")
        res_counts_by_chain[ch][resname_by_key[k]] += 1

    chain_summaries: Dict[str, Dict[str, object]] = {
        ch: _metrics_from_res_counts(cnts) for ch, cnts in res_counts_by_chain.items()
    }

    return {
        "du_label": anchor.label,
        "anchor_x": round(anchor.x, 3),
        "anchor_y": round(anchor.y, 3),
        "anchor_z": round(anchor.z, 3),
        "radius_A": float(radius),
        **metrics,
        "chain_summaries": chain_summaries,
    }


def _infer_pdb_id_from_filename(p: Path) -> Optional[str]:
    # Common stems look like: 1C4Z_with_DU, 1C4Z_DU, 1C4Z...
    m = re.match(r"^([0-9][A-Za-z0-9]{3})", p.stem)
    if not m:
        return None
    return m.group(1).upper()


def infer_pdb_id(p: Path, pdb_text: str) -> Optional[str]:
    pid = _infer_pdb_id_from_filename(p)
    if pid:
        return pid

    for ln in pdb_text.splitlines()[:50]:
        if ln.startswith("HEADER") and len(ln) >= 66:
            cand = ln[62:66].strip()
            if re.fullmatch(r"[0-9][A-Za-z0-9]{3}", cand or ""):
                return cand.upper()
    return None


def _http_get_json(url: str, timeout_s: float = 10.0) -> Optional[object]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "cavity-env-analyzer/1.0 (mailto:none)",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
        return json.loads(data.decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def _load_json_cache(path: Path) -> Dict[str, object]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json_cache(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _pdbe_chain_to_uniprot(pdb_id: str) -> Dict[str, str]:
    # Returns {chain_id -> uniprot_accession} for polymer chains.
    pdb_id_l = pdb_id.lower()
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id_l}"
    data = _http_get_json(url)
    if not isinstance(data, dict) or pdb_id_l not in data:
        return {}

    out: Dict[str, str] = {}
    root = data.get(pdb_id_l, {})
    uni = root.get("UniProt", {}) if isinstance(root, dict) else {}
    if not isinstance(uni, dict):
        return {}

    for acc, acc_block in uni.items():
        if not isinstance(acc_block, dict):
            continue
        mappings = acc_block.get("mappings", [])
        if not isinstance(mappings, list):
            continue
        for m in mappings:
            if not isinstance(m, dict):
                continue
            chain_id = m.get("chain_id")
            if isinstance(chain_id, str) and chain_id.strip():
                # First hit wins; PDBs can have multiple mappings but gene stays same.
                out.setdefault(chain_id.strip(), acc)
    return out


def _uniprot_gene_symbol(uniprot_acc: str) -> Optional[str]:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_acc}.json"
    data = _http_get_json(url)
    if not isinstance(data, dict):
        return None

    # Prefer HGNC-approved gene symbol when present (human proteins)
    xrefs = data.get("uniProtKBCrossReferences", [])
    if isinstance(xrefs, list):
        for xr in xrefs:
            if not isinstance(xr, dict):
                continue
            if xr.get("database") != "HGNC":
                continue
            props = xr.get("properties", [])
            if not isinstance(props, list):
                continue
            for p in props:
                if not isinstance(p, dict):
                    continue
                if p.get("key") == "GeneName" and isinstance(p.get("value"), str) and p["value"].strip():
                    return p["value"].strip()

    genes = data.get("genes", [])
    if not isinstance(genes, list):
        return None
    for g in genes:
        if not isinstance(g, dict):
            continue
        gn = g.get("geneName")
        if isinstance(gn, dict) and isinstance(gn.get("value"), str) and gn["value"].strip():
            return gn["value"].strip()
    return None


def lookup_chain_gene_symbol(
    pdb_id: str,
    chain_id: str,
    cache: Dict[str, object],
    pdb_chainmap_cache: Optional[Dict[str, Dict[str, str]]] = None,
    uniprot_gene_cache: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return (uniprot_acc, gene_symbol) for a given PDB ID and chain.

    Best-effort, uses PDBe for PDB→UniProt mapping and UniProt for gene.
    """

    key = f"{pdb_id.upper()}:{chain_id.strip()}"
    cached = cache.get(key)
    if isinstance(cached, dict):
        ua = cached.get("uniprot")
        gs = cached.get("gene_symbol")
        return (ua if isinstance(ua, str) else None, gs if isinstance(gs, str) else None)

    pdb_id_u = pdb_id.upper()
    if pdb_chainmap_cache is not None and pdb_id_u in pdb_chainmap_cache:
        chain_to_uni = pdb_chainmap_cache[pdb_id_u]
    else:
        chain_to_uni = _pdbe_chain_to_uniprot(pdb_id)
        if pdb_chainmap_cache is not None:
            pdb_chainmap_cache[pdb_id_u] = chain_to_uni

    uniprot_acc = chain_to_uni.get(chain_id)

    gene_symbol: Optional[str] = None
    if uniprot_acc:
        # Check persistent cache by UniProt too (lets us reuse across PDBs)
        uc_key = f"UNIPROT:{uniprot_acc}"
        cached_u = cache.get(uc_key)
        if isinstance(cached_u, dict) and isinstance(cached_u.get("gene_symbol"), str):
            gene_symbol = str(cached_u["gene_symbol"]) or None

        if gene_symbol is None and uniprot_gene_cache is not None and uniprot_acc in uniprot_gene_cache:
            gene_symbol = uniprot_gene_cache[uniprot_acc]

        if gene_symbol is None:
            gene_symbol = _uniprot_gene_symbol(uniprot_acc)
            if uniprot_gene_cache is not None:
                uniprot_gene_cache[uniprot_acc] = gene_symbol
            cache[uc_key] = {"gene_symbol": gene_symbol}

    cache[key] = {"uniprot": uniprot_acc, "gene_symbol": gene_symbol}
    return uniprot_acc, gene_symbol


def _as_int(v: object, default: int = 0) -> int:
    try:
        return int(v)  # type: ignore[arg-type]
    except Exception:
        return default


def iter_pdb_paths(
    pdb: Optional[Sequence[str]],
    pdb_dir: Optional[str],
    pattern: str,
) -> List[Path]:
    paths: List[Path] = []
    if pdb:
        for p in pdb:
            paths.append(Path(p))
    if pdb_dir:
        d = Path(pdb_dir)
        paths.extend(sorted(d.glob(pattern)))

    # Deduplicate while preserving order
    seen: set[Path] = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Summarize residue chemistry around DU cavity markers in PDB files "
            "(pos/neg, polar/apolar, aromatic fractions) within a radius."
        )
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdb", nargs="+", help="One or more PDB files to analyze")
    g.add_argument("--pdb-dir", help="Directory of PDB files to analyze")

    p.add_argument("--pattern", default="*.pdb", help="Glob pattern when using --pdb-dir")
    p.add_argument("--radius", type=float, default=8.0, help="Å radius around each DU anchor")
    p.add_argument(
        "--no-hetatm",
        action="store_true",
        help="Exclude HETATM records from analysis (protein ATOM only)",
    )
    p.add_argument(
        "--out-csv",
        default="cavity_residue_environment.csv",
        help="Output CSV path",
    )

    p.add_argument(
        "--split-by-chain",
        action="store_true",
        help=(
            "Also report per-chain residue chemistry around each DU (top 2 chains by nearby residues) "
            "in the same CSV row."
        ),
    )
    p.add_argument(
        "--annotate-genes",
        action="store_true",
        help=(
            "Annotate the top 2 chains with best-effort UniProt accession and gene symbol using the PDB ID "
            "(requires internet; cached; safe to rerun)."
        ),
    )
    p.add_argument(
        "--gene-cache-json",
        default=str(Path(__file__).with_name(".cache") / "pdb_chain_gene_cache.json"),
        help="Path to JSON cache used for --annotate-genes",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    pdb_paths = iter_pdb_paths(args.pdb, args.pdb_dir, args.pattern)
    if not pdb_paths:
        raise SystemExit("No PDB files found.")

    out_rows: List[Dict[str, object]] = []

    gene_cache_path = Path(args.gene_cache_json)
    gene_cache: Dict[str, object] = _load_json_cache(gene_cache_path) if args.annotate_genes else {}
    pdb_chainmap_cache: Dict[str, Dict[str, str]] = {}
    uniprot_gene_cache: Dict[str, Optional[str]] = {}

    for pdb_path in pdb_paths:
        text = pdb_path.read_text(encoding="utf-8", errors="replace")
        dus = parse_du_atoms(text)
        if not dus:
            continue

        atoms = parse_protein_atoms(text, include_hetatm=(not args.no_hetatm))

        pdb_id = infer_pdb_id(pdb_path, text) if args.annotate_genes else None

        for du_index, du in enumerate(dus):
            row = analyze_anchor(du, atoms, radius=float(args.radius))
            raw_chain_summaries = row.pop("chain_summaries", {})
            chain_summaries: Dict[str, Dict[str, object]] = (
                raw_chain_summaries if isinstance(raw_chain_summaries, dict) else {}
            )

            row = {
                "pdb_file": pdb_path.name,
                "pdb_stem": pdb_path.stem,
                "du_index": du_index,
                **row,
            }

            if args.split_by_chain:
                # pick top 2 chains by nearby residues, tie-break by chain id
                chain_counts: List[Tuple[str, int]] = []
                for ch, summary in chain_summaries.items():
                    n_res = _as_int(summary.get("n_residues", 0), default=0)
                    chain_counts.append((ch, n_res))
                chain_counts.sort(key=lambda t: (-t[1], t[0]))
                top_chains: List[Optional[str]] = [
                    cast(Optional[str], c) for c, n in chain_counts if n > 0
                ][:2]
                while len(top_chains) < 2:
                    top_chains.append(None)

                for i, ch in enumerate(top_chains, start=1):
                    prefix = f"chain{i}_"
                    row[f"chain{i}_id"] = ch

                    summary = chain_summaries.get(ch, {}) if ch else {}
                    for k in (
                        "n_residues",
                        "n_pos",
                        "n_neg",
                        "pos_over_posneg",
                        "n_polar",
                        "n_apolar",
                        "polar_over_polarapolar",
                        "n_aromatic",
                        "n_nonaromatic",
                        "aromatic_fraction",
                        "residue_hist",
                    ):
                        row[prefix + k] = summary.get(k)

                    if args.annotate_genes and pdb_id and ch:
                        uniprot_acc, gene_symbol = lookup_chain_gene_symbol(
                            pdb_id=pdb_id,
                            chain_id=ch,
                            cache=gene_cache,
                            pdb_chainmap_cache=pdb_chainmap_cache,
                            uniprot_gene_cache=uniprot_gene_cache,
                        )
                    else:
                        uniprot_acc, gene_symbol = None, None

                    row[f"chain{i}_uniprot"] = uniprot_acc
                    row[f"chain{i}_gene_symbol"] = gene_symbol

            out_rows.append(row)

    if not out_rows:
        print("No DU anchors found in the provided PDBs.")
        return 0

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(out_rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    if args.annotate_genes:
        _write_json_cache(gene_cache_path, gene_cache)

    print(f"Wrote {len(out_rows)} DU-environment rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
