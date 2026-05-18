#!/usr/bin/env python3
"""
cavity_analyzer.py
==================
Batch analysis of PDB_DU complex files for PROTAC/TPD interface cavity characterization.

Usage
-----
  python cavity_analyzer.py --folder /path/to/pdbs [options]

Required
--------
  --folder          Directory containing *_DU.pdb files

Optional
--------
  --cutoff          Distance cutoff in Å (default: 8.0)
  --chain_map       JSON file: {"C":"VHL","A":"ELOB","B":"ELOC",...}
  --ligase_chains   Comma-separated chain IDs for ligase, e.g. C,F,I,L
  --partner_chains  Comma-separated chain IDs for partner, e.g. A,D,G,J
  --hbond           Estimate inter-chain H-bond count (geometry-based)
  --dssp            DSSP secondary structure (requires mkdssp in PATH)
  --output          Output CSV path (default: cavity_analysis.csv)
  --verbose         Per-file progress output

Output columns
--------------
  pdb_id, du_chain, du_resnum, du_x, du_y, du_z,
  hgnc_ligase, ligase_chains, hgnc_partner, partner_chains,
  n_ligase,  apolar_lig,  polar_lig,  aromatic_lig,  aliphatic_lig,  mean_bfactor_lig,
  n_partner, apolar_par,  polar_par,  aromatic_par,  aliphatic_par,  mean_bfactor_par,
  ratio_N, ratio_apolar, ratio_polar, ratio_aromatic, ratio_aliphatic,
  hbond_count (optional),
  dssp_helix_lig, dssp_sheet_lig, dssp_loop_lig,
  dssp_helix_par, dssp_sheet_par, dssp_loop_par (optional)
"""

import os
import sys
import json
import argparse
import warnings
import io
import shutil
import subprocess
import tempfile
import contextlib
import functools
import re
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import make_dssp_dict
    from Bio import BiopythonWarning
    warnings.simplefilter("ignore", BiopythonWarning)
except ImportError:
    sys.exit("ERROR: BioPython not found.  pip install biopython")


@contextlib.contextmanager
def _suppress_stdio():
    """Suppress Python-level stdout/stderr (does not affect subprocess output)."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


_DSSP_DISABLED_REASON = None
_DSSP_DISABLE_REPORTED = False


def _dssp_executable():
    # Try common executable names. Windows installs vary.
    for exe in ("mkdssp", "dssp"):
        exe_path = shutil.which(exe)
        if exe_path:
            return exe_path
    return None


def _find_mmcif_dictionary(dssp_exe_path):
    """Best-effort discovery of mmcif_pdbx.dic for mkdssp/dssp.

    Conda typically installs:
      <prefix>/Library/bin/mkdssp(.exe)
      <prefix>/share/libcifpp/mmcif_pdbx.dic
    """
    # Allow explicit override.
    override = os.environ.get("DSSP_MMCIF_DICTIONARY")
    if override and os.path.exists(override):
        return override

    try:
        exe_path = Path(dssp_exe_path)
        # .../Library/bin/mkdssp.exe -> prefix is parents[2]
        prefix = exe_path.resolve().parents[2]
        candidates = [
            prefix / "share" / "libcifpp" / "mmcif_pdbx.dic",
            prefix / "Library" / "share" / "libcifpp" / "mmcif_pdbx.dic",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    except Exception:
        pass

    return None


def _maybe_report_dssp_disabled(verbose):
    global _DSSP_DISABLE_REPORTED
    if not verbose:
        return
    if _DSSP_DISABLED_REASON and not _DSSP_DISABLE_REPORTED:
        print(f"  [WARN] DSSP disabled: {_DSSP_DISABLED_REASON}")
        _DSSP_DISABLE_REPORTED = True


def _run_mkdssp_to_file(pdb_file):
    """Run mkdssp/dssp and return path to DSSP output file, or None on failure."""
    global _DSSP_DISABLED_REASON

    exe = _dssp_executable()
    if not exe:
        _DSSP_DISABLED_REASON = "mkdssp/dssp not found in PATH"
        return None

    mmcif_dic = _find_mmcif_dictionary(exe)

    # Create a temp output file path that mkdssp can write.
    fd, out_path = tempfile.mkstemp(prefix="dssp_", suffix=".dssp")
    os.close(fd)

    try:
        # mkdssp v4+ typically uses positional args:
        #   mkdssp [options] input-file [output-file]
        # (Some older wrappers use -i/-o, but the conda build on Windows is positional.)
        cmd = [exe]
        if mmcif_dic:
            cmd += ["--mmcif-dictionary", mmcif_dic]
        cmd += [str(pdb_file), out_path]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            # Special-case the common dictionary error to avoid endless repeats.
            if "mmcif_pdbx.dic" in err or "Could not load dictionary" in err:
                _DSSP_DISABLED_REASON = "mkdssp cannot load mmcif_pdbx.dic (install DSSP data files)"
            else:
                _DSSP_DISABLED_REASON = err or f"{exe} exited with code {proc.returncode}"
            return None

        # Some failures still produce an empty file.
        try:
            if os.path.getsize(out_path) <= 0:
                _DSSP_DISABLED_REASON = "mkdssp produced empty output"
                return None
        except OSError:
            _DSSP_DISABLED_REASON = "mkdssp output file missing"
            return None

        return out_path
    except FileNotFoundError:
        _DSSP_DISABLED_REASON = f"{exe} not found"
        return None
    except Exception as e:
        _DSSP_DISABLED_REASON = str(e)
        return None
    finally:
        # Caller deletes out_path only if it was successfully returned.
        pass


@functools.lru_cache(maxsize=256)
def _get_dssp_dict_for_pdb(pdb_file, verbose=False):
    """Return DSSP dict for a PDB file (cached)."""
    global _DSSP_DISABLED_REASON

    if _DSSP_DISABLED_REASON:
        _maybe_report_dssp_disabled(verbose)
        return None

    out_path = _run_mkdssp_to_file(pdb_file)
    if not out_path:
        _maybe_report_dssp_disabled(verbose)
        return None

    try:
        with _suppress_stdio():
            dssp_dict, _ = make_dssp_dict(out_path)
        return dssp_dict
    except Exception as e:
        _DSSP_DISABLED_REASON = f"failed to parse DSSP output ({e})"
        _maybe_report_dssp_disabled(verbose)
        return None
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass

# ── Residue property sets ──────────────────────────────────────────────────────
APOLAR    = {"ALA","VAL","LEU","ILE","MET","PRO","GLY","PHE","TRP","TYR"}
POLAR     = {"SER","THR","CYS","ASN","GLN","HIS","LYS","ARG","ASP","GLU","TYR"}
AROMATIC  = {"PHE","TRP","TYR","HIS"}
ALIPHATIC = {"ALA","VAL","LEU","ILE","MET"}
SKIP      = {"HOH","DU","WAT","H2O"}

HBOND_DONORS    = {"N","NZ","NH1","NH2","NE","NE1","NE2","ND1","ND2","OG","OG1","OH","SG"}
HBOND_ACCEPTORS = {"O","OD1","OD2","OE1","OE2","ND1","NE2","OG","OG1","OH","SG"}
HBOND_DIST_MAX  = 3.5

DSSP_HELIX = {"H","G","I"}
DSSP_SHEET = {"E","B"}


def _parse_compnd_chain_molecules(pdb_file):
    """Parse PDB COMPND records to map chain ID -> molecule name.

    This is a best-effort parser and may return an empty dict for files
    lacking COMPND or using unusual formatting.
    """
    try:
        compnd_lines = []
        with open(pdb_file, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("COMPND"):
                    compnd_lines.append(line[10:].rstrip())
                elif compnd_lines:
                    # COMPND blocks are near the top; stop once it ends.
                    if line and (line[0].isalpha() and not line.startswith("COMPND")):
                        break

        if not compnd_lines:
            return {}

        text = " ".join(compnd_lines)
        # Split into MOL_ID blocks.
        blocks = re.split(r"\bMOL_ID:\s*\d+\s*;", text)
        chain_to_mol = {}
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            mol_m = re.search(r"\bMOLECULE:\s*([^;]+);", block, flags=re.IGNORECASE)
            chain_m = re.search(r"\bCHAIN:\s*([^;]+);", block, flags=re.IGNORECASE)
            if not mol_m or not chain_m:
                continue
            mol_name = mol_m.group(1).strip()
            chains_raw = chain_m.group(1)
            chains = [c.strip() for c in chains_raw.split(",") if c.strip()]
            for c in chains:
                # Chains are single characters in classic PDB.
                chain_to_mol[c] = mol_name
        return chain_to_mol
    except Exception:
        return {}


def _infer_roles_from_compnd(present_chains, chain_to_molecule, ligase_keywords):
    if not chain_to_molecule:
        return []
    kws = [k.strip().upper() for k in ligase_keywords if k.strip()]
    if not kws:
        return []
    hits = []
    for c in present_chains:
        mol = chain_to_molecule.get(c, "")
        mol_u = mol.upper()
        if any(k in mol_u for k in kws):
            hits.append(c)
    return sorted(set(hits))


def classify(resname):
    r = resname.strip()
    return {
        "apolar":    int(r in APOLAR),
        "polar":     int(r in POLAR),
        "aromatic":  int(r in AROMATIC),
        "aliphatic": int(r in ALIPHATIC),
    }


def safe_ratio(num, den):
    return round(num / den, 3) if den > 0 else float("nan")


def count_hbonds(res_list_a, res_list_b):
    """Geometry-based H-bond estimation (no H atoms required)."""
    def atoms(rlist, role_set):
        out = []
        for res in rlist:
            for atom in res.get_atoms():
                aname = atom.get_name().strip()
                if aname in role_set:
                    out.append(atom.get_vector().get_array())
        return out

    donors_a    = atoms(res_list_a, HBOND_DONORS)
    acceptors_a = atoms(res_list_a, HBOND_ACCEPTORS)
    donors_b    = atoms(res_list_b, HBOND_DONORS)
    acceptors_b = atoms(res_list_b, HBOND_ACCEPTORS)

    count = 0
    for d in donors_a:
        for a in acceptors_b:
            if np.linalg.norm(d - a) <= HBOND_DIST_MAX:
                count += 1
    for d in donors_b:
        for a in acceptors_a:
            if np.linalg.norm(d - a) <= HBOND_DIST_MAX:
                count += 1
    return count


def get_dssp_counts(pdb_file, residues, verbose=False, dssp_dict=None):
    """Return DSSP secondary structure counts for a residue list.

    If mkdssp/dssp is missing or broken (e.g. missing mmcif_pdbx.dic),
    DSSP is disabled and this returns zeros without spamming stderr.
    """
    global _DSSP_DISABLED_REASON

    if dssp_dict is None:
        dssp_dict = _get_dssp_dict_for_pdb(pdb_file, verbose=verbose)
    if not dssp_dict:
        return {"helix": 0, "sheet": 0, "loop": 0}

    counts = {"helix": 0, "sheet": 0, "loop": 0}
    for res in residues:
        key = (res.get_parent().get_id(), res.get_id())
        # make_dssp_dict returns tuples where index 1 is the DSSP SS code.
        ss = dssp_dict[key][1] if key in dssp_dict else "-"
        if ss in DSSP_HELIX:
            counts["helix"] += 1
        elif ss in DSSP_SHEET:
            counts["sheet"] += 1
        else:
            counts["loop"] += 1
    return counts


def analyze_pdb(pdb_file, cutoff, chain_map,
                ligase_chains, partner_chains,
                do_dssp, do_hbond, verbose,
                role_mode="largest",
                ligase_keywords=None,
                chain_info_rows=None):

    parser    = PDBParser(QUIET=True)
    # Biopython's PDBParser can emit noisy parse diagnostics to stderr.
    with _suppress_stdio():
        structure = parser.get_structure("s", pdb_file)
    pdb_id    = Path(pdb_file).stem.split("_")[0].upper()
    rows      = []
    chain_to_molecule = _parse_compnd_chain_molecules(pdb_file) if (role_mode == "compnd" or chain_info_rows is not None) else {}

    if chain_info_rows is not None:
        # One row per chain (first model) to help identify "who is who".
        try:
            first_model = structure[0]
            for chain in first_model:
                cid = chain.get_id()
                n_res = 0
                het_res = set()
                has_du = False
                for res in chain:
                    rname = res.get_resname().strip()
                    hetflag = res.get_id()[0].strip()
                    if rname == "DU":
                        has_du = True
                    if hetflag:
                        if rname not in {"HOH", "WAT", "H2O"}:
                            het_res.add(rname)
                        continue
                    if "CA" in res and rname not in SKIP:
                        n_res += 1

                chain_info_rows.append({
                    "pdb_id": pdb_id,
                    "chain_id": cid,
                    "molecule": chain_to_molecule.get(cid, ""),
                    "n_residues_ca": n_res,
                    "has_DU": int(has_du),
                    "het_resnames": ",".join(sorted(het_res)),
                })
        except Exception:
            pass

    # ── locate DU centres ────────────────────────────────────────────────────
    du_list = []
    het_resnames = set()
    for model in structure:
        for chain in model:
            for res in chain:
                rname = res.get_resname().strip()
                # Track HET residue names (for debugging when DU is missing)
                if res.get_id()[0].strip() and rname not in {"HOH", "WAT", "H2O"}:
                    het_resnames.add(rname)
                if rname == "DU":
                    coords = np.array([a.get_vector().get_array()
                                       for a in res.get_atoms()])
                    du_list.append({
                        "chain":  chain.get_id(),
                        "resnum": res.get_id()[1],
                        "centre": coords.mean(axis=0),
                        "model":  model.get_id(),
                    })

    if not du_list:
        if verbose:
            extra = ""
            if het_resnames:
                extra = f" (non-water HET found: {','.join(sorted(list(het_resnames))[:12])})"
            print(f"  [SKIP] No DU residue in {Path(pdb_file).name}{extra}")
        return rows

    # ── collect protein residues (first model only, CA-present) ─────────────
    first_model  = structure[0]
    all_residues = []
    for chain in first_model:
        for res in chain:
            if res.get_resname().strip() in SKIP: continue
            if "CA" not in res: continue
            all_residues.append(res)

    # ── process each DU ──────────────────────────────────────────────────────
    dssp_dict = None
    if do_dssp:
        # Compute DSSP once per PDB file (cached) to avoid repeated mkdssp calls.
        dssp_dict = _get_dssp_dict_for_pdb(pdb_file, verbose=verbose)

    for du in du_list:
        if du["model"] != 0:
            continue
        centre = du["centre"]

        nearby = [r for r in all_residues
                  if np.linalg.norm(
                      r["CA"].get_vector().get_array() - centre) <= cutoff]
        if not nearby:
            continue

        present_chains = {r.get_parent().get_id() for r in nearby}

        # For heuristics, count how many nearby residues each chain contributes.
        chain_counts = {}
        for r in nearby:
            cid = r.get_parent().get_id()
            chain_counts[cid] = chain_counts.get(cid, 0) + 1

        # Ligase/Partner assignment
        # Priority:
        # 1) explicit args
        # 2) COMPND inference via keywords (optional)
        # 3) fallback heuristic: pick chain with most nearby residues as ligase
        lig_chains = [c for c in ligase_chains if c in present_chains] if ligase_chains else []
        par_chains = [c for c in partner_chains if c in present_chains] if partner_chains else []

        if not lig_chains and not par_chains:
            if role_mode == "compnd":
                inferred = _infer_roles_from_compnd(
                    present_chains,
                    chain_to_molecule,
                    ligase_keywords or [],
                )
                if inferred:
                    lig_chains = inferred
                    par_chains = [c for c in sorted(present_chains) if c not in lig_chains]

        if not lig_chains and not par_chains:
            # Largest-contribution chain becomes ligase; rest partner.
            lig = max(sorted(present_chains), key=lambda c: chain_counts.get(c, 0))
            lig_chains = [lig]
            par_chains = [c for c in sorted(present_chains) if c != lig]

        if lig_chains and not par_chains:
            par_chains = [c for c in sorted(present_chains) if c not in lig_chains]
        if par_chains and not lig_chains:
            lig_chains = [c for c in sorted(present_chains) if c not in par_chains]

        lig_res = [r for r in nearby if r.get_parent().get_id() in lig_chains]
        par_res = [r for r in nearby if r.get_parent().get_id() in par_chains]

        # Labels (prefer explicit chain_map; else COMPND molecule name; else chain IDs)
        lig_chain_str = ",".join(sorted({r.get_parent().get_id() for r in lig_res})) or "?"
        par_chain_str = ",".join(sorted({r.get_parent().get_id() for r in par_res})) or "?"

        def label_for(chain_list, chain_str):
            if not chain_list:
                return chain_str
            c0 = chain_list[0]
            if c0 in chain_map:
                return chain_map[c0]
            if role_mode == "compnd" and chain_to_molecule and c0 in chain_to_molecule:
                return chain_to_molecule[c0]
            return chain_str

        hgnc_lig = label_for(lig_chains, lig_chain_str)
        hgnc_par = label_for(par_chains, par_chain_str)

        # Property tallies
        def tally(res_list):
            n = apo = pol = aro = ali = 0
            bfactors = []
            for r in res_list:
                n += 1
                c = classify(r.get_resname())
                apo += c["apolar"]; pol += c["polar"]
                aro += c["aromatic"]; ali += c["aliphatic"]
                bfactors.append(np.mean([a.get_bfactor() for a in r.get_atoms()]))
            mean_b = round(float(np.mean(bfactors)), 2) if bfactors else float("nan")
            return n, apo, pol, aro, ali, mean_b

        n_l, apo_l, pol_l, aro_l, ali_l, bf_l = tally(lig_res)
        n_p, apo_p, pol_p, aro_p, ali_p, bf_p = tally(par_res)

        row = {
            # Coordinates & identity
            "pdb_id":           pdb_id,
            "du_chain":         du["chain"],
            "du_resnum":        du["resnum"],
            "du_x":             round(float(centre[0]), 3),
            "du_y":             round(float(centre[1]), 3),
            "du_z":             round(float(centre[2]), 3),
            # HGNC labels
            "hgnc_ligase":      hgnc_lig,
            "ligase_chains":    lig_chain_str,
            "hgnc_partner":     hgnc_par,
            "partner_chains":   par_chain_str,
            # Ligase residue counts
            "n_ligase":         n_l,
            "apolar_lig":       apo_l,
            "polar_lig":        pol_l,
            "aromatic_lig":     aro_l,
            "aliphatic_lig":    ali_l,
            "mean_bfactor_lig": bf_l,
            # Partner residue counts
            "n_partner":        n_p,
            "apolar_par":       apo_p,
            "polar_par":        pol_p,
            "aromatic_par":     aro_p,
            "aliphatic_par":    ali_p,
            "mean_bfactor_par": bf_p,
            # Ratios (ligase / partner)
            "ratio_N":          safe_ratio(n_l,   n_p),
            "ratio_apolar":     safe_ratio(apo_l, apo_p),
            "ratio_polar":      safe_ratio(pol_l, pol_p),
            "ratio_aromatic":   safe_ratio(aro_l, aro_p),
            "ratio_aliphatic":  safe_ratio(ali_l, ali_p),
        }

        if do_hbond:
            row["hbond_count"] = count_hbonds(lig_res, par_res)

        if do_dssp:
            ss_l = get_dssp_counts(pdb_file, lig_res, verbose=verbose, dssp_dict=dssp_dict)
            ss_p = get_dssp_counts(pdb_file, par_res, verbose=verbose, dssp_dict=dssp_dict)
            row.update({
                "dssp_helix_lig": ss_l["helix"],
                "dssp_sheet_lig": ss_l["sheet"],
                "dssp_loop_lig":  ss_l["loop"],
                "dssp_helix_par": ss_p["helix"],
                "dssp_sheet_par": ss_p["sheet"],
                "dssp_loop_par":  ss_p["loop"],
            })

        rows.append(row)

        if verbose:
            print(f"  [OK] {pdb_id} | DU@{du['chain']}{du['resnum']} | "
                  f"Ligase({hgnc_lig}) N={n_l} | Partner({hgnc_par}) N={n_p} | "
                  f"ratio_N={safe_ratio(n_l, n_p)}")
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Batch PDB cavity analysis — PROTAC/TPD ternary complexes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--folder",          required=True)
    ap.add_argument("--cutoff",          type=float, default=8.0)
    ap.add_argument("--chain_map",       default=None)
    ap.add_argument("--ligase_chains",   default=None)
    ap.add_argument("--partner_chains",  default=None)
    ap.add_argument("--hbond",           action="store_true")
    ap.add_argument("--dssp",            action="store_true")
    ap.add_argument("--role_mode",        choices=["largest", "compnd"], default="largest",
                    help="How to infer ligase/partner when chains not provided. 'largest' picks the chain with most nearby residues as ligase; 'compnd' uses PDB COMPND records + ligase keywords.")
    ap.add_argument("--ligase_keywords",  default="VHL,CRBN,CEREBLON,DDB1,CUL2,CUL4,ELOB,ELOC,RBX1,SKP1",
                    help="Comma-separated keywords used when --role_mode compnd.")
    ap.add_argument("--chain_info_output", default=None,
                    help="Optional CSV to write per-PDB chain->molecule info parsed from COMPND (helps identify which chain is which).")
    ap.add_argument("--output",          default="cavity_analysis.csv")
    ap.add_argument("--verbose",         action="store_true")
    args = ap.parse_args()

    folder    = Path(args.folder)
    pdb_files = sorted(folder.glob("*_DU.pdb"))
    if not pdb_files:
        sys.exit(f"No *_DU.pdb files in {folder}")

    print(f"Found {len(pdb_files)} PDB file(s) in {folder}")

    chain_map = {}
    if args.chain_map:
        with open(args.chain_map) as f:
            chain_map = json.load(f)
        print(f"Chain map: {chain_map}")

    ligase_chains  = [c.strip() for c in args.ligase_chains.split(",")]  \
                     if args.ligase_chains  else []
    partner_chains = [c.strip() for c in args.partner_chains.split(",")] \
                     if args.partner_chains else []

    ligase_keywords = [k.strip() for k in args.ligase_keywords.split(",")] if args.ligase_keywords else []

    all_rows = []
    chain_info_rows = [] if args.chain_info_output else None
    for pdb_file in pdb_files:
        if args.verbose:
            print(f"\nProcessing {pdb_file.name} ...")
        try:
            rows = analyze_pdb(
                pdb_file       = str(pdb_file),
                cutoff         = args.cutoff,
                chain_map      = chain_map,
                ligase_chains  = ligase_chains,
                partner_chains = partner_chains,
                do_dssp        = args.dssp,
                do_hbond       = args.hbond,
                verbose        = args.verbose,
                role_mode       = args.role_mode,
                ligase_keywords = ligase_keywords,
                chain_info_rows = chain_info_rows,
            )
            all_rows.extend(rows)
        except Exception as e:
            print(f"  [ERROR] {pdb_file.name}: {e}")

    if not all_rows:
        sys.exit("No results. Check PDB files contain DU residues.")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nDone. {len(df)} row(s) → {args.output}")
    print(df.to_string(index=False))

    if args.chain_info_output and chain_info_rows is not None:
        try:
            ci = pd.DataFrame(chain_info_rows).drop_duplicates(subset=["pdb_id", "chain_id"])
            ci.to_csv(args.chain_info_output, index=False)
            if args.verbose:
                print(f"\nWrote chain info → {args.chain_info_output}")
        except Exception as e:
            print(f"  [WARN] Failed to write chain info CSV: {e}")


if __name__ == "__main__":
    main()