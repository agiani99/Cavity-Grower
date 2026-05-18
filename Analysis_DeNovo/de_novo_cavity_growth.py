#!/usr/bin/env python3
"""
de_novo_cavity_growth.py
========================
Beam-search fragment-growing de novo design of small molecules anchored to
DU cavity-centre markers previously placed by dogsite_interface_cavities.py.

Pipeline
--------
  1. Read DU HETATM atom(s) from PDB  →  cavity anchor coordinate(s)
  2. Load all non-DU / non-water heavy-atom coords  →  clash detector
  3. Seed beam with a single C atom at the anchor position
  4. Iteratively grow each beam member:
      • Single-atom additions  (C / N / O / S / F / Cl / Br / C=O / C=N / C≡N)
      • Fixed fragments        (C(=O)N, C(=O)NC, C(=O)N(C)C, OC, NC, N(C)C)
       • Ring-fragment additions (benzene, pyridine, piperidine, …)
  5. Score every grown molecule with:
       • QED  (RDKit)                     ← drug-likeness,   higher → better
       • SA   (RDKit contrib SA_Score)    ← synthetic acc.,  lower  → better
       • clash_frac   (% atoms inside protein VdW shell)
       • out_frac     (% atoms outside cavity sphere)
       composite = 0.40·QED + 0.35·SA_norm + 0.25·(1−clash) − 0.30·out_frac
  6. Keep top-K states (beam search); harvest molecules in target HAC window
  7. Write accepted molecules to SDF + summary CSV

Dependencies
------------
  pip install rdkit numpy          # scipy optional – speeds up clash check

SA scorer: uses rdkit.Contrib.SA_Score (bundled with RDKit).
           Falls back to a ring/chiral heuristic if not importable.

Quick start
-----------
  # full run on first DU atom, target 16-20 heavy atoms
  python de_novo_cavity_growth.py --pdb 1ABC_DU.pdb

  # fine-grained control
  python de_novo_cavity_growth.py --pdb 1ABC_DU.pdb         \\
      --du-index 0          \\   # which DU cavity to target
      --target-min 16       \\
      --target-max 20       \\
      --beam-width 80       \\   # molecules kept per generation
      --max-attach 3        \\   # attachment points sampled per molecule
      --max-frags  8        \\   # fragments tried per attachment point
      --qed-min 0.30        \\   # discard if QED below this
      --sa-max  4.5         \\   # discard if SA above this
      --clash-radius 1.5    \\   # Å: atom counts as clashing if closer
      --cavity-radius 14.0  \\   # Å: atoms beyond this are penalised
      --n-steps 30          \\
      --seed 42             \\
      --out-sdf designs.sdf \\
      --out-csv designs.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── RDKit core ─────────────────────────────────────────────────────────────────
from rdkit import Chem, Geometry, rdBase
from rdkit.Chem import AllChem, QED
from rdkit.Chem import ChemicalFeatures, RDConfig
from rdkit.Chem.rdchem import BondStereo, BondType

# ── SA scorer (RDKit contrib — part of every standard RDKit install) ───────────
def _load_sascorer():
    """Try every known import path for the SA scorer."""
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return sascorer
    except ImportError:
        pass
    try:
        from rdkit.Chem import RDConfig
        sys.path.insert(0, os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer          # noqa: F401  (re-import after path update)
        return sascorer
    except ImportError:
        return None

_SA = _load_sascorer()

# ──────────────────────────────────────────────────────────────────────────────
# Fragment / growth library
# ──────────────────────────────────────────────────────────────────────────────
#
# Single-atom additions
# (display_name, atomic_num, bond_to_parent, extras)
# *extras* is a list of (atomic_num, bond_type) atoms added *to the new atom*.
# Example: C=O attaches C (single) to the parent, then adds O with a DOUBLE
# bond to that new carbon.
#
SINGLE_ATOM_FRAGS: List[Tuple[str, int, BondType, Optional[List[Tuple[int, BondType]]]]] = [
    ("C",    6,  BondType.SINGLE,  None),   # methylene / methyl
    ("N",    7,  BondType.SINGLE,  None),   # amine
    ("O",    8,  BondType.SINGLE,  None),   # ether / alcohol
    ("S",   16,  BondType.SINGLE,  None),   # thioether
    ("F",    9,  BondType.SINGLE,  None),   # fluoro
    ("Cl",  17,  BondType.SINGLE,  None),   # chloro
    ("Br",  35,  BondType.SINGLE,  None),   # bromo
    ("C=O",  6,  BondType.SINGLE,  [(8,  BondType.DOUBLE)]),                  # carbonyl / carboxyl precursor
    ("C=N",  6,  BondType.SINGLE,  [(7,  BondType.DOUBLE)]),                  # imine
    ("C#N",  6,  BondType.SINGLE,  [(7,  BondType.TRIPLE)]),                  # nitrile: -C#N
    ("C#C",  6,  BondType.SINGLE,  [(6,  BondType.TRIPLE)]),                  # alkyne: -C#C
    ("S=O", 16,  BondType.SINGLE,  [(8,  BondType.DOUBLE)]),                  # sulfoxide-like motif (one oxo)
    ("S(=O)=O", 16, BondType.SINGLE, [(8, BondType.DOUBLE), (8, BondType.DOUBLE)]),  # sulfone (two oxo)
]


# Fixed (non-ring) SMILES fragments with a designated attachment atom.
# The attachment atom is indicated via an RDKit atom-map number in the SMILES
# (e.g. "[N:1](C)C"). The map number is stripped after parsing.
FIXED_SMILES_FRAGS: List[Tuple[str, str, int]] = [
    ("C(=O)N",       "[C:1](=O)N",       1),
    ("C(=O)NC",      "[C:1](=O)NC",      1),
    ("C(=O)N(C)C",   "[C:1](=O)N(C)C",   1),
    ("OC",           "[O:1]C",           1),
    ("NC",           "[N:1]C",           1),
    ("N(C)C",        "[N:1](C)C",        1),
]


# Explicit internal alkene additions with defined E/Z stereochemistry.
# These create the motif: parent-R — CH = CH — CH2 —
# so both alkene carbons have two different substituents (R vs H, and CH2 vs H),
# making stereochemistry well-defined.
ALKENE_FRAGS: List[Tuple[str, BondStereo]] = [
    ("C=C(E)", BondStereo.STEREOE),
    ("C=C(Z)", BondStereo.STEREOZ),
]

# Ring fragments (SMILES).
# By default the grower can rotate the attachment atom across multiple positions
# in the ring (preferring atoms with an H) until a sanitizable/kekulizable
# product is found.
RING_FRAGS: List[Tuple[str, str]] = [
    ("Ph",    "c1ccccc1"   ),   # benzene
    ("Py",    "c1ccncc1"   ),   # pyridine
    ("Nap",   "c1ccc2ccccc2c1"), # naphthalene
    ("Ind",   "c1ccc2[nH]ccc2c1"),# indole
    ("Qui",   "c1ccc2ncccc2c1"), # quinoline
    ("iQui",  "c1ccc2ccncc2c1"), # isoquinoline
    ("Bzt",   "c1ccc2sccc2c1"),  # benzothiophene
    ("Pyz",   "n1ccncc1"   ),   # pyrazine (1,4-diazine)
    ("Pym",   "n1cnccc1"   ),   # pyrimidine (1,3-diazine)
    ("Pydz",  "n1ncccc1"   ),   # pyridazine (1,2-diazine)
    ("Fur",   "c1ccoc1"    ),   # furan
    ("Thio",  "c1ccsc1"    ),   # thiophene
    ("Tz",    "c1nccs1"    ),   # thiazole
    ("Imid",  "c1cnc[nH]1" ),   # imidazole
    ("Oxz",   "c1cnoc1"    ),   # oxazole
    ("Pyr",   "c1cc[nH]c1" ),   # pyrrole (C-attachment)
    ("Indn",  "c1ccc2CCCc2c1"), # indane
    ("Tet",   "c1ccc2CCCCc2c1"),# tetralin
    ("cPr",   "C1CC1"      ),   # cyclopropyl
    ("cPen",  "C1CCCC1"    ),   # cyclopentane
    ("cHex",  "C1CCCCC1"   ),   # cyclohexane
    ("Dec",   "C1CCC2CCCCC2C1"),# decalin
    ("Nor",   "C1CC2CCC1C2"),   # norbornane
    ("BCP",   "C1C2CC1C2"  ),   # bicyclo[1.1.1]pentane
    ("Pip",   "C1CCNCC1"   ),   # piperidine
    ("Mor",   "N1CCOCC1"   ),   # morpholine 
    ("Pipz",  "N1CCNCC1"   ),   # piperazine
    ("Pyrr",  "C1CCNC1"    ),   # pyrrolidine
]

# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclass (mirrors CLI args)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    pdb_path:          str
    du_index:          int   = 0
    target_min:        int   = 16
    target_max:        int   = 30
    beam_width:        int   = 50
    n_steps:           int   = 30
    max_output:        int   = 200
    qed_min:           float = 0.65
    sa_max:            float = 4.5
    clash_radius:      float = 1.8     # Å
    cavity_radius:     float = 14.0    # Å  – atoms beyond this get penalised
    n_embed_attempts:  int   = 15      # ETDG re-seeds on failure
    max_attach:        int   = 10       # attachment points sampled per mol
    max_frags:         int   = 20      # fragments tried per attachment point
    seed:              int   = 42
    use_rings:         bool  = True
    ring_attach_rotate: bool = True    # try multiple ring atoms as attachment points
    ring_attach_max:     int  = 0      # 0 = try all candidate ring atoms
    mmff_opt:          bool  = True
    verbose:           bool  = False

    # Pharmacophore complement scoring (optional; enable by setting ph4_weight > 0)
    ph4_weight:          float = 0.5   # added to composite as: composite += w * ph4
    ph4_protein_radius:  float = 8.0   # Å around anchor to gather protein sites
    ph4_match_dist:      float = 3.5   # Å feature-feature distance cutoff
    ph4_unmatched_weight: float = 0.5  # penalty weight for unmatched ligand features (0 keeps legacy behaviour)
    ph4_include_backbone: bool = True # include peptide N/O sites (can dominate)
    vina_enable:       bool  = False
    vina_strict:       bool  = False
    vina_receptor_backend: str = "meeko"
    vina_prepare_receptor_exe: Optional[str] = None
    vina_reduce_exe: Optional[str] = None
    vina_beam_top_n:   int   = 64      # 0 = score all deduplicated candidates each step
    vina_cache_dir:    Optional[str] = ".vina_receptor_cache"
    out_sdf:           str   = "cavity_designs.sdf"
    out_csv:           str   = "cavity_designs.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

def calc_sa(mol: Chem.Mol) -> float:
    """SA score in [1, 6.5]; lower = more synthesisable."""
    if _SA is not None:
        try:
            return float(_SA.calculateScore(mol))
        except Exception:
            pass
    # Heuristic fallback: ring count + chiral centres
    ri = mol.GetRingInfo()
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    return min(6.5, 2.0 + 0.25 * ri.NumRings() + 0.6 * n_chiral)


def calc_qed(mol: Chem.Mol) -> float:
    """QED in [0, 1]; higher = more drug-like."""
    try:
        return float(QED.qed(mol))
    except Exception:
        return 0.0


def composite_score(
    qed: float,
    sa: float,
    clash_frac: float,
    out_frac: float,
) -> float:
    """
    Combined objective — all components in [0, 1], higher = better.

    Weights                 Rationale
    -------                 ---------
    0.40 × QED              drug-likeness is the primary goal
    0.35 × SA_norm          SA ∈ [1, 6.5] → [1, 0] normalised
    0.25 × (1−clash)        cavity fit / no protein penetration
    −0.30 × out_frac        penalise atoms escaping the cavity sphere
    """
    sa_norm = max(0.0, (6.5 - sa) / 5.5)
    return (
        0.40 * qed
        + 0.35 * sa_norm
        + 0.25 * (1.0 - clash_frac)
        - 0.30 * out_frac
    )


_VINA_PROTEIN_RESIDUES: frozenset = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "MSE", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
})


def _filter_protein_for_vina(protein_content: str, include_polymer_hetatm: bool = True) -> str:
    filtered_lines: List[str] = []
    for line in protein_content.splitlines():
        if line.startswith("ATOM"):
            filtered_lines.append(line)
            continue
        if include_polymer_hetatm and line.startswith("HETATM"):
            resname = line[17:20].strip().upper()
            if resname in _VINA_PROTEIN_RESIDUES:
                filtered_lines.append(line)
            continue
        if line.startswith(("TER", "END")):
            filtered_lines.append(line)
    return "\n".join(filtered_lines) + "\n"


def _compute_vina_box(mol: Chem.Mol) -> Tuple[List[float], List[float]]:
    if mol is None or mol.GetNumConformers() == 0:
        return [0.0, 0.0, 0.0], [20.0, 20.0, 20.0]

    conformer = mol.GetConformer()
    coords = np.array([
        [
            conformer.GetAtomPosition(atom_idx).x,
            conformer.GetAtomPosition(atom_idx).y,
            conformer.GetAtomPosition(atom_idx).z,
        ]
        for atom_idx in range(mol.GetNumAtoms())
    ])
    center = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0)
    size = np.maximum(extent + 8.0, np.array([14.0, 14.0, 14.0]))
    return center.tolist(), size.tolist()


def _run_meeko_cli(arguments: List[str]) -> None:
    command = [sys.executable, "-m", *arguments]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Meeko preparation timed out after 120s: {' '.join(command)}"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Meeko preparation failed.")


def _run_external_command(command: List[str], timeout: int, failure_message: str) -> None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{failure_message} timed out after {timeout}s: {' '.join(command)}") from exc
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or failure_message)


def _looks_like_pdb_text(text: str) -> bool:
    if not text:
        return False
    pdb_prefixes = ("ATOM", "HETATM", "TER", "END", "USER  MOD")
    return any(line.startswith(pdb_prefixes) for line in text.splitlines())


def _prepare_receptor_pdbqt_meeko(protein_pdb: str, receptor_pdbqt_path: str) -> None:
    protein_pdb_path = os.path.splitext(receptor_pdbqt_path)[0] + ".pdb"

    def _write_filtered_input(include_polymer_hetatm: bool) -> None:
        with open(protein_pdb_path, "w", encoding="utf-8") as handle:
            handle.write(_filter_protein_for_vina(protein_pdb, include_polymer_hetatm=include_polymer_hetatm))

    cmd = [
        "meeko.cli.mk_prepare_receptor",
        "--read_pdb", protein_pdb_path,
        "-a",
        "-p", receptor_pdbqt_path,
    ]

    _write_filtered_input(include_polymer_hetatm=True)
    try:
        _run_meeko_cli(cmd)
        return
    except RuntimeError as exc:
        first_error = str(exc)

    # Fallback for modified residues / polymer-like HETATM entries that Meeko
    # fails to pad correctly. Retrying with ATOM-only records is safer than
    # aborting the full Vina-enabled run.
    _write_filtered_input(include_polymer_hetatm=False)
    try:
        _run_meeko_cli(cmd)
    except RuntimeError as exc:
        raise RuntimeError(
            "Meeko receptor preparation failed for both polymer-aware and "
            f"ATOM-only inputs. First error: {first_error}. ATOM-only retry: {exc}"
        ) from exc


def _prepare_receptor_pdbqt_adfr(
    protein_pdb: str,
    receptor_pdbqt_path: str,
    prepare_receptor_exe: Optional[str],
    reduce_exe: Optional[str],
) -> None:
    protein_pdb_path = os.path.splitext(receptor_pdbqt_path)[0] + ".pdb"
    hydrogenated_pdb_path = os.path.splitext(receptor_pdbqt_path)[0] + ".H.pdb"

    with open(protein_pdb_path, "w", encoding="utf-8") as handle:
        handle.write(_filter_protein_for_vina(protein_pdb, include_polymer_hetatm=False))

    prepare_receptor_cmd = prepare_receptor_exe or shutil.which("prepare_receptor")
    if not prepare_receptor_cmd:
        raise RuntimeError(
            "ADFR receptor backend selected, but 'prepare_receptor' was not found. "
            "Install ADFR Suite or pass --vina-prepare-receptor-exe."
        )

    reduce_cmd = reduce_exe or shutil.which("reduce")
    receptor_input_path = protein_pdb_path
    if reduce_cmd:
        reduce_result = subprocess.run(
            [reduce_cmd, "-BUILD", "-Quiet", protein_pdb_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if reduce_result.returncode != 0 and not _looks_like_pdb_text(reduce_result.stdout):
            raise RuntimeError(reduce_result.stderr.strip() or reduce_result.stdout.strip() or "REDUCE hydrogenation failed")
        if reduce_result.returncode != 0:
            warnings.warn(
                "REDUCE reported a non-zero exit code but produced a hydrogenated receptor; continuing. "
                f"Details: {reduce_result.stderr.strip()}",
                RuntimeWarning,
            )
        with open(hydrogenated_pdb_path, "w", encoding="utf-8") as handle:
            handle.write(reduce_result.stdout)
        receptor_input_path = hydrogenated_pdb_path
    else:
        warnings.warn(
            "ADFR receptor backend selected without REDUCE; prepare_receptor will use the filtered PDB as-is. "
            "Pass --vina-reduce-exe or install 'reduce' for more reliable hydrogenation.",
            RuntimeWarning,
        )

    _run_external_command(
        [prepare_receptor_cmd, "-r", receptor_input_path, "-o", receptor_pdbqt_path],
        timeout=180,
        failure_message="ADFR receptor preparation failed",
    )


def _prepare_receptor_pdbqt(
    protein_pdb: str,
    receptor_pdbqt_path: str,
    backend: str = "meeko",
    prepare_receptor_exe: Optional[str] = None,
    reduce_exe: Optional[str] = None,
) -> None:
    backend_name = backend.lower().strip()
    if backend_name == "meeko":
        _prepare_receptor_pdbqt_meeko(protein_pdb, receptor_pdbqt_path)
        return
    if backend_name == "adfr":
        _prepare_receptor_pdbqt_adfr(
            protein_pdb,
            receptor_pdbqt_path,
            prepare_receptor_exe=prepare_receptor_exe,
            reduce_exe=reduce_exe,
        )
        return
    raise RuntimeError(f"Unsupported receptor backend: {backend}")


def _prepare_ligand_pdbqt(mol: Chem.Mol, ligand_pdbqt_path: str) -> None:
    ligand_sdf_path = os.path.splitext(ligand_pdbqt_path)[0] + ".sdf"
    ligand_mol = Chem.Mol(mol)
    ligand_mol = Chem.AddHs(ligand_mol, addCoords=True)
    writer = Chem.SDWriter(ligand_sdf_path)
    writer.write(ligand_mol)
    writer.close()

    _run_meeko_cli([
        "meeko.cli.mk_prepare_ligand",
        "-i", ligand_sdf_path,
        "-o", ligand_pdbqt_path,
    ])


def _parse_vina_score(vina_stdout: str) -> Optional[float]:
    patterns = [
        r"Estimated Free Energy of Binding\s*:\s*(-?\d+(?:\.\d+)?)",
        r"Affinity:\s*(-?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, vina_stdout)
        if match:
            return float(match.group(1))
    return None


def _vina_value(score: Optional[float]) -> float:
    return float("inf") if score is None else float(score)


def _dominates(a: "GrowthState", b: "GrowthState") -> bool:
    a_vina = _vina_value(a.vina_score)
    b_vina = _vina_value(b.vina_score)
    better_or_equal = a.composite >= b.composite and a_vina <= b_vina
    strictly_better = a.composite > b.composite or a_vina < b_vina
    return better_or_equal and strictly_better


def _pareto_rank(states: List["GrowthState"]) -> List["GrowthState"]:
    if not states:
        return []

    remaining = list(states)
    ranked: List[GrowthState] = []
    front_idx = 1
    while remaining:
        front: List[GrowthState] = []
        for state in remaining:
            if not any(_dominates(other, state) for other in remaining if other is not state):
                front.append(state)

        front_sorted = sorted(
            front,
            key=lambda s: (-s.composite, _vina_value(s.vina_score), s.sa, s.smiles),
        )
        for state in front_sorted:
            state.pareto_front = front_idx
        ranked.extend(front_sorted)
        remaining = [state for state in remaining if state not in front]
        front_idx += 1

    return ranked


def _rank_by_vina(states: List["GrowthState"]) -> List["GrowthState"]:
    return sorted(
        states,
        key=lambda s: (_vina_value(s.vina_score), -s.composite, s.sa, s.smiles),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Simple cavity pharmacophore complement scoring
# ──────────────────────────────────────────────────────────────────────────────

_FEAT_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)


def _protein_atom_to_sites(
    resname: str,
    atom_name: str,
    elem: str,
    is_hetatm: bool,
    include_backbone: bool,
) -> List[str]:
    """Map a PDB atom to coarse pharmacophore site types.

    Site types: donor, acceptor, pos, neg, hydrophobe, aromatic.
    Uses residue/atom-name heuristics (PDB has no bond perception).
    """
    res = resname.upper()
    an = atom_name.strip().upper()
    el = elem.strip().upper()

    # Backbone atoms are ubiquitous; exclude by default to avoid dominating.
    if not include_backbone and (not is_hetatm) and an in {"N", "C", "O", "OXT", "CA"}:
        return []

    sites: List[str] = []

    # Charged sidechains
    if res in {"ASP", "GLU"} and an in {"OD1", "OD2", "OE1", "OE2"}:
        return ["acceptor", "neg"]
    if res == "LYS" and an == "NZ":
        return ["donor", "pos"]
    if res == "ARG" and an in {"NE", "NH1", "NH2"}:
        return ["donor", "pos"]

    # Amides
    if res in {"ASN", "GLN"}:
        if an in {"OD1", "OE1"}:
            sites.append("acceptor")
        if an in {"ND2", "NE2"}:
            sites.append("donor")
        return sites

    # Hydroxyls
    if res in {"SER", "THR", "TYR"} and an in {"OG", "OG1", "OH"}:
        return ["donor", "acceptor"]

    # Histidine: ambiguous; treat as aromatic + acceptor on ring nitrogens
    if res == "HIS" and an in {"ND1", "NE2", "CE1", "CD2", "CG"}:
        if an in {"ND1", "NE2"}:
            sites.append("acceptor")
        sites.append("aromatic")
        return sites

    # Aromatics
    if res in {"PHE", "TYR", "TRP"}:
        if el in {"C", "N"}:
            sites.append("aromatic")
        if res == "TRP" and an == "NE1":
            sites.append("donor")
        return sites

    # HETATM cofactors/ions/ligands: conservative generic mapping
    if is_hetatm:
        if el == "O":
            sites.append("acceptor")
        elif el == "N":
            sites.extend(["donor", "acceptor"])
        elif el in {"C", "S"}:
            sites.append("hydrophobe")
        return sites

    # Hydrophobes: sidechain C/S on hydrophobic residues
    if res in {"ALA", "VAL", "LEU", "ILE", "MET", "PRO"} and el in {"C", "S"}:
        return ["hydrophobe"]

    # Fallbacks for sidechain hetero atoms
    if el == "O":
        sites.append("acceptor")
    elif el == "N":
        sites.append("donor")
    elif el == "S":
        sites.append("hydrophobe")
    return sites


def parse_protein_ph4_sites(
    pdb_text: str,
    anchor: np.ndarray,
    radius: float,
    include_backbone: bool,
) -> Dict[str, np.ndarray]:
    """Extract coarse protein pharmacophore sites within *radius* of *anchor*."""
    sites: Dict[str, List[List[float]]] = {
        "donor": [],
        "acceptor": [],
        "pos": [],
        "neg": [],
        "hydrophobe": [],
        "aromatic": [],
    }

    r2 = float(radius) * float(radius)
    ax, ay, az = float(anchor[0]), float(anchor[1]), float(anchor[2])

    for ln in pdb_text.splitlines():
        if len(ln) < 54:
            continue
        record = ln[:6]
        if record not in ("ATOM  ", "HETATM"):
            continue

        resname = ln[17:20].strip()
        atom_name = ln[12:16].strip()

        if resname == "DU" or atom_name == "DU":
            continue
        if resname in _WATER_RESNAMES:
            continue

        elem = ln[76:78].strip() if len(ln) >= 78 else ""
        if elem == "H" or (not elem and atom_name.startswith("H")):
            continue

        try:
            x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
        except ValueError:
            continue

        dx, dy, dz = x - ax, y - ay, z - az
        if dx*dx + dy*dy + dz*dz > r2:
            continue

        kinds = _protein_atom_to_sites(
            resname=resname,
            atom_name=atom_name,
            elem=elem,
            is_hetatm=(record == "HETATM"),
            include_backbone=include_backbone,
        )
        for k in kinds:
            if k in sites:
                sites[k].append([x, y, z])

    return {k: np.array(v, dtype=np.float32) for k, v in sites.items()}


def _nearest_dist(point: np.ndarray, spatial_idx) -> float:
    """Nearest-neighbor distance from *point* to coords/KDTree; inf if empty."""
    if spatial_idx is None:
        return float("inf")
    if hasattr(spatial_idx, "query"):
        d, _ = spatial_idx.query(point, k=1)
        return float(d)
    coords = spatial_idx
    if coords is None or len(coords) == 0:
        return float("inf")
    diffs = coords - point[None, :]
    return float(np.sqrt(np.min(np.sum(diffs**2, axis=1))))


def calc_ph4_complement(
    mol: Chem.Mol,
    protein_sites_idx: Dict[str, object],
    match_dist: float,
    unmatched_weight: float = 0.5,
) -> float:
    """Compute a simple [0,1] complement score ligand->protein.

    The score rewards ligand pharmacophore features that land close to a
    complementary protein site, and (optionally) penalizes ligand features that
    do NOT find a complementary site.

    With unmatched_weight=0, this reduces to the legacy fraction matched:
      score = n_match / n_total

    With unmatched_weight>0:
      raw = (n_match - unmatched_weight * n_unmatched) / n_total
      score = clamp(raw, 0, 1)
    """
    if mol.GetNumConformers() == 0 or not protein_sites_idx:
        return 0.0

    feats = _FEAT_FACTORY.GetFeaturesForMol(mol)
    if not feats:
        return 0.0

    complement = {
        "Donor": "acceptor",
        "Acceptor": "donor",
        "PosIonizable": "neg",
        "NegIonizable": "pos",
        "Hydrophobe": "hydrophobe",
        "Aromatic": "aromatic",
    }

    n_total = 0
    n_match = 0
    for f in feats:
        fam = f.GetFamily()
        if fam not in complement:
            continue
        prot_kind = complement[fam]
        idx = protein_sites_idx.get(prot_kind)
        if idx is None:
            continue
        pos = np.array(list(f.GetPos()), dtype=np.float32)
        n_total += 1
        if _nearest_dist(pos, idx) <= float(match_dist):
            n_match += 1

    if n_total == 0:
        return 0.0
    n_unmatched = int(n_total - n_match)
    raw = (float(n_match) - float(unmatched_weight) * float(n_unmatched)) / float(n_total)
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return float(raw)

# ──────────────────────────────────────────────────────────────────────────────
# PDB parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_du_atoms(pdb_text: str) -> List[Tuple[np.ndarray, str]]:
    """
    Return list of (xyz_array, label) for every DU HETATM marker.

    *label* is the human-readable chain+residue identifier, e.g. "Z/1" or
    "Z/2", derived from PDB columns 22 (chain) and 23-26 (residue sequence).
    This lets the caller print which cavities are available and which one is
    selected via --du-index.
    """
    entries: List[Tuple[np.ndarray, str]] = []
    for ln in pdb_text.splitlines():
        if len(ln) < 54:
            continue
        if not (ln.startswith("HETATM") or ln.startswith("ATOM")):
            continue
        resname   = ln[17:20].strip()
        atom_name = ln[12:16].strip()
        if resname == "DU" or atom_name == "DU":
            try:
                xyz   = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
                chain = ln[21:22].strip() or "?"
                resseq = ln[22:26].strip() or "?"
                label = f"{chain}/{resseq}"
                entries.append((xyz, label))
            except ValueError:
                pass
    return entries


# All residue names that unambiguously denote water molecules across common
# PDB dialects, MD-engine outputs, and CIF-derived files.
_WATER_RESNAMES: frozenset = frozenset({
    "HOH",   # standard PDB / wwPDB
    "WAT",   # CNS, CHARMM
    "H2O",   # uncommon but seen in deposited files
    "TIP",   # TIP3P short form (CHARMM)
    "TIP3",  # TIP3P long form
    "TIP4",  # TIP4P
    "SPC",   # SPC/E water
    "SOL",   # GROMACS default
    "DOD",   # deuterated water (neutron structures)
    "TP3",   # AMBER TIP3P
    "OHH",   # alternate ordering sometimes seen in CIF
})


def parse_protein_coords(
    pdb_text:     str,
    anchor:       Optional[np.ndarray] = None,
    cavity_radius: float               = 0.0,
) -> np.ndarray:
    """
    Return (N, 3) float32 array of protein / cofactor heavy-atom positions
    suitable for clash detection.

    Exclusions
    ----------
    • All DU marker atoms (any resname or atom_name == "DU")
    • Hydrogen atoms (element H, or atom name starting with H when element
      column is absent)
    • All water molecules — recognised by *_WATER_RESNAMES* (covers HOH, WAT,
      TIP3, SOL, SPC, etc.) **plus** any remaining HETATM oxygen that falls
      within *cavity_radius* Å of *anchor* (catches crystal waters placed
      inside the cavity under non-standard names, or waters that DoGSite
      identified as part of the binding site volume).

    Parameters
    ----------
    anchor        : (3,) array — selected DU cavity centre; used only if
                    cavity_radius > 0.
    cavity_radius : Å — HETATM oxygens closer than this to *anchor* are
                    removed regardless of residue name.  Pass 0 (default)
                    to skip this extra filter.
    """
    pts: List[List[float]] = []
    anchor_f = anchor.astype(float) if anchor is not None else None

    for ln in pdb_text.splitlines():
        if len(ln) < 54:
            continue
        record = ln[:6]
        if record not in ("ATOM  ", "HETATM"):
            continue

        resname   = ln[17:20].strip()
        atom_name = ln[12:16].strip()

        # ── Skip all DU markers ───────────────────────────────────────────────
        if resname == "DU" or atom_name == "DU":
            continue

        # ── Skip all recognised water residues ───────────────────────────────
        if resname in _WATER_RESNAMES:
            continue

        # ── Skip hydrogens ────────────────────────────────────────────────────
        elem = ln[76:78].strip() if len(ln) >= 78 else ""
        if elem == "H" or (not elem and atom_name.startswith("H")):
            continue

        try:
            x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
        except ValueError:
            continue

        # ── Skip HETATM oxygens inside the selected cavity ───────────────────
        # This removes crystal waters / solvent molecules that sit inside the
        # binding site but were deposited under non-standard residue names.
        # Only HETATM records are considered (ATOM oxygens are protein atoms).
        if (
            record == "HETATM"
            and (elem == "O" or (not elem and atom_name.startswith("O")))
            and anchor_f is not None
            and cavity_radius > 0.0
        ):
            dx, dy, dz = x - anchor_f[0], y - anchor_f[1], z - anchor_f[2]
            if dx*dx + dy*dy + dz*dz < cavity_radius * cavity_radius:
                continue

        pts.append([x, y, z])

    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# Clash / cavity geometry
# ──────────────────────────────────────────────────────────────────────────────

def _build_spatial_index(coords: np.ndarray):
    """Build a KDTree if scipy is available, else keep raw array."""
    try:
        from scipy.spatial import KDTree
        return KDTree(coords) if len(coords) > 0 else None
    except ImportError:
        return coords if len(coords) > 0 else None


def _mol_heavy_coords(mol: Chem.Mol) -> Optional[np.ndarray]:
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    return np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )


def calc_clash_frac(mol: Chem.Mol, spatial_idx, radius: float) -> float:
    """Fraction of mol heavy atoms within *radius* Å of any protein atom."""
    mc = _mol_heavy_coords(mol)
    if mc is None or spatial_idx is None or len(mc) == 0:
        return 0.0

    if hasattr(spatial_idx, "query_ball_point"):         # KDTree
        hits = spatial_idx.query_ball_point(mc, r=radius)
        n_clash = sum(1 for h in hits if len(h) > 0)
    else:                                                 # numpy brute force
        prot = spatial_idx                               # (M, 3)
        diffs = mc[:, None, :] - prot[None, :, :]       # (N, M, 3)
        n_clash = int(np.any(np.sum(diffs**2, axis=2) < radius**2, axis=1).sum())

    return n_clash / len(mc)


def calc_out_frac(mol: Chem.Mol, anchor: np.ndarray, radius: float) -> float:
    """Fraction of mol heavy atoms farther than *radius* Å from anchor."""
    mc = _mol_heavy_coords(mol)
    if mc is None or len(mc) == 0:
        return 0.0
    dists = np.linalg.norm(mc - anchor[None, :], axis=1)
    return float((dists > radius).sum()) / len(mc)

# ──────────────────────────────────────────────────────────────────────────────
# Molecule manipulation
# ──────────────────────────────────────────────────────────────────────────────

def extendable_atoms(mol: Chem.Mol) -> List[int]:
    """Heavy-atom indices that carry at least one implicit hydrogen."""
    return [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() > 1 and a.GetTotalNumHs() > 0
    ]


def grow_single_atom(
    mol:         Chem.Mol,
    attach_idx:  int,
    new_anum:    int,
    bond_type:   BondType,
    extras:      Optional[List[Tuple[int, BondType]]],
) -> Optional[Chem.Mol]:
    """
    Attach *new_anum* to *attach_idx* via *bond_type*.
    If *dbl_partner* is given, also add that atom to the new atom with a
    double bond (producing C=O, C=N, etc.).
    Returns sanitised mol or None on failure.
    """
    rw = Chem.RWMol(mol)
    ni = rw.AddAtom(Chem.Atom(new_anum))
    rw.AddBond(attach_idx, ni, bond_type)
    if extras:
        for extra_anum, extra_bt in extras:
            pi = rw.AddAtom(Chem.Atom(int(extra_anum)))
            rw.AddBond(ni, pi, extra_bt)
    try:
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


def grow_ring(
    mol:        Chem.Mol,
    attach_idx: int,
    ring_smiles: str,
    rotate_attach: bool = True,
    max_attach_atoms: int = 0,
    verbose: bool = False,
) -> Optional[Chem.Mol]:
    """
    Attach *ring_smiles* to *attach_idx* via a single bond.

    If *rotate_attach* is True, try multiple attachment atoms on the ring until
    the combined molecule sanitizes (including kekulization/aromaticity).
    Candidate atoms are (by default) heavy atoms that carry at least one H
    (i.e., substitutable positions). Returns the first successful product.
    """
    ring_mol = Chem.MolFromSmiles(ring_smiles)
    if ring_mol is None:
        return None

    def _candidate_ring_atoms(rm: Chem.Mol) -> List[int]:
        # Prefer substitutable positions (carry at least one H).
        cand = []
        for a in rm.GetAtoms():
            if a.GetAtomicNum() <= 1:
                continue
            if a.GetTotalNumHs() <= 0:
                continue
            # Avoid aromatic [nH] (pyrrole/indole) as an attachment site.
            if a.GetAtomicNum() == 7 and a.GetIsAromatic():
                continue
            cand.append(a.GetIdx())

        if not cand:
            cand = [a.GetIdx() for a in rm.GetAtoms() if a.GetAtomicNum() > 1]

        # Keep legacy behaviour: try atom 0 first if it is eligible.
        if 0 in cand:
            cand = [0] + [i for i in cand if i != 0]
        return cand

    candidates = [0] if not rotate_attach else _candidate_ring_atoms(ring_mol)
    if max_attach_atoms and max_attach_atoms > 0:
        candidates = candidates[: int(max_attach_atoms)]

    n_orig = mol.GetNumAtoms()
    combo = Chem.CombineMols(mol, ring_mol)

    log_ctx = nullcontext() if verbose else rdBase.BlockLogs()
    with log_ctx:
        for ring_atom_idx in candidates:
            rw = Chem.RWMol(combo)
            rw.AddBond(attach_idx, n_orig + int(ring_atom_idx), BondType.SINGLE)
            try:
                Chem.SanitizeMol(rw)
                return rw.GetMol()
            except Exception:
                continue
    return None


def grow_fixed_fragment(
    mol: Chem.Mol,
    attach_idx: int,
    frag_smiles: str,
    attach_map_num: int = 1,
    bond_type: BondType = BondType.SINGLE,
) -> Optional[Chem.Mol]:
    """Attach a non-ring fragment described by SMILES to *attach_idx*.

    The fragment SMILES must include exactly one atom with atom-map number
    *attach_map_num* to indicate the attachment atom.
    """
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None:
        return None

    frag_attach = None
    for a in frag.GetAtoms():
        if int(a.GetAtomMapNum()) == int(attach_map_num):
            frag_attach = int(a.GetIdx())
        a.SetAtomMapNum(0)
    if frag_attach is None:
        return None

    n_orig = mol.GetNumAtoms()
    combo = Chem.CombineMols(mol, frag)
    rw = Chem.RWMol(combo)
    rw.AddBond(int(attach_idx), n_orig + int(frag_attach), bond_type)
    try:
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


def grow_alkene_internal_stereo(
    mol: Chem.Mol,
    attach_idx: int,
    stereo: BondStereo,
) -> Optional[Chem.Mol]:
    """Add R-CH=CH-CH2- with explicit E/Z on the C=C bond.

    The substituents used to define E/Z are:
      - on the first alkene carbon: the existing parent atom (attach_idx)
      - on the second alkene carbon: the newly added terminal carbon (c3)
    """
    rw = Chem.RWMol(mol)
    c1 = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(attach_idx, c1, BondType.SINGLE)
    c2 = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(c1, c2, BondType.DOUBLE)
    c3 = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(c2, c3, BondType.SINGLE)

    try:
        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)
        b = m2.GetBondBetweenAtoms(int(c1), int(c2))
        if b is None:
            return None
        b.SetStereoAtoms(int(attach_idx), int(c3))
        b.SetStereo(stereo)
        Chem.AssignStereochemistry(m2, force=True, cleanIt=True)
        return m2
    except Exception:
        return None


def _rand_rotation(seed: int) -> np.ndarray:
    """
    Return a uniformly random 3×3 rotation matrix using a quaternion
    parameterisation (Shoemake 1992) seeded by *seed*.
    """
    rng = np.random.default_rng(seed)
    u   = rng.random(3)
    q   = np.array([
        np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
        np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
        np.sqrt(      u[0]) * np.sin(2 * np.pi * u[2]),
        np.sqrt(      u[0]) * np.cos(2 * np.pi * u[2]),
    ])                                              # [x, y, z, w]
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def embed_constrained(
    mol:        Chem.Mol,
    anchor_idx: int,
    anchor_xyz: np.ndarray,
    n_attempts: int  = 8,
    mmff_opt:   bool = True,
    base_seed:  int  = 42,
    verbose:    bool = False,
) -> Tuple[Optional[Chem.Mol], bool]:
    """
    Embed *mol* in 3-D so that *anchor_idx* sits exactly at *anchor_xyz*.

    Strategy (avoids RDKit coordMap translation ambiguity):
      1. Embed mol freely with ETKDGv3 + optional MMFF minimisation.
      2. Translate entire conformer so that *anchor_idx* lands on *anchor_xyz*.
      3. On each retry > 0, also apply a different random rotation around the
         anchor so the beam explores diverse orientations inside the cavity.

    Returns (mol_no_H_with_conformer, True) or (None, False) on failure.
    """
    mol_h = Chem.AddHs(mol)

    # RDKit may emit UFFTYPER warnings during parts of embedding/FF setup for
    # some functional groups (e.g. hypervalent sulfur). These are not fatal for
    # our pipeline, so we silence them unless verbose is requested.
    log_ctx = nullcontext() if verbose else rdBase.BlockLogs()

    with log_ctx:
        for attempt in range(n_attempts):
            params            = AllChem.ETKDGv3()
            params.randomSeed = base_seed + attempt * 17

            if AllChem.EmbedMolecule(mol_h, params) < 0:
                # If ETDG fails, retry with random initial coords
                params.useRandomCoords = True
                if AllChem.EmbedMolecule(mol_h, params) < 0:
                    continue

            if mmff_opt:
                try:
                    mp = AllChem.MMFFGetMoleculeProperties(mol_h)
                    ff = AllChem.MMFFGetMoleculeForceField(mol_h, mp)
                    if ff is not None:
                        ff.Minimize(maxIts=300)
                except Exception:
                    pass

            conf   = mol_h.GetConformer()
            n_all  = mol_h.GetNumAtoms()

            # Collect all atom positions as (N, 3) array
            coords = np.array(
                [list(conf.GetAtomPosition(i)) for i in range(n_all)],
                dtype=float,
            )

            # Current position of anchor atom
            anchor_current = coords[anchor_idx].copy()

            # For attempt > 0 apply a random rotation around current anchor position
            if attempt > 0:
                R      = _rand_rotation(base_seed + attempt * 1000)
                coords = (R @ (coords - anchor_current).T).T + anchor_current

            # Translate so anchor lands on anchor_xyz
            offset = anchor_xyz - coords[anchor_idx]
            coords += offset

            # Write back
            for i in range(n_all):
                conf.SetAtomPosition(i, coords[i].tolist())

            return Chem.RemoveHs(mol_h), True

    return None, False

# ──────────────────────────────────────────────────────────────────────────────
# Growth state
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GrowthState:
    mol:        Chem.Mol
    smiles:     str
    composite:  float
    vina_score: Optional[float]
    qed:        float
    sa:         float
    clash_frac: float
    out_frac:   float
    ph4:        float
    step:       int
    pareto_front: Optional[int] = None

    def hac(self) -> int:
        """Heavy-atom count."""
        return sum(1 for a in self.mol.GetAtoms() if a.GetAtomicNum() > 1)

# ──────────────────────────────────────────────────────────────────────────────
# Main grower (beam search)
# ──────────────────────────────────────────────────────────────────────────────

class CavityGrower:
    """
    Beam-search fragment grower.

    Parameters
    ----------
    anchor        : (3,) ndarray — DU atom position = starting / constraint point
    protein_coords: (N, 3) ndarray — all protein heavy-atom positions
    cfg           : Config
    """

    def __init__(
        self,
        anchor:         np.ndarray,
        protein_coords: np.ndarray,
        protein_pdb_text: str,
        cfg:            Config,
    ) -> None:
        self.anchor      = anchor
        self.spatial_idx = _build_spatial_index(protein_coords)
        self.protein_pdb_text = protein_pdb_text
        self.cfg         = cfg
        self.rng         = random.Random(cfg.seed)

        # Per-kind protein pharmacophore site indices (only built if enabled)
        self.ph4_sites_idx: Dict[str, object] = {}
        self.vina_cache: Dict[str, Optional[float]] = {}
        self._vina_tempdir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._vina_receptor_pdbqt: Optional[str] = None
        if self.cfg.vina_enable:
            try:
                self._prepare_vina_receptor()
            except RuntimeError as exc:
                if self.cfg.vina_strict:
                    raise
                warnings.warn(
                    "Vina receptor preparation failed; continuing without Vina scoring for this run. "
                    f"Reason: {exc}",
                    RuntimeWarning,
                )
                self.cfg.vina_enable = False

    def set_protein_ph4_sites(self, sites: Dict[str, np.ndarray]) -> None:
        """Set protein pharmacophore sites (coords), building per-kind indices."""
        self.ph4_sites_idx = {}
        if not sites:
            return
        for k, coords in sites.items():
            if coords is None or len(coords) == 0:
                continue
            self.ph4_sites_idx[k] = _build_spatial_index(coords)

    def _vina_receptor_cache_key(self) -> str:
        cache_input = "\n".join(
            [
                self.cfg.vina_receptor_backend,
                str(self.cfg.vina_prepare_receptor_exe or ""),
                str(self.cfg.vina_reduce_exe or ""),
                self.protein_pdb_text,
            ]
        )
        return hashlib.sha256(cache_input.encode("utf-8")).hexdigest()[:16]

    def _prepare_vina_receptor(self) -> None:
        vina_exe = shutil.which("vina")
        if vina_exe is None:
            raise RuntimeError("--vina-enable requires the 'vina' executable on PATH.")

        cache_dir = self.cfg.vina_cache_dir
        if cache_dir:
            cache_root = Path(cache_dir).resolve()
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_key = self._vina_receptor_cache_key()
            receptor_dir = cache_root / cache_key
            receptor_dir.mkdir(parents=True, exist_ok=True)
            self._vina_receptor_pdbqt = str(receptor_dir / "protein.pdbqt")
            if os.path.exists(self._vina_receptor_pdbqt):
                return
        else:
            self._vina_tempdir = tempfile.TemporaryDirectory()
            self._vina_receptor_pdbqt = os.path.join(self._vina_tempdir.name, "protein.pdbqt")

        _prepare_receptor_pdbqt(
            self.protein_pdb_text,
            self._vina_receptor_pdbqt,
            backend=self.cfg.vina_receptor_backend,
            prepare_receptor_exe=self.cfg.vina_prepare_receptor_exe,
            reduce_exe=self.cfg.vina_reduce_exe,
        )

    def _score_vina(self, state: GrowthState) -> Optional[float]:
        if not self.cfg.vina_enable:
            return None
        if state.smiles in self.vina_cache:
            return self.vina_cache[state.smiles]
        if self._vina_receptor_pdbqt is None:
            raise RuntimeError("Vina receptor was not prepared.")

        center, size = _compute_vina_box(state.mol)
        with tempfile.TemporaryDirectory() as temp_dir:
            ligand_pdbqt = os.path.join(temp_dir, "ligand.pdbqt")
            _prepare_ligand_pdbqt(state.mol, ligand_pdbqt)
            vina_cmd = [
                "vina",
                "--receptor", self._vina_receptor_pdbqt,
                "--ligand", ligand_pdbqt,
                "--center_x", str(center[0]),
                "--center_y", str(center[1]),
                "--center_z", str(center[2]),
                "--size_x", str(size[0]),
                "--size_y", str(size[1]),
                "--size_z", str(size[2]),
                "--score_only",
            ]
            result = subprocess.run(vina_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Vina scoring failed.")

        score = _parse_vina_score(result.stdout)
        if score is None:
            raise RuntimeError("Could not parse a Vina score from the output.")

        self.vina_cache[state.smiles] = score
        return score

    def _ensure_vina_scores(self, states: List[GrowthState], limit: Optional[int] = None) -> None:
        if not self.cfg.vina_enable:
            return

        subset = states if limit is None or limit <= 0 else states[: min(len(states), limit)]
        for state in subset:
            if state.vina_score is None:
                state.vina_score = self._score_vina(state)

    def _rank_states(self, states: List[GrowthState], beam_stage: bool) -> List[GrowthState]:
        if not states:
            return []

        ranked = sorted(states, key=lambda s: s.composite, reverse=True)
        if not self.cfg.vina_enable:
            return ranked

        self._ensure_vina_scores(ranked)

        if beam_stage:
            negative_vina = [state for state in ranked if state.vina_score is not None and state.vina_score < 0.0]
            if negative_vina:
                return _rank_by_vina(negative_vina)
            return _rank_by_vina(ranked)[:10]

        return _pareto_rank(ranked)

    # ── scoring ──────────────────────────────────────────────────────────────

    def _score(self, mol: Chem.Mol) -> Tuple[float, float, float, float, float]:
        """Returns (composite, qed, sa, clash_frac, out_frac)."""
        qed  = calc_qed(mol)
        sa   = calc_sa(mol)
        cf   = calc_clash_frac(mol, self.spatial_idx, self.cfg.clash_radius)
        of   = calc_out_frac(mol, self.anchor, self.cfg.cavity_radius)
        comp = composite_score(qed, sa, cf, of)

        if float(self.cfg.ph4_weight) > 0.0 and self.ph4_sites_idx:
            ph4 = calc_ph4_complement(
                mol,
                self.ph4_sites_idx,
                self.cfg.ph4_match_dist,
                unmatched_weight=self.cfg.ph4_unmatched_weight,
            )
            comp += float(self.cfg.ph4_weight) * float(ph4)
        return comp, qed, sa, cf, of

    def _make_state(self, mol: Chem.Mol, step: int) -> Optional[GrowthState]:
        """Embed mol, apply hard filters, compute scores → GrowthState or None."""
        embedded, ok = embed_constrained(
            mol,
            anchor_idx  = 0,           # atom 0 is always the anchor carbon
            anchor_xyz  = self.anchor,
            n_attempts  = self.cfg.n_embed_attempts,
            mmff_opt    = self.cfg.mmff_opt,
            base_seed   = self.cfg.seed + step * 31,
            verbose     = self.cfg.verbose,
        )
        if not ok or embedded is None:
            return None

        cf = calc_clash_frac(embedded, self.spatial_idx, self.cfg.clash_radius)
        of = calc_out_frac(embedded, self.anchor, self.cfg.cavity_radius)

        # Hard geometric filters (cheap; skip scoring if violated)
        if cf > 0.30:   # > 30 % of atoms clash with protein → discard
            return None
        if of > 0.25:   # > 25 % of atoms outside cavity sphere → discard
            return None

        comp, qed, sa, cf2, of2 = self._score(embedded)
        ph4 = 0.0
        if float(self.cfg.ph4_weight) > 0.0 and self.ph4_sites_idx:
            ph4 = calc_ph4_complement(
                embedded,
                self.ph4_sites_idx,
                self.cfg.ph4_match_dist,
                unmatched_weight=self.cfg.ph4_unmatched_weight,
            )

        return GrowthState(
            mol        = embedded,
            smiles     = Chem.MolToSmiles(embedded),
            composite  = comp,
            vina_score = None,
            qed        = qed,
            sa         = sa,
            clash_frac = cf2,
            out_frac   = of2,
            ph4        = ph4,
            step       = step,
        )

    # ── seed ─────────────────────────────────────────────────────────────────

    def _seed_beam(self) -> List[GrowthState]:
        """Create the initial beam: a single C atom placed at the anchor."""
        seed_mol = Chem.RWMol(Chem.MolFromSmiles("C"))
        # Manually create a 1-atom conformer at the anchor position
        from rdkit.Chem.rdchem import Conformer
        conf = Conformer(1)
        conf.SetAtomPosition(0, self.anchor.tolist())
        seed_mol.AddConformer(conf, assignId=True)
        seed_mol = seed_mol.GetMol()
        comp, qed, sa, cf, of = self._score(seed_mol)
        ph4 = 0.0
        if float(self.cfg.ph4_weight) > 0.0 and self.ph4_sites_idx:
            ph4 = calc_ph4_complement(
                seed_mol,
                self.ph4_sites_idx,
                self.cfg.ph4_match_dist,
                unmatched_weight=self.cfg.ph4_unmatched_weight,
            )
        return [GrowthState(
            mol        = seed_mol,
            smiles     = "C",
            composite  = comp,
            vina_score = None,
            qed        = qed,
            sa         = sa,
            clash_frac = cf,
            out_frac   = of,
            ph4        = ph4,
            step       = 0,
        )]

    # ── expand one state ─────────────────────────────────────────────────────

    def _expand(self, state: GrowthState) -> List[GrowthState]:
        """Generate valid child states from *state*."""
        cfg      = self.cfg
        children: List[GrowthState] = []
        atoms    = extendable_atoms(state.mol)
        if not atoms:
            return children

        # Sample a limited number of attachment points to keep the beam tractable
        n_attach = min(len(atoms), cfg.max_attach)
        attach_pts = self.rng.sample(atoms, n_attach)

        # Build the full action list once per expand call
        actions: List[Tuple] = []
        for frag in SINGLE_ATOM_FRAGS:
            actions.append(("atom", frag))
        for frag in FIXED_SMILES_FRAGS:
            actions.append(("fixed", frag))
        for name, stereo in ALKENE_FRAGS:
            actions.append(("alkene", (name, stereo)))
        if cfg.use_rings:
            for rfrag in RING_FRAGS:
                actions.append(("ring", rfrag))

        for attach_idx in attach_pts:
            # Sample a subset of fragments for this attachment point
            sampled = self.rng.sample(actions, min(len(actions), cfg.max_frags))

            for kind, frag in sampled:
                if kind == "atom":
                    name, at_num, bt, extras = frag
                    new_mol = grow_single_atom(state.mol, attach_idx, at_num, bt, extras)
                elif kind == "fixed":
                    _name, frag_smiles, map_num = frag
                    new_mol = grow_fixed_fragment(state.mol, attach_idx, frag_smiles, attach_map_num=map_num)
                elif kind == "alkene":
                    _name, stereo = frag
                    new_mol = grow_alkene_internal_stereo(state.mol, attach_idx, stereo)
                else:
                    rname, rsmiles = frag
                    new_mol = grow_ring(
                        state.mol,
                        attach_idx,
                        rsmiles,
                        rotate_attach=cfg.ring_attach_rotate,
                        max_attach_atoms=cfg.ring_attach_max,
                        verbose=cfg.verbose,
                    )

                if new_mol is None:
                    continue

                child = self._make_state(new_mol, state.step + 1)
                if child is not None:
                    children.append(child)

        return children

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> List[GrowthState]:
        """
        Execute the beam search.

        Returns a list of GrowthState objects for molecules that reached the
        target heavy-atom count and passed QED / SA thresholds.
        """
        cfg       = self.cfg
        beam      = self._seed_beam()
        completed : List[GrowthState] = []
        seen      : set[str]          = set()
        t0        = time.time()

        try:
            for step in range(1, cfg.n_steps + 1):
                if not beam:
                    print("  Beam exhausted - stopping early.")
                    break

                # ── Expand ───────────────────────────────────────────────────
                candidates: List[GrowthState] = []
                for state in beam:
                    candidates.extend(self._expand(state))

                if not candidates:
                    print(f"  No valid expansions at step {step}.")
                    break

                # ── De-duplicate by canonical SMILES ─────────────────────────
                best: Dict[str, GrowthState] = {}
                for c in candidates:
                    if c.smiles not in best or c.composite > best[c.smiles].composite:
                        best[c.smiles] = c
                candidates = self._rank_states(list(best.values()), beam_stage=True)

                # ── Harvest molecules that hit the target HAC window ──────────
                for c in candidates:
                    if cfg.target_min <= c.hac() <= cfg.target_max:
                        if c.smiles not in seen:
                            if c.qed >= cfg.qed_min and c.sa <= cfg.sa_max:
                                completed.append(c)
                                seen.add(c.smiles)

                # ── Update beam: keep only molecules still below target ───────
                growing = [c for c in candidates if c.hac() < cfg.target_min]
                beam = growing[: cfg.beam_width]

                # ── Progress reporting ────────────────────────────────────────
                elapsed = time.time() - t0
                if cfg.verbose or step % 5 == 0:
                    top = candidates[0]
                    vina_text = "NA" if top.vina_score is None else f"{top.vina_score:.2f}"
                    front_text = "NA" if top.pareto_front is None else str(top.pareto_front)
                    print(
                        f"  step {step:3d} | beam {len(beam):4d} | "
                        f"candidates {len(candidates):5d} | "
                        f"done {len(completed):4d} | "
                        f"top score {top.composite:.3f} "
                        f"(QED={top.qed:.2f}, SA={top.sa:.2f}, Vina={vina_text}, Front={front_text}) | "
                        f"{elapsed:.1f}s"
                    )

                if len(completed) >= cfg.max_output:
                    print(f"  Reached max_output={cfg.max_output}.")
                    break

            completed = self._rank_states(completed, beam_stage=False)
            return completed[: cfg.max_output]
        finally:
            if self._vina_tempdir is not None:
                self._vina_tempdir.cleanup()
                self._vina_tempdir = None
                self._vina_receptor_pdbqt = None

# ──────────────────────────────────────────────────────────────────────────────
# Output writers
# ──────────────────────────────────────────────────────────────────────────────

def write_sdf(states: List[GrowthState], path: str) -> None:
    """Write all molecules with SDF properties to *path*."""
    writer = Chem.SDWriter(path)
    for i, s in enumerate(states):
        mol = copy.deepcopy(s.mol)
        mol.SetProp("_Name",       f"design_{i+1:04d}")
        mol.SetProp("SMILES",      s.smiles)
        mol.SetProp("composite",   f"{s.composite:.4f}")
        if s.vina_score is not None:
            mol.SetProp("vina_score",  f"{s.vina_score:.4f}")
        mol.SetProp("QED",         f"{s.qed:.4f}")
        mol.SetProp("SA_score",    f"{s.sa:.4f}")
        mol.SetProp("clash_frac",  f"{s.clash_frac:.4f}")
        mol.SetProp("out_frac",    f"{s.out_frac:.4f}")
        mol.SetProp("ph4",         f"{s.ph4:.4f}")
        mol.SetProp("heavy_atoms", str(s.hac()))
        mol.SetProp("growth_step", str(s.step))
        if s.pareto_front is not None:
            mol.SetProp("pareto_front", str(s.pareto_front))
        writer.write(mol)
    writer.close()
    print(f"  Wrote {len(states):4d} molecules  ->  {path}")


def write_csv(states: List[GrowthState], path: str) -> None:
    """Write summary CSV with key descriptors."""
    rows = [
        {
            "rank":        i + 1,
            "id":          f"design_{i+1:04d}",
            "smiles":      s.smiles,
            "pareto_front": s.pareto_front if s.pareto_front is not None else "",
            "composite":   round(s.composite,  4),
            "vina_score":  round(s.vina_score, 4) if s.vina_score is not None else "",
            "QED":         round(s.qed,         4),
            "SA_score":    round(s.sa,          4),
            "clash_frac":  round(s.clash_frac,  4),
            "out_frac":    round(s.out_frac,    4),
            "ph4":         round(s.ph4,         4),
            "heavy_atoms": s.hac(),
            "growth_step": s.step,
        }
        for i, s in enumerate(states)
    ]
    if not rows:
        print("  No molecules to write.")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows):4d} rows        ->  {path}")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "De novo fragment-growing molecule design anchored to DU cavity "
            "markers in a protein PDB, scored with QED and SA (RDKit)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    p.add_argument("--pdb",          required=True,
                   help="PDB file containing DU HETATM markers from dogsite_interface_cavities.py")
    p.add_argument("--du-index",     type=int,   default=0,
                   help="0-based index of the DU atom to use as anchor")

    # Target
    p.add_argument("--target-min",   type=int,   default=16,
                   help="Minimum heavy-atom count of accepted molecules")
    p.add_argument("--target-max",   type=int,   default=20,
                   help="Maximum heavy-atom count of accepted molecules")

    # Beam search
    p.add_argument("--beam-width",   type=int,   default=50,
                   help="Molecules retained in the beam per generation")
    p.add_argument("--n-steps",      type=int,   default=30,
                   help="Maximum growth steps")
    p.add_argument("--max-output",   type=int,   default=200,
                   help="Stop after this many molecules are accepted")
    p.add_argument("--max-attach",   type=int,   default=3,
                   help="Attachment points sampled per molecule per step (limits combinatorics)")
    p.add_argument("--max-frags",    type=int,   default=8,
                   help="Fragments tried per attachment point per step")

    # Score thresholds
    p.add_argument("--qed-min",      type=float, default=0.25,
                   help="Minimum QED for a completed molecule (0–1, higher = more drug-like)")
    p.add_argument("--sa-max",       type=float, default=5.5,
                   help="Maximum SA score (1–6.5, lower = more synthesisable)")

    # Geometry
    p.add_argument("--clash-radius", type=float, default=1.5,
                   help="Clash detection radius in Å")
    p.add_argument("--cavity-radius",type=float, default=14.0,
                   help="Atoms beyond this distance from anchor are penalised (Å)")

    # Pharmacophore complement reward (optional)
    p.add_argument(
        "--ph4-weight",
        type=float,
        default=0.25,
        help=(
            "Add w*ph4_complement to the composite score (0 disables). "
            "ph4_complement rewards ligand pharmacophore features that land near "
            "complementary protein sites around the anchor (and can penalize unmatched features)."
        ),
    )
    p.add_argument(
        "--ph4-protein-radius",
        type=float,
        default=8.0,
        help="Protein atoms within this radius (Å) of the DU anchor define cavity pharmacophore sites",
    )
    p.add_argument(
        "--ph4-match-dist",
        type=float,
        default=3.5,
        help="Max distance (Å) for a ligand feature to match a complementary protein site",
    )
    p.add_argument(
        "--ph4-unmatched-weight",
        type=float,
        default=0.5,
        help=(
            "Penalty weight for ligand pharmacophore features that do not find a complementary protein site. "
            "0 reproduces the legacy behaviour (only reward matches)."
        ),
    )
    p.add_argument(
        "--ph4-include-backbone",
        action="store_true",
        help="Include backbone N/O sites when extracting protein pharmacophore sites (can dominate)",
    )

    p.add_argument(
        "--vina-enable",
        action="store_true",
        help=(
            "Rescore candidate poses with AutoDock Vina and use Pareto fronts over "
            "(maximize composite, minimize vina_score) for beam selection and final ranking"
        ),
    )
    p.add_argument(
        "--vina-strict",
        action="store_true",
        help="Fail the run if Vina receptor preparation or scoring fails instead of falling back to composite-only ranking",
    )
    p.add_argument(
        "--vina-receptor-backend",
        choices=("meeko", "adfr"),
        default="meeko",
        help="Backend used to prepare receptor PDBQT files for Vina",
    )
    p.add_argument(
        "--vina-prepare-receptor-exe",
        default=None,
        help="Path to the ADFR Suite prepare_receptor executable when using --vina-receptor-backend adfr",
    )
    p.add_argument(
        "--vina-reduce-exe",
        default=None,
        help="Path to the REDUCE executable used to add receptor hydrogens before ADFR preparation",
    )
    p.add_argument(
        "--vina-beam-top-n",
        type=int,
        default=64,
        help=(
            "Deprecated and ignored. Beam-stage Vina selection now scores all deduplicated candidates, "
            "forwards only compounds with negative Vina scores, and otherwise falls back to the 10 best "
            "Vina-scored compounds."
        ),
    )
    p.add_argument(
        "--vina-cache-dir",
        default=".vina_receptor_cache",
        help=(
            "Directory for persistent receptor PDBQT caching across runs. "
            "Set to an empty string to disable persistent caching."
        ),
    )

    # Embedding
    p.add_argument("--n-embed",      type=int,   default=8, dest="n_embed_attempts",
                   help="ETDG re-seeding attempts per molecule")
    p.add_argument("--no-mmff",      action="store_true",
                   help="Skip MMFF optimisation after ETDG embedding")
    p.add_argument("--no-rings",     action="store_true",
                   help="Disable ring-fragment additions")
    p.add_argument(
        "--no-ring-attach-rotate",
        action="store_true",
        help=(
            "Disable ring attachment-point rotation (use the ring SMILES atom 0 only). "
            "Rotation is slower but reduces sanitize/kekulize failures for some rings"
        ),
    )
    p.add_argument(
        "--ring-attach-max",
        type=int,
        default=0,
        help=(
            "Max ring atoms to try as attachment points per ring fragment when rotation is enabled. "
            "0 = try all candidate atoms (slowest, most robust)"
        ),
    )

    # Misc
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--verbose",      action="store_true")
    p.add_argument("--out-sdf",      default="cavity_designs.sdf")
    p.add_argument("--out-csv",      default="cavity_designs.csv")

    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    pdb_text = Path(args.pdb).read_text(encoding="utf-8", errors="replace")

    # ── DU atoms ──────────────────────────────────────────────────────────────
    du_entries = parse_du_atoms(pdb_text)   # List[Tuple[xyz, label]]
    if not du_entries:
        print(f"ERROR: No DU atoms found in {args.pdb}", file=sys.stderr)
        return 1
    if args.du_index >= len(du_entries):
        print(
            f"ERROR: --du-index {args.du_index} out of range "
            f"(found {len(du_entries)} DU atom(s): indices 0-{len(du_entries)-1})",
            file=sys.stderr,
        )
        return 1

    # List every available DU cavity so the user can verify the selection
    print(f"\nDU cavities found in {Path(args.pdb).name}:")
    for i, (xyz, lbl) in enumerate(du_entries):
        marker = " <selected>" if i == args.du_index else ""
        print(f"  [{i}]  chain/res {lbl:8s}  "
              f"({xyz[0]:7.2f}, {xyz[1]:7.2f}, {xyz[2]:7.2f}){marker}")

        # Single anchor for this run - no other DU atoms are touched
    anchor, anchor_label = du_entries[args.du_index]
    print(f"\nGrowing from DU[{args.du_index}] ({anchor_label})  "
            f"->  anchor ({anchor[0]:.2f}, {anchor[1]:.2f}, {anchor[2]:.2f})")
    if len(du_entries) > 1:
        print(f"  (remaining {len(du_entries)-1} DU cavity/cavities ignored "
              f"for this run - use --du-index to target a different one)")

    # ── Protein coords (waters fully excluded) ────────────────────────────────
    # All recognised water residues are stripped.
    # Additionally, any HETATM oxygen within cavity_radius of the selected
    # anchor is removed so that crystal waters sitting inside the cavity do
    # not generate false clashes against the growing molecule.
    prot = parse_protein_coords(
        pdb_text,
        anchor        = anchor,
        cavity_radius = args.cavity_radius,
    )
    print(f"Protein heavy atoms (waters & DU removed): {len(prot)}")

    if _SA is None:
          print("\nWARNING: rdkit.Contrib.SA_Score not found - "
              "using ring-complexity heuristic for SA.\n"
              "For accurate SA scoring install/link the RDKit contrib module.")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = Config(
        pdb_path          = args.pdb,
        du_index          = args.du_index,
        target_min        = args.target_min,
        target_max        = args.target_max,
        beam_width        = args.beam_width,
        n_steps           = args.n_steps,
        max_output        = args.max_output,
        qed_min           = args.qed_min,
        sa_max            = args.sa_max,
        clash_radius      = args.clash_radius,
        cavity_radius     = args.cavity_radius,
        n_embed_attempts  = args.n_embed_attempts,
        max_attach        = args.max_attach,
        max_frags         = args.max_frags,
        seed              = args.seed,
        use_rings         = not args.no_rings,
        ring_attach_rotate = not args.no_ring_attach_rotate,
        ring_attach_max    = args.ring_attach_max,
        mmff_opt          = not args.no_mmff,
        verbose           = args.verbose,
        ph4_weight         = args.ph4_weight,
        ph4_protein_radius = args.ph4_protein_radius,
        ph4_match_dist     = args.ph4_match_dist,
        ph4_unmatched_weight = args.ph4_unmatched_weight,
        ph4_include_backbone = args.ph4_include_backbone,
        vina_enable       = args.vina_enable,
        vina_strict       = args.vina_strict,
        vina_receptor_backend = args.vina_receptor_backend,
        vina_prepare_receptor_exe = args.vina_prepare_receptor_exe,
        vina_reduce_exe   = args.vina_reduce_exe,
        vina_beam_top_n   = args.vina_beam_top_n,
        vina_cache_dir    = args.vina_cache_dir or None,
        out_sdf           = args.out_sdf,
        out_csv           = args.out_csv,
    )

    # ── Print run summary ─────────────────────────────────────────────────────
    print(
        f"\n{'='*65}\n"
        f"  Beam width     : {cfg.beam_width}    "
        f"  Steps max      : {cfg.n_steps}\n"
        f"  HAC target     : {cfg.target_min}-{cfg.target_max}    "
        f"  Max output     : {cfg.max_output}\n"
        f"  QED >= {cfg.qed_min:.2f}          "
        f"  SA  <= {cfg.sa_max:.1f}\n"
        f"  Clash radius   : {cfg.clash_radius} A    "
        f"  Cavity radius  : {cfg.cavity_radius} A\n"
        f"  Attach/mol     : {cfg.max_attach}    "
        f"  Frags/attach   : {cfg.max_frags}    "
        f"  Use rings      : {cfg.use_rings}\n"
        f"  Vina Pareto    : {cfg.vina_enable}    "
        f"  Vina strict    : {cfg.vina_strict}    "
        f"  Vina backend   : {cfg.vina_receptor_backend}    "
        f"  Vina top-N     : {cfg.vina_beam_top_n}\n"
        f"{'='*65}\n"
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    grower  = CavityGrower(anchor, prot, pdb_text, cfg)
    if float(cfg.ph4_weight) > 0.0:
        sites = parse_protein_ph4_sites(
            pdb_text,
            anchor=anchor,
            radius=cfg.ph4_protein_radius,
            include_backbone=cfg.ph4_include_backbone,
        )
        grower.set_protein_ph4_sites(sites)
    results = grower.run()

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Accepted molecules : {len(results)}")
    if results:
        top = results[0]
        vina_scores = [state.vina_score for state in results if state.vina_score is not None]
        print(f"  Best composite     : {top.composite:.4f}")
        if vina_scores:
            print(f"  Best Vina          : {min(vina_scores):.4f}")
        if top.pareto_front is not None:
            print(f"  Best Pareto front  : {top.pareto_front}")
        print(f"  Best QED           : {top.qed:.4f}")
        print(f"  Best SA            : {top.sa:.4f}")
        print(f"  Best SMILES        : {top.smiles}")
    print(f"{'='*65}\n")

    # ── Write outputs ─────────────────────────────────────────────────────────
    write_sdf(results, cfg.out_sdf)
    write_csv(results, cfg.out_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
