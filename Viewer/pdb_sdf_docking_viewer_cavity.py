import streamlit as st
import pandas as pd
import py3Dmol
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import streamlit.components.v1 as components
import subprocess
import tempfile
import os
import io
import re
import shutil
import hashlib
import sys
import numpy as np
import itertools

try:
    # `stmol` tends to embed py3Dmol more reliably in Streamlit than raw components.html.
    from stmol import showmol  # type: ignore
except Exception:
    showmol = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(APP_DIR)
REPO_PYTHON = os.path.join(WORKSPACE_DIR, '.venv', 'Scripts', 'python.exe')

# Page config
st.set_page_config(page_title="PDB-SDF Docking Viewer", layout="wide")
st.title("🧬 PDB-SDF Docking Viewer with VINA")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] label {
        font-size: 0.8rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'ligands' not in st.session_state:
    st.session_state.ligands = []
if 'protein' not in st.session_state:
    st.session_state.protein = None
if 'protein_content' not in st.session_state:
    st.session_state.protein_content = None
if 'protein_name' not in st.session_state:
    st.session_state.protein_name = None
if 'protein_upload_id' not in st.session_state:
    st.session_state.protein_upload_id = None
if 'show_cavity_residues' not in st.session_state:
    st.session_state.show_cavity_residues = True
if 'cavity_cutoff' not in st.session_state:
    st.session_state.cavity_cutoff = 4.5
if 'sdf_content' not in st.session_state:
    st.session_state.sdf_content = None
if 'sdf_upload_id' not in st.session_state:
    st.session_state.sdf_upload_id = None
if 'vina_signature' not in st.session_state:
    st.session_state.vina_signature = None
if 'vina_status' not in st.session_state:
    st.session_state.vina_status = None

VINA_PIPELINE_VERSION = '2026-04-25-2'

SCORE_PROPERTY_HINTS = (
    'vina_score',
    'docking_score',
    'binding_energy',
    'affinity',
    'minimizedaffinity',
    'cnnscore',
    'delta_g',
    'composite',
    'composite_grower',
    'score',
    'energy',
    'SA_score',
    'SA',
)

PROTEIN_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'MSE', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
}

WATER_RESIDUES = {'HOH', 'WAT', 'SOL'}


def _coerce_score(value):
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def extract_score(props, mol_block):
    """Extract the most likely docking score stored in an SDF entry."""
    for key in SCORE_PROPERTY_HINTS:
        if key in props:
            score = _coerce_score(props[key])
            if score is not None:
                return score, key

    for key, value in props.items():
        key_text = str(key).lower()
        if any(hint in key_text for hint in ('score', 'affinity', 'energy')):
            score = _coerce_score(value)
            if score is not None:
                return score, str(key)

    lines = mol_block.splitlines()
    for idx, line in enumerate(lines):
        lowered = line.strip().lower()
        if lowered.startswith('>') and any(hint in lowered for hint in ('score', 'affinity', 'energy')):
            if idx + 1 < len(lines):
                score = _coerce_score(lines[idx + 1])
                if score is not None:
                    return score, line.strip()
        if any(hint in lowered for hint in ('score', 'affinity', 'energy')):
            tokens = line.replace('=', ' ').replace(':', ' ').split()
            for token in reversed(tokens):
                score = _coerce_score(token)
                if score is not None:
                    return score, 'inline_text'

    return 0.0, 'default_0.0'


def extract_named_numeric_property(props, *keys):
    for key in keys:
        if key in props:
            value = _coerce_score(props[key])
            if value is not None:
                return value
    return None


def rank_ligands(ligands, score_key='score'):
    ranked_ligands = sorted(
        ligands,
        key=lambda ligand: float('inf') if ligand.get(score_key) is None else ligand[score_key],
    )
    for rank, ligand in enumerate(ranked_ligands, start=1):
        ligand['rank'] = rank
    return ranked_ligands


def compute_input_signature(protein_content, sdf_content):
    hasher = hashlib.md5()
    hasher.update(VINA_PIPELINE_VERSION.encode('utf-8'))
    hasher.update(b'\0')
    hasher.update(protein_content.encode('utf-8'))
    hasher.update(b'\0')
    hasher.update(sdf_content.encode('utf-8'))
    return hasher.hexdigest()


def compute_uploaded_file_id(file_name, file_bytes):
    hasher = hashlib.md5()
    hasher.update((file_name or '').encode('utf-8'))
    hasher.update(b'\0')
    hasher.update(file_bytes)
    return hasher.hexdigest()

def parse_sdf_with_scores(sdf_content):
    """Parse SDF file and extract ligands with embedded scores"""
    ligands = []
    mol_blocks = [block.strip() for block in sdf_content.split('$$$$') if block.strip()]
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(sdf_content.encode('utf-8')), removeHs=False)

    for i, (mol_block, mol) in enumerate(zip(mol_blocks, supplier)):
        mol_block = mol_block.strip()
        if mol_block and len(mol_block.split('\n')) > 4:
            try:
                if mol is not None:
                    props = {prop_name: mol.GetProp(prop_name) for prop_name in mol.GetPropNames()}
                    if mol.HasProp('_Name'):
                        props['_Name'] = mol.GetProp('_Name')
                    score, score_source = extract_score(props, mol_block)
                    vina_score = extract_named_numeric_property(props, 'vina_score', 'docking_score', 'binding_energy', 'affinity')
                    sa_score = extract_named_numeric_property(props, 'SA_score', 'SA')
                    composite_score = extract_named_numeric_property(props, 'composite', 'composite_grower')
                    
                    ligand_entry = {
                        'mol': mol,
                        'mol_block': mol_block,
                        'index': i,
                        'score': score,
                        'score_source': score_source,
                        'composite_score': composite_score,
                        'sa_score': sa_score,
                        'properties': props,
                        'mw': rdMolDescriptors.CalcExactMolWt(mol),
                        'logp': rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
                        'name': props.get('_Name', f'Ligand_{i+1}')
                    }
                    if vina_score is not None:
                        ligand_entry['vina_score'] = vina_score

                    ligands.append(ligand_entry)
            except Exception as e:
                st.warning(f"Error parsing molecule {i}: {e}")
                continue

    return rank_ligands(ligands)


def filter_protein_for_vina(protein_content: str):
    filtered_lines = []
    for line in protein_content.splitlines():
        if line.startswith('ATOM'):
            filtered_lines.append(line)
            continue
        if line.startswith('HETATM'):
            resname = line[17:20].strip().upper()
            if resname in PROTEIN_RESIDUES:
                filtered_lines.append(line)
            continue
        if line.startswith(('TER', 'END')):
            filtered_lines.append(line)
    return '\n'.join(filtered_lines) + '\n'


def compute_vina_box(mol):
    if mol is None or mol.GetNumConformers() == 0:
        return [0.0, 0.0, 0.0], [20.0, 20.0, 20.0]

    conformer = mol.GetConformer()
    coords = np.array([
        [conformer.GetAtomPosition(atom_idx).x, conformer.GetAtomPosition(atom_idx).y, conformer.GetAtomPosition(atom_idx).z]
        for atom_idx in range(mol.GetNumAtoms())
    ])
    center = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0)
    size = np.maximum(extent + 8.0, np.array([14.0, 14.0, 14.0]))
    return center.tolist(), size.tolist()


def run_meeko_cli(arguments):
    python_executable = REPO_PYTHON if os.path.exists(REPO_PYTHON) else sys.executable
    command = [python_executable, '-m', *arguments]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'Meeko preparation failed.')


def prepare_receptor_pdbqt(protein_pdb, receptor_pdbqt_path):
    protein_pdb_path = os.path.splitext(receptor_pdbqt_path)[0] + '.pdb'
    with open(protein_pdb_path, 'w', encoding='utf-8') as handle:
        handle.write(filter_protein_for_vina(protein_pdb))

    run_meeko_cli([
        'meeko.cli.mk_prepare_receptor',
        '--read_pdb', protein_pdb_path,
        '-a',
        '-p', receptor_pdbqt_path,
    ])


def find_workspace_receptor_pdbqt(protein_name):
    if not protein_name:
        return None

    protein_stem = os.path.splitext(os.path.basename(protein_name))[0]
    candidates = []
    for root, _, files in os.walk(WORKSPACE_DIR):
        for file_name in files:
            if file_name.lower() == f'{protein_stem}.pdbqt'.lower():
                candidates.append(os.path.join(root, file_name))

    if not candidates:
        return None

    def candidate_priority(path):
        normalized = path.replace('\\', '/').lower()
        return (
            0 if 'optimized_vina_batch_full_rerun' in normalized else 1,
            0 if '/_receptor_pdbqt/' in normalized else 1,
            len(path),
            normalized,
        )

    return sorted(candidates, key=candidate_priority)[0]


def build_prepared_receptor(protein_pdb, protein_name=None):
    workspace_receptor = find_workspace_receptor_pdbqt(protein_name)
    if workspace_receptor is not None:
        return None, workspace_receptor, f'Using prepared receptor: {os.path.basename(workspace_receptor)}'

    temp_dir = tempfile.TemporaryDirectory()
    receptor_pdbqt_path = os.path.join(temp_dir.name, 'protein.pdbqt')
    prepare_receptor_pdbqt(protein_pdb, receptor_pdbqt_path)
    return temp_dir, receptor_pdbqt_path, 'Prepared receptor from uploaded PDB'


def prepare_ligand_pdbqt(ligand_sdf, ligand_pdbqt_path):
    ligand_sdf_path = os.path.splitext(ligand_pdbqt_path)[0] + '.sdf'
    ligand_mol = Chem.MolFromMolBlock(ligand_sdf, removeHs=False)
    if ligand_mol is None:
        raise RuntimeError('Could not parse ligand SDF for Vina preparation.')

    ligand_mol = Chem.AddHs(ligand_mol, addCoords=True)
    writer = Chem.SDWriter(ligand_sdf_path)
    writer.write(ligand_mol)
    writer.close()

    run_meeko_cli([
        'meeko.cli.mk_prepare_ligand',
        '-i', ligand_sdf_path,
        '-o', ligand_pdbqt_path,
    ])


def parse_vina_score(vina_stdout):
    patterns = [
        r'Estimated Free Energy of Binding\s*:\s*(-?\d+(?:\.\d+)?)',
        r'Affinity:\s*(-?\d+(?:\.\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, vina_stdout)
        if match:
            return float(match.group(1))
    return None


def extract_chains_from_pdb(pdb_content: str):
    """Extract unique chain IDs from PDB ATOM/HETATM records."""
    chains = set()
    for line in pdb_content.splitlines():
        if line.startswith('ATOM') or line.startswith('HETATM'):
            # PDB chain ID is column 22 (1-based), i.e. index 21 (0-based)
            if len(line) > 21:
                chain_id = line[21].strip()
                if chain_id:
                    chains.add(chain_id)
    return sorted(chains)


def find_cavity_residues(protein_content: str, mol_block: str, cutoff: float = 4.5):
    """Find protein residues within a cutoff distance from the current ligand pose."""
    ligand = Chem.MolFromMolBlock(mol_block, removeHs=False)
    if ligand is None or ligand.GetNumConformers() == 0:
        return []

    conformer = ligand.GetConformer()
    ligand_coords = np.array([
        [conformer.GetAtomPosition(atom_idx).x, conformer.GetAtomPosition(atom_idx).y, conformer.GetAtomPosition(atom_idx).z]
        for atom_idx in range(ligand.GetNumAtoms())
    ])
    if ligand_coords.size == 0:
        return []

    cutoff_sq = cutoff * cutoff
    residues = {}

    for line in protein_content.splitlines():
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            continue
        if len(line) < 54:
            continue

        resname = line[17:20].strip().upper()
        if resname in WATER_RESIDUES:
            continue
        if line.startswith('HETATM') and resname not in PROTEIN_RESIDUES:
            continue

        try:
            atom_coord = np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])
        except ValueError:
            continue

        distances_sq = np.sum((ligand_coords - atom_coord) ** 2, axis=1)
        min_distance_sq = float(np.min(distances_sq))
        if min_distance_sq > cutoff_sq:
            continue

        chain_id = line[21].strip() or '_'
        residue_id = line[22:26].strip()
        insertion_code = line[26].strip()
        residue_key = (chain_id, residue_id, insertion_code, resname)
        residue_label = f"{resname} {chain_id}:{residue_id}{insertion_code}".strip()
        residue_entry = residues.get(residue_key)
        if residue_entry is None or min_distance_sq < residue_entry['min_distance_sq']:
            residues[residue_key] = {
                'chain': chain_id,
                'resi': f"{residue_id}{insertion_code}" if insertion_code else residue_id,
                'resn': resname,
                'label': residue_label,
                'min_distance_sq': min_distance_sq,
            }

    ordered_residues = sorted(
        residues.values(),
        key=lambda residue: (residue['min_distance_sq'], residue['chain'], residue['resi']),
    )

    for residue in ordered_residues:
        residue['distance'] = float(np.sqrt(residue['min_distance_sq']))

    return ordered_residues


def build_residue_selection(cavity_residues):
    if not cavity_residues:
        return None

    return {
        'or': [
            {
                'chain': residue['chain'],
                'resi': residue['resi'],
                'resn': residue['resn'],
            }
            for residue in cavity_residues
        ]
    }

def show_3d_viewer(protein_content, mol_block, ligand_name, cavity_residues=None, show_cavity_residues=True):
    """Display 3D molecular viewer"""
    view = py3Dmol.view(width=900, height=700)
    chain_colors = {}
    
    # Add protein
    view.addModel(protein_content, 'pdb')
    # Some "protein" files are HETATM-only or otherwise not cartoon-able; add a line fallback.
    # Also color chains distinctly when multiple chains exist.
    base_style = {
        'cartoon': {'color': 'lightblue', 'opacity': 0.8},
        'line': {'color': 'lightblue', 'opacity': 0.6},
    }
    cavity_selection = build_residue_selection(cavity_residues)
    if cavity_selection is not None:
        view.setStyle({'model': 0, 'not': cavity_selection}, base_style)
    else:
        view.setStyle({'model': 0}, base_style)

    chains = extract_chains_from_pdb(protein_content)
    if len(chains) > 1:
        # Use named colors supported by 3Dmol.js (avoid custom hex/theme coupling).
        palette = [
            'red', 'green', 'blue', 'orange', 'purple', 'cyan',
            'magenta', 'yellow', 'grey', 'salmon', 'lime', 'teal',
        ]
        for chain_id, color in zip(chains, itertools.cycle(palette)):
            chain_colors[chain_id] = color
            chain_selection = {'model': 0, 'chain': chain_id}
            if cavity_selection is not None:
                chain_selection['not'] = cavity_selection
            view.setStyle(
                chain_selection,
                {
                    'cartoon': {'color': color, 'opacity': 0.8},
                    'line': {'color': color, 'opacity': 0.6},
                },
            )
    else:
        for chain_id in chains:
            chain_colors[chain_id] = 'lightblue'

    highlighted_residues = cavity_residues or []
    if cavity_selection is not None and show_cavity_residues and highlighted_residues:
        for residue in highlighted_residues:
            residue_selection = {
                'model': 0,
                'chain': residue['chain'],
                'resi': residue['resi'],
                'resn': residue['resn'],
            }
            chain_color = chain_colors.get(residue['chain'], 'lightblue')
            view.setStyle(
                residue_selection,
                {
                    'stick': {'colorscheme': 'default', 'radius': 0.18},
                    'line': {'color': chain_color, 'opacity': 0.75},
                },
            )
            view.setStyle(
                {**residue_selection, 'elem': 'C'},
                {'stick': {'color': chain_color, 'radius': 0.18}},
            )
    
    # Add ligand
    view.addModel(mol_block, 'sdf')
    view.setStyle({'model': 1}, {'stick': {'colorscheme': 'default', 'radius': 0.2}})
    # NOTE: Avoid adding a surface by default; a bad/empty selection can blank the viewer in some cases.
    
    # Always zoom/center on the ligand pose so Next/Previous keeps the ligand in view.
    # Fallback to full-scene zoom if selection-based zoom isn't supported.
    try:
        view.zoomTo({'model': 1})
    except Exception:
        view.zoomTo()
    view.spin(False)  # Add rotation animation

    if showmol is not None:
        showmol(view, height=700, width=900)
    else:
        components.html(view._make_html(), height=700)

def run_vina_rescoring(protein_pdb, ligand_sdf, center=None, size=None, receptor_pdbqt_path=None):
    """Run VINA rescoring for a ligand pose"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            vina_exe = shutil.which('vina')
            if vina_exe is None:
                raise RuntimeError('AutoDock Vina is not available on PATH.')

            ligand_pdbqt = os.path.join(temp_dir, 'ligand.pdbqt')

            receptor_file = receptor_pdbqt_path
            if receptor_file is None:
                receptor_file = os.path.join(temp_dir, 'protein.pdbqt')
                prepare_receptor_pdbqt(protein_pdb, receptor_file)
            prepare_ligand_pdbqt(ligand_sdf, ligand_pdbqt)

            if center is None or size is None:
                ligand_mol = Chem.MolFromMolBlock(ligand_sdf, removeHs=False)
                center, size = compute_vina_box(ligand_mol)

            vina_cmd = [
                vina_exe,
                '--receptor', receptor_file,
                '--ligand', ligand_pdbqt,
                '--center_x', str(center[0]),
                '--center_y', str(center[1]),
                '--center_z', str(center[2]),
                '--size_x', str(size[0]),
                '--size_y', str(size[1]),
                '--size_z', str(size[2]),
                '--score_only',
            ]

            result = subprocess.run(vina_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'vina scoring failed.')

            vina_score = parse_vina_score(result.stdout)
            if vina_score is None:
                raise RuntimeError('Could not parse a Vina score from the output.')

            return vina_score
            
    except subprocess.TimeoutExpired:
        return None, 'Vina scoring timed out.'
    except Exception as e:
        return None, str(e)


def score_all_ligands_with_vina(protein_pdb, ligands, protein_name=None, progress_callback=None):
    scored_ligands = []
    failures = []
    ligands_needing_vina = [ligand for ligand in ligands if ligand.get('vina_score') is None]

    if not ligands_needing_vina:
        return rank_ligands(list(ligands), score_key='vina_score'), failures

    try:
        receptor_temp_dir, receptor_pdbqt_path, receptor_status = build_prepared_receptor(protein_pdb, protein_name=protein_name)
        if progress_callback is not None:
            progress_callback(0, len(ligands_needing_vina), receptor_status)
    except Exception as exc:
        failures.append(f"Receptor preparation failed: {exc}")
        for ligand in ligands:
            updated_ligand = dict(ligand)
            updated_ligand['vina_score'] = None
            updated_ligand['vina_error'] = f"Receptor preparation failed: {exc}"
            scored_ligands.append(updated_ligand)
        return rank_ligands(scored_ligands, score_key='vina_score'), failures

    total_ligands = len(ligands_needing_vina)
    try:
        for ligand in ligands:
            if ligand.get('vina_score') is not None:
                scored_ligands.append(dict(ligand))

        for index, ligand in enumerate(ligands_needing_vina, start=1):
            if progress_callback is not None:
                progress_callback(index - 1, total_ligands, ligand['name'])

            center, size = compute_vina_box(ligand['mol'])
            vina_score = run_vina_rescoring(
                protein_pdb,
                ligand['mol_block'],
                center=center,
                size=size,
                receptor_pdbqt_path=receptor_pdbqt_path,
            )

            updated_ligand = dict(ligand)
            if isinstance(vina_score, tuple):
                updated_ligand['vina_score'] = None
                updated_ligand['vina_error'] = vina_score[1]
                failures.append(f"{ligand['name']}: {vina_score[1]}")
            else:
                updated_ligand['vina_score'] = vina_score
                updated_ligand['score'] = vina_score
            scored_ligands.append(updated_ligand)

        if progress_callback is not None:
            progress_callback(total_ligands, total_ligands, 'Completed')
    finally:
        if receptor_temp_dir is not None:
            receptor_temp_dir.cleanup()

    return rank_ligands(scored_ligands, score_key='vina_score'), failures

# Sidebar controls
with st.sidebar:
    st.header("📁 File Upload")
    
    # PDB upload
    pdb_file = st.file_uploader("Upload PDB File", type=['pdb'])
    if pdb_file:
        pdb_bytes = pdb_file.getvalue()
        pdb_upload_id = compute_uploaded_file_id(pdb_file.name, pdb_bytes)
        if st.session_state.protein_upload_id != pdb_upload_id:
            st.session_state.protein_upload_id = pdb_upload_id
            st.session_state.protein_name = pdb_file.name
            st.session_state.protein_content = pdb_bytes.decode('utf-8')
            st.session_state.vina_signature = None
            st.session_state.vina_status = None
        st.success("✅ PDB loaded")
        
        # Basic protein info
        protein_content = st.session_state.protein_content
        if protein_content is not None:
            lines = protein_content.split('\n')
            atom_count = sum(1 for l in lines if l.startswith('ATOM'))
            hetatm_count = sum(1 for l in lines if l.startswith('HETATM'))
            st.info(f"ATOM: {atom_count} | HETATM: {hetatm_count}")
    
    # SDF upload  
    sdf_file = st.file_uploader("Upload SDF File", type=['sdf'])
    if sdf_file:
        sdf_bytes = sdf_file.getvalue()
        sdf_upload_id = compute_uploaded_file_id(sdf_file.name, sdf_bytes)
        if st.session_state.sdf_upload_id != sdf_upload_id:
            sdf_content = sdf_bytes.decode('utf-8')
            st.session_state.sdf_upload_id = sdf_upload_id
            st.session_state.sdf_content = sdf_content
            st.session_state.ligands = parse_sdf_with_scores(sdf_content)
            st.session_state.current_idx = 0
            st.session_state.vina_signature = None
            st.session_state.vina_status = None
        st.success(f"✅ {len(st.session_state.ligands)} ligands loaded")

if st.session_state.protein_content and st.session_state.ligands and st.session_state.sdf_content:
    current_signature = compute_input_signature(
        st.session_state.protein_content,
        st.session_state.sdf_content,
    )
    vina_count = sum(1 for ligand in st.session_state.ligands if ligand.get('vina_score') is not None)
    all_have_vina_scores = vina_count == len(st.session_state.ligands)
    if st.session_state.vina_signature != current_signature:
        if all_have_vina_scores:
            st.session_state.ligands = rank_ligands(st.session_state.ligands, score_key='vina_score')
            st.session_state.vina_signature = current_signature
            st.session_state.vina_status = f"Detected embedded Vina scores for all {vina_count} ligands. Rescoring skipped."
        else:
            st.session_state.vina_signature = None
            st.session_state.vina_status = None

    if st.session_state.vina_signature != current_signature:
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0, text='Preparing receptor for Vina...')

        def update_vina_progress(done_count, total_count, ligand_name):
            if total_count <= 0:
                progress_bar.progress(0, text='Preparing receptor for Vina...')
                return
            fraction = min(max(done_count / total_count, 0.0), 1.0)
            progress_bar.progress(
                fraction,
                text=f"Vina rescoring {done_count}/{total_count}: {ligand_name}",
            )

        with st.spinner("Scoring all ligands with Vina..."):
            scored_ligands, failures = score_all_ligands_with_vina(
                st.session_state.protein_content,
                st.session_state.ligands,
                protein_name=st.session_state.protein_name,
                progress_callback=update_vina_progress,
            )
            st.session_state.ligands = scored_ligands
            st.session_state.vina_signature = current_signature
            success_count = sum(1 for ligand in scored_ligands if ligand.get('vina_score') is not None)
            reused_count = sum(1 for ligand in st.session_state.ligands if ligand.get('vina_score') is not None) - (success_count - vina_count)
            if failures:
                st.session_state.vina_status = (
                    f"Scored {success_count}/{len(scored_ligands)} ligands with Vina. "
                    f"First failure: {failures[0]}"
                )
            else:
                if vina_count > 0:
                    st.session_state.vina_status = (
                        f"Reused embedded Vina scores for {vina_count} ligands and scored the remaining "
                        f"{len(scored_ligands) - vina_count} ligands with Vina."
                    )
                else:
                    st.session_state.vina_status = f"Scored {success_count} ligands with Vina."
        progress_placeholder.empty()

# Main interface
if st.session_state.protein_content and st.session_state.ligands:
    total_ligands = len(st.session_state.ligands)
    current_ligand = st.session_state.ligands[st.session_state.current_idx]
    cavity_residues = find_cavity_residues(
        st.session_state.protein_content,
        current_ligand['mol_block'],
        cutoff=st.session_state.cavity_cutoff,
    )

    controls_col1, controls_col2 = st.columns([1, 1])
    with controls_col1:
        st.session_state.show_cavity_residues = st.checkbox(
            "Show cavity residues",
            value=st.session_state.show_cavity_residues,
            help="Show or hide protein residues within the selected ligand cutoff distance.",
        )
    with controls_col2:
        st.session_state.cavity_cutoff = st.slider(
            "Cavity cutoff (A)",
            min_value=2.5,
            max_value=8.0,
            value=float(st.session_state.cavity_cutoff),
            step=0.5,
            help="Residues with any atom inside this distance from the ligand are considered cavity residues.",
        )
        cavity_residues = find_cavity_residues(
            st.session_state.protein_content,
            current_ligand['mol_block'],
            cutoff=st.session_state.cavity_cutoff,
        )
    
    # Navigation controls
    col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
    
    with col1:
        if st.button("⬅️ Previous") and st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
            st.rerun()
    
    with col2:
        new_idx = st.selectbox(
            "Select Ligand:",
            range(total_ligands),
            index=st.session_state.current_idx,
            format_func=lambda x: (
                f"#{st.session_state.ligands[x]['rank']} | "
                f"{st.session_state.ligands[x]['name']} "
                f"(VINA: {st.session_state.ligands[x].get('vina_score', float('nan')):.3f})"
                if st.session_state.ligands[x].get('vina_score') is not None
                else (
                    f"#{st.session_state.ligands[x]['rank']} | "
                    f"{st.session_state.ligands[x]['name']} (VINA: n/a)"
                )
            )
        )
        if new_idx != st.session_state.current_idx:
            st.session_state.current_idx = new_idx
            st.rerun()
    
    with col3:
        if st.button("➡️ Next") and st.session_state.current_idx < total_ligands - 1:
            st.session_state.current_idx += 1
            st.rerun()
    
    with col4:
        if st.button("🔄 Sort by VINA"):
            st.session_state.ligands = rank_ligands(st.session_state.ligands, score_key='vina_score')
            st.session_state.current_idx = 0
            st.rerun()
    
    # Current ligand information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"Ligand {st.session_state.current_idx + 1}/{total_ligands}")
        st.caption(f"Rank #{current_ligand['rank']}")
        if current_ligand.get('vina_score') is not None:
            st.metric("VINA Score", f"{current_ligand['vina_score']:.3f}")
        else:
            st.metric("VINA Score", "n/a")
    
    with col2:
        st.metric("Molecular Weight", f"{current_ligand['mw']:.1f}")
        st.metric("LogP", f"{current_ligand['logp']:.2f}")
        if current_ligand.get('composite_score') is not None:
            st.metric("Composite", f"{current_ligand['composite_score']:.3f}")
        else:
            st.metric("Composite", "n/a")
    
    with col3:
        if current_ligand.get('sa_score') is not None:
            st.metric("SA Score", f"{current_ligand['sa_score']:.3f}")
        else:
            st.metric("SA Score", "n/a")
        st.metric("Cavity Residues", str(len(cavity_residues)))

    if st.session_state.vina_status:
        if any(ligand.get('vina_score') is not None for ligand in st.session_state.ligands):
            st.caption(st.session_state.vina_status)
        else:
            st.error(st.session_state.vina_status)

    ranking_df = pd.DataFrame(
        [
            {
                'Rank': ligand['rank'],
                'Name': ligand['name'],
                'VINA Score': ligand.get('vina_score'),
                'Composite': ligand.get('composite_score'),
                'SA Score': ligand.get('sa_score'),
                'MW': ligand['mw'],
                'LogP': ligand['logp'],
            }
            for ligand in st.session_state.ligands
        ]
    )
    selected_rank = current_ligand['rank']
    st.subheader("🏅 Score Ranking")
    st.dataframe(
        ranking_df.style.apply(
            lambda row: [
                'background-color: rgba(255, 235, 59, 0.35)' if row['Rank'] == selected_rank else ''
                for _ in row
            ],
            axis=1,
        ).set_table_styles(
            [
                {'selector': 'th', 'props': [('font-size', '0.8rem')]},
                {'selector': 'td', 'props': [('font-size', '0.8rem')]},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    vina_failures = [
        {
            'Name': ligand['name'],
            'VINA Error': ligand['vina_error'],
        }
        for ligand in st.session_state.ligands
        if ligand.get('vina_score') is None and ligand.get('vina_error')
    ]
    if vina_failures:
        st.subheader("⚠️ VINA Failures")
        st.dataframe(pd.DataFrame(vina_failures), use_container_width=True, hide_index=True)

    if cavity_residues:
        st.subheader("🧱 Cavity Residues")
        cavity_df = pd.DataFrame(
            [
                {
                    'Residue': residue['label'],
                    'Distance (A)': round(residue['distance'], 2),
                }
                for residue in cavity_residues
            ]
        )
        st.dataframe(cavity_df, use_container_width=True, hide_index=True)
    else:
        st.info("No cavity residues found for the current ligand at the selected cutoff.")
    
    # 3D Viewer
    st.subheader("🔬 3D Visualization")
    show_3d_viewer(
        st.session_state.protein_content, 
        current_ligand['mol_block'],
        current_ligand['name'],
        cavity_residues=cavity_residues,
        show_cavity_residues=st.session_state.show_cavity_residues,
    )
    
    # Ligand properties table
    if current_ligand['properties']:
        st.subheader("📊 Ligand Properties")
        props_df = pd.DataFrame(list(current_ligand['properties'].items()), 
                               columns=['Property', 'Value'])
        st.dataframe(props_df, use_container_width=True)

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to the PDB-SDF Docking Viewer
    
    This app provides a streamlined interface for visualizing protein-ligand docking results with integrated scoring tools.
    
    ### Features:
    - 📁 **Simple Input**: Just upload PDB (protein) and SDF (ligands) files
    - 🔬 **3D Visualization**: Interactive molecular viewer with py3Dmol
    - 🎯 **Bulk VINA Scoring**: Score all uploaded ligand poses using AutoDock Vina
    - 📊 **Score Analysis**: Rank and compare ligand scores
    
    ### Getting Started:
    1. Upload your PDB protein structure file
    2. Upload your SDF ligand poses file
    3. Wait for automatic VINA scoring to complete
    4. Browse through ligands using the navigation controls
    
    ### Requirements:
    - AutoDock Vina installed and accessible via command line
    - Meeko installed in the selected Python environment
    - RDKit for molecular processing
    """)
    
    st.info("👆 Please upload PDB and SDF files using the sidebar to get started!")
