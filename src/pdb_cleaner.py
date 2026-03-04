# type: ignore
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from typing import List, Tuple, Optional, Dict
from Bio.PDB import PDBParser
from rdkit import Chem
import tempfile
import os, io

# Common “ignore” list for non-drug-like HET groups
DEFAULT_EXCLUDE = {
    # solvents / cryoprotectants
    "HOH", "WAT", "DOD", "H2O", "GOL", "EDO", "PG4", "PEG", "MPD", "PGE",
    # inorganic ions / small anions
    "NA", "K", "CL", "CA", "MG", "MN", "ZN", "FE", "CO", "CU",
    "IOD", "I", "BR", "F", "LI", "SR", "BA", "CD", "PB", "NI",
    "SO4", "SUL", "PO4", "NO3", "NH4", "SCN",
    # common buffers / additives
    "TRS", "HEP", "MES", "BOG", "ACE",
    # saccharides (you may keep or drop depending on target)
    "NAG", "BMA", "MAN", "GLC",
}

def _rdkit_sdf_names(sdf_path: Optional[str]) -> List[str]:
    """Collect candidate names from SDF: title (_Name) + RESNAME-like props if present."""
    if not sdf_path:
        return []
    names = []
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for mol in suppl:
        if mol is None:
            continue
        if mol.HasProp("_Name"):
            names.append(mol.GetProp("_Name").strip())
        for key in mol.GetPropNames():
            if key.lower() in {"resname", "pdb_resname", "res_name"}:
                try:
                    names.append(mol.GetProp(key).strip())
                except KeyError:
                    pass
        break  # only the first molecule is used to choose the binding site
    return [n for n in {n.upper() for n in names} if n]  # unique, uppercase names

def _is_hetero(res) -> bool:
    """Biopython marks hetero residues with a non-blank id[0]."""
    # res.id is a tuple like (' ', resseq, icode) for standard residues
    # HET residues have id[0] != ' '
    hetflag = res.id[0]
    return hetflag not in (" ", "")

def _heavy_count(res) -> int:
    """Count heavy atoms in a residue."""
    count = 0
    for atom in res.get_atoms():
        elem = atom.element.strip().upper() if hasattr(atom, "element") else atom.get_name()[0]
        if elem != "H":
            count += 1
    return count

def _coords_of_residue(res) -> List[Tuple[float, float, float]]:
    return [tuple(atom.get_coord()) for atom in res.get_atoms()]

def _center_and_box(coords: List[Tuple[float,float,float]], padding: float = 5.0):
    xs = [c[0] for c in coords]; ys = [c[1] for c in coords]; zs = [c[2] for c in coords]
    cx = sum(xs)/len(xs); cy = sum(ys)/len(ys); cz = sum(zs)/len(zs)
    size = [(max(xs)-min(xs))+padding, (max(ys)-min(ys))+padding, (max(zs)-min(zs))+padding]
    return [cx, cy, cz], size

def auto_pick_ligand_from_pdb(
    pdb_path: str,
    sdf_path: Optional[str] = None,
    exclude: Optional[set] = None,
    min_heavy_atoms: int = 8,
    padding: float = 5.0,
    prefer_multiatom: bool = True,
) -> Dict:
    """
    Auto-pick a ligand (HET residue) from a PDB and compute center/box.
    - If sdf_path is given, try to match resname to SDF title/properties.
    - Otherwise, pick the largest non-excluded HET by heavy-atom count.
    Returns a dict with keys: resname, chain_id, resseq, icode, atom_count, center, box.
    """
    exclude = set() if exclude is None else {e.upper() for e in exclude}
    exclude |= DEFAULT_EXCLUDE

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = next(structure.get_models())  # first model

    # Collect HET candidates
    candidates = []
    for chain in model:
        for res in chain:
            if not _is_hetero(res):
                continue
            rn = res.get_resname().strip().upper()
            if rn in exclude:
                continue
            hvy = _heavy_count(res)
            if prefer_multiatom and hvy < min_heavy_atoms:
                # ignore small ions/fragments
                continue
            coords = _coords_of_residue(res)
            if not coords:
                continue
            candidates.append({
                "resname": rn,
                "chain_id": chain.id,
                "resseq": res.id[1],
                "icode": res.id[2].strip() if isinstance(res.id[2], str) else "",
                "atom_count": len(coords),
                "heavy_count": hvy,
                "coords": coords,
            })

    if not candidates:
        raise RuntimeError("No suitable HET ligands found in PDB (after exclusions).")

    # If SDF is provided, try name-based match first
    sdf_names = _rdkit_sdf_names(sdf_path)
    matched = [c for c in candidates if c["resname"] in sdf_names] if sdf_names else []

    pool = matched if matched else candidates
    # Choose the largest by heavy-atom count; tie-break by total atoms
    pool.sort(key=lambda c: (c["heavy_count"], c["atom_count"]), reverse=True)
    chosen = pool[0]

    center, box = _center_and_box(chosen["coords"], padding=padding)
    return {
        "resname": chosen["resname"],
        "chain_id": chosen["chain_id"],
        "resseq": chosen["resseq"],
        "icode": chosen["icode"],
        "atom_count": chosen["atom_count"],
        "heavy_count": chosen["heavy_count"],
        "center": center,
        "box": box,
    }


def _filter_altloc(pdb_text: str, preferred: str = "A") -> str:
    """
    Keep only altLoc 'preferred' (or blank) atoms from a PDB text block.
    AltLoc is column 17 (1-based) in PDB format.
    """
    lines_out = []
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            lines_out.append(line)
            continue
        altloc = line[16]  # zero-based index
        if altloc == " " or altloc == preferred:
            # Normalize the altLoc to blank (optional)
            # Replace only if it's the preferred letter.
            if altloc == preferred:
                line = line[:16] + " " + line[17:]
            lines_out.append(line)
        # else: drop atom with non-preferred altloc
    return "\n".join(lines_out) + "\n"


def clean_pdb_to_string(
    pdb_path: str,
    keep_water: bool = False,
    default_altloc: Optional[str] = "A",
) -> str:
    """
    Update of your attached cleaner to *not* save an intermediate file.
    - (Optionally) pre-filter alternate locations to a preferred ID (e.g., 'A').
    - Remove heterogens (optionally keep water).
    - Return cleaned PDB as a string.
    """
    # Read raw PDB text (so we can filter altloc first, like --default_altloc in Meeko CLI)
    with open(pdb_path, "r") as f:
        pdb_text = f.read()

    if default_altloc and len(default_altloc) == 1:
        pdb_text = _filter_altloc(pdb_text, preferred=default_altloc)

    # Run PDBFixer on the (possibly altloc-filtered) structure
    # PDBFixer requires a filename or PDB ID; we feed it from a temporary file.
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as tmp_in:
        tmp_in.write(pdb_text)
        tmp_in_path = tmp_in.name

    try:
        fixer = PDBFixer(filename=tmp_in_path)
        # Normalize chemistry as in your script
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=keep_water)
        # (Optional) fixer.findMissingResidues(); fixer.findMissingAtoms(); fixer.addMissingAtoms()
        # Skip H addition; Meeko will handle protonation / templates.

        # Write to an in-memory string (no cleaned file written)
        handle = io.StringIO()
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
        cleaned_pdb_str = handle.getvalue()
        handle.close()
    finally:
        # Ensure the temporary input is removed
        try:
            os.remove(tmp_in_path)
        except OSError:
            pass

    return cleaned_pdb_str
