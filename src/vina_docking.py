# type: ignore
import argparse
import io
import os
import sys
import json
import tempfile
from typing import List, Optional, Tuple
import subprocess
import shutil

from vina import Vina
from meeko import MoleculePreparation, PDBQTWriterLegacy 

# RDKit for SDF handling
from rdkit import Chem
from rdkit.Chem import AllChem
  
from pdb_cleaner import auto_pick_ligand_from_pdb, clean_pdb_to_string
# Meeko Ligand prep 

def sdf_to_pdbqt_strings(
    sdf_path: str,
    add_h_if_missing: bool = True,
    embed_if_missing: bool = True,
) -> List[Tuple[str, str]]:
    """
    Convert one or more ligands in SDF to PDBQT strings using Meeko, entirely in memory.
    Returns: list of (lig_name, ligand_pdbqt_string)
    """
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    if not supplier or len(supplier) == 1:
        raise RuntimeError(f"No molecules read from SDF: {sdf_path}")

    out = []
    prep = MoleculePreparation() # dafult config, can change

    for i, mol in enumerate(supplier):
        if mol is None:
            continue

        # Ensure we have H and 3D if the SDF lacks them (Meeko expects H and 3D)
        if add_h_if_missing:
            # AddHs won't add duplicates, so it's safe if they exist
            mol = Chem.AddHs(mol, addCoords=True)

        if embed_if_missing and mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol)

        mol_setups = prep.prepare(mol)  # list of setups (typically 1 unless reactive docking)
        if not mol_setups:
            raise RuntimeError("Meeko could not prepare ligand (no setups produced).")

        # Write PDBQT for the *first* setup; expand if you want multiple setups per ligand
        lig_pdbqt = PDBQTWriterLegacy.write_string(mol_setups[0])  # string 

        # Try to name the ligand
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"lig_{i+1}"
        out.append((name, lig_pdbqt))

    if len(out) == 0:
        raise RuntimeError(f"No valid ligands found in: {sdf_path}")

    return out

def receptor_pdb_to_pdbqt_tempfile(
    cleaned_pdb_str: str,
    default_altloc: Optional[str] = "A",
    delete_bad_res: bool = True,
) -> str:
    """
    Prepare receptor with Meeko's CLI under the hood and return the path to a
    temporary receptor PDBQT. This file is created in a TemporaryDirectory
    and deleted when the process ends. Vina requires a *filename* for the receptor.

    Why CLI? Meeko’s receptor Python API is less documented; Vina requires a file path anyway.
    The outcome stays ephemeral and is immediately cleaned up by our temp context.
    """


    # Ensure the CLI script is available in PATH (mk_prepare_receptor.py)
    cli = shutil.which("mk_prepare_receptor.py")
    if cli is None:
        raise RuntimeError(
            "mk_prepare_receptor.py not found in PATH. Please ensure Meeko is installed."
        )

    # Work inside a temp directory
    tmpdir = tempfile.mkdtemp(prefix="meeko_rec_")
    tmp_cleaned_pdb = os.path.join(tmpdir, "cleaned.pdb")
    with open(tmp_cleaned_pdb, "w") as f:
        f.write(cleaned_pdb_str)

    rec_pdbqt = os.path.join(tmpdir, "receptor.pdbqt")

    # Build CLI args:
    #   -i input, -p write_pdbqt, --default_altloc, and (optionally) delete bad residues.
    args = [
        sys.executable,
        cli,
        "-i",
        tmp_cleaned_pdb,
        "-p",
        rec_pdbqt,
    ]
    if default_altloc and len(default_altloc) == 1:
        args.extend(["--default_altloc", default_altloc])
    if delete_bad_res:
        args.append("-a")

    # Run Meeko receptor preparation
    subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # We return the path; caller is responsible for keeping this directory alive
    # while Vina runs (we'll manage that via a context manager in main()).
    return tmpdir, rec_pdbqt

# Vina Docking

def dock_one_ligand(
    v: Vina,
    lig_name: str,
    lig_pdbqt: str,
    n_poses: int,
    energy_range: float = 3.0,
) -> Tuple[str, List[Tuple[float, float]]]:
    """
    Load ligand PDBQT string, dock with existing receptor/maps in Vina,
    and return (lig_name, energies list[(affinity, intermol)]).

    Energies from Vina API can be retrieved via v.energies(...).
    """

    v.set_ligand_from_string(lig_pdbqt) 
    v.dock(n_poses=n_poses)
    # Get energies for the top poses
    energies = v.energies(n_poses=n_poses, energy_range=energy_range)
    return lig_name, energies


def parse_args():
    p = argparse.ArgumentParser(
        description="Dock ligand(s) from SDF into a receptor PDB using Meeko + Vina with in-memory intermediates."
    )
    p.add_argument("-r", "--receptor_pdb", required=True, help="Input receptor PDB file")
    p.add_argument("-l", "--ligand_sdf", required=True, help="Input ligand SDF (can contain multiple molecules)")
    p.add_argument("--center", nargs=3, type=float, required=False, metavar=("CX", "CY", "CZ"),
                   help="Grid center (Å) for Vina [required]")
    p.add_argument("--size", nargs=3, type=float, required=False, metavar=("SX", "SY", "SZ"),
                   help="Grid box sizes (Å) for Vina [required]")
    p.add_argument("--exhaustiveness", type=int, default=8, help="Vina exhaustiveness (default: 8)")
    p.add_argument("--n_poses", type=int, default=9, help="Number of poses to keep (default: 9)")
    p.add_argument("--sf_name", choices=["vina", "vinardo", "ad4"], default="vina", help="Scoring function (default: vina)")
    p.add_argument("--cpu", type=int, default=0, help="CPUs (0=all available; Vina default)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (0=Vina default)")
    p.add_argument("--default_altloc", default="A", help="Preferred altLoc ID (default: A)")
    p.add_argument("--keep_water", action="store_true", help="Keep crystallographic water (default: drop)")
    p.add_argument("--out_pdbqt", default=None, help="Optional output PDBQT file for docked poses")
    p.add_argument("--out_sdf", default=None, help="Optional output SDF (converts top pose per ligand)")
    return p.parse_args()


def vina_from_args(sf_name: str, cpu: int, seed: int) -> Vina:
    v = Vina(sf_name=sf_name, cpu=cpu, seed=seed) 
    return v


def main():
    args = parse_args()
    #  Clean receptor PDB to string (no file saved)
    cleaned_pdb_str = clean_pdb_to_string(
        args.receptor_pdb,
        keep_water=args.keep_water,
        default_altloc=args.default_altloc,
    )

    # Prepare ligand(s) -> PDBQT strings 
    lig_list = sdf_to_pdbqt_strings(args.ligand_sdf)

    # Prepare receptor -> ephemeral PDBQT file path (Meeko CLI)
    #    keep temp dir alive until after docking
    tmpdir, receptor_pdbqt = receptor_pdb_to_pdbqt_tempfile(
        cleaned_pdb_str=cleaned_pdb_str,
        default_altloc=args.default_altloc,
        delete_bad_res=True,
    )


    if args.center is None or args.size is None:
        picked = auto_pick_ligand_from_pdb(args.receptor_pdb, sdf_path=args.ligand_sdf)
        center = picked["center"]
        size = picked["box"]
        
        # vina requires list and cant handle numpy floats
        center = [float(x) for x in center]
        size = [float(x) for x in size]

        # print(f"# Auto-selected ligand {picked['resname']} at {picked['chain_id']}:{picked['resseq']}; "
        #     f"center={center}, box={size}")
    else:
        center = list(map(float, args.center))
        size = list(map(float, args.size))

            
            
    # set up Vina and compute maps
    v = vina_from_args(args.sf_name, args.cpu, args.seed)
    # Vina requires receptor as filename (no in-memory receptor at present)    
    v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt)

    v.compute_vina_maps(center=center,
                        box_size=size)  
    # dock each ligand, report scores
    all_results = []
    combined_out = args.out_pdbqt is not None
    if combined_out:
        # If writing combined results, we dock all, but Vina's write_poses
        # writes the *current* ligand; we will append by calling write_poses
        # after each docking, then merge the files at the end.
        # Simpler: write once at the end by concatenating v.poses(...) outputs,
        # but Vina API provides write_poses per current ligand; we'll just append.
        # NOT SET UP
        if os.path.exists(args.out_pdbqt):
            os.remove(args.out_pdbqt)

    for lig_name, lig_pdbqt in lig_list:
        #print(lig_name, lig_pdbqt[0])
        name, energies = dock_one_ligand(
            v,
            lig_name=lig_name,
            lig_pdbqt=lig_pdbqt[0],
            n_poses=args.n_poses,
        )
        all_results.append((name, energies))

        # # Append poses amd save for this ligand if requested
        # if combined_out:
        #     # Write/append top poses for current ligand
        #     v.write_poses(pdbqt_filename=args.out_pdbqt, n_poses=args.n_poses, overwrite=False)

    # summary to check 
    print("# Docking summary (kcal/mol):")
    for lig_name, energies in all_results:
        energies_list = [tuple(row) for row in energies]
        # energies: list of tuples (affinity, intermol, intramol, torsional ? depending on Vina version)
        scores = [f"{e[0]:.3f}" for e in energies_list] if len(energies_list) > 0 else []
        print(f"{lig_name}\tbest={scores[0] if scores else 'NA'}\tall={','.join(scores)}")

if __name__ == "__main__":
    main()
