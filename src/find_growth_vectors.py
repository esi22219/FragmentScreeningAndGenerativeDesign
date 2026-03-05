# exit_finder.py
# Geometry-first exit identification with virtual roles (V-SYNTHES style)
# type: ignore
from __future__ import annotations
import argparse, sys
from typing import List, Tuple, Optional
import os, tempfile, numpy as np
from pathlib import Path
from data_models import ExitCandidate, ExitRole, RSite
from drawing import visualize_rsites
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


# PyVOL for pocket surface/vertex normals
import pyvol
from pyvol.spheres import Spheres

# Vina/Meeko for cap-probe docking
try:
    from vina import Vina
    #from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer, ReceptorWriterLegacy
    HAVE_DOCK = True
except Exception:
    HAVE_DOCK = False

# --------------------------- Helpers: coordinates, SASA, pocket ---------------------------

def ligand_coords(m: Chem.Mol) -> np.ndarray:
    conf = m.GetConformer()
    return np.array([[conf.GetAtomPosition(i).x,
                      conf.GetAtomPosition(i).y,
                      conf.GetAtomPosition(i).z] for i in range(m.GetNumAtoms())], dtype=float)

def calc_atom_sasa(m: Chem.Mol) -> np.ndarray:
    """Labute ASA per atom (include Hs for robust geometry)."""
    mol3d = Chem.Mol(m)
    if mol3d.GetNumConformers() == 0:
        mol3d = Chem.AddHs(mol3d)
        AllChem.EmbedMolecule(mol3d, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol3d)
    contribs, h_contrib = rdMolDescriptors._CalcLabuteASAContribs(mol3d, includeHs=True)
    return np.asarray(contribs, dtype=float)

def pyvol_pocket_vertices(protein_pdb: str, lig_xyz: np.ndarray, probe: float=1.4, within: float=8.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices, normals) for the pocket surface nearest the ligand, clipped to 'within' Å."""
    comps = Spheres(pdb=protein_pdb).calculate_surface(probe_radius=probe, largest_only=False, all_components=True, noh=True)
    if not comps:
        comps = Spheres(pdb=protein_pdb).calculate_surface(probe_radius=probe, largest_only=False, all_components=False, noh=True)
    best = None; best_min = 1e9
    for comp in comps:
        verts = np.asarray(comp.mesh.vertices, dtype=float)
        norms = np.asarray(comp.mesh.vertex_normals, dtype=float)
        if verts.size == 0: continue
        dmin = np.linalg.norm(verts[:,None,:] - lig_xyz[None,:,:], axis=2).min()
        if dmin < best_min:
            best_min = dmin; best = (verts, norms)
    if best is None:
        raise RuntimeError("No PyVOL pocket near ligand.")
    verts, norms = best
    # clip to region near ligand
    d = np.linalg.norm(verts[:,None,:] - lig_xyz[None,:,:], axis=2).min(axis=1)
    keep = d <= within
    verts = verts[keep]; norms = norms[keep]
    # normalize normals
    nrm = np.linalg.norm(norms, axis=1, keepdims=True) + 1e-9
    norms = norms / nrm
    return verts, norms

def protein_heavy_xyz(protein_pdb: str) -> np.ndarray:
    xyz = []
    with open(protein_pdb, "r") as fh:
        for ln in fh:
            if not ln.startswith(("ATOM", "HETATM")): continue
            an = ln[12:16].strip()
            if an.startswith("H"): continue
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return np.asarray(xyz, dtype=float)


# virtual role assignment

def assign_virtual_role(m: Chem.Mol, atom_idx: int) -> Optional[ExitRole]:
    """A: Aromatic sp2 C; B: aliphatic sp3 C; C: heteroatom (N/O/S with LP); D: heteroaromatic C–H (Minisci-like)"""
    a = m.GetAtomWithIdx(atom_idx)
    sym = a.GetSymbol()
    hyb = a.GetHybridization()
    is_arom = a.GetIsAromatic()
    # Role C: Heteroatom lone-pair (N/O/S not quaternary)
    if sym in ("N","O","S"):
        if a.GetFormalCharge() <= 1 and a.GetDegree() >= 1:
            return ExitRole("HETERO_NOS")
    # Role A: Aromatic carbon
    if sym == "C" and is_arom:
        return ExitRole("AROMATIC_SP2_C")
    # Role D: Heteroaromatic C (aromatic C next to heteroatom) — simple heuristic
    if sym == "C" and is_arom:
        for nbr in a.GetNeighbors():
            if nbr.GetIsAromatic() and nbr.GetSymbol() in ("N","O","S"):
                return ExitRole("HETEROAROMATIC_MINISCI")
    # Role B: Aliphatic sp3 carbon (growth)
    if sym == "C" and (hyb == Chem.HybridizationType.SP3) and not is_arom:
        return ExitRole("ALIPH_SP3_C")
    return None


# candidate generation (geometry)

def nearest_pocket_vector(atom_xyz: np.ndarray, verts: np.ndarray) -> Tuple[np.ndarray, float, int]:
    diffs = verts - atom_xyz
    dists = np.linalg.norm(diffs, axis=1)
    j = int(np.argmin(dists))
    vec = diffs[j]; dist = float(dists[j])
    return vec, dist, j

def score_candidates(lig: Chem.Mol,
                     pocket_verts: np.ndarray, pocket_normals: np.ndarray,
                     min_dist: float=2.0, max_dist: float=8.0,
                     w_angle: float=0.6, w_dist: float=0.3, w_sasa: float=0.1,
                     sasa_min: float=1.0) -> List[ExitCandidate]:
    lig_h = Chem.AddHs(lig, addCoords=True)
    coords = ligand_coords(lig_h)
    sasa = calc_atom_sasa(lig_h)
    max_sasa = max(float(np.max(sasa)), 1e-6)
    centroid = coords.mean(axis=0)

    out: List[ExitCandidate] = []
    for i, atom in enumerate(lig_h.GetAtoms()):
        role = assign_virtual_role(lig_h, i)
        print('h')
        print(sasa[i])
        if role is None:
            continue
        if sasa[i] < sasa_min:
            continue
        
        vec, dist, j = nearest_pocket_vector(coords[i], pocket_verts)
        if not (min_dist <= dist <= max_dist):
            continue
        pv = vec / (np.linalg.norm(vec) + 1e-9)
        # orient normal inward toward ligand centroid
        n = pocket_normals[j]
        to_lig = centroid - pocket_verts[j]
        if np.dot(n, to_lig) < 0: n = -n
        cosang = float(np.dot(pv, n))
        if cosang < 0:  # outward or dead-end
            cosang *= 0.5
        ndist = 1.0 - (dist - min_dist) / (max_dist - min_dist + 1e-9)  # closer is better
        nsasa = float(sasa[i]) / max_sasa
        geom = w_angle*cosang + w_dist*ndist + w_sasa*nsasa
        out.append(ExitCandidate(
            atom_idx=i,
            atom_symbol=atom.GetSymbol(),
            role=role,
            pos=coords[i],
            vec_to_pocket=pv,
            dist_to_pocket=dist,
            sasa=sasa[i],
            geom_score=geom
        ))
    out.sort(key=lambda x: x.geom_score, reverse=True)
    return out

def raycast_steric_ok(protein_xyz: np.ndarray, start: np.ndarray, direction: np.ndarray,
                      max_len: float=4.0, step: float=0.25, clash_dist: float=2.0) -> bool:
    v = direction / (np.linalg.norm(direction) + 1e-9)
    ts = np.arange(0.0, max_len + 1e-9, step)
    for t in ts:
        p = start + v*t
        if np.min(np.linalg.norm(protein_xyz - p, axis=1)) < clash_dist:
            return False
    return True


# cap-probe docking to confirm pocket space UNFINISHED

def rdkit_to_pdbqt_string(mol3d: Chem.Mol) -> str:
    prep = MoleculePreparation()
    setups = prep.prepare(mol3d)
    chunks = []
    for st in setups:
        s, ok, msg = PDBQTWriterLegacy.write_string(st)
        if not ok:
            raise RuntimeError(msg)
        chunks.append(s)
    return ''.join(chunks)

# def receptor_pdb_to_pdbqt_string(pdb_text: str) -> str:
#     polymer = Polymer.from_pdb_string(pdb_text)
#     rprep = ReceptorPreparation(polymer)
#     rsetup = rprep.prepare()
#     s, ok, msg = ReceptorWriterLegacy.write_string(rsetup)
#     if not ok:
#         raise RuntimeError(msg)
#     return s

def vina_cap_probe_energy(protein_pdb: str, lig: Chem.Mol, exit_atom_idx: int,
                          box_center: Tuple[float,float,float], box_size: Tuple[float,float,float]=(20,20,20)) -> Optional[float]:
    """
    Attach a tiny 'cap' (methyl) along the exit vector (~probe) and dock the capped ligand briefly.
    Returns best Vina score (kcal/mol) or None if Vina/Meeko not available.
    NOTE: This is an *optional* pocket-space sanity check, not chemistry.
    """

    # As vSytnhe does it, use a simple cap probe to confirm growth areas based on where pocket space is 
    if not HAVE_DOCK:
        return None
    # Place a methyl probe along exit vector ~2.0 Å away
    lig3d = Chem.Mol(lig)
    if lig3d.GetNumConformers() == 0:
        lig3d = Chem.AddHs(lig3d)
        AllChem.EmbedMolecule(lig3d, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(lig3d)
    rw = Chem.RWMol(lig3d)
    conf = lig3d.GetConformer()
    p0 = np.array([conf.GetAtomPosition(exit_atom_idx).x,
                   conf.GetAtomPosition(exit_atom_idx).y,
                   conf.GetAtomPosition(exit_atom_idx).z])
    # crude outward direction: use principal axis from prior scoring if stored externally; otherwise place a methyl ~2 Å away in +Z and RDKit will optimize upon docking
    # Add a carbon atom (cap) and connect single bond
    cap_idx = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(exit_atom_idx, cap_idx, Chem.BondType.SINGLE)
    out = rw.GetMol()
    out = Chem.AddHs(out, addCoords=True)
    AllChem.EmbedMolecule(out, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(out)

    # have all below in vina_test i think, just need to change this and import those to use
    # Prepare ligand PDBQT (string)
    lig_pdbqt = rdkit_to_pdbqt_string(out)

    # Prepare receptor PDBQT as string
    with open(protein_pdb, "r") as fh:
        rec_txt = fh.read()
    rec_pdbqt = receptor_pdb_to_pdbqt_string(rec_txt)

    # Vina requires receptor from a path; keep ephemeral, auto-deleted
    tmpdir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    with tempfile.NamedTemporaryFile("w", suffix=".pdbqt", dir=tmpdir, delete=True) as rf:
        rf.write(rec_pdbqt); rf.flush()
        v = Vina(sf_name='vina', verbosity=0)
        v.set_receptor(rf.name)
        v.set_ligand_from_string(lig_pdbqt)
        v.compute_vina_maps(center=list(box_center), box_size=list(box_size))
        v.dock(exhaustiveness=4, n_poses=3)
        # collect best score
        e = v.score()[0]
    return float(e)


def read_sdf_first_mol(sdf:Path):
    with open(sdf, "rb") as f:
            supplier = Chem.ForwardSDMolSupplier(f, sanitize=True, removeHs=False, strictParsing=True)
    if next(supplier):
        return next(supplier)
    else:
        return None

# Main Entrypoint/Orchestrator

def find_exit_sites(protein_pdb: str,
                    ligand: Chem.Mol,
                    use_docking_probe: bool=False, # wire this shit up my guy
                    top_k: int=5) -> Tuple[List[ExitCandidate], List[RSite]]:
    """
    Main entry: geometry-first exits + virtual roles, optional docking probe.
    """


    lig3d = Chem.Mol(ligand)
    if lig3d.GetNumConformers() == 0:
        lig3d = Chem.AddHs(lig3d)
        AllChem.EmbedMolecule(lig3d, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(lig3d)

    lig_xyz = ligand_coords(lig3d)
    verts, norms = pyvol_pocket_vertices(protein_pdb, lig_xyz)
    cands = score_candidates(lig3d, verts, norms)

    # Steric ray-cast
    prot_xyz = protein_heavy_xyz(protein_pdb)
    for c in cands:
        c.steric_ok = raycast_steric_ok(prot_xyz, c.pos, c.vec_to_pocket)

    # Optional docking-based pocket check (lightweight)
    if use_docking_probe and HAVE_DOCK and len(cands) > 0:
        box_center = tuple(lig_xyz.mean(axis=0))
        for c in cands[:top_k*2]:  # check a few more than top_k
            try:
                c.vina_score = vina_cap_probe_energy(protein_pdb, lig3d, c.atom_idx, box_center)
            except Exception:
                c.vina_score = None

    # Rank: steric_ok first, then vina (if available), then geometry
    def rank_key(x: ExitCandidate):
        vina_term = 0.0 if x.vina_score is None else (-x.vina_score)  # lower kcal/mol is better
        return (x.steric_ok is True, vina_term, x.geom_score)

    cands.sort(key=rank_key, reverse=True)

    # Produce R-site labels
    labels: List[RSite] = []
    for i, c in enumerate(cands[:top_k]):
        labels.append(RSite(
            label=f"R{i+1}",
            atom_idx=c.atom_idx,
            role=c.role,
            vector=c.vec_to_pocket / (np.linalg.norm(c.vec_to_pocket) + 1e-9),
            score=(c.geom_score + (0.25 if c.steric_ok else -0.25 if c.steric_ok is False else 0.0))
        ))
    return cands, labels

if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="V‑SYNTHES‑style exit site identification (geometry‑first + virtual roles)."
    )
    parser.add_argument("-p", "--protein_pdb", required=True,
                        help="Path to the receptor PDB (same one used for PyVOL pocket detection).")
    parser.add_argument("-s", "--ligand_sdf", required=True,
                        help="Path to the ligand SDF (first molecule will be used).")
    parser.add_argument("--dock", action="store_true",
                        help="Optional cap‑probe docking (Vina+Meeko). If not installed, this is ignored.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="How many R‑sites to label (default: 5).")
    args = parser.parse_args()

    # Load ligand from SDF (first valid entry)
    suppl = Chem.SDMolSupplier(args.ligand_sdf, removeHs=False)
    ligands = [m for m in suppl if m is not None]
    if not ligands:
        sys.exit(f"[error] No valid molecules found in SDF: {args.ligand_sdf}")
    lig = ligands[0]

    # Ensure 3D (embed if missing)
    print(lig.GetNumConformers())
    if lig.GetNumConformers() == 0:
        lig = Chem.AddHs(lig)
        AllChem.EmbedMolecule(lig, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(lig)

    # Run exit identification
    try:
        cands, labels = find_exit_sites(
            protein_pdb=args.protein_pdb,
            ligand=lig,
            use_docking_probe=args.dock,
            top_k=args.top_k
        )
    except Exception as e:
        sys.exit(f"[error] Exit identification failed: {e}")

    # Report
    print("Top candidates:")
    if not cands:
        print("  (none)")
    else:
        for i, c in enumerate(cands):#[:max(args.top_k, 30)]):  # show a few more than top_k
            print(
                f"  #{i+1:02d} atom={c.atom_idx:>3} {c.atom_symbol:<2} "
                f"role={c.role.name:<24} geom={c.geom_score:+.3f} "
                f"dist={c.dist_to_pocket:.2f}Å SASA={c.sasa:.1f} "
                f"steric_ok={c.steric_ok} "
                f"{'' if c.vina_score is None else f'vina={c.vina_score:+.2f} kcal/mol'}"
            )

    print("\nR‑site labels:")
    if not labels:
        print("  (none)")
    else:
        for r in labels:
            print(
                f"  {r.label}: atom {r.atom_idx}  role={r.role.name}  score={r.score:+.3f}"
            )
        visualize_rsites(lig, labels, out_svg="new.svg")