from typing import List, Optional, Dict, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# simple color palette (RGB in [0,1])
_PALETTE = [
    (0.90, 0.17, 0.31),  # red
    (0.00, 0.45, 0.70),  # blue
    (0.00, 0.62, 0.38),  # green
    (0.80, 0.47, 0.74),  # purple
    (0.95, 0.90, 0.25),  # yellow
    (0.34, 0.67, 0.90),  # light blue
]

def visualize_rsites(
    lig: Chem.Mol,
    labels: List["RGroupLabel"],
    img_size: int = 600,
    remove_hs_for_drawing: bool = True,
    out_svg: Optional[str] = None
) -> str:
    """
    Draw the ligand and highlight atoms labeled as R-groups.
    Each R-site atom is colored and annotated with its label (e.g., R1, R2).

    Returns an SVG string; also writes to 'out_svg' if provided.
    """
    # 1) Make a copy for drawing
    mol = Chem.Mol(lig)

    # 2) Optionally remove Hs for a cleaner picture
    #    NOTE: if your labels refer to explicit H atoms, set remove_hs_for_drawing=False
    draw_mol = Chem.RemoveHs(mol) if remove_hs_for_drawing else mol

    # 3) Generate 2D coordinates for a clean depiction
    rdDepictor.Compute2DCoords(draw_mol)
    rdMolDraw2D.PrepareMolForDrawing(draw_mol, kekulize=False)

    # 4) Build a mapping if we removed Hs (for most ligands, heavy-atom indices are stable)
    #    In many cases, RemoveHs preserves the ordering of heavy atoms, so we can use a direct map.
    #    If your systems are tricky, you can replace this with a more elaborate mapping via substructure matches.
    idx_map = {i: i for i in range(draw_mol.GetNumAtoms())}
    if remove_hs_for_drawing and draw_mol.GetNumAtoms() != mol.GetNumAtoms():
        # naive heavy-atom index map: iterate heavy atoms in the original and match order in draw_mol
        heavy_src = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        heavy_dst = list(range(draw_mol.GetNumAtoms()))
        if len(heavy_src) == len(heavy_dst):
            idx_map = dict(zip(heavy_src, heavy_dst))
        else:
            # fallback: identity where possible
            idx_map = {i: i for i in range(min(len(heavy_src), len(heavy_dst)))}

    # 5) Mark highlights + atom notes (labels)
    highlight_atoms = []
    atom_colors: Dict[int, Tuple[float, float, float]] = {}
    for i, lab in enumerate(labels):
        src_idx = lab.atom_idx
        dst_idx = idx_map.get(src_idx, None)
        if dst_idx is None or dst_idx >= draw_mol.GetNumAtoms():
            continue
        highlight_atoms.append(dst_idx)
        atom_colors[dst_idx] = _PALETTE[i % len(_PALETTE)]
        # Set an atom note so the label is rendered near the atom
        draw_mol.GetAtomWithIdx(dst_idx).SetProp("atomNote", lab.label)

    # 6) Draw
    drawer = rdMolDraw2D.MolDraw2DSVG(img_size, img_size)
    dopts = drawer.drawOptions()
    dopts.addAtomIndices = False
    dopts.addBondIndices = False
    dopts.fixedBondLength = 25  # consistent scaling
    dopts.useBWAtomPalette()    # base palette BW, highlights provide color
    # slightly larger highlight halos
    highlight_radii = {a: 0.6 for a in highlight_atoms}

    drawer.DrawMolecule(
        draw_mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=highlight_radii,
        highlightBonds=[]
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    if out_svg:
        with open(out_svg, "w") as fh:
            fh.write(svg)
    return svg