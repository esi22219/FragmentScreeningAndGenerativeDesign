ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT)
import xml.etree.ElementTree as ET
from rdkit import Chem, DataStructs
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem import AddHs, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdmolops import *        
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from rdkit.Chem.Descriptors import *
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.Crippen import MolLogP
import datetime, os, time, random, re, resource, sys
from multiprocessing import Process, Queue
from collections import Counter
import re

_MAPNUM_RE = re.compile(r':(\d+)') 

def readMol(smiles):
    try:
        targetMol = Chem.MolFromSmiles(smiles)
        RemoveStereochemistry(targetMol) # type: ignore
    except:
        if "[nH]" in smiles:
            modifiedSmiles = smiles.replace("[nH]", "n", 1)

        elif "n" in smiles:
            modifiedSmiles = smiles.replace("n", "[nH]", 1)
        elif "[B]" in smiles:
            modifiedSmiles = smiles.replace("[B]", "[B-]")
        else:
            modifiedSmiles = smiles
        try:
            targetMol = Chem.MolFromSmiles(modifiedSmiles)
        except:
            if "[nH]" in modifiedSmiles:
                try:
                    modifiedSmiles = smiles.replace("[nH]", "n")
                    targetMol = Chem.MolFromSmiles(modifiedSmiles)
                    RemoveStereochemistry(targetMol) # type: ignore
                except:
                    print(smiles + " was not processed by rdkit")
            else:
                print(smiles + " was not processed by rdkit")
    else:
        if targetMol == None:
            if "[nH]" in smiles:
                modifiedSmiles = smiles.replace("[nH]", "n", 1)
            elif "n" in smiles:
                modifiedSmiles = smiles.replace("n", "[nH]", 1)
            elif "[B]" in smiles:
                modifiedSmiles = smiles.replace("[B]", "[B-]")
            else:
                modifiedSmiles = smiles
            try:
                targetMol = Chem.MolFromSmiles(modifiedSmiles)
                RemoveStereochemistry(targetMol) # type: ignore
            except:
                if "[nH]" in modifiedSmiles:
                    try:
                        modifiedSmiles = smiles.replace("[nH]", "n")
                        targetMol = Chem.MolFromSmiles(modifiedSmiles)
                        RemoveStereochemistry(targetMol) # type: ignore
                    except:
                        print(smiles + " was not processed by rdkit")
                else:
                    print(smiles + " was not processed by rdkit")
    if targetMol != None:
        return targetMol
    else:
        return None

def countLines(file):
    lines = 0
    f = open(file, "rb")
    buf_size = 1024*1024
    read_f = f.raw.read
    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)
    f.close()
    return lines

def splitFileByLines(inpFile, outName, linesPerFile):
    n = 0
    smallFile = None
    outNamesList = []
    for ind, line in enumerate(open(inpFile)):
        if ind % linesPerFile == 0:
            if smallFile:
                smallFile.close()
            n+=1
            smallFile = open(outName + "_" + str(n), "w")
            outNamesList.append(outName + "_" + str(n))
        smallFile.write(line)
    if smallFile:
        smallFile.close()
    return outNamesList

def listDir(path):
    d_names = []
    f_names = []
    for a, b, c in os.walk(path):
        main_dir = str(a)
        d_names = b
        f_names = c
        break
    return d_names, f_names, main_dir 

def Ro2Filtration(synthonSmiles):
    mol = Chem.MolFromSmiles(synthonSmiles)
    functionality = synthonSmiles.count(":")
    Bivalent_electrophilic = synthonSmiles.count(":30")
    Bivalent_nucleophilic = synthonSmiles.count(":40")
    Bivalent_neutral = synthonSmiles.count(":50")
    MolW = ExactMolWt(mol) - functionality - Bivalent_electrophilic - Bivalent_nucleophilic - Bivalent_neutral
    LogP = MolLogP(mol)
    HDC = CalcNumHBD(mol) # type: ignore
    HAC = CalcNumHBA(mol) # type: ignore
    if "[NH:20]" in synthonSmiles or "[OH:20]" in synthonSmiles or "[SH:20]" in synthonSmiles:
        marksForCount = ["[NH:20]", "[OH:20]", "[SH:20]"]
        count = 0
        for m in marksForCount:
            count += synthonSmiles.count(m)
        HDC -= count
    if MolW > 200 or LogP > 2 or HDC > 2 or HAC > 4:
        return False, ["MolW=" + str(MolW), "LogP=" + str(LogP), "HDC=" + str(HDC), "HAC=" + str(HAC)]
    else:
        return True, ["MolW=" + str(MolW), "LogP=" + str(LogP), "HDC=" + str(HDC), "HAC=" + str(HAC)]

def CheckMolStructure(goodValenceSmiles, label):
    vallences = {"C": 4, "N": 3, "N+": 4, "O": 2, "S:10": 6, "S:20": 2}
    try:
        mol = Chem.MolFromSmiles(goodValenceSmiles)
    except:
        return False
    else:
        if mol:
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() != 0:
                    symbol = atom.GetSymbol()
                    if symbol == "C" or symbol == "O" or (symbol == "N" and "+" not in label):
                        if atom.GetTotalValence() < vallences[symbol]:
                            return False
                    elif symbol == "S":
                        if atom.GetTotalValence() < vallences[symbol + ":" + str(atom.GetAtomMapNum())]:
                            return False
                    elif atom.GetTotalValence() < vallences["N+"]:
                            return False
            return True
        else:
            return False

def generateMajorTautFromSynthonSmiles(initSmiles):
    enumerator = rdMolStandardize.TautomerEnumerator()
    initMol = Chem.MolFromSmiles(initSmiles)
    nHinit = []
    for atom in initMol.GetAtoms():
        if atom.GetAtomMapNum() != 0:
            nHinit.append(atom.GetTotalNumHs())
    initMol.UpdatePropertyCache()
    Chem.GetSymmSSSR(initMol)
    tautMol = enumerator.Canonicalize(initMol)
    tautSmiles = Chem.MolToSmiles(tautMol, canonical=True)
    initSmiles = Chem.MolToSmiles(Chem.MolFromSmiles(initSmiles), canonical=True)
    if tautSmiles == initSmiles:
        return tautSmiles
    nHtaut = []
    for atom in tautMol.GetAtoms():
        if atom.GetAtomMapNum() != 0:
            nHtaut.append(atom.GetTotalNumHs())
    if nHinit == nHtaut:
        return tautSmiles
    else:
        return initSmiles

def checkLable(productSmiles:str, Label:str):
        goodValenceSmiles = None
        if Label.split("->")[0][1] == "S":
            hCount = 1
            out = productSmiles.replace(Label.split("->")[0],
                                            "[" + Label.split("->")[1].split(":")[0] + "H" + str(hCount) + ":" +
                                            Label.split("->")[1].split(":")[1] + "]")
            goodValenceSmiles = out
        else:
            for hCount in range(1, 5):
                if hCount == 1:
                    if "+" in Label and "H" not in Label:
                        out = productSmiles.replace(Label.split("->")[0],
                                                    "[" + Label.split("->")[1].split(":")[0] + "+:" +
                                                    Label.split("->")[1].split(":")[1] + "]")

                    else:
                        out = productSmiles.replace(Label.split("->")[0],
                                                    "[" + Label.split("->")[1].split(":")[0] + ":" +
                                                    Label.split("->")[1].split(":")[1] + "]")

                    check = CheckMolStructure(out, Label)
                    if check:
                        goodValenceSmiles = out
                        break
                if "+" in Label and "H" not in Label:
                    out = productSmiles.replace(Label.split("->")[0],
                                                "[" + Label.split("->")[1].split(":")[0] + "H" + str(hCount) + "+:" +
                                                Label.split("->")[1].split(":")[1] + "]")
                else:
                    out = productSmiles.replace(Label.split("->")[0],
                                                "[" + Label.split("->")[1].split(":")[0] + "H" + str(hCount) + ":" +
                                                Label.split("->")[1].split(":")[1] + "]")

                newMol = Chem.MolFromSmiles(out)
                if not newMol:
                    break
                else:
                    goodValenceSmiles = out
                    check = CheckMolStructure(goodValenceSmiles, Label)
                    if check:
                        break
        if not goodValenceSmiles:
            print("Problem with structure check: " + productSmiles + " " + out)
        else:
            return generateMajorTautFromSynthonSmiles(goodValenceSmiles)

def readSyntonLib(synthLibFile, Ro2Filtration=False, FindAnaloguesOfMissingBBs=False):
    fragBegTime = datetime.datetime.now()
    availableBBs = {}
    re.compile(r"\[\w*:\w*\]")
    for line in open(synthLibFile):
        sline = line.strip()
        if sline:
            mol = Chem.MolFromSmiles(sline.split()[0])
            if Ro2Filtration:
                mol = AddHs(mol)
                MolW = ExactMolWt(mol)
                LogP = MolLogP(mol)
                HDC = CalcNumHBD(mol) # type: ignore
                HAC = CalcNumHBA(mol) # type: ignore
                if MolW>200 or LogP > 2 or HDC > 2 or HAC > 4:
                    continue
            availableBBs[sline.split()[0]] = {}
            availableBBs[sline.split()[0]]["BBs"] = sline.split()[1]
            if FindAnaloguesOfMissingBBs:
                availableBBs[sline.split()[0]]["fp_b"] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            availableBBs[sline.split()[0]]["n_atoms"] = mol.GetNumAtoms()
            availableBBs[sline.split()[0]]["n_rings"] = CalcNumRings(mol)
            availableBBs[sline.split()[0]]["marks"] = sorted(
                [sline.split()[0][m.start():m.start() + 2] + sline.split()[0][m.end() - 4:m.end()] for m in
                 re.finditer(pat, sline.split()[0])]) # type: ignore
            availableBBs[sline.split()[0]]["marksVallences"] = "+".join(sorted([atom.GetSymbol() + ":" +
                        str(atom.GetTotalDegree()) for atom in mol.GetAtoms() if atom.GetAtomMapNum() != 0]))
    print("Lib BB reading time:")
    print(datetime.datetime.now() - fragBegTime)
    return availableBBs

def _split_smirks(line: str):
    """Return (left, right) parts of a SMIRKS; agents (if any) are ignored."""
    if '>>' not in line:
        return line, ''
    left, right = line.split('>>', 1)
    return left.strip(), right.strip()

def _get_mapnums(text: str):
    """Set of integer map numbers in a SMILES/SMARTS/SMIRKS side."""
    return {int(x) for x in _MAPNUM_RE.findall(text)}

def _rdkit_leaving_group_fragment(left_side: str, left_only_maps: set, charge_halide: bool):
    """
    Best-effort LG reconstruction with RDKit:
      - Parse the reactant side SMARTS.
      - Keep atoms whose map number is in left_only_maps, plus any bonds between them.
      - Optionally assign -1 charge to terminal halides (F/Cl/Br/I) when detached.
      - Return SMARTS for the LG fragment (possibly multi-fragment).
    Returns None if RDKit unavailable or parsing fails.
    """
    if Chem is None:
        return None
    try:
        mol = Chem.MolFromSmarts(left_side)
        if mol is None:
            return None

        keep_idx = []
        for a in mol.GetAtoms():
            n = a.GetAtomMapNum()
            if n and n in left_only_maps:
                keep_idx.append(a.GetIdx())
        if not keep_idx:
            return None

        keep_set = set(keep_idx)
        rw = Chem.RWMol()
        old2new = {}

        # Add kept atoms (preserve basic properties + map numbers)
        for oi in sorted(keep_idx):
            a = mol.GetAtomWithIdx(oi)
            na = Chem.Atom(a.GetAtomicNum())
            na.SetIsAromatic(a.GetIsAromatic())
            na.SetFormalCharge(a.GetFormalCharge())
            na.SetNumExplicitHs(a.GetNumExplicitHs())
            na.SetAtomMapNum(a.GetAtomMapNum())
            ni = rw.AddAtom(na)
            old2new[oi] = ni

        # Add bonds only when BOTH ends are also LG atoms
        for b in mol.GetBonds():
            bi, ei = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if bi in keep_set and ei in keep_set:
                rw.AddBond(old2new[bi], old2new[ei], b.GetBondType())
                rb = rw.GetBondBetweenAtoms(old2new[bi], old2new[ei])
                if b.GetIsAromatic():
                    rb.SetIsAromatic(True)

        sub = rw.GetMol()

        # Optional charge heuristic for halides
        if charge_halide:
            for a in sub.GetAtoms():
                if a.GetAtomicNum() in (9, 17, 35, 53) and a.GetFormalCharge() == 0:
                    a.SetFormalCharge(-1)

        return Chem.MolToSmarts(sub)
    except Exception:
        return None

def _regex_leaving_group_fragment(left_side: str, left_only_maps: set):
    """
    Fallback when RDKit is not available:
      - Collect bracketed atoms that carry any of the left_only map numbers.
      - Output them as disconnected atoms ('.' joined).
      - Preserves atom queries + map numbers but loses internal connectivity.
    """
    atoms = re.findall(r'\[[^\]]+\]', left_side)
    chosen = []
    for ba in atoms:
        maps = {int(x) for x in _MAPNUM_RE.findall(ba)}
        if maps & left_only_maps:
            chosen.append(ba)
    return '.'.join(chosen) if chosen else ''

def _make_products_with_explicit_lg(smirks: str, charge_halide: bool):
    """
    If some map numbers appear only on the reactant side, build an LG fragment
    with those mapped atoms and append it to the product side.
    """
    if '>>' not in smirks:
        return smirks, False

    left, right = _split_smirks(smirks)
    if not left:
        return smirks, False

    lmaps = _get_mapnums(left)
    rmaps = _get_mapnums(right)
    left_only = lmaps - rmaps
    if not left_only:
        return smirks, False

    frag = _rdkit_leaving_group_fragment(left, left_only, charge_halide)
    if not frag:
        frag = _regex_leaving_group_fragment(left, left_only)
        if not frag:
            return smirks, False

    new_right = frag if not right else f"{right}.{frag}"
    return f"{left}>>{new_right}", True

def make_leaving_groups_explicit_in_xml(xml_text: str, charge_halide: bool = False):
    """
    Transform an XML string so that in every SMARTS=\"...\" attribute, each SMIRKS line
    gains an explicit product fragment for any reactant-only mapped atoms.
    
    Parameters
    ----------
    xml_text : str
        The input XML content as a string.
    charge_halide : bool, default False
        If True, set -1 charge on halide LG atoms (F/Cl/Br/I) when they are carried over.

    Returns
    -------
    str
        The updated XML content.
    """
    # Parse the XML from string; preserve element order and attributes.
    root = ET.fromstring(xml_text)

    for elem in root.iter():
        smarts = elem.attrib.get("SMARTS")
        if smarts is None:
            continue

        # Preserve multi-line layout: process per line
        lines = smarts.split('\n')
        new_lines = []
        for line in lines:
            raw = line.strip()
            if '>>' not in raw:
                new_lines.append(line)
                continue
            fixed, changed = _make_products_with_explicit_lg(raw, charge_halide)
            # Keep original indentation on that line
            indent = line[:len(line) - len(line.lstrip(' \t'))]
            new_lines.append(indent + fixed)
        elem.set("SMARTS", '\n'.join(new_lines))

    # Serialize back to a string with XML declaration
    # Note: ET.tostring doesn't include the declaration; add it manually.
    xml_bytes = ET.tostring(root, encoding="utf-8")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + xml_bytes.decode('utf-8')