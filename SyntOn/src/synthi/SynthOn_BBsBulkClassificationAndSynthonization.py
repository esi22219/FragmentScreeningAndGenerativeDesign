from src.data_models import MissSink, SynthonRecord, BBProcessingResult
from typing import Iterable, Iterator, List, Optional, Set, Dict, Tuple, Callable
import re

_MARK_RE = re.compile(r"\[\w*:\w*\]") # same as used in SyntOn.py for matching marks

import os,sys, io
import argparse
import hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT)
from SyntOn_Classifier import *
from SyntOn_BBs import *
from UsefulFunctions import readMol
from rdkit import Chem
"""from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.rdmolops import *"""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import threading
"""from rdkit.Chem.Descriptors import *
from rdkit.Chem.rdMolDescriptors import *"""
from functools import partial
from tqdm import tqdm
from rdkit import rdBase
from rdkit import RDLogger
# logging
RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdChemReactions')
blocker = rdBase.BlockLogs()

def _process_chunk(
    pairs_chunk,
    keepPG: bool,
    Ro2Filtr: bool
):
    out = []
    for (bb_smiles, bb_id) in pairs_chunk:
        res = process_bb_record(bb_smiles, bb_id, keepPG=keepPG, Ro2Filtr=Ro2Filtr)
        if res is not None:
            out.append(res)
    return out

def _extract_marks(smi: str) -> Tuple[str, ...]:
    """ Convert label tokens in SMILES to the compact "X:NN" form used by enumeration

    Args:
        smi (str): smiles string to parse

    Returns:
        Tuple[str, ...]: collection of marks
    """

    # Examples in SyntOn.py also use this map-style extraction.
    marks = []
    plain = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
    for m in _MARK_RE.finditer(plain):
        # "[C:10]" -> 'C:10' ; "[nH:20]" -> 'n:20' (drop H)
        token = plain[m.start()+1:m.end()-1]   # C:10 or nH:20
        atom, lbl = token.split(":")
        atom = atom.replace("H","")  # keep 'n' from 'nH'
        marks.append(f"{atom}:{lbl}")
    return tuple(sorted(marks))

def process_bb_record(
    bb_smiles: str,
    bb_id: str,
    keepPG: bool = False,
    Ro2Filtr: bool = False,
) -> Optional[BBProcessingResult]:
    """
    Process a single BB (SMILES + ID) into synthons and metadata.

    Returns:
        BBProcessingResult or None if the BB couldn't be parsed.
    """
    fragments = [frag for frag in bb_smiles.split(".") if frag] # keep splitting by "." as in synthi repo, will merge synthons from all fragments into a single result

    all_classes: List[str] = []
    final_synthon_map: Dict[str, Set[str]] = {}  # synthon_smiles -> annotations (set of strings)
    classified = False
    without_synthons = True
    azoles_flag = False

    for frag in fragments:
        # replicate the [nH+] -> [nH] from og synton
        if "[nH+]" in frag:
            frag = frag.replace("[nH+]", "[nH]:", 1)

        initMol = readMol(frag)
        if initMol is None:
            # BB not readable
            continue

        # recreated workflow from Synt-On og BulkClassificationandSynthonization using same methods
        # classification
        Classes = BBClassifier(mol=initMol)
        if Classes:
            classified = True
            all_classes.extend(Classes)

        # mark whether any non-MedChemHighlights/DEL classes exist
        frag_has_synthons = any(("MedChemHighlights" not in c and "DEL" not in c) for c in (Classes or []))
        without_synthons = without_synthons and (not frag_has_synthons)

        # remove 'MedChemHighlights'/'DEL'
        filtered_classes = [c for c in (Classes or []) if "MedChemHighlights" not in c and "DEL" not in c]

        # synthon gen
        azoles, fSynt = mainSynthonsGenerator(Chem.MolToSmiles(initMol), keepPG, filtered_classes, returnBoolAndDict=True)
        azoles_flag = azoles_flag or bool(azoles)

        # merge synthons and annotations
        for syn, info in fSynt.items():
            if syn not in final_synthon_map:
                final_synthon_map[syn] = set(info)
            else:
                final_synthon_map[syn].update(info)

    # post-processing
    if not fragments:
        return None

    if azoles_flag:
        all_classes.append("nHAzoles_nHAzoles")
    all_classes = [c for c in all_classes if "MedChemHighlights" not in c and "DEL" not in c]

    # build SynthonRecord list with optional Ro2 filtration
    synthon_records: List[SynthonRecord] = []
    for syn_smi, annotations_set in final_synthon_map.items():
        if Ro2Filtr:
            ok, _params = Ro2Filtration(syn_smi)  # same as legacy usage
            if not ok:
                continue
        # if all passed make into a record class for passage to later steps
        marks = _extract_marks(syn_smi)
        synthon_records.append(
            SynthonRecord(
                synthon_smiles=syn_smi,
                marks=marks,
                n_marks=len(marks),
                annotations=tuple(sorted(annotations_set)),
                source_bb_smiles=bb_smiles,
                source_bb_id=bb_id,
                classes=tuple(sorted(set(all_classes))),
            )
        )
    # return an instance of a processed result class
    return BBProcessingResult(
        bb_smiles=bb_smiles,
        bb_id=bb_id,
        classes=tuple(sorted(set(all_classes))),
        synthons=tuple(synthon_records),
        classified=classified,
        had_synthons=bool(synthon_records),
        azoles_flag=azoles_flag,
    )

def iter_bb_processing_results(
    records: Iterable[Tuple[str, str]],
    keepPG: bool = False,
    Ro2Filtr: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
    progress_step: int = 100,
) -> Iterator[BBProcessingResult]:
    """
    Streaming generator: yields BBProcessingResult for each (bb_smiles, bb_id).

    - No file I/O.
    - Designed to be mapped across workers.
    - Optional progress callback called every `progress_step` processed BBs. for progress bar and updates
    """
    count = 0
    for bb_smiles, bb_id in records:
        res = process_bb_record(bb_smiles, bb_id, keepPG=keepPG, Ro2Filtr=Ro2Filtr)
        count += 1
        if progress_callback and (count % progress_step == 0):
            progress_callback(progress_step)
        if res is not None:
            yield res
            
# all changed to properly yield/stream intermediate calculations and actions
def iter_synthons(
    records: Iterable[Tuple[str, str]],
    keepPG: bool = False,
    Ro2Filtr: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
    progress_step: int = 100,
) -> Iterator[SynthonRecord]:
    """go through all BBs and yield synthon object

    Args:
        records (Iterable[Tuple[str, str]]): BB file
        keepPG (bool, optional): keep protected groups in Synthons. Defaults to False.
        Ro2Filtr (bool, optional): Filter out synthons/BBs based on simple filters like MW and num atoms. Defaults to False.
        progress_callback (Optional[Callable[[int], None]], optional): progress handler for debugging. Defaults to None.
        progress_step (int, optional): how many steps to report back. Defaults to 100.

    Yields:
        Iterator[SynthonRecord]: synthon generator for better handling
    """
    for bb_result in iter_bb_processing_results(records, keepPG, Ro2Filtr, progress_callback, progress_step):
        for syn in bb_result.synthons:
            yield syn

def _iter_input_pairs(path: str):
    """Take input file and give a BB smiles and id

    Args:
        path (str): path to BB .smi file

    Yields:
        smi, bbid: smiles of BB + its id
    """
    with open(path) as fh:
        for idx, line in enumerate(fh, start=1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            smi = parts[0]
            bbid = parts[1] if len(parts) > 1 else f"BB_{idx}"
            yield smi, bbid

def _chunked(iterable, size):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

def main(inp: str,
         keepPG: bool,
         output: Optional[str] = None,
         Ro2Filtr: bool = False,
         save_files: bool = False,
         sink: Optional[Callable[[BBProcessingResult], None]] = None,
         nCores: int = -1,
         chunk_size: int = 1000) -> int:

    # Resolve cores
    if nCores is None or nCores == 0:
        nCores = 1
    elif nCores < 0:
        try:
            import multiprocessing as mp
            nCores = max(1, mp.cpu_count())
        except Exception:
            nCores = 1

    # Prepare legacy writers if requested
    if save_files:
        if not output:
            output = inp
        out_bb = open(output + "_BBmode.smi", "w")
        out_syn = open(output + "_Synthmode.smi", "w")
        out_np  = open(output + "_NotProcessed", "w")
        out_nc  = open(output + "_NotClassified", "w")
    else:
        out_bb = out_syn = out_np = out_nc = None

    try:
        # Single-core fast path (no executor overhead)
        if nCores == 1:
            for pairs in _chunked(_iter_input_pairs(inp), chunk_size):
                results = _process_chunk(pairs, keepPG, Ro2Filtr)
                _consume_results(results, out_bb, out_syn, out_np, out_nc, sink, save_files)
        else:
            # Multi-core: map chunks across workers
            with ProcessPoolExecutor(max_workers=nCores) as ex:
                futures = (
                    ex.submit(_process_chunk, pairs, keepPG, Ro2Filtr)
                    for pairs in _chunked(_iter_input_pairs(inp), chunk_size)
                )
                for fut in futures:
                    results = fut.result()  # propagate exceptions here
                    _consume_results(results, out_bb, out_syn, out_np, out_nc, sink, save_files)
    finally:
        # Close files if used
        for h in (out_bb, out_syn, out_np, out_nc):
            try:
                h and h.close()
            except Exception:
                pass

    return 0

def _consume_results(results, out_bb, out_syn, out_np, out_nc, sink, save_files):
    """
    Single place that:
      - writes legacy files (optional)
      - forwards to sink (optional)
    """
    for bb_res in results:
        # Optional sink first (so in-memory pipeline stays hot)
        if sink is not None:
            sink(bb_res)

        if not save_files:
            continue

        # Legacy writing (same formatting as before)
        sline = f"{bb_res.bb_smiles} {bb_res.bb_id}"

        if not bb_res.classified:
            out_nc.write(sline + "\n"); out_nc.flush()
            continue

        if not bb_res.had_synthons:
            out_bb.write(sline + " " + ",".join(bb_res.classes) + " -\n"); out_bb.flush()
            continue

        out_bb.write(
            sline + " " +
            ",".join(bb_res.classes) + " " +
            ".".join([syn.synthon_smiles for syn in bb_res.synthons]) + " " +
            str(len(bb_res.synthons)) + "\n"
        ); out_bb.flush()

        for syn in bb_res.synthons:
            out_syn.write(
                syn.synthon_smiles + " " +
                "+".join(syn.annotations) + " " +
                sline + "\n"
            ); out_syn.flush()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="BBs classification and Synthons generation for large BBs libraries",
                                        epilog="Code implementation:                Yuliana Zabolotna, Alexandre Varnek\n"
                                            "                                    Laboratoire de Chémoinformatique, Université de Strasbourg.\n\n"
                                            "Knowledge base (SMARTS library):    Dmitriy M.Volochnyuk, Sergey V.Ryabukhin, Kostiantyn Gavrylenko, Olexandre Oksiuta\n"
                                            "                                    Institute of Organic Chemistry, National Academy of Sciences of Ukraine\n"
                                            "                                    Kyiv National Taras Shevchenko University\n"
                                            "2021 Strasbourg, Kiev",
                                        prog="SyntOn_BBsBulkClassificationAndSynthonization", formatter_class=argparse.RawTextHelpFormatter)
    # files
    parser.add_argument("-i", "--input", type=str, help="Input file containing 2 columns building blocks smiles and ids.")
    parser.add_argument("-o", "--output", type=str, help="Output files suffix name.")
    parser.add_argument("--save_files", action="store_true", help="If set, write output files (_BBmode.smi, _Synthmode.smi, _NotProcessed, _NotClassified). ")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output prefix (used only when --save-files is set).")

    # filtering
    parser.add_argument("--keepPG", action="store_true", help="Write both protected and unprotected synthons to the output (concerns Boc, Bn, Fmoc, Cbz and Esters protections).")
    parser.add_argument("--Ro2Filtr", action="store_true", help="Write only synthons satisfying Ro2 (MW <= 200, logP <= 2, H-bond donors count <= 2 and H-bond acceptors count <= 4)")

    # threading
    parser.add_argument("--nCores", default=-1, type=int, help="Number of available cores for parallel calculations. Memory usage is optimized, so maximal number of parallel processes can be launched.")
    parser.add_argument("--chunk_size", type=int, default=1000,
        help="Number of BB lines per task sent to a worker (balances IPC/CPU). Default: 1000")

    args = parser.parse_args()
    sink = MissSink(dedupe=True)

    main(inp=args.input,
        keepPG=args.keepPG,
        Ro2Filtr=args.Ro2Filtr,
        nCores=args.nCores,
        chunk_size=args.chunk_size,
        save_files=args.save_files,
        output=args.output,
        sink=sink)
