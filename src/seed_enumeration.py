# builds all 
import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import List, Set
from tqdm import tqdm
import time
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*') # just unmapped reactions due to smarts setup in Config/setup.xml from Synt-On original Repo
from rdkit.Chem import rdChemReactions as Reactions
# needed for indexing proper directories
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT)
from SyntOn.src.synthi.SyntOn import enumeration, fragmentation  # type: ignore


def _read_smiles_file(path: str, skip_with_star: bool = True) -> List[str]:
    """Reads synthons file ensuring consistency. i.e. atom map numbers are used to model reaction sites instead of * or other
        Ensures lines are not malformed and do not start with unusual characters

    Args:
        path (str): path to file
        skip_with_star (bool, optional): whether to skip synthons with * notation. Defaults to True.

    Returns:
        List[str]: list of mol smiles
    """
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            tok = s.split()[0]
            if skip_with_star and '*' in tok:
                continue
            out.append(tok)
    return out


def _ensure_outdir(out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

def _collect_temp_results(out_dir: str) -> List[str]:
    """Concatenate and remove all temp_* files produced by worker processes.""" 
    results = []
    d = Path(out_dir)
    for p in sorted(d.glob('temp_*'), key=lambda x: int(x.name.split('_')[1])):
        with open(p, 'r') as f:
            for line in f:
                s = line.strip()
                if s:
                    results.append(s)
        try:
            p.unlink()
        except OSError:
            pass
    return results


def _prepare_seed(enumerator: enumeration, smiles: str):# keep thid here
    """Convert a labeled seed SMILES into a prepared RDKit Mol. Prep the starting seed/scaffold for enumeration with synthons

    We reuse the internal preparation method so that the seed gets the same hydrogen mark-up as library synthons and are compatible
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"Cannot parse seed SMILES: {smiles}")
    prep = enumerator._enumeration__PrepMolForReconstruction([m])
    if not prep:
        raise RuntimeError(f"Failed to prepare seed synthon: {smiles}")
    return prep[0]


def enumerate_from_seed(
    seed_smiles: str,
    library_synthons: List[str],
    out_dir: str,
    ncores: int,
    max_stages: int,
    desired_per_seed: int,
    mw_lower: int,
    mw_upper: int,
    reconstruction_smarts: List[str],
) -> Set[str]:
    """Enumerate products starting from the given labeled seed. main call

    Returns the unique SMILES set (size may exceed desired_per_seed during merging,
    we cap before writing).
    """
    # Build a fresh enumerator using the library synthons and reconstruction rules
    enum = enumeration(
        outDir=out_dir,
        Synthons=list(set(library_synthons)),
        reactionSMARTS=reconstruction_smarts,
        maxNumberOfReactedSynthons=max_stages + 1,
        MWupperTh=mw_upper if mw_upper is not None else None,
        MWlowerTh=mw_lower if mw_lower is not None else None,
        desiredNumberOfNewMols=10 ** 12,  # effectively unlimited; we control per-seed cap
        nCores=max(1, ncores),
    )

    # Prepare the seed mol using the same internal method as the library entries
    seed_mol = _prepare_seed(enum, seed_smiles)

    # Build a partner pool from enum internals
    partners: List[Chem.Mol] = []
    partners.extend(enum._enumeration__poliFuncBB)  # type: ignore[attr-defined]
    partners.extend(enum._enumeration__biFuncBB)   # type: ignore[attr-defined]
    partners.extend(enum._enumeration__monoFuncBB) # type: ignore[attr-defined]
    #print([Chem.MolToSmiles(m) for m in partners])
    
    
    # # Optionally pre-filter partners by allowed marks (minor optimization)
    # import re
    # pat = re.compile(r"\[\w\*:\w\]")
    # seed_sm = Chem.MolToSmiles(seed_mol, canonical=True)
    # print(seed_sm)
    # allowed = set()
    # for m in re.finditer(pat, seed_sm):
    #     key = seed_sm[m.start()+1] + ":" + seed_sm[m.end()-3:m.end()-1]
    #     if key in enum._enumeration__marksCombinations:  # type: ignore[attr-defined]
    #         allowed.update(enum._enumeration__marksCombinations[key])  # type: ignore[attr-defined]
    # print(f"allowed: {allowed}")
    # filtered_partners = []
    # for p in partners:
    #     ps = Chem.MolToSmiles(p, canonical=True)
    #     marks = set(
    #         [ps[m.start()+1] + ":" + ps[m.end()-3:m.end()-1] for m in re.finditer(pat, ps)]
    #     )
    #     if allowed & marks:
    #         filtered_partners.append(p)
    
    
    token_re = re.compile(r"\[([^\[\]]+?):(\d+)\]")
    
    
    def _to_key(descriptor: str, mapnum: str) -> str:
        """
        Convert bracket descriptor (e.g., 'CH3', 'cH', 'NH', 'O', 'nH') + mapnum ('10','20',...)
        into the Synt-On key expected by _marksCombinations:
          - 'C' for aliphatic carbon (starts with 'C')
          - 'c' for aromatic carbon (starts with 'c')
          - single-letter elements for N/O/S/n (starts with 'N','O','S','n')
        Fallback: uppercase of first char if an unusual descriptor appears.
        """
        ch0 = descriptor[0]  # first character encodes element/aromaticity in RDKit bracket notation
        if ch0 in ("C", "c", "N", "n", "O", "S"):
            base = ch0
        else:
            base = ch0.upper()
        return f"{base}:{mapnum}"

    # Build 'allowed' from the seed's marks
    seed_sm = Chem.MolToSmiles(seed_mol, canonical=True)
    allowed = set()
    for m in token_re.finditer(seed_sm):
        descriptor, mapnum = m.group(1), m.group(2)
        key = _to_key(descriptor, mapnum)
        # If the key is recognized, pull its compatible marks from Synt-On
        if key in enum._enumeration__marksCombinations:
            allowed.update(enum._enumeration__marksCombinations[key])  # type: ignore[attr-defined]

    # If no marks detected on the seed (e.g., unlabeled seed),
    # skip filtering to avoid dropping all partners; reconstruction
    # will still enforce chemical feasibility.
    print(f"allow: {allowed}")
    if not allowed:
        filtered_partners = partners
    else:
        filtered_partners = []
        for p in partners:
            ps = Chem.MolToSmiles(p, canonical=True)
            partner_marks = set()
            for m in token_re.finditer(ps):
                descriptor, mapnum = m.group(1), m.group(2)
                print(descriptor, mapnum)
                partner_marks.add(_to_key(descriptor, mapnum))
            if allowed & partner_marks:
                filtered_partners.append(p)

    
    print(f" len part: {len(partners)}, len filtered: {len(filtered_partners)}")
    # Spawn child processes that call the internal reconstruction function (like Synt-On)
    results: Set[str] = set()
    Pool = []  # list of [process, queue, joined?]

    # Clean any leftover temp files before starting
    for tmp in Path(out_dir).glob('temp_*'):
        try:
            tmp.unlink()
        except OSError:
            pass

    gen_nonuniq = 0
    for partner in filtered_partners:
        if ncores > 1:
            # mirror Synt-On's multiprocessing pattern
            from multiprocessing import Process, Queue
            # keep pool bounded by ncores
            # count & merge active threads using Synt-On's helper
            Pool, nAlive = enum._enumeration__countAndMergeActiveThreads(Pool)  # type: ignore[attr-defined]
            while nAlive >= ncores:
                time.sleep(0.2)
                Pool, nAlive = enum._enumeration__countAndMergeActiveThreads(Pool)  # type: ignore[attr-defined]

            q = Queue()
            p = Process(
                target=enum._enumeration__molReconsrtuction,  # type: ignore[attr-defined]
                args=(seed_mol, partner, 1, None, q),
            )
            p.start()
            Pool.append([p, q, False])
        else:
            # single-process path (no temp files)
            sub = enum._enumeration__molReconsrtuction(seed_mol, partner, 1)  # type: ignore[attr-defined]
           # print(len(sub))
            if sub:
                for s in sub:
                    if s not in results:
                        results.add(s)
                        gen_nonuniq += 1
                        print(gen_nonuniq)
                        if len(results) >= desired_per_seed:
                            break
        if len(results) >= desired_per_seed:
            break

    if ncores > 1:
        # Drain pool
        Pool, nAlive = enum._enumeration__countAndMergeActiveThreads(Pool)  # type: ignore[attr-defined]
        while nAlive > 0:
            time.sleep(0.2)
            Pool, nAlive = enum._enumeration__countAndMergeActiveThreads(Pool)  # type: ignore[attr-defined]
        # Merge temp files and deduplicate
        merged = _collect_temp_results(out_dir)
        for s in merged:
            results.add(s)
            if len(results) >= desired_per_seed:
                break

    return results



# all_results = set()

# for s in seed_smiles_list:
#     seed_mol = Chem.MolFromSmiles(s)

#     # Run ONLY from this seed:
#     res = enum.getReconstructedMols(
#         allowedToRunSubprocesses=False,   # or True if you want subprocess batching to temp files
#         randomSeed=seed_mol,              # <- start exactly from this seed
#         seed=(0, 0),                      # ignored because randomSeed is provided
#         mainRun=False                     # <- do NOT auto-advance to other seeds
#     )
#     all_results.update(res)  # union results from each seed





def main():
    parser = argparse.ArgumentParser(
        description="Seed-driven enumeration using Synt-On reconstruction reactions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--seeds', required=True, help='File with labeled seed synthons (SMILES).')
    parser.add_argument('--synthonLib', required=True, help='File with library synthons (SMILES).')
    parser.add_argument('-oD', '--outDir', required=True, help='Output directory.')
    parser.add_argument('--nCores', type=int, default=1, help='Parallel processes for partner expansion.')
    parser.add_argument('--MaxNumberOfStages', type=int, default=5, help='Maximum number of stages (depth).')
    parser.add_argument('--desiredPerSeed', type=int, default=1000, help='Target unique products per seed.')
    parser.add_argument('--MWupperTh', type=int, default=460, help='Upper MW threshold.')
    parser.add_argument('--MWlowerTh', type=int, default=100, help='Lower MW threshold.')
    parser.add_argument('--reactionsToWorkWith', type=str, default='R1-R13', help='Subset of reactions to allow.')

    args = parser.parse_args()

    _ensure_outdir(args.outDir)

    seeds = _read_smiles_file(args.seeds, skip_with_star=True)
    if not seeds:
        raise SystemExit("No usable seeds found (ensure labeled synthons, no '*' lines)")
 
    
    lib = _read_smiles_file(args.synthonLib, skip_with_star=True)
    if not lib:
        raise SystemExit("No usable synthons found in synthonLib (first token per line is used)")
  
    # Instantiate a fragmentor only to obtain reconstruction reactions
    frag = fragmentation()
    # If the user specified a subset, pass it in; otherwise take all non-R14
    reactionSMARTS = frag.getReactionForReconstruction()

    # Iterate seeds and enumerate
    all_unique: List[str] = []
    for idx, seed in enumerate(seeds, start=1):
       # print(f"[Seed {idx}] {seed}")
        print("start\n")
        s = time.time()
        uniq = enumerate_from_seed(
            seed_smiles=seed,
            library_synthons=lib,
            out_dir=args.outDir,
            ncores=max(1, args.nCores),
            max_stages=args.MaxNumberOfStages,
            desired_per_seed=args.desiredPerSeed,
            mw_lower=args.MWlowerTh,
            mw_upper=args.MWupperTh,
            reconstruction_smarts=reactionSMARTS,
        )
        # Write per-seed file
        print(len(uniq))
        e = time.time()
        print(f"total time {round(e - s, 2)}")
        if uniq:
            print(uniq)
            per_seed_path = Path(args.outDir) / f"Seed_{idx}_enumerated.smi"
            # with open(per_seed_path, 'w') as out:
            #     for s in list(uniq)[: args.desiredPerSeed]:
            #         out.write(s + "\n")
            all_unique.extend(uniq)
            #print(f"  -> wrote {min(len(uniq), args.desiredPerSeed)} unique molecules to {per_seed_path}")

    # # Write aggregate ooga booga its just a random erstsdfggfgsudhu HEp me get over myself an dfind success in my life
    # agg = Path(args.outDir) / "AllSeeds_enumerated.smi"
    # seen = set()
    # with open(agg, 'w') as out:
    #     for s in all_unique:
    #         if s not in seen:
    #             out.write(s + "\n")
    #             seen.add(s)
    # print(f"[Done] Aggregate written: {agg} (unique={len(seen)})")


if __name__ == '__main__':
    # srcPath = os.path.split(os.path.realpath(__file__))[0]
    # sys.path.insert(1, srcPath) just checking with these but useless now as set on top
    # print(srcPath)
    main()
