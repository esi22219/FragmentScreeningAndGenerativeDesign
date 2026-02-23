from rdkit import Chem
from rdkit import RDLogger
import sys
import argparse
import time
import subprocess
import logging
import tempfile
from itertools import islice
from pathlib import Path



def sdf_to_smiles_tempfile(sdf_path, n, use_ids=True):
    """Get intermediary smiles file from input of sdf to running Bulk Generation

    Args:
        sdf_path (path/string): path to sdf input file
        n (int): number of mols to collect from sdf
    """
    
    sdf_path = Path(sdf_path)

    with open(sdf_path, "rb") as f:
        supplier = Chem.ForwardSDMolSupplier(f, sanitize=True, removeHs=False, strictParsing=True)
        mols = [mol for mol in islice(supplier, n) if mol is not None]

    with tempfile.NamedTemporaryFile("w", suffix = ".smi", delete=False) as temp:
        for i, mol in enumerate(mols, start=1):
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            if not smi:
                continue
            if use_ids:
                id = mol.GetProp("Catalog_ID") if mol.HasProp("Catalog_ID") else f"BB_{i}"
                temp.write(f"{smi} {id}\n")
            else:
                temp.write(f"{smi}")
# this needs to be changed to yeild in some form of another instead of writing any sort of temp file for faster processing
    # ie yield mol, id
    return temp.name

def main(args):  
    #sdf_path = "/home3/complib/enamine/building_block/Enamine_Building_Blocks_Stock_530316cmpd_20251211.sdf"
    # hard code to start
    out_dir = Path(args.out).resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(out_dir.parent)
    # generator
    logging.debug("Starting\n")

    temp_path = sdf_to_smiles_tempfile(args.input, args.n)

    # call of Bulk synthonization module
    cmd = ["python", "/home2/esi22219/projects/FragmentScreeningAndGenerativeDesign/SyntOn/src/synthi/SynthOn_BBsBulkClassificationAndSynthonization.py", "-i", temp_path, "-o", out_dir]
    
    if args.keepPG:
        cmd.append("--keepPG")
    if args.Ro2Filtr:
        cmd.append("--Ro2Filtr")
    if args.progress:
        cmd.append("--progress")
        if args.pstep:
            cmd.append("--pstep")
            cmd.append(str(args.pstep))
    # if args.n_cores != -1:
    #     cmd += ["--nCores", str(args.n_cores)]

    # subprocess.run(
    #     cmd,
    #     check=True
    # )

    
# Ensure output directory exists and run tool there so os.rename doesn't cross filesystems

    subprocess.run(
        cmd,
        check=True,
        cwd=str(out_dir.parent)  # <-- key change
    )


    ### FULL SDF ENUMERATION (FOR ANY N LESS THAN TOTAL DO NOT USE) list comp is very long with 500k+mols in sdf
    # suppl = Chem.SDMolSupplier(args.input, sanitize=True, removeHs=False, strictParsing=True)
    # mols_all = [m for m in suppl if m is not None]
    # first_N = mols_all[:args.n]



    return cmd

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate synthons and scaffolds from an SDF using Synt-On functions",
                                epilog="Analysis and Code Implementation: Eli Paul, Sung-Hun Bae\n"
                                    "Eisai Center for Genetics Guided Dementia Discovery (G2D2)\n"
                                    "Original Synt-On Code implementation:                Yuliana Zabolotna, Alexandre Varnek\n"
                                    "                                    Laboratoire de Chémoinformatique, Université de Strasbourg.\n\n"
                                    "Knowledge base (SMARTS library):    Dmitriy M.Volochnyuk, Sergey V.Ryabukhin, Kostiantyn Gavrylenko, Olexandre Oksiuta\n"
                                    "                                    Institute of Organic Chemistry, National Academy of Sciences of Ukraine\n"
                                    "                                    Kyiv National Taras Shevchenko University\n"
                                    "2021 Strasbourg, Kiev",
                                prog="test_read", formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", '-i', required=True, help="Path to sdf file with Building blocks (BBs)")
    p.add_argument("--out", "-o", required=True, help="Output prefix (files will be prefix_synthons.smi, prefix_bb_scaffolds.smi, prefix_classification.tsv)")
    p.add_argument("-n", type = int, default = None, help="Number of BB's to process (Default to None, i.e. process all)")
    #p.add_argument("--id-prop", default=None, help="SDF property name to use as BB id (default: _Name then fallback to record index)")
    p.add_argument("--keepPG", action="store_true", help="Pass keepPG True to mainSynthonsGenerator (keep protected synthons)")
    p.add_argument("--Ro2Filtr", action="store_true", help="Filter produced synthons by Ro2 using Synt-On Ro2Filtration")
    p.add_argument("--n_cores", type=int, default = -1, help = "Number of available cores for parallel calculations. Memory usage is optimized, so maximal number of parallel processes can be launched.")
    #p.add_argument("--dedup", action="store_true", help = "Deduplicate seen pairs of synthons and scaffolds for in run memory")
    p.add_argument("--progress", action="store_true",
                        help="Show a progress bar (tqdm if available; otherwise a simple counter).")
    p.add_argument("--pstep", type=int, default=50,
                        help="Batch updates every N lines in multiprocessing mode (default: 50)")
    args = p.parse_args()

    # global messaging
    # logging.basicConfig(
    #     level=logging.DEBUG if args.verbose else logging.INFO,
    #     format="%(message)s",
    #     stream=sys.stdout
    # )

    # if args.surpress and not args.verbose:
    #     RDLogger.DisableLog('rdApp.*')
    #     RDLogger.DisableLog('rdApp.warning')
    #     RDLogger.DisableLog('rdChemReactions')
    #     from rdkit import rdBase
    #     blocker = rdBase.BlockLogs()
    main(args)