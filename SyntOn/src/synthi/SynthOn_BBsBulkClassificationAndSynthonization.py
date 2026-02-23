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
RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdChemReactions')

blocker = rdBase.BlockLogs()

# for progress bar (small useful helper)
class _DummyTqdm:
    def __init__(self, total=None, **kwargs):
        self.total = total
        self.n = 0
    def update(self, n=1):
        self.n += n
    def close(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()


def _get_pbar(total: int, enable: bool, **kwargs):
    """helper for progress bar initialization

    Args:
        total (int): final count for progress check
        enable (bool): whether to use pbar or not

    Returns:
        _DummyTqdm: tqdm progress bar object
    """
    if enable and tqdm is not None:
        return tqdm(total=total, **kwargs)
    return _DummyTqdm(total=total, **kwargs)

def main(inp,keepPG, output=None, Ro2Filtr=False, progress_queue=None, show_progress=False, total_lines=None,
         progress_step: int = 50):
    """Main Bulk BB Classification and Synthonization with multiprocessing.
    Currently saves 4 output files. Will be changed to streaming most likely

    Args:
        inp (path): path to input sdf BB file
        keepPG (bool): whether to keep protected groups in synthonization
        output (str, optional): output file prefix declaration. Defaults to None.
        Ro2Filtr (bool, optional): Filter generated synthons simly by MW<=200, logP<=2, H-bond donors<=2 and acceptors<=4. Defaults to False.
    """
    # if no specified prefix, set to input file name
    if not output:
        output=inp


    # set up progres bar tracking
    batch_counter = 0
    pbar = None
    
    if show_progress and total_lines is not None:
        pbar = _get_pbar(total=total_lines, enable=True, desc=f"Processing {os.path.basename(inp)}", unit="line")

    # Change from saving a file to string.io or something better in memory so there is no need to save and open intermediate files
    # for example
    #io.StringIO("output text") and then can push text to a manager of some kind to hold and batch push to next step

    # but to utilize, need to have all information as string, which is mostly already done at some point
    # \n.join[list of information per line that is saved into output file]
    with open(inp) as inp, open(output + "_BBmode.smi", "w") as out, open(output + "_Synthmode.smi", "w") as outS,\
        open(output + "_NotProcessed", "w") as notProc, open(output + "_NotClassified", "w") as notClassified: # create log/info files
        # process input line by line and update files accordingly
        
        for line in inp:
            sline = line.strip()
            if not sline:
                continue
            
            # prog bar updates
            if progress_queue is not None:
                progress_step = int(progress_step)
                batch_counter += 1
                if batch_counter % progress_step == 0:
                    try:
                        progress_queue.put(progress_step)
                    except Exception:
                        pass
            elif pbar is not None:
                pbar.update(1)


            finalSynthon = {}
            classified = False
            withoutSynthons = True
            allClasses = []
            for molSmiles in line.split()[0].split("."):
                if "[nH+]" in molSmiles:
                    molSmiles = molSmiles.replace("[nH+]", "[nH]:", 1)
                initMol = readMol(molSmiles)
                if initMol == None: # BB not able to be read
                    notProc.write(line)
                    notProc.flush()
                    continue

                # BB Classification call
                Classes = BBClassifier(mol=initMol)
                allClasses.extend(Classes)
                if Classes:
                    classified = True
                    for clas in Classes:
                        if "MedChemHighlights" not in clas and "DEL" not in clas:
                            withoutSynthons = False
                            break
                ind2del = []
                for ind,Class in enumerate(Classes):
                    if "MedChemHighlights" in Class or "DEL" in Class:
                        ind2del.append(ind)
                if ind2del:
                    for ind in ind2del[::-1]:
                        Classes.pop(ind)
                
                # synthon generation from BB
                azoles,fSynt = mainSynthonsGenerator(Chem.MolToSmiles(initMol), keepPG, Classes,returnBoolAndDict=True)
                for synth1 in fSynt:
                    if synth1 not in finalSynthon: 
                        finalSynthon[synth1] = fSynt[synth1].copy()
                    else:
                        finalSynthon[synth1].update(fSynt[synth1].copy())
            
            # BB unable to be classfied
            if not classified:
                notClassified.write(line)
                notClassified.flush()
                continue

            # no synthons were found for BB, but BB classification succeeded
            if not finalSynthon and withoutSynthons:
                
                out.write(sline + " " + ",".join(allClasses) + " -\n")
                io.StringIO(sline + " " + ",".join(allClasses) + " -\n") # essentially
                out.flush() # what is this? just removing anything left over from memory? gotta be just clearing up any space not being
                # used downstream
                continue

            # all previous calls worked, save to BBfile. if multiple synthons, they are separated by "." in output file
            #print(finalSynthon)
            
            if finalSynthon:
                if azoles:
                    allClasses.append("nHAzoles_nHAzoles")
                out.write(sline + " " + ",".join(allClasses) + " " + ".".join(list(finalSynthon)) + " " + str(len(finalSynthon)) + "\n")
                
                # optional filter and write Synthonfile
                for synth2 in finalSynthon:
                    if Ro2Filtr: # filter
                        goodSynth, paramsValList = Ro2Filtration(synth2)
                        if not goodSynth:
                            continue
                        else:
                            outS.write(synth2 + " " + "+".join(finalSynthon[synth2]) + " " + sline + "\n") # all of this appened to list and done above or maybe better method
                    else:
                        outS.write(synth2 + " " + "+".join(finalSynthon[synth2]) + " " + sline + "\n")
            elif not withoutSynthons:
                out.write(sline + " " + ",".join(Classes) + " NoSynthonsWereGenerated\n") 
            
                # can pretty much just change any line with write to io and then no file opening and writing required,
                # just in memory
        # flush any remaining batched progress updates
        if progress_queue is not None:
            remainder = batch_counter % progress_step
            if remainder:
                try:
                    progress_queue.put(remainder)
                except Exception:
                    pass

    if pbar is not None:
        pbar.close()


def cli() -> int:
    """
    Console entry point: parse CLI args and delegate to main() or the parallel processing p xzath.
    Returns an exit code (0 on success).
    """

    parser = argparse.ArgumentParser(description="BBs classification and Synthons generation for large BBs libraries",
                                     epilog="Code implementation:                Yuliana Zabolotna, Alexandre Varnek\n"
                                            "                                    Laboratoire de Chémoinformatique, Université de Strasbourg.\n\n"
                                            "Knowledge base (SMARTS library):    Dmitriy M.Volochnyuk, Sergey V.Ryabukhin, Kostiantyn Gavrylenko, Olexandre Oksiuta\n"
                                            "                                    Institute of Organic Chemistry, National Academy of Sciences of Ukraine\n"
                                            "                                    Kyiv National Taras Shevchenko University\n"
                                            "2021 Strasbourg, Kiev",
                                     prog="SyntOn_BBsBulkClassificationAndSynthonization", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--input", type=str, help="Input file containing 2 columns building blocks smiles and ids.")
    parser.add_argument("-o", "--output", type=str, help="Output files suffix name.")
    parser.add_argument("--keepPG", action="store_true", help="Write both protected and unprotected "
                                            "synthons to the output (concerns Boc, Bn, Fmoc, Cbz and Esters protections).")
    parser.add_argument("--Ro2Filtr", action="store_true", help="Write only synthons satisfying Ro2 (MW <= 200, logP <= 2, H-bond donors count <= 2 and H-bond acceptors count <= 4)")

    parser.add_argument("--nCores", default=-1, type=int, help="Number of available cores for parallel calculations. Memory usage is optimized, so maximal number of parallel processes can be launched.")
    parser.add_argument("--progress", action="store_true",
                        help="Show a progress bar (tqdm if available; otherwise a simple counter).")
    parser.add_argument("--pstep", type=int, default=50,
                        help="Batch updates every N lines in multiprocessing mode (default: 50)")
    
    args = parser.parse_args()
    with open(args.input) as f:
        colNumb = len(f.readline().strip().split())
    if colNumb == 1:
        with open(args.input + "_withIDs", "w") as out:
            for ind,line in enumerate(open(args.input)):
                sline = line.strip()
                if sline:
                    out.write(sline + " BB_" + str(ind+1) + "\n")
        inPut = args.input + "_withIDs"
        colNumb = 2
    else:
        inPut = args.input
    if args.nCores == -1:
        wc = countLines(inPut)
        main(inPut, 
            args.keepPG, 
            args.output, 
            args.Ro2Filtr,
            show_progress=args.progress, 
            progress_step=args.pstep, 
            total_lines=wc)
    else:
        wc = countLines(inPut)
        linesPerFile = wc // args.nCores
        outNamesList = splitFileByLines(inPut, inPut, linesPerFile)
        
        manager = Manager()
        q = manager.Queue(maxsize=10000)

        def _progress_consumer(q_, total_, enable_):
            with _get_pbar(total=total_, enable=enable_, desc="Processing", unit="line") as pbar:
                while True:
                    item = q_.get()
                    if item is None:  # sentinel
                        break
                    try:
                        pbar.update(int(item))
                    except Exception:
                        pass

        consumer_t = threading.Thread(target=_progress_consumer, args=(q, wc, args.progress), daemon=True)
        consumer_t.start()
        
        
        fixed_main = partial(main,  keepPG=args.keepPG,
                             Ro2Filtr=args.Ro2Filtr,
                             show_progress=args.progress,
                            progress_step=args.pstep,
                            total_lines=wc)
        nCores = args.nCores
        finalLog = []
        with ProcessPoolExecutor(max_workers=nCores) as executor:
            for out in executor.map(fixed_main, outNamesList):
                finalLog.append(out)
        with open(args.output + "_BBmode.smi", "w") as out, open(args.output + "_Synthmode.smi", "w") as outS, \
                open(args.output + "_NotProcessed", "w") as notProc, open(args.output + "_NotClassified", "w") as notClassified:
            for inp in outNamesList:
                for line in open(inp + "_BBmode.smi"):
                    if line.strip():
                        out.write(line)
                os.remove(inp + "_BBmode.smi")
                for line in open( inp + "_Synthmode.smi"):
                    if line.strip():
                        outS.write(line)
                os.remove(inp + "_Synthmode.smi")
                for line in open( inp + "_NotProcessed"):
                    if line.strip():
                        notProc.write(line)
                os.remove(inp + "_NotProcessed")
                for line in open( inp + "_NotClassified"):
                    if line.strip():
                        notClassified.write(line)
                os.remove(inp + "_NotClassified")
        for file in outNamesList:
            os.remove(file)
    with open(args.output + "_Synthmode.smi") as inpS, open("ttt", "w") as outS:
        synthons = {}
        for line in inpS:
            sline = line.strip()
            if sline and len(sline.split()) == colNumb+2:
                if sline.split()[0] not in synthons:
                    synthons[sline.split()[0]] = {"Class": set(sline.split()[1].split(",")),
                                                  "BBs": {sline.split()[2]}, "BB_Ids": {sline.split()[3]},
                                                  "Count": 1}
                else:
                    synthons[sline.split()[0]]["Class"].update(sline.split()[1].split(","))
                    synthons[sline.split()[0]]["BBs"].add(sline.split()[2])
                    synthons[sline.split()[0]]["BB_Ids"].add(sline.split()[3])
                    synthons[sline.split()[0]]["Count"] += 1
        for synth in sorted(synthons):
            hashcode = hashlib.sha1(synth.encode("utf-8")).hexdigest()[:10]
            uid = f"SYN_{hashcode}"
            outS.write(synth + " " + 
                       "+".join(synthons[synth]['BB_Ids']) + " " +
                        ",".join(synthons[synth]['Class']) + " " + 
                        ";".join(synthons[synth]['BBs']) + " " + 
                        str(synthons[synth]['Count']) + " " + 
                        uid + "\n") # replace pathing used as id with hased synthon (SYN_hash)

    os.remove(args.output + "_Synthmode.smi")
    os.rename("ttt", args.output + "_Synthmode.smi")
    return 0

if __name__ == '__main__':
    raise SystemExit(cli())

