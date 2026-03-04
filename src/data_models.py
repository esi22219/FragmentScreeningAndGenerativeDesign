from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np


@dataclass(frozen=True)
class SynthonRecord:
    synthon_smiles: str                     # labeled synthon SMILES
    marks: Tuple[str, ...]                  # e.g. ('C:10','N:20') extracted from synthon_smiles
    n_marks: int                            # len(marks)
    annotations: Tuple[str, ...] = tuple()  #"+".join(finalSynthon[s])
    source_bb_smiles: str = ""
    source_bb_id: str = ""
    classes: Tuple[str, ...] = tuple()      # BB classes

@dataclass
class BBProcessingResult:
    bb_smiles: str
    bb_id: str
    classes: Tuple[str, ...]                # all retained classes
    synthons: Tuple[SynthonRecord, ...]     # 0..N synthons
    classified: bool                        # True if BBClassifier returned any classes
    had_synthons: bool                      # True if any synthons generated (after filters)
    azoles_flag: bool = False               # kept for parity with legacy "nHAzoles_nHAzoles"


@dataclass
class ExitSite:
    atom_idx: int
    families: Tuple[str, ...]
    vector: Tuple[float, float, float]
    score: float
    steric_ok: bool
    notes: str = ""


@dataclass
class FragmentRecord:
    frag_id: str
    frag_smiles: str
    exits: List[ExitSite] = field(default_factory=list)
    # optional: cached capped variants or MEL seeds
    mel_entries: List[str] = field(default_factory=list)  # store SMILES or RDKit PickleToString



@dataclass
class ExitRole:
    name: str    # 'AROMATIC_SP2_C', 'ALIPH_SP3_C', 'HETERO_NOS', 'HETEROAROMATIC_MINISCI'


@dataclass
class ExitCandidate:
    atom_idx: int
    atom_symbol: str
    role: ExitRole
    pos: np.ndarray
    vec_to_pocket: np.ndarray     # unit vector toward nearest pocket vertex
    dist_to_pocket: float
    sasa: float
    geom_score: float
    steric_ok: Optional[bool] = None
    vina_score: Optional[float] = None  # optional docking-based pocket check


@dataclass
class RSite:
    label: str             # R1, R2, ...
    atom_idx: int
    role: ExitRole
    vector: np.ndarray     # unit vector
    score: float



# add a sort of sink for fragments as well and then can link to synthon sink with reaction types available or something like that
# data manager for each step and passign data through
class MissSink:
    def __init__(self, dedupe=True):
        self.miss_synthon = defaultdict(list)
        self.miss_frag = {}  # frag_id -> FragmentRecord
        self._seen = set() if dedupe else None

    def __call__(self, bb_result):
        # bb_result: BBProcessingResult
        for syn in bb_result.synthons:
            s = syn.synthon_smiles
            if self._seen is not None:
                if s in self._seen:
                    continue
                self._seen.add(s)
            for mark in syn.marks:
                self.miss[mark].append(s)
    
    def add_fragment_record(self, rec: FragmentRecord):
        self.miss_frag[rec.frag_id] = rec
