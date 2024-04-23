import collections
import json
import numpy as np
import rdkit.Chem
from rdkit.Chem import AllChem
import scipy.constants
import xtb.interface
import xtb.libxtb
import xtb.utils

REFERENCE_STATES_FILEPATH = '../reference_states.json'
BOHR_RADIUS = scipy.constants.physical_constants['Bohr radius'][0] * 1e10

class Molecule:
    def __init__(self, smiles):
        self.mol = rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(self.mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(self.mol)
        conformer = self.mol.GetConformer()
        self.atomic_numbers = []
        self.elements = []
        for atom in self.mol.GetAtoms():
            self.atomic_numbers.append(atom.GetAtomicNum())
            self.elements.append(atom.GetSymbol())
        self.atomic_numbers = np.array(self.atomic_numbers)
        self.xyz = np.array([conformer.GetAtomPosition(i)
            for i in range(self.mol.GetNumAtoms())])
        self.singlepoint = collections.defaultdict(
            lambda: collections.deque(maxlen=2))

    def minimizeEnergy(self, solvent, scalar=1, max_iterations=1000):
        self.appendSinglepoint(solvent)
        while not self.energyMinimized(solvent) and max_iterations > 0:
            self.xyz -= scalar * self.singlepoint[solvent][-1].get_gradient()
            self.appendSinglepoint(solvent)
            max_iterations -= 1
        return int(self.energyMinimized(solvent) ^ True)

    def appendSinglepoint(self, solvent):
        calculator = xtb.interface.Calculator(xtb.interface.Param.GFN2xTB,
            self.atomic_numbers,
            self.xyz)
        calculator.set_solvent(solvent)
        calculator.set_verbosity(xtb.libxtb.VERBOSITY_MUTED)
        self.singlepoint[solvent].append(calculator.singlepoint())

    def energyMinimized(self, solvent):
        return len(self.singlepoint[solvent]) > 1 and np.isclose(
            self.getEnergy(solvent, i=0),
            self.getEnergy(solvent, i=1))

    def getEnergy(self, solvent, i=-1):
        return self.singlepoint[solvent][i].get_energy()

    def formationEnergy(self, reference_states, iterations=10):
        formation_energy_list = []
        for i in range(iterations):
            self.minimizeEnergy(xtb.utils.Solvent.h2o)
            formation_energy = self.getEnergy(xtb.utils.Solvent.h2o)
            for element, num_atoms in collections.Counter(
                self.elements).items():
                for j in range(num_atoms // 2):
                    formation_energy -= np.random.normal(
                        reference_states[element]['energy'],
                        reference_states[element]['stddev'])
            formation_energy_list.append(formation_energy)
        formation_energy_list = np.array(formation_energy_list)
        return np.mean(formation_energy_list), np.std(formation_energy_list)

    generateFp = lambda self: np.array(rdkit
        .Chem
        .rdFingerprintGenerator
        .GetRDKitFPGenerator(fpSize=512)
        .GetFingerprint(self.mol), dtype=np.intc)

def dumpReferenceStates(iterations=100):
    solvent = xtb.utils.Solvent.h2o
    reference_states = collections.defaultdict(lambda: dict())
    for element, smiles in zip(['H',
        'C',
        'N',
        'O',
        'F',
        'P',
        'S',
        'Cl'], ['[H][H]',
        '[C-]#[C+]', # UFFTYPER: Unrecognized atom type: C_ (1)
        'N#N',
        'O=O',
        'FF',
        'P#P', # UFFTYPER: Warning: hybridization set to SP3 for atom 0
        # UFFTYPER: Warning: hybridization set to SP3 for atom 1
        'S=S',
        'ClCl']):
        reference_states[element]['smiles'] = smiles
        energy = []
        for _ in range(iterations):
            molecule = Molecule(smiles)
            molecule.minimizeEnergy(solvent)
            energy.append(molecule.getEnergy(solvent))
        # TODO: Verify that energy is normally distributed
        reference_states[element]['energy'] = np.mean(energy)
        reference_states[element]['stddev'] = np.std(energy)
    json.dump(reference_states, open(REFERENCE_STATES_FILEPATH, 'w'))


loadReferenceStates = lambda REFERENCE_STATES_FILEPATH: json.load(
    open(REFERENCE_STATES_FILEPATH, 'r'))

# dumpReferenceStates()
# reference_states = loadReferenceStates(REFERENCE_STATES_FILEPATH)
