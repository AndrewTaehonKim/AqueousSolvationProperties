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
KCALPERMOL_PER_HARTREE = 627.509
BOHR_RADIUS = scipy.constants.physical_constants['Bohr radius'][0] * 1e10

class Molecule:
    def __init__(self, smiles):
        self.mol = rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles(smiles))
        self.preOptimize()
        self.atomic_numbers = []
        self.elements = []
        for atom in self.mol.GetAtoms():
            self.atomic_numbers.append(atom.GetAtomicNum())
            self.elements.append(atom.GetSymbol())
        self.atomic_numbers = np.array(self.atomic_numbers)
        self.singlepoint = collections.defaultdict(
            lambda: collections.deque(maxlen=2))
        self.n = len(self.atomic_numbers)
        self.calculator = xtb.interface.Calculator(xtb.interface.Param.GFN2xTB,
            self.atomic_numbers,
            np.reshape(self.xyz, (self.n, 3)))

    def preOptimize(self):
        AllChem.EmbedMolecule(self.mol)
        AllChem.MMFFOptimizeMolecule(self.mol, maxIters=400)
        conformer = self.mol.GetConformer()
        self.xyz = np.array([conformer
            .GetAtomPosition(i) for i in range(self.mol.GetNumAtoms())])

    def appendSinglepoint(self, solvent=None):
        calculator = xtb.interface.Calculator(xtb.interface.Param.GFN2xTB,
            self.atomic_numbers,
            self.xyz)
        calculator.set_solvent(solvent)
        calculator.set_verbosity(xtb.libxtb.VERBOSITY_MUTED)
        self.singlepoint[solvent].append(calculator.singlepoint())

    def minimizeEnergy(self, solvent=None, randomize=False):
        if randomize:
            self.preOptimize()
        start = timeit.default_timer()
        solution = scipy.optimize.minimize(self.getEnergy,
            self.xyz.flatten(),
            method='L-BFGS-B')
        return 1 - solution.success

    def getEnergy(self, xyz, solvent=None):
        calculator = xtb.interface.Calculator(xtb.interface.Param.GFN2xTB,
            self.atomic_numbers,
            xyz)
        calculator.set_solvent(solvent)
        calculator.set_verbosity(xtb.libxtb.VERBOSITY_MUTED)
        return calculator.singlepoint().get_energy()

    def energyMinimized(self, solvent=None):
        return len(self.singlepoint[solvent]) > 1 and np.isclose(
            self.getEnergy(solvent, i=0),
            self.getEnergy(solvent, i=1))

    def formationEnergy(self, reference_states, iterations=10):
        formation_energy_list = []
        for i in range(iterations):
            # TODO: Pre-optimize
            self.minimizeEnergy(xtb.utils.Solvent.h2o, randomize=True)
            formation_energy = self.getEnergy(xtb.utils.Solvent.h2o)
            for element, num_atoms in collections.Counter(
                self.elements).items():
                for j in range(num_atoms // 2):
                    formation_energy -= np.random.normal(
                        reference_states[element]['energy'],
                        reference_states[element]['stddev'])
            formation_energy_list.append(formation_energy)
        # TODO: Convert to kcal/mol
        formation_energy_list = np.array(formation_energy_list)
        return np.mean(formation_energy_list), np.std(formation_energy_list)

    def hydrationEnergy(self, iterations=10):
        solvent = xtb.utils.Solvent.h2o
        hydration_energy_list = []
        for i in range(iterations):
            self.minimizeEnergy(solvent=solvent, randomize=True)
            solvent_energy = self.getEnergy(solvent=solvent)
            self.minimizeEnergy(randomize=True)
            vacuum_energy = self.getEnergy()
            hydration_energy_list.append(solvent_energy - vacuum_energy)
        hydration_energy_list = np.array(
            hydration_energy_list) * KCALPERMOL_PER_HARTREE
        return np.mean(hydration_energy_list), np.std(hydration_energy_list)

    generateFp = lambda self: np.array(rdkit
        .Chem
        .rdFingerprintGenerator
        .GetRDKitFPGenerator(fpSize=512)
        .GetFingerprint(self.mol), dtype=np.intc)

def dumpReferenceStates(iterations=100):
    # TODO: Verify accuracy of reference states
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

loadReferenceStates = lambda filepath: json.load(open(filepath, 'r'))

# dumpReferenceStates()
# reference_states = loadReferenceStates(REFERENCE_STATES_FILEPATH)
