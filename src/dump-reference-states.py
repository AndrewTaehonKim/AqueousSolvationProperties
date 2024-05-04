import concurrent.futures
import json
import numpy as np
import rdkit.Chem
from rdkit.Chem import AllChem
import scipy.constants
import scipy.optimize
import xtb.interface
import xtb.libxtb
import xtb.utils

ITERS = 1000
ANGSTROM_PER_BOHR = scipy.constants.physical_constants['Bohr radius'][0] / scipy.constants.angstrom
REFERENCE_STATES_FILEPATH = '../reference-states.json'

class Molecule:
    def __init__(self, smiles):
        self.mol = rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles(smiles))
        self.n = self.mol.GetNumAtoms()
        self.atomic_numbers = np.array([atom.GetAtomicNum() for atom in self.mol.GetAtoms()])
        self.formation_energy = []

    def minimizeEnergy(self):
        AllChem.EmbedMolecule(self.mol)
        AllChem.MMFFOptimizeMolecule(self.mol, maxIters=400)
        conformer = self.mol.GetConformer()
        return scipy.optimize.minimize(self.getEnergyGradient,
            np.array([conformer.GetAtomPosition(i) for i in range(self.n)])
                .flatten() / ANGSTROM_PER_BOHR,
            method='L-BFGS-B',
            jac=True,
            options = {'gtol': 3e-3, 'maxiter': 3 * self.n}).fun

    def getEnergyGradient(self, xyz):
        calculator = xtb.interface.Calculator(xtb.interface.Param.GFN2xTB,
            self.atomic_numbers,
            xyz.reshape(self.n, 3))
        calculator.set_solvent(xtb.utils.Solvent.h2o)
        calculator.set_verbosity(xtb.libxtb.VERBOSITY_MUTED)
        singlepoint = calculator.singlepoint()
        return singlepoint.get_energy(), singlepoint.get_gradient().flatten()

thread_pool_executor = concurrent.futures.ThreadPoolExecutor()
json.dump({symbol: [thread_pool_executor
    .submit(lambda smiles: Molecule(smiles).minimizeEnergy(), smiles)
    .result() for _ in range(ITERS)] for symbol, smiles in zip(
        ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl'],
        ['[H][H]', '[C-]#[C+]', 'N#N', 'O=O', 'FF', 'P#P', 'S=S', 'ClCl'])},
    open(REFERENCE_STATES_FILEPATH, 'w'))
