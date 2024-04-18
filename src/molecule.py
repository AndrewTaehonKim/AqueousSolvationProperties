import collections
import numpy as np
import rdkit.Chem
from rdkit.Chem import AllChem
import scipy.constants
import xtb.interface
import xtb.libxtb
import xtb.utils

BOHR_RADIUS = scipy.constants.physical_constants['Bohr radius'][0] * 1e10

class Molecule:
    def __init__(self, smiles, scalar=1):
        mol = rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)
        conformer = mol.GetConformer()
        self.atomic_numbers = np.array([atom.GetAtomicNum()
            for atom in mol.GetAtoms()])
        self.elements = np.array([atom.GetSymbol()
            for atom in mol.GetAtoms()])
        self.xyz = np.array([conformer.GetAtomPosition(i)
            for i in range(mol.GetNumAtoms())])
        self.singlepoint = collections.deque(maxlen=2)
        self.appendSinglepoint()
        while not self.isOptimized():
            self.xyz -= scalar * self.singlepoint[-1].get_gradient()
            self.appendSinglepoint()
        self.xyz *= BOHR_RADIUS

    def appendSinglepoint(self):
        calculator = xtb.interface.Calculator(xtb.interface.Param.GFN2xTB,
            self.atomic_numbers,
            self.xyz)
        calculator.set_solvent(xtb.utils.Solvent.h2o)
        calculator.set_verbosity(xtb.libxtb.VERBOSITY_MUTED)
        self.singlepoint.append(calculator.singlepoint())

    def isOptimized(self):
        return len(self.singlepoint) > 1 \
            and np.isclose(self.singlepoint[0].get_energy(),
                self.singlepoint[1].get_energy())
