from rdkit import Chem
from rdkit.Chem import AllChem
from xtb.libxtb import VERBOSITY_MINIMAL
from xtb.interface import Calculator, Param, Molecule
from xtb.utils import Solvent
from scipy import constants
import numpy as np

### Global Conversion Factors ###
ANGSTROM_PER_BOHR = constants.physical_constants['Bohr radius'][0] * 1.0e10 # Because RDK works in Bohrs ...
print(ANGSTROM_PER_BOHR)
### Methods ###
# Converts the SMILES string into a Mol object (Andrew & Lareine)
def smilesToMol(smiles):
    # import mol object using rdkit.Chem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError (f"{smiles} is not a valid SMILES input")
    AllChem.EmbedMolecule(mol)
    # basic optimize using MMFF94 force field to get starting structure in Bohr
    AllChem.MMFFOptimizeMolecule(mol) 
    return mol

# converts Mol object to temporary xyz file for optimization using xtb (Andrew)
def rdkitmolToXTBMol(mol, save=False):
    # get positions and number of atoms in the molecule
    conformer = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    # get the atomic numbers and positions to use as input
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    atomic_positions = np.array([conformer.GetAtomPosition(i) for i in range(num_atoms)])
    return Molecule(atomic_numbers, atomic_positions)  # initialize molecule object... might be slower than just taking in the information directly

# Returns the atomic numbers and positions for input into calculator
def rdkitmolToXTBInputs(mol, save=False):
    # get positions and number of atoms in the molecule
    conformer = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    # get the atomic numbers and positions to use as input
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    atomic_positions = np.array([conformer.GetAtomPosition(i) for i in range(num_atoms)])
    return atomic_numbers, atomic_positions

# Optimization Method
def optimize(atomic_numbers, atomic_positions, gradient, max_iterations=100):
    pass

# General purpose optimizer that performs optimization
def geom_opt(smiles, max_iterations=50, has_solvent=False, solvent=Solvent.h2o):
    molecule_properties = rdkitmolToXTBInputs(smilesToMol(smiles))
    # initialize Calculator Object to perform calcuations with XTB
    calc = Calculator(Param.GFN2xTB, *molecule_properties)
    # Set Calculator Properties
    calc.set_verbosity(VERBOSITY_MINIMAL)
    calc.set_max_iterations(max_iterations)
    if has_solvent:
        calc.set_solvent(Solvent.h2o)
    result = calc.singlepoint()
    grad = result.get_gradient()
    print(result.get_energy())
    # for i in range(num_atoms):
    #     atom_position = conformer.GetAtomPosition(i)
    #     xyz_numpy_array_in_angstroms[i] = [atom_position.x, atom_position.y, atom_position.z]
    # # return a list of atom symbols and their corresponding positions
    # return [atom.GetSymbol() for atom in mol.GetAtoms()], xyz_numpy_array_in_angstroms
    return 0
