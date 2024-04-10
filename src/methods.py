from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import numpy as np

def aqueousPhaseStructureLowestEnergy(smiles):
    # import mol object using rdkit.Chem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol)
    # basic optimize using MMFF94 force field
    AllChem.MMFFOptimizeMolecule(mol)
    # get positions and number of atoms in the molecule
    conformer = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    # initialize matrix to store xyz coordinates
    xyz_numpy_array_in_angstroms = np.zeros((num_atoms, 3))
    # set the atom positions in above array
    for i in range(num_atoms):
        atom_position = conformer.GetAtomPosition(i)
        xyz_numpy_array_in_angstroms[i] = [atom_position.x, atom_position.y, atom_position.z]
    # return a list of atom symbols and their corresponding positions
    return [atom.GetSymbol() for atom in mol.GetAtoms()], xyz_numpy_array_in_angstroms
