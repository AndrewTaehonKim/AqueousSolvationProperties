from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def aqueousPhaseStructureLowestEnergy(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    conformer = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    xyz_numpy_array_in_angstroms = np.zeros((num_atoms, 3))
    for i in range(num_atoms):
        atom_position = conformer.GetAtomPosition(i)
        xyz_numpy_array_in_angstroms[i] = [atom_position.x, atom_position.y, atom_position.z]
    return [atom.GetSymbol() for atom in mol.GetAtoms()], xyz_numpy_array_in_angstroms
