from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import numpy as np

# Converts the SMILES string into a Mol object (Andrew & Lareine)
def smilesToMol(smiles):
    # import mol object using rdkit.Chem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError (f"{smiles} is not a valid SMILES input")
    AllChem.EmbedMolecule(mol)
    # basic optimize using MMFF94 force field to get starting structure
    AllChem.MMFFOptimizeMolecule(mol)
    return mol

# converts Mol object to temporary xyz file for optimization using xtb (Andrew & Lareine)
def molToXYZ(mol, save=False):
    # get positions and number of atoms in the molecule
    conformer = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    # Create a temporary xyz file
    with open('xyz/temp.xyz', 'w') as file:
        # write the number of atoms
        file.write(str(num_atoms) + '\n')
        file.write('\n')
        for symbol, pos in zip([atom.GetSymbol() for atom in mol.GetAtoms()], [conformer.GetAtomPosition(i) for i in range(num_atoms)]):
            file.write(f"{symbol} \t {str(pos.x)} \t {str(pos.y)} \t {str(pos.z)}\n")
        file.close()
    # initialize matrix to store xyz coordinates
    xyz_numpy_array_in_angstroms = np.zeros((num_atoms, 3))
    # set the atom positions in above array
    for i in range(num_atoms):
        atom_position = conformer.GetAtomPosition(i)
        xyz_numpy_array_in_angstroms[i] = [atom_position.x, atom_position.y, atom_position.z]
    # return a list of atom symbols and their corresponding positions
    return [atom.GetSymbol() for atom in mol.GetAtoms()], xyz_numpy_array_in_angstroms

def aqueousPhaseStructureLowestEnergy(smiles):
    
    return 0
