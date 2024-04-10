from rdkit.Chem import Mol
from src.methods import smilesToMol

def test_smilesToMol(test_smiles_list):
    for test_smiles in test_smiles_list:
        mol = smilesToMol(test_smiles)
        assert isinstance(mol, Mol)
    