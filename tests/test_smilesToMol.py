from rdkit.Chem import Mol
from src.methods import smilesToMol

def test_smilesToMol(test_smiles):
    mol = smilesToMol(test_smiles)
    assert isinstance(mol, Mol)
    