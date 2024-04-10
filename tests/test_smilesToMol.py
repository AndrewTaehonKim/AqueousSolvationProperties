from rdkit.Chem import Mol
from src.methods import smilesToMol

test_smiles = 'C(=O)=O'

def test_smilesToMol():
    mol = smilesToMol(test_smiles)
    assert isinstance(mol, Mol)
    