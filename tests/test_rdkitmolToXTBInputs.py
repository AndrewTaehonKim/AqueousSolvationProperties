from src.methods import smilesToMol, rdkitmolToXTBInputs
import numpy as np
# (Andrew)
def test_rdkitmolToXTBInputs(test_smiles_list):
    for test_smiles in test_smiles_list:
        mol = smilesToMol(test_smiles)
        atomic_numbers, atomic_symbols, atomic_positions = rdkitmolToXTBInputs(mol)
        assert isinstance(atomic_numbers, np.ndarray) # check types
        assert isinstance(atomic_symbols, np.ndarray) # check types
        assert isinstance(atomic_positions, np.ndarray) # check types
        assert len(atomic_numbers) != 0 # check not empty
        assert len(atomic_numbers) == len(atomic_symbols) # check symbols and numbers 
        assert atomic_positions.shape == (len(atomic_numbers), 3) # check shape of the xyz is correct
        
