from src.methods import getEnergyAndGradient, rdkitmolToXTBInputs, smilesToMol
from xtb.utils import Solvent
import numpy as np

def test_getEnergyAndGradient(test_smiles_list):
    for smiles in test_smiles_list:
        atomic_numbers, atomic_symbols, atomic_positions = rdkitmolToXTBInputs(smilesToMol(smiles))
        # test vacuum
        energy, grad = getEnergyAndGradient(atomic_numbers, atomic_positions, calc_solvent=False, solvent=None)
        assert isinstance(energy, float) # energy should be a scalar float
        assert isinstance(grad, np.ndarray) # gradient should be a matrix
        assert grad.shape == (len(atomic_symbols), 3) # gradient size should match # atoms * 3 dimensions
        assert energy < 0 # energy should be negative
        assert not np.allclose(grad, np.zeros(grad.shape)) # there should exist a gradient
        # test aqueous
        solv_energy, solv_grad = getEnergyAndGradient(atomic_numbers, atomic_positions, calc_solvent=True, solvent=Solvent.h2o)
        assert isinstance(solv_energy, float) # energy should be a scalar float
        assert isinstance(solv_grad, np.ndarray) # gradient should be a matrix
        assert solv_grad.shape == (len(atomic_symbols), 3) # gradient size should match # atoms * 3 dimensions
        assert solv_energy < 0 # energy should be negative
        assert not np.allclose(solv_grad, np.zeros(solv_grad.shape)) # there should exist a gradient
        # the two should be sufficiently different in energy
        assert not np.isclose(energy, solv_energy, 1e-7)