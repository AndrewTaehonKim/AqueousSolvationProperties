from rdkit.Chem import Mol
from xtb.utils import Solvent
from src.methods import updateXYZ, smilesToMol, rdkitmolToXTBInputs, getEnergyAndGradient
import random
import numpy as np

def test_updateXYZ(test_smiles_list):
    test_cases = (random.sample(test_smiles_list[:3], 1)) # test one normal case
    test_cases.append(random.sample(test_smiles_list[3:], 1)[0]) # test one random case
    for smiles in test_cases:
        mol = smilesToMol(smiles)
        atomic_numbers, atomic_symbols, atomic_positions = rdkitmolToXTBInputs(mol)
        energy, grad = getEnergyAndGradient(atomic_numbers, atomic_positions, calc_solvent=False, solvent=None)
        H = np.eye(len(atomic_symbols.reshape(-1, 1)))
        # test vacuum
        new_atomic_positions, new_energy, new_gradient, new_H = updateXYZ(atomic_numbers, atomic_positions, grad, H, calc_solvent=False, solvent=None)
        assert isinstance(new_atomic_positions, np.ndarray)
        assert isinstance(new_energy, float)
        assert isinstance(new_gradient, np.ndarray)
        assert isinstance(new_H, np.ndarray)
        assert new_atomic_positions.shape == (len(atomic_positions), 3)
        assert new_energy < energy
        assert new_gradient.shape == (len(atomic_positions), 3)
        assert new_H.shape == (len(atomic_positions)*3, len(atomic_positions)*3)
        assert not np.allclose(new_atomic_positions, atomic_positions)
        assert not np.allclose(new_gradient, grad)
        assert not np.allclose(new_H, H)
        # test solvent
        energy, grad = getEnergyAndGradient(atomic_numbers, atomic_positions, calc_solvent=True, solvent=Solvent.h2o)
        H = np.eye(len(atomic_symbols.reshape(-1, 1)))
        # test vacuum
        solv_atomic_positions, solv_energy, solv_gradient, solv_H = updateXYZ(atomic_numbers, atomic_positions, grad, H, calc_solvent=False, solvent=None)
        assert isinstance(solv_atomic_positions, np.ndarray)
        assert isinstance(solv_energy, float)
        assert isinstance(solv_gradient, np.ndarray)
        assert isinstance(solv_H, np.ndarray)
        assert solv_atomic_positions.shape == (len(atomic_positions), 3)
        assert solv_energy < energy
        assert solv_gradient.shape == (len(atomic_positions), 3)
        assert solv_H.shape == (len(atomic_positions)*3, len(atomic_positions)*3)
        assert not np.allclose(solv_atomic_positions, atomic_positions)
        assert not np.allclose(solv_gradient, grad)
        assert not np.allclose(solv_H, H)
        # test difference
        assert not np.allclose(solv_atomic_positions, new_atomic_positions)