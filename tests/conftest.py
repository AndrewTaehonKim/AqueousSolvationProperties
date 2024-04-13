import pytest
import random
import numpy as np

# Contains pytest fixtures (Andrew)

@pytest.fixture
# holds the smiles to be tested
def test_smiles_list():
    # test molecules: CO2, H2O, C2H6 - well known
    test_smiles_list = ['C(=O)=O', 'O', 'CC']
    # add random generated smiles for safety - in case user puts in weird smiles
    elements = ['C','C','C','C','C','O','O','O','O','N','N','N','N','P', 'P']
    for i in range(3): # 3 random test smiles
        len = random.randint(2,8) # don't want these random smiles to be too long
        test_smiles_list.append(''.join(random.sample(elements, len)))
        if random.randint(0,1) == 1:
            test_smiles_list.append(''.join(random.sample(['Cl', 'F'], 1)))
    return test_smiles_list

@pytest.fixture
# holds a converged history
def converged_check_list():
    xyz_history = np.array([
        np.array([0, 0, 0]),
        np.array([1e-8,1e-8,1e-8])
    ])
    gradient_history = np.array([
        np.zeros(3),
        np.array([1e-8,1e-8,1e-8])
    ])
    energy_history = np.array([
        0, -1e-9
    ])
    return (xyz_history, energy_history, gradient_history)

@pytest.fixture
# holds a non-converged history
def not_converged_check_list():
    xyz_history = np.array([
        np.zeros(3),
        np.ones(3)
    ])
    gradient_history = np.array([
        np.zeros(3),
        np.ones(3)
    ])
    energy_history = np.array([
        0, -1
    ])
    return (xyz_history, energy_history, gradient_history)
