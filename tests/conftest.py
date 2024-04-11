import pytest
# Contains pytest fixtures (Andrew)

@pytest.fixture
# holds the smiles to be tested
def test_smiles_list():
    # test molecules: CO2, H2O, 
    test_smiles_list = ['C(=O)=O', 'O']
    return test_smiles_list

@pytest.fixture
# holds the correct number of atoms in the test smiles above
def atom_count_list():
    # test molecules: CO2, H2O, 
    test_smiles_list = [3, 3]
    return test_smiles_list