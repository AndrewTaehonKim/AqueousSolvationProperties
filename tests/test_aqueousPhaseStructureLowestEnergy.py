from src.methods import aqueousPhaseStructureLowestEnergy

def test_aqueousPhaseStructureLowestEnergy(test_smiles_list, atom_count_list):
    for test_smiles, atom_count in zip(test_smiles_list, atom_count_list):
        dimensions = 3
        elements, pos = aqueousPhaseStructureLowestEnergy(test_smiles)
        assert len(elements) == atom_count
        assert pos.shape[0] == atom_count
        assert pos.shape[1] == dimensions