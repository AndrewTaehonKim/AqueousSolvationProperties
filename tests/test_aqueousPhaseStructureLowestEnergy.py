from src.methods import aqueousPhaseStructureLowestEnergy

def test_aqueousPhaseStructureLowestEnergy(test_smiles):
    atom_count = 3
    dimensions = 3
    elements, pos = aqueousPhaseStructureLowestEnergy(test_smiles)
    assert len(elements) == atom_count
    assert pos.shape[0] == atom_count
    assert pos.shape[1] == dimensions