from src.methods import *
from src.molecule import Molecule, dumpReferenceStates
CO2 = 'C(=O)=O'
H2O = 'O'
BENZENE = 'C1=CC=CC=C1'
ETHANE = 'CC'

dumpReferenceStates()

water = Molecule(H2O)
# H2O =smiles_to_properties(H2O)
# ETHANE =smiles_to_properties(ETHANE)
# CO2 =smiles_to_properties(CO2)
# BENZENE = smiles_to_properties(BENZENE)

# make_reference_files()
# smiles_to_properties('COONP')
