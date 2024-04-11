from src.methods import *

CO2 = 'C(=O)=O'
H2O = 'O'

mol = smilesToMol(CO2)
rdkitmolToXTBMol(mol)
H2O =aqueousPhaseStructureLowestEnergy(H2O)
print(H2O)