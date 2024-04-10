from src.methods import *

CO2 = 'C(=O)=O'
H2O = 'O'

CO2 = aqueousPhaseStructureLowestEnergy(CO2)
H2O =aqueousPhaseStructureLowestEnergy(H2O)
print(H2O)