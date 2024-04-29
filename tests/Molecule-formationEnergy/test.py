import os
from os import path
import sys

sys.path.append(path.abspath(path.join(os.getcwd(), '../../')))

from src.molecule import *

REFERENCE_STATES_FILEPATH = '../../reference_states.json'
CASES_DIR = 'cases'

def test_Molecule_formationEnergy():
    for case in os.listdir(CASES_DIR):
        if not case.startswith('.'):
            (input_file,
                output_file,
                expected_file) = [open(path.join(CASES_DIR,
                case,
                filename + '.txt'),
                mode) for filename, mode in zip(['input',
                    'output',
                    'expected'],
                ['r', 'w', 'r'])]
            smiles = next(input_file).strip()
            molecule = Molecule(smiles)
            output = (formation_energy_output,
                formation_energy_stddev_output) = Molecule(
                smiles).formationEnergy(
                reference_states)
            for data in output:
                output_file.write(str(data) + '\n')
            formation_energy_expected = float(next(expected_file).strip())
            formation_energy_stddev_expected = float(next(expected_file)
                .strip())
            assert np.isclose(formation_energy_output,
                formation_energy_expected,
                rtol=0.01) # TODO: Adjust rtol
            assert np.isclose(formation_energy_stddev_output,
                formation_energy_stddev_expected)

reference_states = loadReferenceStates(REFERENCE_STATES_FILEPATH)
test_Molecule_formationEnergy()
