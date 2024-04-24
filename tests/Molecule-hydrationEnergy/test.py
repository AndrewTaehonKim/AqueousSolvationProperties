import os
from os import path
import sys

sys.path.append(path.abspath(path.join(os.getcwd(), '../../')))

from src.molecule import *

CASES_DIR = 'cases'

def test_Molecule_hydrationEnergy():
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
            output = (hydration_energy_output,
                hydration_energy_stddev_output) = Molecule(
                smiles).hydrationEnergy()
            for data in output:
                output_file.write(str(data) + '\n')
            hydration_energy_expected = float(next(expected_file).strip())
            hydration_energy_stddev_expected = float(next(expected_file)
                .strip())
            assert np.isclose(hydration_energy_output,
                hydration_energy_expected,
                atol=0.01)
            assert np.isclose(hydration_energy_stddev_output,
                hydration_energy_stddev_expected,
                atol=0.01)

test_Molecule_hydrationEnergy()
