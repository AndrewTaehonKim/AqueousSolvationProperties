import os
from os import path
import sys

sys.path.append(path.abspath(path.join(os.getcwd(), '../../')))

from src.molecule import *

CASES_DIR = 'cases'

def test_Molecule_generateFp():
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
            fp_output = molecule.generateFp()
            for bit in fp_output:
                output_file.write(str(bit) + '\n')
            output_file.write('\n')
            fp_expected = []
            for bit in expected_file:
                bit = bit.strip()
                if bit:
                    fp_expected.append(int(bit))
            assert np.array_equal(fp_output, fp_expected)

test_Molecule_generateFp()
