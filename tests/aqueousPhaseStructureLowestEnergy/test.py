import sys
import os
from os import path

sys.path.append(path.abspath(path.join(os.getcwd(), '../../')))

from src.methods import aqueousPhaseStructureLowestEnergy
from collections import Counter
import numpy as np

CASES_DIR = 'cases'

def test_aqueousPhaseStructureLowestEnergy():
    CASES_DIR = 'cases'
    for case in os.listdir(CASES_DIR):
        if not case.startswith('.'):
            (input_file,
                expected_file,
                output_file) = map(lambda x: open(path.join(CASES_DIR,
                    case,
                    x[0] + '.txt'),
                    x[1]),
                    [('input', 'r'), ('expected', 'r'), ('output', 'w')])
            smiles = next(input_file).strip()
            elements_output, pos_output = aqueousPhaseStructureLowestEnergy(smiles)
            n_output = len(elements_output)
            output_file.write(str(len(elements_output)) + '\n')
            for element in elements_output:
                output_file.write(element + '\n')
            for pos in pos_output:
                output_file.write(' '.join(map(str, pos)) + '\n')
            n_expected = int(next(expected_file).strip())
            assert n_expected == n_output
            elements_expected = Counter()
            for i in range(n_expected):
                elements_expected[next(expected_file).strip()] += 1
            assert elements_expected == Counter(elements_output)
            pos_expected = []
            for i in range(n_expected):
                pos_expected.append([float(x) for x in next(expected_file).strip().split()])
            assert np.allclose(pos_expected, pos_output)

test_aqueousPhaseStructureLowestEnergy()