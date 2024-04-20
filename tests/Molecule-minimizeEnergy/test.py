import sys
import os
from os import path

sys.path.append(path.abspath(path.join(os.getcwd(), '../../')))

from src.molecule import *
import collections
import numpy as np
import scipy.optimize

CASES_DIR = 'cases'

def test_Molecule_minimizeEnergy():
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
            molecule.minimizeEnergy(xtb.utils.Solvent.h2o)
            molecule.xyz *= BOHR_RADIUS
            n_output = len(molecule.elements)
            output_file.write(str(n_output) + '\n')
            for element in molecule.elements:
                output_file.write(element + '\n')
            for position in molecule.xyz:
                output_file.write(' '.join(map(str, position)) + '\n')
            n_expected = int(next(expected_file).strip())
            elements = []
            for i in range(n_expected):
                elements.append(next(expected_file).strip())
            xyz = []
            for i in range(n_expected):
                row = [float(x) for x in next(expected_file).strip().split()]
                xyz.append(row)
            xyz = np.array(xyz)
            molecule.xyz = transform(xyz, molecule.xyz)
            assert n_output == n_expected
            assert collections.Counter(molecule
                .elements) == collections.Counter(elements)
            assert np.allclose(xyz, transform(xyz, molecule.xyz))

def transform(a, b):
    a -= np.mean(a, axis=0)
    b -= np.mean(b, axis=0)
    return rotate(scipy
        .optimize
        .minimize(lambda theta: np.linalg.norm(a - rotate(theta, b)),
            [0] * 3)
        .x,
    b)

def rotate(theta, positions):
    r = np.array([
        [[1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])]],
        [[np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])]],
        [[np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1]]
    ])
    return [position @ r[0] @ r[1] @ r[2] for position in positions]

test_Molecule_minimizeEnergy()
