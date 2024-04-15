import sys
import os
from os import path

sys.path.append(path.abspath(path.join(os.getcwd(), '../../')))

from src.methods import geomOpt
import numpy as np
from xtb.utils import Solvent

CASES_DIR = 'cases'

def test_geomOpt():
    CASES_DIR = 'cases'
    for case in os.listdir(CASES_DIR):
        if not case.startswith('.'):
            input_file = open(path.join(CASES_DIR, case, 'input.txt'), 'r')
            range_input = range(int(next(input_file).strip()))
            atomic_numbers_input = []
            for _ in range_input:
                atomic_numbers_input.append(int(next(input_file).strip()))
            atomic_numbers_input = np.array(atomic_numbers_input)
            atomic_symbols_input = []
            for _ in range_input:
                atomic_symbols_input.append(next(input_file).strip())
            atomic_positions_input = []
            for _ in range_input:
                atomic_positions_input.append([float(x) for x in next(input_file).strip().split()])
            atomic_positions_input = np.array(atomic_positions_input)
            max_optimize_iterations_input = int(next(input_file).strip())
            calc_solvent_input = next(input_file).strip() == 'True'
            if next(input_file).strip() == 'Solvent.h2o':
                solvent_input = Solvent.h2o
            else:
                solvent_input = None
            plot_input = next(input_file).strip() == 'True'
            verbose_input = next(input_file).strip() == 'True'
            geomOpt_output = geomOpt(atomic_numbers_input,
                atomic_symbols_input,
                atomic_positions_input,
                max_optimize_iterations=max_optimize_iterations_input,
                calc_solvent=calc_solvent_input,
                solvent=solvent_input,
                plot=plot_input,
                verbose=verbose_input)
            print(geomOpt_output)

test_geomOpt()