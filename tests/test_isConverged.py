from src.methods import isConverged
import random
import numpy as np

def test_isConverged(converged_check_list, not_converged_check_list):
    # converged check
    assert isConverged(*converged_check_list, criteria='Normal')
    assert isConverged(*converged_check_list, criteria='Tight')
    # assert isConverged(*converged_check_list, criteria='VeryTight')
    # not converged check
    assert not isConverged(*not_converged_check_list, criteria='Normal')
    assert not isConverged(*not_converged_check_list, criteria='Tight')
    assert not isConverged(*not_converged_check_list, criteria='VeryTight')