from rdkit.Chem import AllChem, MolFromSmiles, AddHs
from rdkit.Chem import Draw
from xtb.libxtb import VERBOSITY_MINIMAL, VERBOSITY_MUTED
from xtb.interface import Calculator, Param, Molecule
from xtb.utils import Solvent
from scipy import constants
from scipy.optimize import minimize
import numpy as np
import timeit
import pickle
import os
import random

import matplotlib.pyplot as plt

### Global Conversion Factors ###
ANGSTROM_PER_BOHR = constants.physical_constants['Bohr radius'][0] * 1.0e10 # Because RDK works in Bohrs ...
KCALPERMOL_PER_HARTREE = 627.509
### Methods ###
# Converts the SMILES string into a Mol object (Andrew & Lareine)
def smilesToMol(smiles, draw_img=False):
    # import mol object using rdkit.Chem
    mol = AddHs(MolFromSmiles(smiles))
    if mol is None:
        raise ValueError (f"{smiles} is not a valid SMILES input")
    # Draw for debugging purposes
    if draw_img:
        img = Draw.MolToImage(mol)
        img.save(f"images/{smiles}.jpg")
    AllChem.EmbedMolecule(mol)
    # basic optimize using MMFF94 force field to get starting structure in Bohr
    AllChem.MMFFOptimizeMolecule(mol)
    return mol

# Returns the atomic numbers and positions for input into calculator (Andrew)
def rdkitmolToXTBInputs(mol, save=False):
    # get positions and number of atoms in the molecule
    conformer = mol.GetConformer() 
    num_atoms = mol.GetNumAtoms()
    # get the atomic numbers and positions to use as input
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    atomic_symbols = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    atomic_positions = np.array([conformer.GetAtomPosition(i) for i in range(num_atoms)]) / ANGSTROM_PER_BOHR # Convert force field optimized xyz from angstrom to bohr
    return atomic_numbers, atomic_symbols, atomic_positions

def aqueousPhaseStructureLowestEnergy(smiles):
    atomic_numbers, atomic_symbols, atomic_positions = rdkitmolToXTBInputs(smilesToMol(smiles))
    return atomic_symbols, atomic_positions

# Runs the SCF Calculation and gets very expensive as the number of atoms increases... need to call this as little as possible (Andrew & Lareine)
def getEnergyAndGradient(atomic_numbers,
    atomic_positions,
    calc_solvent,
    solvent,
    max_SCF_iterations=125,
    verbose=False): # max SCF based on ORCA
    # initialize Calculator Object to perform calcuations with XTB
    calc = Calculator(Param.GFN2xTB, atomic_numbers, atomic_positions)
    # Set Calculator Properties
    calc.set_verbosity(VERBOSITY_MINIMAL if verbose else VERBOSITY_MUTED)
    # max iterations for each SCF energy calculation
    calc.set_max_iterations(max_SCF_iterations)
    # if solvent is required, run optimize with solvent with the starting position as the optimal version of the vacuum optimization
    calc.set_solvent(solvent)
    # perform SCF calculation
    result = calc.singlepoint()
    return result.get_energy() * KCALPERMOL_PER_HARTREE, result.get_gradient()

# Controls how the next XYZ coordinates are determined (Andrew & Lareine)
def updateXYZ(atomic_numbers,
    atomic_positions,
    gradient,
    shape,
    H,
    I,
    calc_solvent,
    solvent,
    method='BFGS',
    scalar=0.4):
    # BFGS Update  # LAREINE please check
    if method == 'BFGS':
        # this step size can cause problems ... LAREINE, please take a look and try to improve
        delta_x = -scalar * np.dot(H, gradient.reshape(-1, 1))
        atomic_positions = (atomic_positions.reshape(-1, 1)
            + delta_x).reshape(*shape)
        energy, new_gradient = getEnergyAndGradient(atomic_numbers,
            atomic_positions,
            calc_solvent,
            solvent)
        delta_grad = (new_gradient - gradient).reshape(-1, 1)
        dot = (delta_x * delta_grad).sum()
        J = I - np.outer(delta_x, delta_grad) / dot
        H = J @ H @ J.T + np.outer(delta_x, delta_x) / dot
    else:
        # basic gradient update as comparison and safer default
        energy, new_gradient = getEnergyAndGradient(atomic_numbers,
            atomic_positions,
            calc_solvent,
            solvent)
        atomic_positions -= 0.3 * new_gradient
    return atomic_positions, energy, new_gradient, H

# Convergence Criteria (Andrew)
def isConverged(history,
    criteria,
    TolE=5e-6,
    TolRMSG_square = 1e-8, # TolRMSG=1e-4
    TolMaxG_square=9e-8, # TolMaxG=3e-4,
    TolMaxD=4e-3,
    TolRMSD_square=4e-6): # TolRMSD=2e-3
    # Convergence Criteria based on ORCA 4.2.1. convergence criteria https://www.afs.enea.it/software/orca/orca_manual_4_2_1.pdf page 19
    previous, current = history['xyz'][-2:]
    allclose = np.allclose(previous, current, TolMaxD)
    mean_square = np.sqrt(np.mean(np.sum((previous - current) ** 2)))
    if criteria == 'NORMALOPT':
        return allclose and mean_square < TolRMSD_square
    sum_square = [sum(_**2) for _ in history['grad'][-1]]
    return all([allclose,
        mean_square, # < TolRMS...
        max(sum_square) < TolMaxG_square]) and (criteria == 'TIGHTOPT'
            or all([criteria == 'VERYTIGHTOPT',
                np.isclose(history['energy'][-1],
                    history['energy'][-2],
                    atol=TolE),
                np.mean(sum_square) < TolRMSG_square]))

# Optimization Method (Andrew)
def geomOpt(atomic_numbers,
    atomic_symbols,
    atomic_positions,
    criteria='TIGHTOPT',
    identifier='',
    max_optimize_iterations=1100,
    calc_solvent=False,
    solvent=Solvent.h2o,
    plot=False,
    verbose=True,
    scalar=0.2,
    learning=0.998):
    history_keys = ['xyz', 'energy', 'grad']
    history = {history_key: [] for history_key in history_keys}
    # start timer
    start_time = timeit.default_timer()
    # print("starting position:", atomic_positions)
    # Perform Optimization
    update_energy, update_gradient = getEnergyAndGradient(atomic_numbers,
        atomic_positions,
        calc_solvent,
        solvent)
    n, m = atomic_positions.shape
    H = np.eye(n * m)
    I = H[:]
    if not calc_solvent:
        solvent = None
    for i in range(1, max_optimize_iterations + 1):
        (atomic_positions,
            update_energy,
            update_gradient,
            H) = updateXYZ(atomic_numbers,
            atomic_positions,
            update_gradient,
            (n, m),
            H,
            I,
            calc_solvent=calc_solvent,
            solvent=solvent,
            scalar=scalar)
        for history_key, value in zip(history_keys,
            [atomic_positions, update_energy, update_gradient]):
            history[history_key].append(value)
        # Check Convergence Criteria
        if i > 1 and isConverged(history, criteria=criteria):
            if verbose:
                print(f"Finished calculation after {i} iterations with final energy (kcal/mol): {history['energy'][-1]: .1f}")
            break
        scalar *= learning
    else:
        if verbose:
            print(f"Maximum iterations ({i}) reached for vacuum calculation. Ending with final energy (kcal/mol): {history['energy'][-1]: .1f}")
    # end timer
    end_time = timeit.default_timer()
    if verbose:
        print(f"Total Time: {end_time - start_time: .3f} s")
    # Plotting for Debugging
    if plot:
        fig, ax = plt.subplots()
        ax.plot(np.arange(0,len(history['energy'])-1),
            history['energy'][1:],
            label="Vacuum Energy")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy (Hartree)")
        ax.legend()
        # plt.savefig(f"images/energy_convergence-{' '.join(atomic_symbols)}-{identifier}-{'solvent' if calc_solvent else 'vacuum'}.jpg")
    return history['xyz'][-1], history['energy'][-1]

# Reference Molecule Properties in Solvent (Andrew)
get_solvent_reference_properties = lambda smiles: geomOpt(*rdkitmolToXTBInputs(smilesToMol(smiles)), solvent=True, verbose=False, max_optimize_iterations=200)

# Run this one time to generate the files necessary to hold the reference data (Andrew)
def makeReferenceFiles():
    reference_smiles_list = np.array(['[H][H]','[C-]#[C+]', 'N#N', 'O=O', 'FF', 'P#P', 'S=S', 'ClCl'])
    reference_symbols_list = np.array(['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl'])
    energy_dict = {}
    for ref, symbol in zip(reference_smiles_list, reference_symbols_list):
        xyz, energy = get_solvent_reference_properties(ref)
        energy_dict[symbol] = energy
        np.savetxt(f'references/{ref}.txt', xyz, fmt='%f')
    with open(f'references/reference_energies.pkl', 'wb') as file:
        pickle.dump(energy_dict, file)

# Gets the formation energies by building SMILES molecule based on references (Andrew)
def getFormationEnergies(atomic_symbols, E_products):
    # get the total number of atoms
    count_dictionary = {}
    for symbol in set(atomic_symbols):
        count = np.sum(np.char.count(atomic_symbols, symbol))
        count_dictionary[symbol] = count
    # Specify the filename
    filename = 'references/reference_energies.pkl'
    # Check if the file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    with open(filename, 'rb') as f:
        reference_energies = pickle.load(f)
    # calculate energy used to break bonds
    E_reactants = 0
    # EXAMPLE: H2O is 2H + 1O -> E_H2 + 1/2 E_O2
    #          C2H6 is 2C + 6H -> E_C2 + 3E_H2
    for key, atom_count in count_dictionary.items():
        E_reactants += atom_count/2 * reference_energies[key]
    return E_products - E_reactants

# Perturbed atomic positions (Andrew)
def getPerturbedPositions(atomic_positions, perturb_magnitude):
    perturbation_matrix = np.zeros(atomic_positions.shape)
    # Select a random row
    random_row = np.random.randint(perturbation_matrix.shape[0])
    # Add a random float to move one atom
    perturbation_matrix[random_row, :] += np.random.uniform(-perturb_magnitude, perturb_magnitude, perturbation_matrix.shape[1])
    return perturbation_matrix+atomic_positions

# Final Method (Andrew)
def smiles_to_properties(smiles, criteria='TIGHTOPT', verbose=True, plot=True):
    print("Performing Calculation. Please Wait.")
    # timer
    start_time = timeit.default_timer()
    # Read SMILES to initialize
    atomic_numbers, atomic_symbols, atomic_positions = rdkitmolToXTBInputs(smilesToMol(smiles))
    # Add perturbation and run optimization multiple times
    perturb_count = 5
    perturb_magnitude = .5 # bohrs
    # store results
    vacuum_xyz_history = []
    vacuum_energy_history =[]
    solvent_xyz_history = []
    solvent_energy_history =[]
    solvation_energy_history = []
    for i in range(perturb_count):
        # Run Vacuum calculation
        vacuum_xyz, vacuum_energy = geomOpt(atomic_numbers, atomic_symbols, atomic_positions, criteria=criteria, identifier=str(i), calc_solvent=False, plot=plot, verbose=verbose)
        # Run Solvent calculation
        solvent_xyz, solvent_energy = geomOpt(atomic_numbers, atomic_symbols, atomic_positions, criteria=criteria, identifier=str(i), calc_solvent=True, plot=plot, verbose=verbose)
        # Get solvent Energy
        solvation_energy = (solvent_energy - vacuum_energy)
        # Perturb positions of vacuum
        atomic_positions = getPerturbedPositions(vacuum_xyz, perturb_magnitude)
        # Add to history
        vacuum_xyz_history.append(vacuum_xyz)
        vacuum_energy_history.append(vacuum_energy)
        solvent_xyz_history.append(solvent_xyz)
        solvent_energy_history.append(solvent_energy)
        solvation_energy_history.append(solvation_energy)
    print("solvent optimized xyz coordinates (angstrom)")
    print(random.sample(solvent_xyz_history, 1)[0]*ANGSTROM_PER_BOHR)
    print(f"Raw Vacuum Energy (kcal/mol): {np.mean(vacuum_energy_history): .1f} +- {np.std(vacuum_energy_history): .1f} | Raw Solvent Energy (kcal/mol): {np.mean(solvent_energy) : .1f} +- {np.std(solvent_energy_history): .1f}")
    # formation energies
    formation_energy_history = []
    for solvent_energy in solvent_energy_history:
        formation_energy_history.append(getFormationEnergies(atomic_symbols, solvent_energy))
    print(f"Energy of Formation (kcal/mol): {np.mean(formation_energy_history): .1f}  +- {np.std(formation_energy_history): .1f}")
    
    # solvation energy
    print(f"Solvation Energy (kcal/mol): {np.mean(solvation_energy_history): .3f} +- {np.std(solvation_energy_history): .3f}")
    
    # report time
    end_time = timeit.default_timer()
    print(f"Total Time Taken: {end_time -  start_time: .0f} seconds")
    
    return 0