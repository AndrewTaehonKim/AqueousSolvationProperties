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
def getEnergyAndGradient(atomic_numbers, atomic_positions, calc_solvent, solvent, max_SCF_iterations=125, verbose=False): # max SCF based on ORCA
    # initialize Calculator Object to perform calcuations with XTB
    calc = Calculator(Param.GFN2xTB, atomic_numbers, atomic_positions)
    # Set Calculator Properties
    calc.set_verbosity(VERBOSITY_MINIMAL) if verbose else calc.set_verbosity(VERBOSITY_MUTED)
    calc.set_max_iterations(max_SCF_iterations) # max iterations for each SCF energy calculation
    # if solvent is required, run optimize with solvent with the starting position as the optimal version of the vacuum optimization
    if calc_solvent:
        calc.set_solvent(Solvent.h2o)
    # perform SCF calculation
    result = calc.singlepoint()
    return result.get_energy()*KCALPERMOL_PER_HARTREE, result.get_gradient()

# Controls how the next XYZ coordinates are determined (Andrew & Lareine)
def updateXYZ(atomic_numbers, atomic_positions, gradient, H, calc_solvent, solvent, method='BFGS'):
    epsilon = 1e-7
    # BFGS Update  # LAREINE please check
    if method == 'BFGS':
        H_length = len(atomic_positions.reshape(-1, 1))
        I = np.eye(H_length)
        search = -np.dot(H, gradient.reshape(-1, 1)) # moves based on this
        learning = .5 # this step size can cause problems ... LAREINE, please take a look and try to improve
        new_atomic_positions = atomic_positions.reshape(-1, 1) + learning*search
        new_atomic_positions = new_atomic_positions.reshape(len(atomic_positions), 3)
        energy, new_gradient = getEnergyAndGradient(atomic_numbers, new_atomic_positions, calc_solvent, solvent)
        delta_x = (new_atomic_positions - atomic_positions).reshape(-1, 1)
        delta_grad = (new_gradient - gradient).reshape(-1,1)
        rho = 1.0 / (delta_grad.T@delta_x + epsilon)[0,0] # added epsilon to prevent division by zero
        H = (I-rho*np.outer(delta_x, delta_grad))@H@(I-rho*np.outer(delta_grad, delta_x)) + rho*np.outer(delta_x, delta_x)
    # basic gradient update as comparison and safer default
    else:
        new_atomic_positions = atomic_positions
        energy, new_gradient = getEnergyAndGradient(atomic_numbers, new_atomic_positions, calc_solvent, solvent)
        new_atomic_positions -= 0.3 * new_gradient
    return new_atomic_positions, energy, new_gradient, H

RMS = lambda data: np.sqrt(np.mean(np.square(data)))

# Convergence Criteria (Andrew)
def isConverged(xyz_history, energy_history, gradient_history, criteria, TolE=5e-6, TolRMSG=1e-4, TolMaxG=3e-4, TolMaxD=4e-3,TolRMSD=2e-3): 
    # Convergence Criteria based on ORCA 4.2.1. convergence criteria https://www.afs.enea.it/software/orca/orca_manual_4_2_1.pdf page 19
    match criteria:
        case 'Normal': return True if np.allclose(xyz_history[-1], xyz_history[-2], TolMaxD) and RMS(np.linalg.norm(xyz_history[-1]-xyz_history[-2])) < TolRMSD else False
        case 'Tight': return True if np.allclose(xyz_history[-1], xyz_history[-2], TolMaxD) and RMS(np.linalg.norm(xyz_history[-1]-xyz_history[-2])) and max(np.linalg.norm(gradient_history[-1],axis=1)) < TolMaxG else False
        case 'VeryTight': return True if np.allclose(xyz_history[-1], xyz_history[-2], TolMaxD) and RMS(np.linalg.norm(xyz_history[-1]-xyz_history[-2])) and max(np.linalg.norm(gradient_history[-1],axis=1)) < TolMaxG and abs(energy_history[-1]-energy_history[-2]) < TolE and RMS(np.linalg.norm(gradient_history[-1], axis=1)) < TolRMSG else False
        
# Optimization Method (Andrew)
def geomOpt(atomic_numbers, atomic_symbols, atomic_positions, criteria='Tight', identifier='', max_optimzize_iterations=100, calc_solvent=False, solvent=Solvent.h2o, plot=False, verbose=True):
    xyz_history = []
    energy_history = []
    grad_history = []
    # start timer
    start_time = timeit.default_timer()
    # print("starting position:", atomic_positions)
    # Perform Optimization
    update_energy, update_gradient = getEnergyAndGradient(atomic_numbers, atomic_positions, calc_solvent, solvent)
    H = np.eye(len(atomic_positions.reshape(-1, 1)))
    for iter in range(max_optimzize_iterations):
        atomic_positions, update_energy, update_gradient, H, = updateXYZ(atomic_numbers, atomic_positions, update_gradient, H, calc_solvent=False, solvent=None) if not calc_solvent else updateXYZ(atomic_numbers, atomic_positions, update_gradient, H, calc_solvent=True, solvent=solvent)
        xyz_history.append(atomic_positions)
        energy_history.append(update_energy)
        grad_history.append(update_gradient)
        # Check Convergence Criteria
        if iter > 1 and isConverged(xyz_history, energy_history, grad_history, criteria=criteria):
            print(f"Finished calculation after {iter+1} iterations with final energy (kcal/mol): {energy_history[-1]: .1f}") if verbose is True else ''
            break
        if iter == max_optimzize_iterations-1:
            print(f"Maximum iterations ({iter+1}) reached for vacuum calculation. Ending with final energy (kcal/mol): {energy_history[-1]: .1f}") if verbose is True else ''
    # end timer
    end_time = timeit.default_timer()
    print(f"Total Time: {end_time-start_time: .0f}s") if verbose is True else ''
    
    # Plotting for Debugging
    if plot:
        fig, ax = plt.subplots()
        ax.plot(np.arange(0,len(energy_history)-1), energy_history[1:], label="Vacuum Energy")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy (Hartree)")
        ax.legend()
        plt.savefig(f"images/energy_convergence-{' '.join(atomic_symbols)}-{identifier}-{'solvent' if calc_solvent else 'vacuum'}.jpg")
        
    return xyz_history[-1], energy_history[-1]


# Reference Molecule Properties in Solvent (Andrew)
get_solvent_reference_properties = lambda smiles: geomOpt(*rdkitmolToXTBInputs(smilesToMol(smiles)), solvent=True, verbose=False, max_optimzize_iterations=200)

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
def smiles_to_properties(smiles, criteria='Tight', verbose=True, plot=True):
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