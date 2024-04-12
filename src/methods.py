from rdkit.Chem import AllChem, MolFromSmiles, AddHs
from rdkit.Chem import Draw
from xtb.libxtb import VERBOSITY_MINIMAL, VERBOSITY_MUTED
from xtb.interface import Calculator, Param, Molecule
from xtb.utils import Solvent, get_method
from scipy import constants
from scipy.optimize import minimize
import numpy as np
import timeit
import pickle
import os

import matplotlib.pyplot as plt

### Global Conversion Factors ###
ANGSTROM_PER_BOHR = constants.physical_constants['Bohr radius'][0] * 1.0e10 # Because RDK works in Bohrs ...
KCALPERMOL_PER_HARTREE = 627.509
### Methods ###
# Converts the SMILES string into a Mol object (Andrew & Lareine)
def smilesToMol(smiles):
    # import mol object using rdkit.Chem
    mol = AddHs(MolFromSmiles(smiles))
    if mol is None:
        raise ValueError (f"{smiles} is not a valid SMILES input")
    # Draw for debugging purposes
    img = Draw.MolToImage(mol)
    img.save(f"images/{smiles}.jpg")
    AllChem.EmbedMolecule(mol)
    # basic optimize using MMFF94 force field to get starting structure in Bohr
    AllChem.MMFFOptimizeMolecule(mol)
    return mol

# converts Mol object to temporary xyz file for optimization using xtb (Andrew)
def rdkitmolToXTBMol(mol, save=False):
    # get positions and number of atoms in the molecule
    conformer = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    # get the atomic numbers and positions to use as input
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    atomic_positions = np.array([conformer.GetAtomPosition(i) for i in range(num_atoms)])
    return Molecule(atomic_numbers, atomic_positions)  # initialize molecule object... might be slower than just taking in the information directly

# Returns the atomic numbers and positions for input into calculator
def rdkitmolToXTBInputs(mol, save=False):
    # get positions and number of atoms in the molecule
    conformer = mol.GetConformer() 
    num_atoms = mol.GetNumAtoms()
    # get the atomic numbers and positions to use as input
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    atomic_symbols = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    atomic_positions = np.array([conformer.GetAtomPosition(i) for i in range(num_atoms)]) / ANGSTROM_PER_BOHR # Convert force field optimized xyz from angstrom to bohr
    return atomic_numbers, atomic_symbols, atomic_positions

# Runs the SCF Calculation and gets very expensive as the number of atoms increases... need to call this as little as possible
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

# Controls how the next XYZ coordinates are determined
def update_xyz(atomic_numbers, atomic_positions, gradient, H, calc_solvent, solvent, method='BFGS'):
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

def is_convergenced(xyz_history, energy_history, gradient_history, criteria='Tight', TolE=5e-6, TolRMSG=1e-4, TolMaxG=3e-4, TolMaxD=4e-3,TolRMSD=2e-3):
    # Convergence Criteria based on ORCA 4.2.1. convergence criteria https://www.afs.enea.it/software/orca/orca_manual_4_2_1.pdf page 19
    match criteria:
        case 'Normal': return True if np.allclose(xyz_history[-1], xyz_history[-2], TolMaxD) and RMS(np.linalg.norm(xyz_history[-1]-xyz_history[-2])) < TolRMSD else False
        case 'Tight': return True if np.allclose(xyz_history[-1], xyz_history[-2], TolMaxD) and RMS(np.linalg.norm(xyz_history[-1]-xyz_history[-2])) and max(np.linalg.norm(gradient_history[-1],axis=1)) < TolMaxG else False
        case 'VeryTight': return True if np.allclose(xyz_history[-1], xyz_history[-2], TolMaxD) and RMS(np.linalg.norm(xyz_history[-1]-xyz_history[-2])) and max(np.linalg.norm(gradient_history[-1],axis=1)) < TolMaxG and abs(energy_history[-1]-energy_history[-2]) < TolE and RMS(np.linalg.norm(gradient_history[-1], axis=1)) < TolRMSG else False
        
# Optimization Method
def geom_opt(atomic_numbers, atomic_symbols, atomic_positions, max_optimzize_iterations=100, calc_solvent=False, solvent=Solvent.h2o, plot=True, verbose=True):
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
        atomic_positions, update_energy, update_gradient, H, = update_xyz(atomic_numbers, atomic_positions, update_gradient, H, calc_solvent=False, solvent=None) if not calc_solvent else update_xyz(atomic_numbers, atomic_positions, update_gradient, H, calc_solvent=True, solvent=solvent)
        xyz_history.append(atomic_positions)
        energy_history.append(update_energy)
        grad_history.append(update_gradient)
        # Check Convergence Criteria
        if iter > 1 and is_convergenced(xyz_history, energy_history, grad_history):
            print(f"Finished calculation after {iter} iterations with final energy: {energy_history[-1]}") if verbose is True else ''
            break
        if iter == max_optimzize_iterations-1:
            print(f"Maximum iterations ({iter}) reached for vacuum calculation. Ending with final energy: {energy_history[-1]}") if verbose is True else ''
    # end timer
    end_time = timeit.default_timer()
    print("Total Time: ", end_time-start_time) if verbose is True else ''
    
    # Plotting for Debugging
    if plot:
        fig, ax = plt.subplots()
        ax.plot(np.arange(0,len(energy_history)-1), energy_history[1:], label="Vacuum Energy")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy (Hartree)")
        ax.legend()
        plt.savefig(f"images/energy_convergence-{'solvent' if calc_solvent else 'vacuum'}.jpg")
        
    return xyz_history[-1]*ANGSTROM_PER_BOHR, energy_history[-1]


# Reference Molecule Properties in Solvent
get_solvent_reference_properties = lambda smiles: geom_opt(*rdkitmolToXTBInputs(smilesToMol(smiles)), solvent=True, verbose=True, max_optimzize_iterations=200)

# Run this one time to generate the files necessary to hold the reference data
def make_reference_files():
    reference_smiles_list = np.array(['[H][H]','[C-]#[C+]', 'N#N', 'O=O', 'FF', 'P#P', 'S=S', 'ClCl'])
    energy_dict = {}
    for ref in reference_smiles_list:
        xyz, energy = get_solvent_reference_properties(ref)
        energy_dict[ref] = energy
        np.savetxt(f'references/{ref}.txt', xyz, fmt='%f')
    with open(f'references/reference_energies.pkl', 'wb') as file:
        pickle.dump(energy_dict, file)

# Gets the formation energies by building SMILES molecule based on references
def get_formation_energies(atomic_symbols, E_formed):
    # get the total number of atoms
    count_dictionary = {}
    for symbol in set(atomic_symbols):
        count = np.sum(np.char.count(atomic_symbols, symbol))
        count_dictionary[symbol] = count
    # if any count is odd, multiply by 2 to make it all even
    if any(value % 2 != 0 for value in count_dictionary.values()):
        is_odd = True
        count_dictionary = {key: value * 2 for key, value in count_dictionary.items()}
    # calculate energy used to break bonds
    E_broken = 0
    # Specify the filename
    filename = 'references/reference_energies.pkl'
    # Check if the file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    with open(filename, 'rb') as f:
        reference_energies = pickle.load(f)
    print(reference_energies)
    # for key, value in count_dictionary.items(): # BROKEN for now
    #     E_broken += value/2 * reference_energies[key]*2 if is_odd is True else value/2 * reference_energies[key]
    return E_broken - E_formed

        
# Final Method
def smiles_to_properties(smiles):
    atomic_numbers, atomic_symbols, atomic_positions = rdkitmolToXTBInputs(smilesToMol(smiles))
    # Run Vacuum calculation
    xyz, vacuum_energy = geom_opt(atomic_numbers, atomic_symbols, atomic_positions, calc_solvent=False)
    # Run Solvent calculation
    xyz, solvent_energy = geom_opt(atomic_numbers, atomic_symbols, atomic_positions, calc_solvent=True)
    solvation_energy = (solvent_energy - vacuum_energy) * KCALPERMOL_PER_HARTREE
    print("xyz (angstrom)")
    print(xyz)
    print(f"Solvation Energy (kcal/mol): {solvation_energy: .3}")
    # Get Formation Energies using information in folder
    formation_energy = get_formation_energies(atomic_symbols, solvent_energy)
    print(f"Energy of Formation (kcal/mol): {formation_energy: .3}")