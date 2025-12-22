from collections import defaultdict
import MDAnalysis as mda
from MDAnalysis.analysis.contacts import contact_matrix
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.analysis.rms import rmsd, rmsf
from molecular_simulations.analysis import (
    ipSAE,
    DynamicInteractionEnergy,
    Fingerprinter,
    StaticInteractionEnergy,
)
from molecular_simulations.utils import assign_chainids
import numpy as np
from pathlib import Path
from rust_simulation_tools import kabsch_align
from .funcs import contact_frequency, population_statistics

def basic_simulation_workflow(paths: list[Path], 
                              top_file: Optional[str]=None,
                              traj_file: Optional[str]=None) -> dict[int, Any]:
    import mdtraj as md

    if top_file is None:
        top_file = 'system.prmtop'

    if traj_file is None:
        traj_file = 'prod.dcd'
    
    N = len(inp)
    rmsds = np.zeros((N, 3))
    rmsfs = np.zeros((N, 3))
    rogs = np.zeros((N, 3))
    analysis = {}
    for i, path in enumerate(paths):
        result = {}
        traj = md.load(str(path / traj_file), top=str(path / top_file))
        n_frames = traj.n_frames

        rmsds[i, 2] = n_frames
        rmsfs[i, 2] = n_frames
        rogs[i, 2] = n_frames

        rmsd = md.rmsd(traj, traj[0])

        mean = np.mean(rmsd)
        std = np.std(rmsd)
        
        result['rmsd'] = {
            'mean': mean,
            'std': std,
            'min': np.min(rmsd),
            'max': np.max(rmsd),
            'unit': 'nm',
        }

        rmsds[i, 0] = mean
        rmsds[i, 1] = std

        rmsf = md.rmsf(traj, traj[0])

        mean = np.mean(rmsf)
        std = np.std(rmsf)

        result['rmsf'] = {
            'mean': mean,
            'std': std,
            'min': np.min(rmsf),
            'max': np.max(rmsf),
            'unit': 'nm',
        }

        rmsfs[i, 0] = mean
        rmsfs[i, 1] = std

        rg = md.compute_rg(traj)

        mean = np.mean(rg)
        std = np.std(rg)

        result['radius_of_gyration'] = {
            'mean': mean,
            'std': std,
            'min': np.min(rg),
            'max': np.max(rg),
            'unit': 'nm',
        }

        rogs[i, 0] = mean
        rogs[i, 1] = std

        # Identify flexible/stable residues
        flex_threshold = result['rmsf']['mean'] + result['rmsf']['std']
        flexible_residues = np.where(rmsf > flex_threshold)[0].tolist()

        stable_threshold = result['rmsf']['mean'] - 0.5 * result['rmsf']['std']
        stable_residues = np.where(rmsf < stable_threshold)[0].tolist()

        result['flexibility_analysis'] = {
            'flexible_residues': flexible_residues[:n_flexible],
            'stable_residues': stable_residues[:n_stable],
            'flexibility_threshold': flex_threshold,
            'stability_threshold': stable_threshold,
        }

        result['trajectory_info'] = {
            'path': path,
            'n_frames': n_frames,
            'n_atoms': traj.n_atoms,
            'n_residues': traj.n_residues,
            'time_ns': traj.time[-1] / 1000 if len(traj.time) > 0 else 0.
        }

        analysis[i] = result

    summary_stats = {
        'rmsd': population_statistics(rmsds),
        'rmsf': population_statistics(rmsfs),
        'radius_of_gyration': population_statistics(rogs),
    }

    return {'basic_simulation_analysis': {'summary': summary_stats, 'raw': analysis}}

def matt_workflow(path, chain_of_interest):
    # load universe
    u = mda.Universe(str(path / 'system.prmtop'), str(path / 'prod.dcd'))

    # assign chainIDs in universe
    u = assign_chainids(u)

    # get RMSD
    ref = u.select_atoms(f'{chain_of_interest} and backbone')

    # get RMSF

    # get RoG

    pass

def advanced_simulation_workflow(paths: list[Path],
                                 distance_cutoff: float=8.0,
                                 n_top: int=10,
                                 top_file: Optional[str]=None,
                                 traj_file: Optional[str]=None) -> dict[str, Any]:
    if top_file is None:
        top_file = 'system.prmtop'
    
    if traj_file is None:
        traj_file = 'prod.dcd'

    analysis = {}
    summary = {}
    for i, path in enumerate(paths):
        top = path / top_file
        traj = path / traj_file

        resids_A, resids_B, freq_A, freq_B = contact_frequency(top, traj, distance_cutoff, n_top)
        analysis[i] = {
            'path': path,
            'top_resids_chain_A': resids_A,
            'contact_frequencies_A': freq_A,
            'top_resids_chain_B': resids_B,
            'contact_frequencies_B': freq_B,
        }

        if summary:
            for resid, freq in freq_A:
                summary[resid] += freq

    for resid, freq in summary.items():
        summary[resid] = freq / len(paths)

    return {'advanced_simulation_analysis': {'summary': summary, 'contact_data': analysis}}

def basic_static_workflow(paths: list[Path],
                          distance_cutoff: float=8.0) -> dict[str, Any]:
    analysis = {}
    for i, path in enumerate(paths):
        u = mda.Universe(str(path))
        sel = u.select_atoms(f'(name CA and chainID A) and around {distance_cutoff} (name CA and chainID B)')

        analysis[i] = {
            'path': path,
            'contacts': sel.residues.resids
        }

    return {'basic_structure_analysis': analysis}

def advanced_static_workflow():
    pass

