from collections import defaultdict
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from molecular_simulations.utils import assign_chainids
import numpy as np
from pathlib import Path

def contact_frequency(topology: Path,
                      trajectory: Path,
                      distance_cutoff: float=8.0,
                      n_top: int=10) -> tuple[list, list, dict, dict]:
    u = mda.Universe(str(topology), str(trajectory))
    u = assign_chainids(u)

    
    chainA = u.select_atoms('chainID A and name CA')
    chainB = u.select_atoms('chainID B and name CA')

    resids_A = chainA.resids
    resids_B = chainB.resids

    contact_counts_A = defaultdict(int, {resid: 0. for resid in resids_A})
    contact_counts_B = defaultdict(int, {resid: 0. for resid in resids_B})

    n_frames = u.trajectory.n_frames

    for ts in u.trajectory:
        dist_matrix = distance_array(chainA.positions, chainB.positions)
        idx_A, idx_B = np.where(dist_matrix < distance_cutoff)

        contact_resids_A = set(resids_A[idx_A])
        contact_resids_B = set(resids_B[idx_B])

        for resid in contact_resids_A:
            contact_counts_A[resid] += 1

        for resid in contact_resids_B:
            contact_counts_B[resid] += 1

    freq_A = {resid: count / n_frames for resid, count in contact_counts_A.items()}
    freq_B = {resid: count / n_frames for resid, count in contact_counts_B.items()}

    top_resids_A = sorted(freq_A.keys(), key=lambda r: freq_A[r], reverse=True)[:n_top]
    top_resids_B = sorted(freq_B.keys(), key=lambda r: freq_B[r], reverse=True)[:n_top]

    return top_resids_A, top_resids_B, freq_A, freq_B

def population_statistics(data: np.array) -> np.ndarray:
    """For an input data array of shape (N, 3) where each row
    is an observation of mean, std, and n_observations, compute
    the combined mean, variance and subsequently standard dev.

    Combined mean is simply the geometric mean of means.
    Combined variance is computed according to:
        $\sigma^2 = ( \sum_{n_i} (s_i^2 + (\mu_i - \mu)^2) ) / N$
    """
    means = data[:, 0]
    stds = data[:, 1]
    ns = data[:, 2].astype(float)

    N_total = ns.sum()
    mean_combined = (ns * means).sum() / N_total

    var_combined = (ns * (stds**2 + (means - mean_combined)**2)).sum() / N_total
    std_combined = np.sqrt(var_combined)

    return np.array([mean_combined, std_combined])
