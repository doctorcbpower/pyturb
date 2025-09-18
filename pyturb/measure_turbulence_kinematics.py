import importlib.util
import numpy as np
from numba import njit, prange
from pykdtree.kdtree import KDTree
from tqdm import tqdm

@njit
def calc_epsilon(v1, v2, v3):
    """
    Calculates the relative change between two velocity vectors normalized by a third vector.

    This function is used to compute the epsilon value for convergence checks in the turbulence calculation.

    Args:
        v1 (np.ndarray): First velocity vector.
        v2 (np.ndarray): Second velocity vector.
        v3 (np.ndarray): Velocity vector used for normalization.

    Returns:
        np.ndarray: The computed epsilon value.
    """
    return np.abs(v1 - v2)/np.abs(v3)

@njit
def velocity_smoothing_filter(i, kdtree, g_R_vec, g_V_vec, g_local_density, g_Mach, N_neighbours, itr, tolerance):
    """
    Computes the turbulent velocity and convergence metrics for a single particle using a smoothing filter.

    This function iteratively calculates bulk and turbulent velocities at increasing scales, checks for convergence, and handles shocked cells.

    Args:
        i (int): Index of the particle to process.
        kdtree: KDTree structure for neighbor queries.
        g_R_vec (np.ndarray): Array of particle positions.
        g_V_vec (np.ndarray): Array of particle velocities.
        g_local_density (np.ndarray): Array of local densities for weighting.
        g_Mach (np.ndarray): Array of Mach numbers for each particle.
        N_neighbours (np.ndarray): Array of neighbor counts for each particle.
        itr (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.

    Returns:
        tuple: (g_V_turb_converged, epsilons, iteration) where
            g_V_turb_converged (np.ndarray): Converged turbulent velocity vector.
            epsilons (np.ndarray): Epsilon values for convergence.
            iteration (int): Number of iterations performed.
    """
    # resetting the smoothing process
    epsilons_t = np.zeros(3) + 1
    epsilons_sh = np.zeros(3) + 1
    epsilons = np.zeros(3) + 1
    g_V_bulk_at_scale = np.zeros((itr, 3), dtype=np.float64)
    g_sigma_at_scale = np.zeros((itr, 3), dtype=np.float64)
    g_V_turb_at_scale = np.zeros((itr, 3), dtype=np.float64)
    g_V_turb_converged = np.zeros(3, dtype=np.float64)
    converged = np.zeros(3, dtype=np.bool_)
    for j in range(itr+1):
        # finding neighbours at scale j
        kd_filter = kdtree.query(g_R_vec[i], k=N_neighbours[i]*2**j)[1][0]
        # choosing the weights for calculating average velocities
        weights = g_local_density[kd_filter]
        # the gas bulk velocity at scale j
        g_V_bulk_at_scale[j] = np.sum(np.multiply(g_V_vec[kd_filter], weights[:, np.newaxis]), axis=0)/np.sum(weights)
        # the gas velocity dispersion at scale j
        g_sigma_at_scale[j] = np.sqrt(np.sum(np.multiply(np.square(np.subtract(g_V_vec[kd_filter], g_V_bulk_at_scale[j])), weights[:, np.newaxis]), axis=0)/np.sum(weights))
        # the gas turbulent velocity at scale j
        g_V_turb_at_scale[j] = g_V_vec[i] - g_V_bulk_at_scale[j]
        # calculating epsilons at scale j
        if j >= 1:
            epsilons_t = calc_epsilon(g_V_bulk_at_scale[j], g_V_bulk_at_scale[j-1], g_sigma_at_scale[j-1])
        # checking for convergence in each component
        for k in range(3):
            epsilons[k] = min(epsilons_sh[k], epsilons_t[k])
            if not converged[k] and epsilons[k] <= tolerance:
                #print('Converged in x[' + str(k) + '], with epsilons[' + str(k) + '] = ' + str(epsilons[k]))
                g_V_turb_converged[k] = g_V_turb_at_scale[j-1][k]
                converged[k] = True
            if np.all(converged):
                #print('Turbulence converged')
                return g_V_turb_converged, epsilons, j
        # masking shocked cells
        g_Mach_max = max(g_Mach[kd_filter])
        if g_Mach_max > 1.3:
            epsilons_sh = np.zeros(3) - tolerance
    g_V_turb_converged[~converged] = g_V_turb_at_scale[itr]
    return g_V_turb_converged, epsilons, itr

@njit(parallel=True)
def compute_chunk(start, end, kdtree, g_V_turb_vec, epsilons, iteration):
    """
    Computes the turbulent velocity, epsilon values, and iteration count for a chunk of particles in parallel.
    
    This function applies the velocity smoothing filter to each particle in the specified range, updating the output arrays.

    Args:
        start (int): Starting index of the chunk.
        end (int): Ending index of the chunk.
        kdtree: KDTree structure for neighbor queries.
        g_V_turb_vec (np.ndarray): Array to store turbulent velocities.
        epsilons (np.ndarray): Array to store epsilon values.
        iteration (np.ndarray): Array to store iteration counts.

    Returns:
        None
    """
    for i in prange(start, end):
        g_V_turb_vec[i], epsilons[i], iteration[i] = velocity_smoothing_filter(kdtree, i)
    
class MeasureKinematics():
    """
    Provides methods for setting up kinematic analysis and computing turbulent velocities for gas particles.

    This class manages the configuration and execution of turbulence calculations using KDTree-based neighbor searches.
    """
    def __init__(self,tolerance=0.1,iter=25,n_ngb=10):
        """
        Initializes the MeasureKinematics class with parameters for turbulence calculation.

        Sets the filtering tolerance, maximum number of iterations, and number of neighbors for the analysis.

        Args:
            tolerance (float, optional): Filtering tolerance limit. Defaults to 0.1.
            iter (int, optional): Maximum number of iterations. Defaults to 25.
            n_ngb (int, optional): Number of neighbours. Defaults to 10.

        Returns:
            None
        """
        self.tolerance = tolerance  # Filtering tolerance limit
        self.iter = iter            # Maximum number of iterations
        self.n_ngb = n_ngb          # Number of neighbours
        self.kdtree = KDTree(g_R_vec)
        pass
    def setup_kdtree(self):
        """
        Set up the KDTree
        """
        kdtree = KDTree(g_R_vec)
    # # setting the minimum number of neighbours
    # N_neighbours = np.rint(16*np.mean(2*g_mass)/g_mass).astype(np.int32)
    # @njit

    # function for calculating the turbulence of all gas particles (in parallel chunks)
    def compute_turbulent_velocity(self):
        """
        Calculates the turbulent velocity, epsilon values, and iteration counts for all gas particles.

        This function processes all particles in parallel chunks, applying the velocity smoothing filter and returning the results.

        Args:
            kdtree: KDTree structure for neighbor queries.

        Returns:
            tuple: (g_V_turb_vec, epsilons, iteration) where
                g_V_turb_vec (np.ndarray): Array of turbulent velocity vectors for all particles.
                epsilons (np.ndarray): Array of epsilon values for all particles.
                iteration (np.ndarray): Array of iteration counts for all particles.
        """
        g_V_turb_vec = np.empty((g_N_particles, 3), dtype=np.float64)
        epsilons = np.empty((g_N_particles, 3), dtype=np.float64)
        iteration = np.empty(g_N_particles, dtype=np.float64)
        chunk_size = 10000
        for start in tqdm(range(0, g_N_particles, chunk_size)):
            end = min(start + chunk_size, g_N_particles)
            compute_chunk(start, end, kdtree, g_V_turb_vec, epsilons, iteration)
        return g_V_turb_vec, epsilons, iteration
   
        # # gas particle turbulent velocities
        # g_V_turb_vec, epsilons, iteration = compute_turbulent_velocity(kdtree)
        # g_V_turbx = [float(v[0]) for v in g_V_turb_vec]
        # g_V_turby = [float(v[1]) for v in g_V_turb_vec]
        # g_V_turbz = [float(v[2]) for v in g_V_turb_vec]

        # ## Saving data for halo
        # with open(feedback+'h'+str(halo)+'_g_V_turbx.txt', "w") as output:
        # output.write(str(g_V_turbx))
        # with open(feedback+'h'+str(halo)+'_g_V_turby.txt', "w") as output:
        # output.write(str(g_V_turby))
        # with open(feedback+'h'+str(halo)+'_g_V_turbz.txt', "w") as output:
        # output.write(str(g_V_turbz))
