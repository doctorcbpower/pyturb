import importlib.util
import numpy as np
from numba import njit, prange

# -------------------------
# KDTree for neighbor search
# -------------------------
@njit
def build_kdtree(points):
    """
    For simplicity, just store points; real KDTree can be built recursively 
    if needed.
    """
    return points

@njit
def query_kdtree(kdtree_points, point, k):
    """
    Return the indices of the k nearest neighbors to 'point'.
    """
    N = kdtree_points.shape[0]
    dists = np.empty(N, dtype=np.float64)
    for i in range(N):
        diff = kdtree_points[i] - point
        dists[i] = np.sqrt(np.sum(diff**2))
    idx_sorted = np.argsort(dists)
    return idx_sorted[:k]

@njit
def calc_epsilon(v1, v2, v3):
    return np.abs(v1 - v2) / (np.abs(v3) + 1e-12)

import numpy as np
from numba import njit

@njit
def estimate_local_densities(positions, masses, kdtree_points, N_neighbours):
    """
    Estimate local densities for each particle using a top-hat kernel.
    Positions: (N,3)
    Masses: (N,) or scalar
    kdtree_points: KDTree data for neighbour queries
    """
    n_particles = positions.shape[0]
    densities = np.zeros(n_particles)

    for i in range(n_particles):
        # query N+1 neighbours (first one is the particle itself)
        kd_filter = query_kdtree(kdtree_points, positions[i], N_neighbours + 1)[1:]
        neigh_pos = positions[kd_filter]

        # distance to farthest neighbour in this set
        dists = np.sqrt(np.sum((neigh_pos - positions[i])**2, axis=1))
        h = np.max(dists)

        # total mass inside sphere
        m_sum = np.sum(masses[kd_filter]) if masses.ndim > 0 else masses * N_neighbours

        # density = mass / volume
        densities[i] = m_sum / ((4.0/3.0) * np.pi * h**3)

    print(densities)
    return densities

@njit(parallel=True)
def velocity_smoothing_filter_all(positions, velocities, local_density, mach,
                                  kdtree_points, N_neighbours, itr,
                                  tolerance, shock_threshold):
    n_particles = positions.shape[0]
    V_turb = np.zeros((n_particles, 3))
    epsilons_out = np.zeros((n_particles, 3))
    iterations = np.zeros(n_particles, dtype=np.int32)

    for i in prange(n_particles):
        V_turb[i], epsilons_out[i], iterations[i] = velocity_smoothing_filter_single(
            i, positions, velocities, local_density, mach,
            kdtree_points, N_neighbours, itr, tolerance, shock_threshold
        )

    return V_turb, epsilons_out, iterations
@njit
def velocity_smoothing_filter_single(i, positions, velocities, local_density, mach,
                                     kdtree_points, N_neighbours, itr, tolerance,
                                     shock_threshold):
    epsilons_t = np.ones(3)
    epsilons_sh = np.ones(3)
    epsilons = np.ones(3)
    
    V_bulk_at_scale = np.zeros((itr, 3), dtype=np.float64)
    sigma_at_scale = np.zeros((itr, 3), dtype=np.float64)
    V_turb_at_scale = np.zeros((itr, 3), dtype=np.float64)
    V_turb_converged = np.zeros(3, dtype=np.float64)
    converged = np.zeros(3, dtype=np.bool_)
    
    for j in range(itr):
        k = min(N_neighbours * 2**j, positions.shape[0])
        kd_filter = query_kdtree(kdtree_points, positions[i], k)
        
        weights = local_density[kd_filter]
        V_bulk_at_scale[j] = np.sum(velocities[kd_filter] * weights[:, None], axis=0) / np.sum(weights)
        sigma_at_scale[j] = np.sqrt(np.sum(((velocities[kd_filter] - V_bulk_at_scale[j])**2) * weights[:, None], axis=0) / np.sum(weights))
        V_turb_at_scale[j] = velocities[i] - V_bulk_at_scale[j]
        
        if j >= 1:
            epsilons_t = calc_epsilon(V_bulk_at_scale[j], V_bulk_at_scale[j-1], sigma_at_scale[j-1])
        
        for comp in range(3):
            epsilons[comp] = min(epsilons_sh[comp], epsilons_t[comp])
            if not converged[comp] and epsilons[comp] <= tolerance:
                V_turb_converged[comp] = V_turb_at_scale[j-1][comp]
                converged[comp] = True
        
        if np.all(converged):
            return V_turb_converged, epsilons, j
        
        Mach_max = np.max(mach[kd_filter])
        if Mach_max > 1.3:
            epsilons_sh = -tolerance * np.ones(3)
    
    V_turb_converged[~converged] = V_turb_at_scale[itr-1]
    return V_turb_converged, epsilons, itr

# @njit(parallel=True)
# def compute_all_particles(positions, velocities, local_density, mach, kdtree_points,
#                           N_neighbours, itr, tolerance):
#     N = positions.shape[0]
#     V_turb_vec = np.zeros((N, 3), dtype=np.float64)
#     epsilons_out = np.zeros((N, 3), dtype=np.float64)
#     iteration_out = np.zeros(N, dtype=np.int32)
    
#     for i in prange(N):
#         V_turb_vec[i], epsilons_out[i], iteration_out[i] = velocity_smoothing_filter_single(
#             i, positions, velocities, local_density, mach, kdtree_points, N_neighbours, itr, tolerance
#         )
    
#     return V_turb_vec, epsilons_out, iteration_out

# -------------------------
# Main class
# -------------------------
# class MeasureKinematics:
#     def __init__(self, positions, velocities, local_density, mach,
#                  N_neighbours=32, itr=10, tolerance=0.1):
#         self.positions = positions
#         self.velocities = velocities
#         self.local_density = local_density
#         self.mach = mach
#         self.N_neighbours = N_neighbours
#         self.itr = itr
#         self.tolerance = tolerance
#         # Build a "KDTree" (for now just points, can be replaced by real Numba KDTree later)
#         self.kdtree_points = build_kdtree(positions)
    
#     def compute_turbulent_velocity(self):
#         return compute_all_particles(
#             self.positions, self.velocities, self.local_density, self.mach,
#             self.kdtree_points, self.N_neighbours, self.itr, self.tolerance
#         )

class MeasureVelocityFieldKinematics:
    def __init__(self, positions, velocities, masses, mach, local_density=None, kdtree_points=None ,
                 N_neighbours=32, itr=10, tolerance=0.1, shock_threshold=1.3):
        self.positions = positions
        self.velocities = velocities

        if kdtree_points is None:
            self.kdtree_points = build_kdtree(positions)
        else:
            self.kdtree_points = kdtree_points

        if local_density is None:
            print("Estimating local densities...")
            self.local_density = estimate_local_densities(positions, masses, self.kdtree_points, N_neighbours)
        else:
            self.local_density = local_density
            
        self.mach = mach

        # parameters
        self.N_neighbours = N_neighbours
        self.itr = itr
        self.tolerance = tolerance
        self.shock_threshold = shock_threshold

    def compute_all(self):
        """
        Run velocity smoothing for all particles in parallel.
        """
        return velocity_smoothing_filter_all(
            self.positions, self.velocities, self.local_density, self.mach,
            self.kdtree_points, self.N_neighbours, self.itr,
            self.tolerance, self.shock_threshold
        )