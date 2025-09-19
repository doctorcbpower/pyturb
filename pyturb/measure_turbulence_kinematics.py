#import importlib.util
#import numpy as np
#from numba import njit, prange
#
## -------------------------
## KDTree for neighbor search
## -------------------------
#@njit
#def build_kdtree(points):
#    """
#    For simplicity, just store points; real KDTree can be built recursively 
#    if needed.
#    """
#    return points
#
#@njit
#def query_kdtree(kdtree_points, point, k):
#    """
#    Return the indices of the k nearest neighbors to 'point'.
#    """
#    N = kdtree_points.shape[0]
#    dists = np.empty(N, dtype=np.float64)
#    for i in range(N):
#        diff = kdtree_points[i] - point
#        dists[i] = np.sqrt(np.sum(diff**2))
#    idx_sorted = np.argsort(dists)
#    return idx_sorted[:k]
#
#@njit
#def calc_epsilon(v1, v2, v3):
#    return np.abs(v1 - v2) / (np.abs(v3) + 1e-12)
#
#import numpy as np
#from numba import njit
#
#@njit
#def estimate_local_densities(positions, masses, kdtree_points, N_neighbours):
#    """
#    Estimate local densities for each particle using a top-hat kernel.
#    Positions: (N,3)
#    Masses: (N,) or scalar
#    kdtree_points: KDTree data for neighbour queries
#    """
#    n_particles = positions.shape[0]
#    densities = np.zeros(n_particles)
#
#    for i in range(n_particles):
#        # query N+1 neighbours (first one is the particle itself)
#        kd_filter = query_kdtree(kdtree_points, positions[i], N_neighbours + 1)[1:]
#        neigh_pos = positions[kd_filter]
#
#        # distance to farthest neighbour in this set
#        dists = np.sqrt(np.sum((neigh_pos - positions[i])**2, axis=1))
#        h = np.max(dists)
#
#        # total mass inside sphere
#        m_sum = np.sum(masses[kd_filter]) if masses.ndim > 0 else masses * N_neighbours
#
#        # density = mass / volume
#        densities[i] = m_sum / ((4.0/3.0) * np.pi * h**3)
#
#    print(densities)
#    return densities
#
#@njit(parallel=True)
#def velocity_smoothing_filter_all(positions, velocities, local_density, mach,
#                                  kdtree_points, N_neighbours, itr,
#                                  tolerance, shock_threshold):
#    n_particles = positions.shape[0]
#    V_turb = np.zeros((n_particles, 3))
#    epsilons_out = np.zeros((n_particles, 3))
#    iterations = np.zeros(n_particles, dtype=np.int32)
#
#    for i in prange(n_particles):
#        V_turb[i], epsilons_out[i], iterations[i] = velocity_smoothing_filter_single(
#            i, positions, velocities, local_density, mach,
#            kdtree_points, N_neighbours, itr, tolerance, shock_threshold
#        )
#
#    return V_turb, epsilons_out, iterations
#@njit
#def velocity_smoothing_filter_single(i, positions, velocities, local_density, mach,
#                                     kdtree_points, N_neighbours, itr, tolerance,
#                                     shock_threshold):
#    epsilons_t = np.ones(3)
#    epsilons_sh = np.ones(3)
#    epsilons = np.ones(3)
#    
#    V_bulk_at_scale = np.zeros((itr, 3), dtype=np.float64)
#    sigma_at_scale = np.zeros((itr, 3), dtype=np.float64)
#    V_turb_at_scale = np.zeros((itr, 3), dtype=np.float64)
#    V_turb_converged = np.zeros(3, dtype=np.float64)
#    converged = np.zeros(3, dtype=np.bool_)
#    
#    for j in range(itr):
#        k = min(N_neighbours * 2**j, positions.shape[0])
#        kd_filter = query_kdtree(kdtree_points, positions[i], k)
#        
#        weights = local_density[kd_filter]
#        V_bulk_at_scale[j] = np.sum(velocities[kd_filter] * weights[:, None], axis=0) / np.sum(weights)
#        sigma_at_scale[j] = np.sqrt(np.sum(((velocities[kd_filter] - V_bulk_at_scale[j])**2) * weights[:, None], axis=0) / np.sum(weights))
#        V_turb_at_scale[j] = velocities[i] - V_bulk_at_scale[j]
#        
#        if j >= 1:
#            epsilons_t = calc_epsilon(V_bulk_at_scale[j], V_bulk_at_scale[j-1], sigma_at_scale[j-1])
#        
#        for comp in range(3):
#            epsilons[comp] = min(epsilons_sh[comp], epsilons_t[comp])
#            if not converged[comp] and epsilons[comp] <= tolerance:
#                V_turb_converged[comp] = V_turb_at_scale[j-1][comp]
#                converged[comp] = True
#        
#        if np.all(converged):
#            return V_turb_converged, epsilons, j
#        
#        Mach_max = np.max(mach[kd_filter])
#        if Mach_max > 1.3:
#            epsilons_sh = -tolerance * np.ones(3)
#    
#    V_turb_converged[~converged] = V_turb_at_scale[itr-1]
#    return V_turb_converged, epsilons, itr
#
## @njit(parallel=True)
## def compute_all_particles(positions, velocities, local_density, mach, kdtree_points,
##                           N_neighbours, itr, tolerance):
##     N = positions.shape[0]
##     V_turb_vec = np.zeros((N, 3), dtype=np.float64)
##     epsilons_out = np.zeros((N, 3), dtype=np.float64)
##     iteration_out = np.zeros(N, dtype=np.int32)
#    
##     for i in prange(N):
##         V_turb_vec[i], epsilons_out[i], iteration_out[i] = velocity_smoothing_filter_single(
##             i, positions, velocities, local_density, mach, kdtree_points, N_neighbours, itr, tolerance
##         )
#    
##     return V_turb_vec, epsilons_out, iteration_out
#
## -------------------------
## Main class
## -------------------------
## class MeasureKinematics:
##     def __init__(self, positions, velocities, local_density, mach,
##                  N_neighbours=32, itr=10, tolerance=0.1):
##         self.positions = positions
##         self.velocities = velocities
##         self.local_density = local_density
##         self.mach = mach
##         self.N_neighbours = N_neighbours
##         self.itr = itr
##         self.tolerance = tolerance
##         # Build a "KDTree" (for now just points, can be replaced by real Numba KDTree later)
##         self.kdtree_points = build_kdtree(positions)
#    
##     def compute_turbulent_velocity(self):
##         return compute_all_particles(
##             self.positions, self.velocities, self.local_density, self.mach,
##             self.kdtree_points, self.N_neighbours, self.itr, self.tolerance
##         )
#
#class MeasureVelocityFieldKinematics:
#    def __init__(self, positions, velocities, masses, mach, local_density=None, kdtree_points=None ,
#                 N_neighbours=32, itr=10, tolerance=0.1, shock_threshold=1.3):
#        self.positions = positions
#        self.velocities = velocities
#
#        if kdtree_points is None:
#            self.kdtree_points = build_kdtree(positions)
#        else:
#            self.kdtree_points = kdtree_points
#
#        if local_density is None:
#            print("Estimating local densities...")
#            self.local_density = estimate_local_densities(positions, masses, self.kdtree_points, N_neighbours)
#        else:
#            self.local_density = local_density
#            
#        self.mach = mach
#
#        # parameters
#        self.N_neighbours = N_neighbours
#        self.itr = itr
#        self.tolerance = tolerance
#        self.shock_threshold = shock_threshold
#
#    def compute_all(self):
#        """
#        Run velocity smoothing for all particles in parallel.
#        """
#        return velocity_smoothing_filter_all(
#            self.positions, self.velocities, self.local_density, self.mach,
#            self.kdtree_points, self.N_neighbours, self.itr,
#            self.tolerance, self.shock_threshold
#        )

import numpy as np
from numba import njit, prange
from numba.types import int32, float64
import numba

# -------------------------
# Spatial Grid Hash for O(1) neighbor search
# -------------------------
@njit
def spatial_hash_3d(pos, cell_size, grid_dims):
    """Convert 3D position to grid cell indices."""
    ix = int(pos[0] / cell_size) % grid_dims[0]
    iy = int(pos[1] / cell_size) % grid_dims[1] 
    iz = int(pos[2] / cell_size) % grid_dims[2]
    return ix, iy, iz

@njit
def build_spatial_grid(positions, cell_size):
    """Build spatial hash grid for O(1) neighbor queries."""
    n_particles = positions.shape[0]
    
    # Determine grid bounds - manual axis reduction for Numba compatibility
    mins = np.empty(3, dtype=np.float64)
    maxs = np.empty(3, dtype=np.float64)
    
    for dim in range(3):
        mins[dim] = positions[0, dim]
        maxs[dim] = positions[0, dim]
        for i in range(1, n_particles):
            if positions[i, dim] < mins[dim]:
                mins[dim] = positions[i, dim]
            if positions[i, dim] > maxs[dim]:
                maxs[dim] = positions[i, dim]
    
    # Calculate grid dimensions
    dims = np.empty(3, dtype=np.int32)
    for dim in range(3):
        dims[dim] = max(1, int(np.ceil((maxs[dim] - mins[dim]) / cell_size)))
    
    # Create grid structure
    max_particles_per_cell = max(64, n_particles // 1000)  # Adaptive sizing
    grid_size = dims[0] * dims[1] * dims[2]
    
    # Grid data: [cell_id] -> list of particle indices
    grid_particles = np.full((grid_size, max_particles_per_cell), -1, dtype=np.int32)
    grid_counts = np.zeros(grid_size, dtype=np.int32)
    
    # Populate grid
    for i in range(n_particles):
        pos_shifted = positions[i] - mins
        ix = int(pos_shifted[0] / cell_size) % dims[0]
        iy = int(pos_shifted[1] / cell_size) % dims[1]
        iz = int(pos_shifted[2] / cell_size) % dims[2]
        
        cell_id = ix + iy * dims[0] + iz * dims[0] * dims[1]
        
        if grid_counts[cell_id] < max_particles_per_cell:
            grid_particles[cell_id, grid_counts[cell_id]] = i
            grid_counts[cell_id] += 1
    
    return grid_particles, grid_counts, dims, mins, cell_size

@njit
def query_radius_grid(grid_particles, grid_counts, grid_dims, grid_mins, cell_size,
                                 positions, center_pos, radius, max_neighbors):
    neighbors = np.full(max_neighbors, -1, dtype=np.int32)
    neighbor_count = 0
    
    radius_sq = radius * radius
    dim_x, dim_y, dim_z = grid_dims[0], grid_dims[1], grid_dims[2]
    
    # Cell coordinates of the center particle
    cx = int((center_pos[0] - grid_mins[0]) / cell_size)
    cy = int((center_pos[1] - grid_mins[1]) / cell_size)
    cz = int((center_pos[2] - grid_mins[2]) / cell_size)
    
    radius_cells = int(np.ceil(radius / cell_size))
    
    # Precompute neighbor cell indices
    cell_offsets = []
    for dx in range(-radius_cells, radius_cells + 1):
        ix = cx + dx
        if ix < 0:
            ix += dim_x
        elif ix >= dim_x:
            ix -= dim_x
        for dy in range(-radius_cells, radius_cells + 1):
            iy = cy + dy
            if iy < 0:
                iy += dim_y
            elif iy >= dim_y:
                iy -= dim_y
            for dz in range(-radius_cells, radius_cells + 1):
                iz = cz + dz
                if iz < 0:
                    iz += dim_z
                elif iz >= dim_z:
                    iz -= dim_z
                cell_offsets.append(ix + iy*dim_x + iz*dim_x*dim_y)
    
    # Flatten all particle indices in these cells
    candidates = []
    for cell_id in cell_offsets:
        n_in_cell = grid_counts[cell_id]
        for j in range(n_in_cell):
            pid = grid_particles[cell_id, j]
            if pid == -1:
                break
            candidates.append(pid)
    
    # Convert to array for vectorized distance computation
    candidates = np.array(candidates, dtype=np.int32)
    
    if candidates.size == 0:
        return neighbors, 0
    
    dx = positions[candidates, 0] - center_pos[0]
    dy = positions[candidates, 1] - center_pos[1]
    dz = positions[candidates, 2] - center_pos[2]
    dist_sq = dx*dx + dy*dy + dz*dz
    
    # Select neighbors within radius
    count = 0
    for i in range(candidates.size):
        if dist_sq[i] <= radius_sq and count < max_neighbors:
            neighbors[count] = candidates[i]
            count += 1
            if count >= max_neighbors:
                break
    
    return neighbors, count

#@njit
#def query_radius_grid(grid_particles, grid_counts, grid_dims, grid_mins, cell_size, 
#                     positions, center_pos, radius, max_neighbors):
#    """Fast radius query using spatial grid."""
#    neighbors = np.full(max_neighbors, -1, dtype=np.int32)
#    neighbor_count = 0
#    
#    # Determine grid cell range to search
#    radius_cells = int(np.ceil(radius / cell_size))
#    center_shifted = center_pos - grid_mins
#    
#    cx = int(center_shifted[0] / cell_size)
#    cy = int(center_shifted[1] / cell_size) 
#    cz = int(center_shifted[2] / cell_size)
#    
#    # Search surrounding cells
#    for dx in range(-radius_cells, radius_cells + 1):
#        for dy in range(-radius_cells, radius_cells + 1):
#            for dz in range(-radius_cells, radius_cells + 1):
#                ix = (cx + dx) % grid_dims[0]
#                iy = (cy + dy) % grid_dims[1]
#                iz = (cz + dz) % grid_dims[2]
#                
#                cell_id = ix + iy * grid_dims[0] + iz * grid_dims[0] * grid_dims[1]
#                
#                # Check particles in this cell
#                for j in range(grid_counts[cell_id]):
#                    particle_idx = grid_particles[cell_id, j]
#                    if particle_idx == -1:
#                        break
#                    
#                    dist_sq = np.sum((positions[particle_idx] - center_pos)**2)
#                    if dist_sq <= radius**2 and neighbor_count < max_neighbors:
#                        neighbors[neighbor_count] = particle_idx
#                        neighbor_count += 1
#                        
#                        if neighbor_count >= max_neighbors:
#                            return neighbors[:neighbor_count]
#    
#    return neighbors[:neighbor_count]

# -------------------------
# Memory-optimized velocity smoothing
# -------------------------
@njit
def velocity_smoothing_single_optimized(particle_idx, positions, velocities, 
                                      local_density, mach, grid_data,
                                      N_neighbours, max_itr, tolerance, 
                                      shock_threshold):
    """Optimized single particle processing with minimal memory allocation."""
    grid_particles, grid_counts, grid_dims, grid_mins, cell_size = grid_data
    
    # Pre-allocate working arrays (reused across iterations)
    max_neighbors = min(N_neighbours * (2**max_itr), positions.shape[0])
    
    pos_i = positions[particle_idx]
    vel_i = velocities[particle_idx]
    
    # Previous iteration values (only need current and previous)
    v_bulk_prev = np.zeros(3, dtype=np.float64)
    sigma_prev = np.ones(3, dtype=np.float64)  # Avoid division by zero
    v_turb_result = vel_i.copy()  # Default if no convergence
    
    converged = np.zeros(3, dtype=np.bool_)
    final_epsilon = np.ones(3, dtype=np.float64)
    
    for j in range(max_itr):
        # Current search radius
        k_neighbors = min(N_neighbours * (2**j), max_neighbors)
        search_radius = estimate_search_radius(pos_i, positions, k_neighbors)
        
        # Get neighbors using spatial grid
        neighbor_indices = query_radius_grid(
            grid_particles, grid_counts, grid_dims, grid_mins, cell_size,
            positions, pos_i, search_radius, k_neighbors
        )
        
        if len(neighbor_indices) < 2:  # Need at least some neighbors
            continue
            
        # Compute bulk velocity and dispersion (vectorized)
        weights = local_density[neighbor_indices]
        weight_sum = np.sum(weights)
        
        if weight_sum == 0:
            continue
            
        # Bulk velocity: density-weighted average
        neighbor_vels = velocities[neighbor_indices]
        weighted_vels = neighbor_vels * weights.reshape(-1, 1)
        v_bulk = np.zeros(3, dtype=np.float64)
        for dim in range(3):
            v_bulk[dim] = np.sum(weighted_vels[:, dim]) / weight_sum
        
        # Velocity dispersion around bulk velocity
        sigma = np.zeros(3, dtype=np.float64)
        for dim in range(3):
            vel_diff = neighbor_vels[:, dim] - v_bulk[dim]
            sigma[dim] = np.sqrt(np.sum((vel_diff**2) * weights) / weight_sum)
            sigma[dim] = max(sigma[dim], 1e-12)  # Avoid division by zero
        
        # Check convergence (only after first iteration)
        if j > 0:
            epsilon = np.abs(v_bulk - v_bulk_prev) / sigma_prev
            
            # Check shock condition
            neighbor_mach = mach[neighbor_indices]
            max_mach = neighbor_mach[0]
            for k in range(1, len(neighbor_mach)):
                if neighbor_mach[k] > max_mach:
                    max_mach = neighbor_mach[k]
            if max_mach > shock_threshold:
                epsilon = -tolerance * np.ones(3)  # Force non-convergence
            
            # Per-component convergence check
            for comp in range(3):
                if not converged[comp] and epsilon[comp] <= tolerance:
                    v_turb_result[comp] = vel_i[comp] - v_bulk_prev[comp]
                    converged[comp] = True
                    final_epsilon[comp] = epsilon[comp]
            
            # Early exit if all components converged
            if np.all(converged):
                return v_turb_result, final_epsilon, j + 1
        
        # Update for next iteration
        v_bulk_prev = v_bulk.copy()
        sigma_prev = sigma.copy()
    
    # Use final iteration for non-converged components
    for comp in range(3):
        if not converged[comp]:
            v_turb_result[comp] = vel_i[comp] - v_bulk_prev[comp]
            final_epsilon[comp] = 1.0  # Mark as non-converged
    
    return v_turb_result, final_epsilon, max_itr

@njit
def estimate_search_radius(center_pos, positions, k_neighbors):
    """Estimate search radius needed to find k_neighbors."""
    # Quick distance sampling to estimate radius
    n_sample = min(1000, positions.shape[0])
    sample_distances = np.empty(n_sample, dtype=np.float64)
    
    for i in range(n_sample):
        diff = positions[i] - center_pos
        sample_distances[i] = np.sqrt(np.sum(diff**2))
    
    # Manual sort for Numba compatibility
    for i in range(n_sample - 1):
        min_idx = i
        for j in range(i + 1, n_sample):
            if sample_distances[j] < sample_distances[min_idx]:
                min_idx = j
        if min_idx != i:
            sample_distances[i], sample_distances[min_idx] = sample_distances[min_idx], sample_distances[i]
    
    # Take k-th distance as estimate
    radius_idx = min(k_neighbors, n_sample - 1)
    return sample_distances[radius_idx] * 1.2  # 20% buffer

@njit(parallel=True)
def velocity_smoothing_all_optimized(positions, velocities, local_density, mach,
                                   N_neighbours=32, max_itr=10, tolerance=0.1,
                                   shock_threshold=1.3):
    """Parallel processing of all particles with optimized memory usage."""
    n_particles = positions.shape[0]
    
    # Build spatial grid once for all particles
    max_search_radius = estimate_global_search_radius(positions, N_neighbours * (2**max_itr))
    cell_size = max_search_radius / 4  # 4 cells per max radius
    
    grid_data = build_spatial_grid(positions, cell_size)
    
    # Output arrays
    V_turb = np.zeros((n_particles, 3), dtype=np.float64)
    epsilons_out = np.zeros((n_particles, 3), dtype=np.float64)
    iterations = np.zeros(n_particles, dtype=np.int32)
    
    # Process particles in parallel
    for i in prange(n_particles):
        V_turb[i], epsilons_out[i], iterations[i] = velocity_smoothing_single_optimized(
            i, positions, velocities, local_density, mach, grid_data,
            N_neighbours, max_itr, tolerance, shock_threshold
        )
    
    return V_turb, epsilons_out, iterations

@njit
def estimate_global_search_radius(positions, max_neighbors):
    """Estimate maximum search radius needed globally."""
    n_particles = positions.shape[0]
    if n_particles < max_neighbors:
        # Use bounding box diagonal as estimate
        mins = np.empty(3, dtype=np.float64)
        maxs = np.empty(3, dtype=np.float64)
        
        for dim in range(3):
            mins[dim] = positions[0, dim]
            maxs[dim] = positions[0, dim]
            for i in range(1, n_particles):
                if positions[i, dim] < mins[dim]:
                    mins[dim] = positions[i, dim]
                if positions[i, dim] > maxs[dim]:
                    maxs[dim] = positions[i, dim]
        
        diagonal_sq = 0.0
        for dim in range(3):
            diagonal_sq += (maxs[dim] - mins[dim])**2
        return np.sqrt(diagonal_sq)
    
    # Sample approach for large datasets
    sample_size = min(1000, n_particles)
    max_radius = 0.0
    
    for i in range(sample_size):
        center = positions[i]
        distances = np.empty(sample_size, dtype=np.float64)
        
        for j in range(sample_size):
            diff = positions[j] - center
            distances[j] = np.sqrt(np.sum(diff**2))
        
        # Manual sort
        for p in range(sample_size - 1):
            min_idx = p
            for q in range(p + 1, sample_size):
                if distances[q] < distances[min_idx]:
                    min_idx = q
            if min_idx != p:
                distances[p], distances[min_idx] = distances[min_idx], distances[p]
        
        kth_distance = distances[min(max_neighbors, sample_size - 1)]
        if kth_distance > max_radius:
            max_radius = kth_distance
    
    return max_radius * 1.5  # Safety buffer

# -------------------------
# Optimized main class
# -------------------------
class MeasureVelocityFieldKinematics:
    """Memory and compute optimized version for millions of particles."""
    def __init__(self, positions, velocities, masses, mach, local_density=None,
                 N_neighbours=32, itr=10, tolerance=0.1, shock_threshold=1.3):
        
        self.positions = positions.astype(np.float64)  # Ensure correct dtype
        self.velocities = velocities.astype(np.float64)
        self.mach = mach.astype(np.float64)
        
        # Estimate local density if not provided
        if local_density is None:
            print(f"Estimating local densities for {len(positions):,} particles...")
            self.local_density = self._estimate_densities_optimized(masses, N_neighbours)
        else:
            self.local_density = local_density.astype(np.float64)
        
        # Parameters
        self.N_neighbours = N_neighbours
        self.itr = itr
        self.tolerance = tolerance
        self.shock_threshold = shock_threshold
        
        print(f"Initialized for {len(positions):,} particles")
    
    def _estimate_densities_optimized(self, masses, N_neighbours):
        """Optimized density estimation using spatial grid."""
        # Build coarse grid for density estimation
        n_particles = len(self.positions)
        est_radius = self._estimate_neighbor_radius(N_neighbours)
        
        # Use simple approach for small datasets
        if n_particles < 1000:
            return self._estimate_densities_simple(masses, N_neighbours, est_radius)
        
        grid_data = build_spatial_grid(self.positions, est_radius / 2)
        densities = np.zeros(n_particles, dtype=np.float64)
        
        # Process in chunks to manage memory
        chunk_size = min(10000, n_particles)
        
        for start_idx in range(0, n_particles, chunk_size):
            end_idx = min(start_idx + chunk_size, n_particles)
            
            for i in range(start_idx, end_idx):
                neighbors = query_radius_grid(
                    *grid_data, self.positions, self.positions[i], 
                    est_radius, N_neighbours + 1
                )
                
                if len(neighbors) > 1:
                    neighbor_positions = self.positions[neighbors[1:]]  # Exclude self
                    distances = np.sqrt(np.sum((neighbor_positions - self.positions[i])**2, axis=1))
                    h = np.max(distances) if len(distances) > 0 else est_radius
                    
                    if isinstance(masses, np.ndarray):
                        mass_sum = np.sum(masses[neighbors[1:]])
                    else:
                        mass_sum = masses * len(neighbors[1:])
                    
                    volume = (4.0/3.0) * np.pi * h**3
                    densities[i] = mass_sum / max(volume, 1e-12)
                else:
                    # Fallback for isolated particles
                    densities[i] = masses if np.isscalar(masses) else masses[i]
        
        return densities
    
    def _estimate_densities_simple(self, masses, N_neighbours, est_radius):
        """Simple density estimation for small datasets."""
        n_particles = len(self.positions)
        densities = np.zeros(n_particles, dtype=np.float64)
        
        for i in range(n_particles):
            # Calculate distances to all other particles
            distances = np.sqrt(np.sum((self.positions - self.positions[i])**2, axis=1))
            # Find nearest neighbors (excluding self)
            neighbor_indices = np.argsort(distances)[1:N_neighbours+1]
            
            if len(neighbor_indices) > 0:
                h = distances[neighbor_indices[-1]]  # Distance to furthest neighbor
                
                if isinstance(masses, np.ndarray):
                    mass_sum = np.sum(masses[neighbor_indices])
                else:
                    mass_sum = masses * len(neighbor_indices)
                
                volume = (4.0/3.0) * np.pi * h**3
                densities[i] = mass_sum / max(volume, 1e-12)
            else:
                densities[i] = masses if np.isscalar(masses) else masses[i]
        
        return densities
    
    def _estimate_neighbor_radius(self, N_neighbours):
        """Estimate typical neighbor search radius."""
        n_particles = len(self.positions)
        if n_particles < 1000:
            # Small dataset - use simple approach
            center = np.mean(self.positions, axis=0)
            distances = np.sqrt(np.sum((self.positions - center)**2, axis=1))
            return np.percentile(distances, 90) / 10
        
        # Sample-based estimation for large datasets
        sample_indices = np.random.choice(n_particles, min(500, n_particles), replace=False)
        sample_radii = []
        
        for i in sample_indices[:50]:  # Just use first 50 for speed
            distances = np.sqrt(np.sum((self.positions - self.positions[i])**2, axis=1))
            distances.sort()
            k_idx = min(N_neighbours, len(distances) - 1)
            sample_radii.append(distances[k_idx])
        
        return np.median(sample_radii) * 1.2  # Small buffer
    
    def compute_all(self):
        """
        Run optimized velocity smoothing for all particles.
        Returns: (V_turb, epsilons, iterations)
        """
        print(f"Computing turbulent velocities for {len(self.positions):,} particles...")
        
        return velocity_smoothing_all_optimized(
            self.positions, self.velocities, self.local_density, self.mach,
            self.N_neighbours, self.itr, self.tolerance, self.shock_threshold
        )
