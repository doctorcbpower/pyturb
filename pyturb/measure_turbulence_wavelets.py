import numpy as np
from scipy import ndimage
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy.fft import rfftn, irfftn, fftshift

class MeasureVelocityFieldWavelets:
    """
    Tools to measure properties of a velocity field using wavelets.
    Includes the power spectrum.

    Attributes
    ----------
    alpha_3d : float
        Normalization constant for the 3D Mexican hat wavelet.
    """
    def __init__(self, scales=None, n_scales=10):
        """
        Parameters
        ----------
        scales : array_like, optional
            Wavelet scales to use. If None, creates logarithmically spaced scales.
        n_scales : int
            Number of scales if scales not provided.
        """
        if scales is None:
            # Default scale range - adjust based on your grid resolution
            self.scales = np.logspace(0, 2, n_scales)  # 1 to 100 grid units
        else:
            self.scales = np.array(scales)
        
        # Normalization constant for 3D Mexican hat
        self.alpha_3d = 2 * np.sqrt(5) / (np.pi**(3/4))

    def get_wavelet_coefficients_fft(self, field, box_size):
        # field: (Nx,Ny,Nz) or (Nx,Ny,Nz,3)
        Nx, Ny, Nz = field.shape[:3]
        dx = box_size / Nx

        # pre-allocate coefficients (use float32 if ok)
        dtype = np.float32 if field.dtype == np.float32 else np.float64
        if field.ndim == 4:
            coefficients = np.zeros((Nx, Ny, Nz, 3, len(self.scales)), dtype=dtype)
        else:
            coefficients = np.zeros((Nx, Ny, Nz, len(self.scales)), dtype=dtype)

        # precompute FFTs of components if vector
        if field.ndim == 4:
            F_components = [rfftn(field[...,c]) for c in range(3)]
        else:
            F_field = rfftn(field)

        kernel_F_cache = {} 

        for i, scale in enumerate(self.scales):
            scale_grid = scale / dx
            
            # Fix: Ensure kernel size is always reasonable
            max_kernel_size = min(Nx, Ny, Nz) - 2  # Leave some margin
            theoretical_size = int(6 * scale_grid)
            kernel_size = min(theoretical_size, max_kernel_size)

            if kernel_size < 3:
                continue

            # Make kernel size odd for symmetry
            if kernel_size % 2 == 0:
                kernel_size -= 1
        
            # Now build kernel - guaranteed to fit
            half_size = kernel_size // 2

            cache_key = (scale_grid, kernel_size, Nx, Ny, Nz)
        
            if cache_key in kernel_F_cache:
                Fk = kernel_F_cache[cache_key]
            else:
                # build kernel (use r_sq with broadcasting instead of meshgrid)
                xk = np.arange(-half_size, half_size + 1) * dx
                yk = xk
                zk = xk
                r_sq = (xk[:,None,None]**2) + (yk[None,:,None]**2) + (zk[None,None,:]**2)
                kernel = self.mexican_hat_3d_from_rsq(r_sq, scale_grid)

                # check kernel fits
                s0, s1, s2 = kernel.shape
                if s0 > Nx or s1 > Ny or s2 > Nz:
                    # truncate kernel to fit the box
                    trim0 = (s0 - Nx) // 2
                    trim1 = (s1 - Ny) // 2
                    trim2 = (s2 - Nz) // 2
                    kernel = kernel[trim0:s0-trim0, trim1:s1-trim1, trim2:s2-trim2]
                    s0, s1, s2 = kernel.shape

                # pad kernel into full box
                pad_kernel = np.zeros((Nx,Ny,Nz), dtype=kernel.dtype)
                start = [(Nx - s0)//2, (Ny - s1)//2, (Nz - s2)//2]
                pad_kernel[start[0]:start[0]+s0,
                            start[1]:start[1]+s1,
                            start[2]:start[2]+s2] = kernel

                # center at origin for FFT convolution
                pad_kernel = fftshift(pad_kernel)

                Fk = rfftn(pad_kernel)

                kernel_F_cache[cache_key] = Fk
                
            # convolve in Fourier domain (re-use component FFTs)
            if field.ndim == 4:
                for c in range(3):
                    coefficients[..., c, i] = irfftn(F_components[c] * Fk, s=(Nx,Ny,Nz))
            else:
                coefficients[..., i] = irfftn(F_field * Fk, s=(Nx,Ny,Nz))

        return coefficients
        
    def get_wavelet_coefficients(self, field, box_size):
        """
        Compute local wavelet coefficients at each point.
        
        Parameters
        ----------
        field : ndarray
            3D velocity field component or magnitude
        box_size : float
            Physical size of simulation box
            
        Returns
        -------
        coefficients : ndarray
            For scalar fields (ndim=3): shape (Nx, Ny, Nz, n_scales)
            For vector fields (ndim=4): shape (Nx, Ny, Nz, 3, n_scales)
        """

        if not ((field.ndim == 3) or (field.ndim == 4 and field.shape[-1] == 3)):
            raise ValueError("field must have shape (Nx, Ny, Nz) or (Nx, Ny, Nz, 3)")

        Nx, Ny, Nz = field.shape[:3]
        dx = box_size / Nx
        
        # Physical coordinates
        x = np.linspace(-box_size/2, box_size/2, Nx)
        y = np.linspace(-box_size/2, box_size/2, Ny)
        z = np.linspace(-box_size/2, box_size/2, Nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        if field.ndim == 4:
            coefficients = np.zeros((Nx, Ny, Nz, 3, len(self.scales)), dtype=field.dtype)
        elif field.ndim == 3:
            coefficients = np.zeros((Nx, Ny, Nz, len(self.scales)), dtype=field.dtype)
        else:
            raise ValueError(f"Unsupported field ndim={field.ndim}")

        for i, scale in enumerate(self.scales):
            # Convert scale from physical units to grid units
            scale_grid = scale / dx

            # Kernel size
            kernel_size = min(int(6 * scale_grid), min(Nx, Ny, Nz) // 2)
            if kernel_size < 3:
                continue

            # Build kernel grid
            x_k = np.arange(-kernel_size, kernel_size + 1) * dx
            y_k = np.arange(-kernel_size, kernel_size + 1) * dx
            z_k = np.arange(-kernel_size, kernel_size + 1) * dx
            X_k, Y_k, Z_k = np.meshgrid(x_k, y_k, z_k, indexing="ij")

            # Mexican hat kernel
            kernel = self.mexican_hat_3d(X_k, Y_k, Z_k, scale_grid)

            if field.ndim == 4:  # vector field (Nx, Ny, Nz, 3)
                for comp in range(3):
                    coefficients[..., comp, i] = ndimage.convolve(field[..., comp], kernel, mode="wrap")
            elif field.ndim == 3:  # scalar field (Nx, Ny, Nz)
                coefficients[..., i] = ndimage.convolve(field, kernel, mode="wrap")
            else:
                raise ValueError(f"Unsupported field ndim={field.ndim}")

        return coefficients

        
    def mexican_hat_3d_from_rsq(self,rsq, sigma):
        """
        3D isotropic Mexican hat (Laplacian of Gaussian) wavelet.
    
        Parameters
        ----------
        rsq : ndarray
            Squared radius (x^2 + y^2 + z^2).
        sigma : float
            Scale parameter.
    
        Returns
        -------
        psi : ndarray
            Wavelet evaluated at rsq.
        """
        factor = (3.0 - rsq / sigma**2)
        return factor * np.exp(-0.5 * rsq / sigma**2) 
    
    def mexican_hat_3d(self, x, y, z, scale):
        """
        3D Mexican Hat (2nd derivative of Gaussian) wavelet.
        
        Parameters
        ----------
        x, y, z : ndarray
            Coordinate arrays
        scale : float
            Wavelet scale parameter
            
        Returns
        -------
        psi : ndarray
            Wavelet values
        """
        r_sq = (x**2 + y**2 + z**2) / scale**2
        
        # Mexican hat: (3 - r²) * exp(-r²/2)
        # psi = self.alpha_3d * (3 - r_sq) * np.exp(-r_sq / 2) / (scale**(3/2))
        psi = self.alpha_3d * (3 - r_sq/scale**2) * np.exp(-0.5 * r_sq/scale**2) / (scale**(3/2))
        return psi
    
    def local_power_spectrum(self, field, box_size, radial_bins=None):
        """
        Compute local wavelet power spectra.

        Parameters
        ----------
        field : ndarray
            3D field to analyze
        box_size : float
            Physical box size
        radial_bins : array_like, optional
            Radial bins for averaging. If None, uses whole box.

        Returns
        -------
        k_equiv : ndarray
            Equivalent wavenumbers
        power_local : ndarray
            Local power spectra, shape (n_radial_bins, n_scales) or (n_scales,)
        """
        coefficients = self.get_wavelet_coefficients_fft(field, box_size)

        # Equivalent wavenumbers (following Shi et al.)
        k_equiv = np.sqrt(2 + 3/2) / (self.scales * box_size / field.shape[0])

        # Local power: |coefficients|²
        power_local = coefficients**2

        if radial_bins is not None:
            # Compute power in radial bins
            center = np.array(field.shape) // 2
            Nx, Ny, Nz = field.shape

            # Create radial coordinate array
            i_coords, j_coords, k_coords = np.ogrid[:Nx, :Ny, :Nz]
            r_coords = np.sqrt((i_coords - center[0])**2 + 
                            (j_coords - center[1])**2 + 
                            (k_coords - center[2])**2)

            power_radial = []
            for i in range(len(radial_bins) - 1):
                mask = (r_coords >= radial_bins[i]) & (r_coords < radial_bins[i+1])
                if np.any(mask):
                    power_bin = np.median(power_local[mask], axis=0)
                else:
                    power_bin = np.zeros(len(self.scales))
                power_radial.append(power_bin)
            
            return k_equiv, np.array(power_radial)
        else:
            # Global average
            power_avg = np.mean(power_local, axis=(0, 1, 2))
            return k_equiv, power_avg
#
#def compare_interpolation_methods_wavelet(original_field, interpolated_fields, 
#                                        method_names, box_size):
#    """
#    Compare interpolation methods using wavelet analysis.
#    
#    Parameters
#    ----------
#    original_field : ndarray
#        Ground truth field
#    interpolated_fields : list of ndarray
#        Fields from different interpolation methods
#    method_names : list of str
#        Names of interpolation methods
#    box_size : float
#        Physical box size
#    """
#    
#    # Initialize wavelet analyzer
#    wavelet = MexicanHatWavelet3D()
#    
#    # Radial bins for analysis
#    max_r = min(original_field.shape) // 2
#    radial_bins = np.linspace(0, max_r, 5)
#    
#    # Analyze original field
#    k_orig, power_orig = wavelet.local_power_spectrum(
#        np.linalg.norm(original_field, axis=-1) if original_field.ndim == 4 else original_field,
#        box_size, radial_bins
#    )
#    
#    # Compare with interpolated fields
#    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#    axes = axes.ravel()
#    
#    for bin_idx in range(len(radial_bins) - 1):
#        ax = axes[bin_idx]
#        
#        # Plot original
#        ax.loglog(k_orig, power_orig[bin_idx], 'k-', linewidth=2, 
#                 label='Original', alpha=0.8)
#        
#        # Plot interpolated methods
#        colors = ['red', 'blue', 'green', 'orange', 'purple']
#        for i, (field, name) in enumerate(zip(interpolated_fields, method_names)):
#            field_magnitude = np.linalg.norm(field, axis=-1) if field.ndim == 4 else field
#            k_interp, power_interp = wavelet.local_power_spectrum(
#                field_magnitude, box_size, radial_bins
#            )
#            
#            ax.loglog(k_interp, power_interp[bin_idx], '--', 
#                     color=colors[i % len(colors)], linewidth=2, 
#                     label=name, alpha=0.7)
#        
#        # Kolmogorov slope reference
#        k_ref = k_orig[k_orig > 0]
#        power_ref = k_ref**(-5/3) * power_orig[bin_idx, 0] * k_orig[0]**(5/3)
#        ax.loglog(k_ref, power_ref, ':', color='gray', alpha=0.5, label='-5/3 slope')
#        
#        ax.set_xlabel('k [1/L]')
#        ax.set_ylabel('Power')
#        ax.set_title(f'Radial bin {bin_idx+1}: r = {radial_bins[bin_idx]:.1f}-{radial_bins[bin_idx+1]:.1f}')
#        ax.legend()
#        ax.grid(True, alpha=0.3)
#    
#    plt.tight_layout()
#    return fig
#
#def turbulence_structure_preservation_metric(original_field, interpolated_field, 
#                                           box_size, scales=None):
#    """
#    Quantify how well turbulence structure is preserved across scales.
#    
#    Parameters
#    ----------
#    original_field : ndarray
#        Ground truth field
#    interpolated_field : ndarray
#        Interpolated field
#    box_size : float
#        Physical box size
#    scales : array_like, optional
#        Scales to analyze
#        
#    Returns
#    -------
#    structure_score : float
#        Structure preservation score (higher is better)
#    scale_scores : ndarray
#        Score for each scale
#    """
#    
#    wavelet = MexicanHatWavelet3D(scales=scales)
#    
#    # Get field magnitudes if vector fields
#    if original_field.ndim == 4:
#        orig_mag = np.linalg.norm(original_field, axis=-1)
#        interp_mag = np.linalg.norm(interpolated_field, axis=-1)
#    else:
#        orig_mag = original_field
#        interp_mag = interpolated_field
#    
#    # Wavelet transforms
#    orig_coeffs = wavelet.wavelet_transform_local(orig_mag, box_size)
#    interp_coeffs = wavelet.wavelet_transform_local(interp_mag, box_size)
#    
#    # Compute correlation at each scale
#    scale_scores = np.zeros(len(wavelet.scales))
#    
#    for i in range(len(wavelet.scales)):
#        orig_scale = orig_coeffs[:, :, :, i].ravel()
#        interp_scale = interp_coeffs[:, :, :, i].ravel()
#        
#        # Remove zero variance scales
#        if np.std(orig_scale) > 1e-10 and np.std(interp_scale) > 1e-10:
#            # Normalized correlation
#            correlation = np.corrcoef(orig_scale, interp_scale)[0, 1]
#            scale_scores[i] = max(0, correlation)  # Clip negative correlations
#        else:
#            scale_scores[i] = 0.0
#    
#    # Overall structure score (weighted by scale importance)
#    # Give more weight to energy-containing scales (intermediate scales)
#    weights = np.exp(-(np.log(wavelet.scales) - np.log(np.median(wavelet.scales)))**2)
#    weights /= np.sum(weights)
#    
#    structure_score = np.sum(weights * scale_scores)
#    
#    return structure_score, scale_scores
#
## Example usage function
#def validate_interpolation_with_wavelets(original_velocity, particle_positions, 
#                                       particle_velocities, grid_size, box_size):
#    """
#    Complete validation pipeline using wavelet analysis.
#    """
#    
#    print("=== Wavelet-based Interpolation Validation ===\n")
#    
#    # Test different interpolation methods (you'd implement these)
#    methods = {
#        'Voronoi': lambda: assign_voronoi_basic(particle_positions, particle_velocities, grid_size, box_size),
#        'Gaussian-weighted': lambda: assign_voronoi_optimized(particle_positions, particle_velocities, grid_size, box_size),
#        'Cloud-in-Cell': lambda: cloud_in_cell_interpolation(particle_positions, particle_velocities, grid_size, box_size)
#    }
#    
#    interpolated_fields = []
#    method_names = []
#    structure_scores = []
#    
#    for name, method_func in methods.items():
#        try:
#            interp_field = method_func()
#            interpolated_fields.append(interp_field)
#            method_names.append(name)
#            
#            # Compute structure preservation score
#            score, scale_scores = turbulence_structure_preservation_metric(
#                original_velocity, interp_field, box_size
#            )
#            structure_scores.append(score)
#            
#            print(f"{name:20s}: Structure Score = {score:.3f}")
#            
#        except Exception as e:
#            print(f"Failed to test {name}: {e}")
#    
#    # Visual comparison
#    if interpolated_fields:
#        fig = compare_interpolation_methods_wavelet(
#            original_velocity, interpolated_fields, method_names, box_size
#        )
#        plt.show()
#    
#    # Rank methods
#    if structure_scores:
#        best_idx = np.argmax(structure_scores)
#        print(f"\nBest method: {method_names[best_idx]} (score: {structure_scores[best_idx]:.3f})")
#    
#    return structure_scores, method_names
#
#    import matplotlib.pyplot as plt
