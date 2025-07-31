import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftfreq, fftshift
from scipy.interpolate import RegularGridInterpolator
import random

#def comprehensive_power_spectrum_demo():
#    """
#    Comprehensive demonstration of power spectrum analysis tools.
#    """
#    print("Comprehensive Power Spectrum Analysis Demo")
#    print("=" * 45)
#    
#    # Create shearing box with turbulence
#    shear_box = ShearingBoxTurbulence(
#        grid_size=128,
#        box_size=1.0,
#        shear_rate=1.0,
#        seed=42
#    )
#    
#    # Generate turbulent field
#    print("Generating turbulent field...")
#    shear_box.generate_shearing_box_turbulence(alpha=5/3)
#    
#    # Evolve briefly to develop some structure
#    shear_box.evolve_shearing_box(0.05, 5, remap_frequency=2)
#    
#    # 1. Radial Power Spectrum
#    print("\n1. Computing radial power spectrum...")
#    k_rad, P_rad, errors = shear_box.compute_power_spectrum(
#        component='total_energy', method='radial', normalize=True
#    )
#    
#    # 2. Cylindrical Power Spectrum (good for shearing systems)
#    print("2. Computing cylindrical power spectrum...")
#    k_cyl, P_cyl, _ = shear_box.compute_power_spectrum(
#        component='total_energy', method='cylindrical', normalize=True
#    )
#    
#    # 3. 1D Power Spectra
#    print("3. Computing 1D power spectra...")
#    k_1d_x, P_1d_x, _ = shear_box.compute_power_spectrum(
#        component='total_energy', method='1d_x', normalize=True
#    )
#    k_1d_y, P_1d_y, _ = shear_box.compute_power_spectrum(
#        component='total_energy', method='1d_y', normalize=True
#    )
#    
#    # 4. Individual velocity components
#    print("4. Computing component-wise spectra...")
#    k_vx, P_vx, _ = shear_box.compute_power_spectrum(component='vx', method='radial')
#    k_vy, P_vy, _ = shear_box.compute_power_spectrum(component='vy', method='radial')
#    k_vz, P_vz, _ = shear_box.compute_power_spectrum(component='vz', method='radial')
#    
#    # 5. Structure Functions
#    print("5. Computing velocity structure functions...")
#    struct_funcs = shear_box.compute_velocity_structure_functions(n_lags=15)
#    
#    # 6. Anisotropic Analysis
#    print("6. Computing anisotropic power spectrum...")
#    aniso_data = shear_box.compute_anisotropic_spectrum(n_angles=8)
#    
#    # Create comprehensive plots
#    fig = plt.figure(figsize=(16, 12))
#    
#    # Plot 1: Radial vs Cylindrical Spectra
#    ax1 = plt.subplot(2, 3, 1)
#    mask_rad = (k_rad > 0) & (P_rad > 0)
#    mask_cyl = (k_cyl > 0) & (P_cyl > 0)
#    
#    plt.loglog(k_rad[mask_rad], P_rad[mask_rad], 'b-', linewidth=2, 
#               label='Radial Average')
#    plt.loglog(k_cyl[mask_cyl], P_cyl[mask_cyl], 'r--', linewidth=2, 
#               label='Cylindrical Average')
#    
#    # Plot Kolmogorov reference
#    if np.any(mask_rad):
#        k_ref = k_rad[mask_rad]
#        P_ref = P_rad[mask_rad][5] * (k_ref / k_rad[mask_rad][5])**(-5/3)
#        plt.loglog(k_ref, P_ref, 'k:', alpha=0.7, label=r'$k^{-5/3}$ (Kolmogorov)')
#    
#    plt.xlabel('Wavenumber k')
#    plt.ylabel('Power Spectrum P(k)')
#    plt.title('Radial vs Cylindrical Averaging')
#    plt.legend()
#    plt.grid(True, alpha=0.3)
#    
#    # Plot 2: 1D Spectra Comparison
#    ax2 = plt.subplot(2, 3, 2)
#    mask_x = (k_1d_x > 0) & (P_1d_x > 0)
#    mask_y = (k_1d_y > 0) & (P_1d_y > 0)
#    
#    plt.loglog(k_1d_x[mask_x], P_1d_x[mask_x], 'b-', label='x-direction')
#    plt.loglog(k_1d_y[mask_y], P_1d_y[mask_y], 'r-', label='y-direction')
#    plt.xlabel('Wavenumber k')
#    plt.ylabel('1D Power Spectrum')
#    plt.title('Directional 1D Spectra')
#    plt.legend()
#    plt.grid(True, alpha=0.3)
#    
#    # Plot 3: Component-wise Spectra
#    ax3 = plt.subplot(2, 3, 3)
#    mask_vx = (k_vx > 0) & (P_vx > 0)
#    mask_vy = (k_vy > 0) & (P_vy > 0)
#    mask_vz = (k_vz > 0) & (P_vz > 0)
#    
#    plt.loglog(k_vx[mask_vx], P_vx[mask_vx], 'b-', label=r'$v_x

class MeasureVelocityField:
    """
    Tools to measure properties of a velocity field.
    Includes the power spectrum.
    """
    def __init__(self, grid_size=128):
        """        
        Parameters:
        -----------
        grid_size : int 
            Number of grid points per dimension. Assumed to be a single int.
       """
        self.Nx = self.Ny = self.Nz = grid_size
                   
            
#        # Create coordinate grids (shearing box coordinates)
#        self.x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx, endpoint=False)
#        self.y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny, endpoint=False)
#        self.z = np.linspace(-self.Lz/2, self.Lz/2, self.Nz, endpoint=False)
#        
#        # Create 3D coordinate meshgrids
#        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
#        
#         # Initialize velocity field components
#        self.vx = None
#        self.vy = None
#        self.vz = None
#        self.vx_turb = None  # Store turbulent component
#        self.vy_turb = None
#        self.vz_turb = None
#        
#        # Store initial conditions for remapping
#        self.vx_initial = None
#        self.vy_initial = None
#        self.vz_initial = None

    def compute_power_spectrum(self, component='total_energy', method='radial',
                              k_bins=None, normalize=True):
        """
        Compute power spectrum of velocity field with multiple analysis options.
        
        Parameters:
        -----------
        component : str
            Which component to analyze ('vx', 'vy', 'vz', 'total_energy', 'kinetic_energy')
        method : str
            'radial' for spherically averaged, 'cylindrical' for cylindrically averaged,
            '1d_x', '1d_y', '1d_z' for 1D power spectra along specific axes
        k_bins : array-like, optional
            Custom k bins. If None, uses automatic binning.
        normalize : bool
            Whether to normalize by volume and apply proper scaling
            
        Returns:
        --------
        tuple
            (k_values, power_spectrum, error_bars) if applicable
        """
        if self.vx is None:
            raise ValueError("Generate velocity field first")
        
        # Get velocity components (remove background shear from vy)
        vx = self.vx
        vy = self.vy
        vz = self.vz
        
        # Choose field to analyze
        if component == 'vx':
            field = vx
        elif component == 'vy':
            field = vy
        elif component == 'vz':
            field = vz
        elif component == 'total_energy':
            field = 0.5 * (vx**2 + vy**2 + vz**2)
        elif component == 'kinetic_energy':
            # Total kinetic energy density
            field = vx**2 + vy**2 + vz**2
        else:
            raise ValueError("Invalid component")
        
        self.Nx=np.int32(self.N_gas**(1./3.))
        self.dx=self.box_size/self.Nx
        
        # Create frequency grids
        kx = fftfreq(self.Nx,d=self.dx) * 2 * np.pi
        ky = fftfreq(self.Nx,d=self.dx) * 2 * np.pi
        kz = fftfreq(self.Nx,d=self.dx) * 2 * np.pi
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        
        if method == 'radial':
            return self._compute_radial_spectrum(field, KX, KY, KZ, k_bins, normalize)
        elif method == 'cylindrical':
            return self._compute_cylindrical_spectrum(field, KX, KY, KZ, k_bins, normalize)
        elif method in ['1d_x', '1d_y', '1d_z']:
            return self._compute_1d_spectrum(field, method, normalize)
        else:
            raise ValueError("Invalid method")
    
    def _compute_radial_spectrum(self, field, KX, KY, KZ, k_bins, normalize):
        """Compute spherically averaged power spectrum."""
        # Transform to Fourier space
        field_k = fftn(field)
        power_3d = np.abs(field_k)**2
        
        # Normalize by volume if requested
        if normalize:
            power_3d *= (self.dx * self.dx * self.dx)**2 / (self.box_size * self.box_size * self.box_size)
        
        # Calculate radial wavenumber
        K = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        # Set up k bins
        if k_bins is None:
            k_max = np.min([np.max(np.abs(KX)), np.max(np.abs(KY)), np.max(np.abs(KZ))])
            k_bins = np.logspace(np.log10(2*np.pi/max(self.box_size, self.box_size, self.box_size)),
                               np.log10(k_max), 50)
        
        # Compute radially averaged spectrum
        power_1d = np.zeros(len(k_bins)-1)
        k_centers = np.sqrt(k_bins[1:] * k_bins[:-1])  # Geometric mean
        counts = np.zeros(len(k_bins)-1)
        
        for i in range(len(k_bins)-1):
            mask = (K >= k_bins[i]) & (K < k_bins[i+1]) & (K > 0)
            if np.any(mask):
                power_1d[i] = np.mean(power_3d[mask])
                counts[i] = np.sum(mask)
        
        # Calculate error bars (standard error of mean)
        errors = np.zeros_like(power_1d)
        for i in range(len(k_bins)-1):
            mask = (K >= k_bins[i]) & (K < k_bins[i+1]) & (K > 0)
            if np.sum(mask) > 1:
                errors[i] = np.std(power_3d[mask]) / np.sqrt(np.sum(mask))
        
        return k_centers, power_1d, errors
    
    def _compute_cylindrical_spectrum(self, field, KX, KY, KZ, k_bins, normalize):
        """Compute cylindrically averaged power spectrum (useful for shearing box)."""
        # Transform to Fourier space
        field_k = fftn(field)
        power_3d = np.abs(field_k)**2
        
        if normalize:
            power_3d *= (self.dx * self.dx * self.dx)**2 / (self.box_size * self.box_size * self.box_size)

        # Calculate cylindrical coordinates (kx-ky plane, separate kz)
        K_perp = np.sqrt(KX**2 + KY**2)  # Perpendicular to shear direction
        
        if k_bins is None:
            k_max = np.max(K_perp)
            k_bins = np.logspace(np.log10(2*np.pi/max(self.box_size, self.box_size)),
                               np.log10(k_max), 30)
        
        power_1d = np.zeros(len(k_bins)-1)
        k_centers = np.sqrt(k_bins[1:] * k_bins[:-1])
        
        for i in range(len(k_bins)-1):
            mask = (K_perp >= k_bins[i]) & (K_perp < k_bins[i+1]) & (K_perp > 0)
            if np.any(mask):
                power_1d[i] = np.mean(power_3d[mask])
        
        return k_centers, power_1d, None
    
    def _compute_1d_spectrum(self, field, direction, normalize):
        """Compute 1D power spectrum along specific axis."""
        if direction == '1d_x':
            # Average over y,z then take 1D FFT along x
            field_1d = np.mean(field, axis=(1, 2))
            k_vals = fftfreq(self.Nx, d=self.dx) * 2 * np.pi
        elif direction == '1d_y':
            field_1d = np.mean(field, axis=(0, 2))
            k_vals = fftfreq(self.Nx, d=self.dx) * 2 * np.pi
        elif direction == '1d_z':
            field_1d = np.mean(field, axis=(0, 1))
            k_vals = fftfreq(self.Nx, d=self.dx) * 2 * np.pi
        
        # Compute 1D FFT
        field_k_1d = np.fft.fft(field_1d)
        power_1d = np.abs(field_k_1d)**2
        
        if normalize:
            power_1d *= len(field_1d) * (k_vals[1] - k_vals[0])
        
        # Take positive frequencies only
        n_pos = len(k_vals) // 2
        k_positive = k_vals[1:n_pos+1]  # Exclude k=0 and negative frequencies
        power_positive = power_1d[1:n_pos+1]
        
        return k_positive, power_positive, None
    
    def read_from_file(self, filename):
        """        
        Parameters:
        -----------
            filename: name of HDF5 file containing data
            
        """
         
        # Write HDF5 file
        with h5py.File(filename, 'r') as f:
            print(f"Reading data from {filename}")
            self.N_gas=f['Header'].attrs['NumPart_Total'][0]
            self.time=f['Header'].attrs['Time']
            self.box_size=f['Header'].attrs['BoxSize']
            print(f"Read {self.N_gas} particles")
            print(f"Box size: {self.box_size} pc")
            print(f"Time: {self.time} pc")
            pos=f['PartType0/Coordinates'][()]
            vel=f['PartType0/Velocities'][()]

        self.x=pos[:,0]
        self.y=pos[:,1]
        self.z=pos[:,2]
        self.vx=vel[:,0]
        self.vy=vel[:,1]
        self.vz=vel[:,2]
    
# Run demonstrations
if __name__ == "__main__":
    analysis=MeasureVelocityField(grid_size=128)
    analysis.read_from_file('./turbulent_ics.hdf5')
    k_rad, P_rad, errors = analysis.compute_power_spectrum(
        component='total_energy',
        method='radial',
        normalize=True)
