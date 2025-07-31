import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftfreq, fftshift
from scipy.interpolate import RegularGridInterpolator
import random

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
