import h5py
import numpy as np
from scipy.fft import fftn, fftfreq

class MeasureVelocityField:
    """
    Tools to measure properties of a velocity field.
    Includes the power spectrum.
    """
    def __init__(self):
        pass
        
    def compute_power_spectrum(self, velocity_field, box_size, component='energy',
                              method='radial', k_bins=None, normalize=True):
        """
        Compute power spectrum of velocity field with multiple analysis options.
        
        Parameters:
        -----------
        velocity_field: ndarray - velocity field on a regular grid with shape (Nx,Ny,Nz,3)
        component : str - which component to analyze ('vx', 'vy', 'vz', 'total_energy')
        method : str - 'radial' for spherically averaged, 'cylindrical' for cylindrically 
            averaged, '1d_x', '1d_y', '1d_z' for 1D power spectra along specific axes
        k_bins : array-like, optional - custom k bins. If None, uses automatic binning.
        normalize : bool - whether to normalize by volume and apply proper scaling
            
        Returns:
        --------
        tuple - (k_values, power_spectrum, error_bars) if applicable
        """
       
        if velocity_field.ndim != 4 or velocity_field.shape[-1] != 3:
            raise ValueError("velocity_field must have shape (Nx, Ny, Nz, 3)")

        # Extract components
        vx = velocity_field[..., 0]
        vy = velocity_field[..., 1]
        vz = velocity_field[..., 2]

        # Choose field to analyze
        if component == 'vx':
            field = vx
        elif component == 'vy':
            field = vy
        elif component == 'vz':
            field = vz
        elif component == 'energy':
            field = (vx**2 + vy**2 + vz**2)
        elif component == 'kinetic_energy':
            field = 0.5 * (vx**2 + vy**2 + vz**2)
        else:
            raise ValueError("Invalid component")
            
        # Get number of dimensions along each axis
        Nx,Ny,Nz,_=velocity_field.shape
        dx=box_size/Nx

        # Create grids
        kx = fftfreq(Nx,d=dx) * 2 * np.pi
        ky = fftfreq(Ny,d=dx) * 2 * np.pi
        kz = fftfreq(Nz,d=dx) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        
        if method == 'radial':
            return self._compute_radial_spectrum(field, KX, KY, KZ, dx, box_size, k_bins, normalize)
#        elif method == 'cylindrical':
#            return self._compute_cylindrical_spectrum(field, KX, KY, KZ, k_bins, normalize)
#        elif method in ['1d_x', '1d_y', '1d_z']:
#            return self._compute_1d_spectrum(field, method, normalize)
        else:
            raise ValueError("Invalid method")
    
    def _compute_radial_spectrum(self, field, KX, KY, KZ, dx, box_size, k_bins, normalize):
        """Compute spherically averaged power spectrum."""
        # Transform to Fourier space
        field_k = fftn(field)
        power_3d = np.abs(field_k)**2
        
        # Normalize by volume if requested
        if normalize:
            Ntotal = field.size
            power_3d *= (dx * dx * dx)**2 / (box_size * box_size * box_size)
        
        # Calculate radial wavenumber
        K = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        # Set up k bins
        if k_bins is None:
            k_max = np.min([np.max(np.abs(KX)), np.max(np.abs(KY)), np.max(np.abs(KZ))])
            k_min = 2.*np.pi/box_size
            nbins = 50
            k_bins = np.logspace(np.log10(k_min),np.log10(k_max), nbins)

        # Compute radially averaged spectrum
        power_1d = np.zeros(len(k_bins)-1)
        k_centers = np.sqrt(k_bins[1:] * k_bins[:-1])  # Geometric mean
        errors = np.zeros_like(power_1d)
        counts = np.zeros_like(power_1d)
        
        for i in range(len(k_bins)-1):
            mask = (K >= k_bins[i]) & (K < k_bins[i+1]) & (K > 0)
            if np.any(mask):
                power_values = power_3d[mask]
                power_1d[i] = np.mean(power_values)
                counts[i] = np.sum(mask)

                if len(power_values)>1:
                    errors[i] = np.std(power_values)/np.sqrt(len(power_values))
                else:
                    errors[i]=0

        valid=counts>0

        return k_centers[valid], power_1d[valid], errors[valid]

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
