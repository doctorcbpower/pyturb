import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.fft import rfftn, fftn, irfftn, ifftn, fftfreq
from scipy.interpolate import RegularGridInterpolator
import random

class CreateTurbulentVelocityField:
    """
    Implementation of the Dubinski & Narayan (1995) algorithm for generating
    turbulent velocity fields in molecular clouds with Kolmogorov spectrum.
    """
    
    def __init__(self, grid_size=32, rho0=1.0, box_size=1.0, v_turb=1.0, temperature=1.e4, alpha=5./3., seed=None):
        """
        Initialize the turbulent velocity field generator.
        
        Parameters:
        -----------
        grid_size : int
            Number of grid points per dimension (creates grid_size^3 total points)
        box_size : float
            Physical size of the simulation box
        seed : int, optional
            Random seed for reproducible results
        alpha: float, optinonal
            Index of energy spectrum; assume Kolmogorov value of -5/3 as default.
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.alpha = alpha
        
        self.dx = box_size / grid_size
        self.rho0 = rho0
        self.temperature = temperature
        self.v_turb=v_turb
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # Create coordinate grids
        self.x = np.linspace(0, box_size, grid_size, endpoint=False)
        self.y = np.linspace(0, box_size, grid_size, endpoint=False)
        self.z = np.linspace(0, box_size, grid_size, endpoint=False)
        
        # Create 3D coordinate meshgrids
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Initialize velocity field components
        self.vx = None
        self.vy = None
        self.vz = None
        self.vx_turb = None  # Store turbulent component separately
        self.vy_turb = None
        self.vz_turb = None
        
#        # Precompute k-grid for rFFTs
        k = np.fft.fftfreq(self.grid_size, d=self.dx) * 2*np.pi
        kx = k
        ky = k
        kz = k #np.fft.rfftfreq(self.grid_size, d=self.dx) * 2*np.pi
        self.kx, self.ky, self.kz = np.meshgrid(kx, ky, kz, indexing='ij')
        self.K2 = self.kx**2 + self.ky**2 + self.kz**2
        self.K2[0,0,0] = 1.0  # avoid divide by zero
        
        self.velocity_field=None

    def power_spectrum_index(self,alpha=5./3.):
        """ 
        Take as input the energy spectrum index and return the power spectrum index.
        
        Parameters:
        alpha: float
            Energy spectrum index
        Returns:
        float:
            Power spectrum index
        """
        return alpha+2.
        

    def generate_kolmogorov_field(self, energy_spectrum_index=5./3., power_spectrum_index=None, energy_scale=1.0):
        """
        Generate a 3D turbulent velocity field with Kolmogorov spectrum.
        
        Parameters:
        -----------
        alpha : float
            Spectral index (5/3 for Kolmogorov turbulence)
        energy_scale: float
            Normalisation factor for energy of fluid
        Returns:
        --------
        tuple of 3D arrays
            (vx, vy, vz) velocity components
        """
        
        # Create Kolmogorov power spectrum: P(k) ~ k^(-alpha)
        # For velocity field, we want k^(-alpha/2) amplitude
        if power_spectrum_index==None:
            spectral_index=self.power_spectrum_index(energy_spectrum_index)
        else:
            spectral_index=power_spectrum_index
            
        K_mag=np.sqrt(self.K2)
        amplitude=np.zeros_like(K_mag)
        
        mask=K_mag>0
        amplitude[mask] = K_mag[mask]**(-spectral_index/2)
        amplitude[0, 0, 0] = 0  # Set DC component to zero
        
        # Generate three independent velocity components
        velocity_components = []
        
        for _ in range(3):
            # Generate random phases
            phases = np.random.uniform(0, 2*np.pi, self.kx.shape)
            field_k = amplitude * (np.cos(phases) + 1j*np.sin(phases))
            # Transform to real space using irfftn (Hermitian symmetry automatically enforced)
#            velocity_field = np.real(irfftn(field_k, s=(self.grid_size,self.grid_size,self.grid_size)))
            velocity_field = np.real(ifftn(field_k))
            velocity_components.append(velocity_field)
            
#        velocity_components*/= self.grid_size**3  # ensures Parseval's theorem works

        # Make velocity field divergence-free (solenoidal)
        self.vx_turb, self.vy_turb, self.vz_turb = self._make_solenoidal(velocity_components)
        
        # Initially, total velocity is just turbulent component
        self.vx, self.vy, self.vz = self.vx_turb.copy(), self.vy_turb.copy(), self.vz_turb.copy()

       # Scale to desired energy level
        total_energy = np.mean(self.vx**2 + self.vy**2 + self.vz**2)
        scale_factor = np.sqrt(energy_scale / total_energy)
        
        self.vx *= scale_factor
        self.vy *= scale_factor
        self.vz *= scale_factor

        self.velocity_field = np.stack([self.vx, self.vy, self.vz], axis=-1)

        return self.velocity_field

#        return self.vx,self.vy,self.vz
                
    def _make_solenoidal(self, velocity_components):
        """
        Project velocity field to make it divergence-free (solenoidal).
        """
        # Transform velocity components to Fourier space
#        vx_k = rfftn(velocity_components[0])
#        vy_k = rfftn(velocity_components[1])
#        vz_k = rfftn(velocity_components[2])
        vx_k = fftn(velocity_components[0])
        vy_k = fftn(velocity_components[1])
        vz_k = fftn(velocity_components[2])
                  
        # Calculate divergence in k-space
        div_k = 1j * (self.kx * vx_k + self.ky * vy_k + self.kz * vz_k)

        K2_safe = self.K2.copy()
        K2_safe[ K2_safe == 0 ] = 1.0  # avoid divide-by-zero

        # Project out the compressive component
        vx_k_sol = vx_k - 1j * self.kx * div_k / K2_safe
        vy_k_sol = vy_k - 1j * self.ky * div_k / K2_safe
        vz_k_sol = vz_k - 1j * self.kz * div_k / K2_safe
        
        # Set DC component to zero
        vx_k_sol[0, 0, 0] = 0
        vy_k_sol[0, 0, 0] = 0
        vz_k_sol[0, 0, 0] = 0
        
        # Transform back to real space
#        vx_sol = np.real(irfftn(vx_k_sol,s=(self.grid_size,)*3))
#        vy_sol = np.real(irfftn(vy_k_sol,s=(self.grid_size,)*3))
#        vz_sol = np.real(irfftn(vz_k_sol,s=(self.grid_size,)*3))
        vx_sol = np.real(ifftn(vx_k_sol))
        vy_sol = np.real(ifftn(vy_k_sol))
        vz_sol = np.real(ifftn(vz_k_sol))
        
        return vx_sol, vy_sol, vz_sol
    
    def add_bulk_rotation(self, omega_vector, center=None):
        """
        Add rigid body rotation to the velocity field.
        
        Parameters:
        -----------
        omega_vector : array-like
            Angular velocity vector [omega_x, omega_y, omega_z] in rad/time
        center : array-like, optional
            Center of rotation [x0, y0, z0]. If None, uses box center.
            
        Returns:
        --------
        tuple of 3D arrays
            Updated (vx, vy, vz) velocity components including rotation
        """
        if self.vx_turb is None:
            raise ValueError("Generate turbulent field first using generate_kolmogorov_field()")
        
        omega_vector = np.array(omega_vector)
        
        # Set rotation center
        if center is None:
            center = np.array([self.box_size/2, self.box_size/2, self.box_size/2])
        else:
            center = np.array(center)
        
        # Calculate position vectors relative to rotation center
        x_rel = self.X - center[0]
        y_rel = self.Y - center[1] 
        z_rel = self.Z - center[2]
        
        # Calculate rotational velocity: v_rot = omega × r
        # v_rot = omega × (r - r_center)
        vx_rot = omega_vector[1] * z_rel - omega_vector[2] * y_rel
        vy_rot = omega_vector[2] * x_rel - omega_vector[0] * z_rel
        vz_rot = omega_vector[0] * y_rel - omega_vector[1] * x_rel
        
        # Add rotational component to turbulent velocity
        self.vx = self.vx_turb + vx_rot
        self.vy = self.vy_turb + vy_rot
        self.vz = self.vz_turb + vz_rot
        
        return self.vx, self.vy, self.vz
    
    def add_differential_rotation(self, omega_func, center=None):
        """
        Add differential rotation to the velocity field.
        
        Parameters:
        -----------
        omega_func : callable
            Function that takes radius and returns angular velocity omega(r)
            Should accept radius array and return omega array
        center : array-like, optional
            Center of rotation [x0, y0, z0]. If None, uses box center.
            
        Returns:
        --------
        tuple of 3D arrays
            Updated (vx, vy, vz) velocity components including differential rotation
        """
        if self.vx_turb is None:
            raise ValueError("Generate turbulent field first using generate_kolmogorov_field()")
        
        # Set rotation center
        if center is None:
            center = np.array([self.box_size/2, self.box_size/2, self.box_size/2])
        else:
            center = np.array(center)
        
        # Calculate position vectors relative to rotation center
        x_rel = self.X - center[0]
        y_rel = self.Y - center[1]
        z_rel = self.Z - center[2]
        
        # Calculate cylindrical radius (assuming rotation around z-axis)
        R = np.sqrt(x_rel**2 + y_rel**2)
        
        # Avoid division by zero at center
        R_safe = np.where(R > 1e-10, R, 1e-10)
        
        # Get angular velocity as function of radius
        omega_z = omega_func(R_safe)
        
        # Calculate differential rotational velocity in cylindrical coordinates
        # v_phi = omega(R) * R, then convert to Cartesian
        v_phi = omega_z * R
        
        # Convert to Cartesian components
        # v_x = -v_phi * sin(phi) = -v_phi * y/R
        # v_y = +v_phi * cos(phi) = +v_phi * x/R
        vx_rot = -v_phi * y_rel / R_safe
        vy_rot = +v_phi * x_rel / R_safe
        vz_rot = np.zeros_like(vx_rot)  # No z-component for rotation around z-axis
        
        # Handle center point where R=0
        vx_rot = np.where(R > 1e-10, vx_rot, 0)
        vy_rot = np.where(R > 1e-10, vy_rot, 0)
        
        # Add rotational component to turbulent velocity
        self.vx = self.vx_turb + vx_rot
        self.vy = self.vy_turb + vy_rot
        self.vz = self.vz_turb + vz_rot
        
        return self.vx, self.vy, self.vz
    
    def add_velocity_gradient(self, gradient_matrix, center=None):
        """
        Add a linear velocity gradient (shear) to the velocity field.
        
        Parameters:
        -----------
        gradient_matrix : 3x3 array
            Velocity gradient tensor dv_i/dx_j
        center : array-like, optional
            Reference point for gradient. If None, uses box center.
            
        Returns:
        --------
        tuple of 3D arrays
            Updated (vx, vy, vz) velocity components including gradient
        """
        if self.vx_turb is None:
            raise ValueError("Generate turbulent field first using generate_kolmogorov_field()")
        
        gradient_matrix = np.array(gradient_matrix)
        
        # Set reference center
        if center is None:
            center = np.array([self.box_size/2, self.box_size/2, self.box_size/2])
        else:
            center = np.array(center)
        
        # Calculate position vectors relative to center
        x_rel = self.X - center[0]
        y_rel = self.Y - center[1]
        z_rel = self.Z - center[2]
        
        # Apply linear velocity gradient: v_grad = G · (r - r_center)
        vx_grad = (gradient_matrix[0, 0] * x_rel + 
                   gradient_matrix[0, 1] * y_rel + 
                   gradient_matrix[0, 2] * z_rel)
        
        vy_grad = (gradient_matrix[1, 0] * x_rel + 
                   gradient_matrix[1, 1] * y_rel + 
                   gradient_matrix[1, 2] * z_rel)
        
        vz_grad = (gradient_matrix[2, 0] * x_rel + 
                   gradient_matrix[2, 1] * y_rel + 
                   gradient_matrix[2, 2] * z_rel)
        
        # Add gradient component to turbulent velocity
        self.vx = self.vx_turb + vx_grad
        self.vy = self.vy_turb + vy_grad
        self.vz = self.vz_turb + vz_grad
        
        return self.vx, self.vy, self.vz

    def write_to_file(self, filename, center=None):
        """
        Add a linear velocity gradient (shear) to the velocity field.
        
        Parameters:
        -----------
        center : array-like, optional
            Reference point. If None, uses box center.
            
        """
        if self.vx_turb is None:
            raise ValueError("Generate turbulent field first using generate_kolmogorov_field()")

        pos = np.column_stack([self.X.ravel(), self.Y.ravel(), self.Z.ravel()])
        vel = np.column_stack([self.vx.ravel(), self.vy.ravel(), self.vz.ravel()])
    
        # Add small random perturbations to seed turbulence
        vel += np.random.normal(0, self.v_turb, vel.shape)  # v_turb is in km/s
    
        N_gas = self.grid_size*self.grid_size*self.grid_size
        Volume = self.box_size*self.box_size*self.box_size
        
        # Particle masses (uniform)
        mass = np.full(N_gas, self.rho0 * Volume / N_gas)
    
        # Internal energies (uniform temperature)
        # Zero out and set with parameter file during runtime
        u = np.full(N_gas, 0.0) 
    
        # Particle IDs
        ids = np.arange(1, N_gas + 1, dtype=np.uint64)
    
        print(f"Created {N_gas} particles")
        print(f"Box size: {self.box_size} pc")
        print(f"Mean density: {self.rho0} H/cm^3")
        print(f"Initial temperature: {self.temperature} K")
    
        # Write HDF5 file
        with h5py.File(filename, 'w') as f:
            # Header
            header = f.create_group('Header')
            header.attrs['NumFilesPerSnapshot'] = 1
            header.attrs['NumPart_ThisFile'] = [N_gas, 0, 0, 0, 0, 0]
            header.attrs['NumPart_Total'] = [N_gas, 0, 0, 0, 0, 0]
            header.attrs['NumPart_Total_HighWord'] = [0, 0, 0, 0, 0, 0]
            header.attrs['Time'] = 0.0
            header.attrs['Redshift'] = 0.0
            header.attrs['BoxSize'] = self.box_size
            header.attrs['Omega0'] = 0.0
            header.attrs['OmegaLambda'] = 0.0
            header.attrs['HubbleParam'] = 1.0
            header.attrs['Flag_Sfr'] = 0
            header.attrs['Flag_Cooling'] = 0
            header.attrs['Flag_StellarAge'] = 0
            header.attrs['Flag_Metals'] = 0
            header.attrs['Flag_DoublePrecision'] = 0
            header.attrs['MassTable'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # Gas particles
            part0 = f.create_group('PartType0')
            part0.create_dataset('Coordinates', data=pos)
            part0.create_dataset('Velocities', data=vel)
            part0.create_dataset('Masses', data=mass)
            part0.create_dataset('InternalEnergy', data=u)
            part0.create_dataset('ParticleIDs', data=ids)
        print(f"Initial conditions written to {filename}")

if __name__ == "__main__":
    # Create turbulent velocity field
    turb = CreateTurbulentVelocityField(grid_size=32, box_size=20.0, seed=42)
    # Generate Kolmogorov turbulence
    print("Generating Kolmogorov turbulent velocity field...")
    turb.generate_kolmogorov_field(alpha=5/3)
    turb.write_to_file("./turbulent_ics.hdf5")
