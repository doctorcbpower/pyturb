# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
from scipy import spatial
import numpy as np
import pyturb
from time import sleep, perf_counter
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate and analyse turbulent velocity fields")
    parser.add_argument("--grid", type=int, nargs="+", default=[32, 64, 128, 256],
                        help="Grid resolutions to generate")
    parser.add_argument("--box", type=float, default=10.0,
                        help="Box size")
    parser.add_argument("--vturb", type=float, default=10.0,
                        help="Turbulent velocity scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--outfile", type=str, default="turbulent_ics.hdf5",
                        help="Output HDF5 file")
    args = parser.parse_args()

    ###
    resolution = args.grid
    seed = np.ones(len(resolution),dtype=np.int64)*args.seed
    vel_field = [None]*len(resolution)
    vel_field_smooth = [None]*len(resolution)
    box_size = args.box_size
    
    # Tools to measure velocity fields
    mturb = pyturb.MeasureVelocityField()
    # Tools to generate gridded velocity fields given particle data
    gturb = pyturb.GriddingTools()

    for i,res in enumerate(resolution):
        tstart=perf_counter()
        # Generate a velocity field with turbulence imposed
        turb = pyturb.CreateTurbulentVelocityField(
            grid_size=res, v_turb=args.vturb, box_size=box_size, seed=seed[i]
        )
        vel_field[i] = cturb.generate_kolmogorov_field(energy_spectrum_index=5./3., energy_scale=1.0)
        tend=perf_counter()
        print(f"Generated field in {tend-tstart:.3f}s")
        # Generate a gridded version of the velocity field
        (vx,vy,vz) = (turb.vx.ravel(),turb.vy.ravel(),turb.vz.ravel())
        vel = np.stack([vx, vy, vz], axis=-1)
        (x,y,z) = (turb.X.ravel(),turb.Y.ravel(),turb.Z.ravel())
        pos = np.stack([x, y, z], axis=-1)
        tstart=perf_counter()

        grid_limits =  np.array([0,1,0,1,0,1], dtype=np.float64)*turb.box_size
        grid_size = [res,res,res]
        vel_field_smooth[i] = gt.smooth_to_grid(pos, vel, grid_size, grid_limits, method="CIC")
       tend=perf_counter()
        print(f"Generated field in {tend-tstart:.3f}s")


    turb.write_to_file(args.outfile)


# Tools to measure velocity fields
mturb = pyturb.MeasureVelocityField()
# Tools to generate gridded velocity fields given particle data
gt = pyturb.GriddingTools()

resolution = [32, 64, 128, 256]
seed = np.ones(len(resolution),dtype=np.int64)*42
vel_field = [None]*len(resolution)
vel_field_smooth = [None]*len(resolution)

    (vx,vy,vz) = (turb.vx.ravel(),turb.vy.ravel(),turb.vz.ravel())
    vel = np.stack([vx, vy, vz], axis=-1)
    (x,y,z) = (turb.X.ravel(),turb.Y.ravel(),turb.Z.ravel())
    pos = np.stack([x, y, z], axis=-1)
    grid_limits =  np.array([0,1,0,1,0,1], dtype=np.float64)*turb.box_size
    grid_size = [res,res,res]
    vel_field_smooth[i] = gt.smooth_to_grid(pos, vel, grid_size, grid_limits, method="CIC")

print("Finished generating velocity fields")


# %%
fig, axes = gt.plot_3d_projections(vel_field[0], grid_limits, mode='projection', slice_width=3, projection='mean', cmap='plasma', units='kpc', title="Turbulent Velocity Field - N=32")
fig, axes = gt.plot_3d_projections(vel_field[1], grid_limits, mode='projection', slice_width=3, projection='mean', cmap='plasma', units='kpc', title="Turbulent Velocity Field - N=64")
fig, axes = gt.plot_3d_projections(vel_field[2], grid_limits, mode='projection', slice_width=3, projection='mean', cmap='plasma', units='kpc', title="Turbulent Velocity Field - N=128")
fig, axes = gt.plot_3d_projections(vel_field[3], grid_limits, mode='projection', slice_width=3, projection='mean', cmap='plasma', units='kpc', title="Turbulent Velocity Field - N=256")

# %%
fig, axes = gt.plot_3d_projections(vel_field_smooth[0], grid_limits, mode='projection', slice_width=3, projection='mean', cmap='plasma', units='kpc', title="Turbulent Velocity Field - N=32 (Smoothed)")
fig, axes = gt.plot_3d_projections(vel_field_smooth[1], grid_limits, mode='projection', slice_width=3, projection='mean', cmap='plasma', units='kpc', title="Turbulent Velocity Field - N=64 (Smoothed)")
fig, axes = gt.plot_3d_projections(vel_field_smooth[2], grid_limits, mode='projection', slice_width=3, projection='mean', cmap='plasma', units='kpc', title="Turbulent Velocity Field - N=128 (Smoothed)")
fig, axes = gt.plot_3d_projections(vel_field_smooth[3], grid_limits, mode='projection', slice_width=3, projection='mean', cmap='plasma', units='kpc', title="Turbulent Velocity Field - N=256 (Smoothed)")

# %%
# Compute for the velocity field generated by pyturb...
k0,pk0,_=mturb.compute_power_spectrum(vel_field[0], box_size=turb.box_size, component='energy', method='radial')
# ... and compute for the velocity field constructed from the positions and velocities
k1,pk1,_=mturb.compute_power_spectrum(vel_field[1], box_size=turb.box_size, component='energy', method='radial')
# ... and compute for the velocity field constructed from the positions and velocities
k2,pk2,_=mturb.compute_power_spectrum(vel_field[2], box_size=turb.box_size, component='energy', method='radial')
# ... and compute for the velocity field constructed from the positions and velocities
k3,pk3,_=mturb.compute_power_spectrum(vel_field[3], box_size=turb.box_size, component='energy', method='radial')


# %%
# Compute for the velocity field generated by pyturb...
ks0,pks0,_=mturb.compute_power_spectrum(vel_field_smooth[0], box_size=turb.box_size, component='energy', method='radial')
# ... and compute for the velocity field constructed from the positions and velocities
ks1,pks1,_=mturb.compute_power_spectrum(vel_field_smooth[1], box_size=turb.box_size, component='energy', method='radial')
# ... and compute for the velocity field constructed from the positions and velocities
ks2,pks2,_=mturb.compute_power_spectrum(vel_field_smooth[2], box_size=turb.box_size, component='energy', method='radial')
# ... and compute for the velocity field constructed from the positions and velocities
ks3,pks3,_=mturb.compute_power_spectrum(vel_field_smooth[3], box_size=turb.box_size, component='energy', method='radial')


# %%
plt.xlabel(r"k [$kpc^{-1}$]")
plt.ylabel(r"P$_\text{turb}$(k) [$kpc^{3}$]")
plt.xscale("log")
plt.yscale("log")
plt.plot(k0,pk0,color="red",label="N=32")
plt.plot(k1,pk1,color="blue",label="N=64")
plt.plot(k2,pk2,color="green",label="N=128")
plt.plot(k3,pk3,color="orange",label="N=256")

plt.plot(ks0,pks0,":",color="red",label="N=32 (Smoothed)")
plt.plot(ks1,pks1,":",color="blue",label="N=64 (Smoothed)")
plt.plot(ks2,pks2,":",color="green",label="N=128 (Smoothed)")
plt.plot(ks3,pks3,":",color="orange",label="N=256 (Smoothed)")

plt.grid(True)
plt.legend()

fit_mask = (k3 > 10) & (k3 < 30)
if np.any(fit_mask):
    log_k_fit = np.log10(k3[fit_mask])
    log_P_fit = np.log10(pk3[fit_mask])
    slope, intercept = np.polyfit(log_k_fit, log_P_fit, 1)
        
print(f"Measured power law slope: {slope:.3f}")
print(f"Expected slope (Kolmogorov): {-11/3:.3f}")
print(f"Difference: {abs(slope + 11/3):.3f}")
    
# Add theoretical line
k_theory = np.logspace(0, 2, 50)
P_theory = k_theory**(-11/3)
P_theory *= pk3[10] / P_theory[10]  # Normalize
plt.plot(k_theory, P_theory, '--', color='black', label=r'$k^{-11/3}$')
plt.legend()



# %%
turb.write_to_file("./turbulent_ics.hdf5")


# %%
