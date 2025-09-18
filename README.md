## PyTurb: routines to generate and quantify turbulent velocity field

Developers: Chris Power ([UWA](https://research-repository.uwa.edu.au/en/persons/chris-power), [ICRAR](https://www.icrar.org/people/cpower/)), Balu Sreedhar ([Seville](https://s-balu.github.io)), Andrew Sullivan ([ICRAR](https://www.icrar.org/people/asullivan/)).

More details on the theory and background are in the [wiki](https://github.com/doctorcbpower/pyturb/wiki).

### Installation
Once you have cloned, 

```
cd pyturb
pip install -e .
```
You should be able to load this as a python package using

```
import pyturb
```

and generate a turbulent velocity field and write it to file by doing the following,

```
pturb = pyturb.CreateTurbulentVelocityField(grid_size=128,box_size=10,seed=18732)    
vel_field = pturb.generate_kolmogorov_field(energy_spectrum_index=5./3., energy_scale=1.0)

(vx,vy,vz)=(turb.vx.ravel(),turb.vy.ravel(),turb.vz.ravel())
vel=np.array([vx,vy,vz]).T
(x,y,z)=(turb.X.ravel(),turb.Y.ravel(),turb.Z.ravel())
pos=np.array([x,y,z]).T
pturb.write_to_file("./turbulent_ics.hdf5")
```

You can grid the data and produce projections and slices using,

```
gturb=pyturb.GriddingTools()
grid_size = np.array([64,64,64], dtype=np.int64)  # Numba-safe
grid_limits =  np.array([0,1,0,1,0,1], dtype=np.float64)*turb.box_size
vel_field_smooth = gturb.smooth_to_grid(pos, vel, grid_size, grid_limits,method="GAUSSIAN", filter_sigma=2.0)
fig, axes = gturb.plot_3d_projections(vel_field_smooth, grid_limits, mode='projection', slice_width=3, projection='max', cmap='plasma', units='kpc', title="Turbulent Velocity Field - Magnitude")
```

You can measure the power spectrum of turbulence using,

```
mturb = pyturb.measure_turbulence.MeasureVelocityField()
k,pk,_=mturb.fft().compute_power_spectrum(vel_field, box_size=pturb.box_size, component='energy', method='radial')
```

