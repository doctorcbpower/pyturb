## pyturb
Python routines to generate turbulent velocity fields and to measure their properties

### Installation
Once you have cloned, 

```
cd pyturb
pip install -e .
```
You should be able to load this as a python package using

```
import pturb
```

and generate a turbulent velocity field and write it to file by doing the following,

```
turb=pyturb.CreateTurbulentVelocityField(grid_size=128,v_turb=20.,temperature=1.e4,box_size=10)
turb.generate_kolmogorov_field(alpha=5/3)
(vx,vy,vz)=(turb.vx.ravel(),turb.vy.ravel(),turb.vz.ravel())
vel=np.array([vx,vy,vz]).T
(x,y,z)=(turb.X.ravel(),turb.Y.ravel(),turb.Z.ravel())
pos=np.array([x,y,z]).T
turb.write_to_file("./turbulent_ics.hdf5")
```

You can grid the data and produce projections and slices using,

```
gt=pyturb.GriddingTools()
grid_size = np.array([64,64,64], dtype=np.int64)  # Numba-safe
grid_limits =  np.array([0,1,0,1,0,1], dtype=np.float64)*turb.box_size
data = gt.smooth_to_grid(pos, vel, grid_size, grid_limits,method="GAUSSIAN", filter_sigma=2.0)
fig, axes = gt.plot_3d_projections(data, grid_limits, mode='projection', slice_width=3, projection='max', cmap='plasma', units='kpc', title="Turbulent Velocity Field - Magnitude")
```
