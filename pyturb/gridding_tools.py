from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
from numba import njit


from numba import njit
import numpy as np

@njit
def ngp_assign(grid, coords, values, grid_size):
    """
    Nearest-Grid-Point (NGP) assignment. Assigns both scalars and vectors.

    Args:
        grid (ndarray): Grid to assign values into.
        coords (ndarray): Particle coordinates, shape (N, dim).
        values (ndarray): Particle values, shape (N,) or (N, D).
        grid_size (tuple): Grid dimensions (Nx, Ny, Nz).
    """
    n_particles, dim = coords.shape
    ncomp = 1 if values.ndim == 1 else values.shape[1]

    Nx, Ny, Nz = grid_size

    for p in range(n_particles):
        idx = np.empty(dim, dtype=np.int64)
        for d in range(dim):
            idx[d] = int(np.floor(coords[p, d] + 0.5))

        i, j, k = idx

        if ncomp == 1:
            grid[i % Nx, j % Ny, k % Nz] += values[p]
        else:
            for c in range(ncomp):
                grid[i % Nx, j % Ny, k % Nz, c] += values[p, c]


@njit
def cic_assign(grid, coords, values, grid_size):
    """
    Cloud-In-Cell (CIC) assignment. Assigns both scalars and vectors.

    Args:
        grid (ndarray): Grid to assign values into.
        coords (ndarray): Particle coordinates, shape (N, dim).
        values (ndarray): Particle values, shape (N,) or (N, D).
        grid_size (tuple): Grid dimensions (Nx, Ny, Nz).
    """
    n_particles, dim = coords.shape
    ncomp = 1 if values.ndim == 1 else values.shape[1]

    Nx, Ny, Nz = grid_size

    offsets = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]
    ])

    for p in range(n_particles):
        idx = np.empty(dim, dtype=np.int64)
        fx = np.empty(dim, dtype=np.float64)

        for d in range(dim):
            x = coords[p, d]
            i = int(np.floor(x))
            idx[d] = i
            fx[d] = x - i

        i, j, k = idx
        fx_i, fx_j, fx_k = fx

        # Compute weights
        w = np.empty(8, dtype=np.float64)
        w[0] = (1 - fx_i) * (1 - fx_j) * (1 - fx_k)
        w[1] = fx_i * (1 - fx_j) * (1 - fx_k)
        w[2] = (1 - fx_i) * fx_j * (1 - fx_k)
        w[3] = (1 - fx_i) * (1 - fx_j) * fx_k
        w[4] = fx_i * (1 - fx_j) * fx_k
        w[5] = (1 - fx_i) * fx_j * fx_k
        w[6] = fx_i * fx_j * (1 - fx_k)
        w[7] = fx_i * fx_j * fx_k

        for n in range(8):
            ii = (i + offsets[n, 0]) % Nx
            jj = (j + offsets[n, 1]) % Ny
            kk = (k + offsets[n, 2]) % Nz

            if ncomp == 1:
                grid[ii, jj, kk] += values[p] * w[n]
            else:
                for c in range(ncomp):
                    grid[ii, jj, kk, c] += values[p, c] * w[n]


class GriddingTools:
    def __init__(self):
        pass

    def smooth_to_grid(self, positions, values, grid_size, grid_limits,
                       method="NGP", sigma=1.0, filter_sigma=None):
        """
        Assign particle values to a 2D or 3D grid.
        """
        dim = len(grid_size)

        # Ensure values is at least 2D (N, D)
        values = np.atleast_2d(values)
        N, D = values.shape
        if D == 1:
            values = values[:, 0]  # keep scalar as 1D

        # Grid spacing
        spacing = [(grid_limits[2 * i + 1] - grid_limits[2 * i]) / grid_size[i]
                   for i in range(dim)]
        coords = np.empty((positions.shape[0], dim), dtype=np.float64)
        for i in range(dim):
            coords[:, i] = (positions[:, i] - grid_limits[2 * i]) / spacing[i]

        # Function to assign a single component
        def assign_component(grid, vals):
            if method.upper() == "NGP":
                ngp_assign(grid, coords, vals, grid_size)
            elif method.upper() == "CIC":
                cic_assign(grid, coords, vals, grid_size)
            elif method.upper() == "GAUSSIAN":
                cic_assign(grid, coords, vals, grid_size)
                grid[:] = gaussian_filter(grid, sigma=sigma)
            else:
                raise ValueError(f"Unknown assignment method: {method}")

            if filter_sigma is not None:
                grid[:] = gaussian_filter(grid, sigma=filter_sigma)
            return grid

        # Handle scalar vs vector
        if values.ndim == 1:
            grid = np.zeros(grid_size, dtype=float)
            grid = assign_component(grid, values)
            return grid
        else:
            # Vector values: create one grid per component
            grids = []
            for d in range(D):
                grid = np.zeros(grid_size, dtype=float)
                grid = assign_component(grid, values[:, d])
                grids.append(grid)
            return np.stack(grids, axis=-1)  # shape (Nx,Ny,Nz,D)

    def axis_labels_from_limits(self, grid_limits, units="kpc"):
        names = ["x", "y", "z"]
        return [f"{names[d]} [{units}]" for d in range(len(grid_limits) // 2)]

    def get_field_label(self, field_mode="magnitude", component=0, units="km/s"):
        if field_mode == "magnitude":
            return f"|v| [{units}]"
        else:
            comps = ["vx", "vy", "vz"]
            return f"{comps[component]} [{units}]"

    def prepare_scalar_field(self, grid_3d, mode="magnitude", component=0):
        if grid_3d.ndim == 4:  # vector field
            if mode == "magnitude":
                return np.sqrt((grid_3d**2).sum(axis=3))
            else:
                return grid_3d[..., component]
        elif grid_3d.ndim == 3:  # already scalar
            return grid_3d
        else:
            raise ValueError("grid_3d must be 3D (scalar) or 4D (vector field).")

    def plot_3d_slice(self, grid_3d, grid_limits,
                      slice_axis='z', slice_index=None,
                      slice_width=None, slice_average=True,
                      mode='slice', projection='mean',
                      field_mode="magnitude", component=0,
                      units="km/s", coord_units="kpc",
                      title="3D Grid Slice", cmap='plasma', figsize=(12, 4)):
        scalar_grid = self.prepare_scalar_field(grid_3d, mode=field_mode, component=component)
        nx, ny, nz = scalar_grid.shape
        coord_labels = self.axis_labels_from_limits(grid_limits, coord_units)

        axis_map = {'x': 0, 'y': 1, 'z': 2}
        ax_idx = axis_map[slice_axis]

        if slice_index is None:
            slice_index = scalar_grid.shape[ax_idx] // 2

        # Slice or project
        if mode == "slice":
            if ax_idx == 0:
                data = scalar_grid[slice_index, :, :]
                extent = [grid_limits[2], grid_limits[3], grid_limits[4], grid_limits[5]]
                xlabel, ylabel = coord_labels[1], coord_labels[2]
            elif ax_idx == 1:
                data = scalar_grid[:, slice_index, :]
                extent = [grid_limits[0], grid_limits[1], grid_limits[4], grid_limits[5]]
                xlabel, ylabel = coord_labels[0], coord_labels[2]
            else:  # z
                data = scalar_grid[:, :, slice_index]
                extent = [grid_limits[0], grid_limits[1], grid_limits[2], grid_limits[3]]
                xlabel, ylabel = coord_labels[0], coord_labels[1]

        elif mode == "projection":
            if projection == "mean":
                data = scalar_grid.mean(axis=ax_idx)
            elif projection == "sum":
                data = scalar_grid.sum(axis=ax_idx)
            elif projection == "max":
                data = scalar_grid.max(axis=ax_idx)
            else:
                raise ValueError("projection must be 'mean', 'sum', or 'max'")

            if ax_idx == 0:
                extent = [grid_limits[2], grid_limits[3], grid_limits[4], grid_limits[5]]
                xlabel, ylabel = coord_labels[1], coord_labels[2]
            elif ax_idx == 1:
                extent = [grid_limits[0], grid_limits[1], grid_limits[4], grid_limits[5]]
                xlabel, ylabel = coord_labels[0], coord_labels[2]
            else:
                extent = [grid_limits[0], grid_limits[1], grid_limits[2], grid_limits[3]]
                xlabel, ylabel = coord_labels[0], coord_labels[1]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(data.T, origin="lower", extent=extent, cmap=cmap, aspect="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(self.get_field_label(field_mode, component, units))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return fig, ax

    def plot_3d_projections(self, grid_3d, grid_limits,
                            mode='projection', projection='sum',
                            field_mode="magnitude", component=0,
                            units="km/s", coord_units="kpc",
                            cmap='viridis', figsize=(12, 4), title=None,
                            slice_index=None, slice_width=None, slice_average=True):
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        for ax, axis, lbl in zip(axes, ['z', 'y', 'x'], ['XY', 'XZ', 'YZ']):
            idx = None
            if mode == 'slice' and slice_index is not None:
                if isinstance(slice_index, dict):
                    idx = slice_index.get(axis, None)
                else:
                    idx = slice_index

            _, single_ax = self.plot_3d_slice(
                grid_3d, grid_limits,
                slice_axis=axis,
                slice_index=idx,
                slice_width=slice_width,
                slice_average=slice_average,
                mode=mode,
                projection=projection,
                field_mode=field_mode,
                component=component,
                units=units,
                coord_units=coord_units,
                cmap=cmap,
                figsize=(5, 5)
            )

            im = single_ax.images[0]
            ax.imshow(im.get_array(), origin='lower',
                      extent=im.get_extent(),
                      cmap=im.get_cmap(),
                      aspect='auto')
            ax.set_title(lbl)
            ax.set_xlabel(single_ax.get_xlabel())
            ax.set_ylabel(single_ax.get_ylabel())
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.close(single_ax.figure)

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        return fig, axes
