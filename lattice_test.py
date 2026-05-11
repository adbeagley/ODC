from typing import Tuple
from functools import partial

import numpy as np
import torch as th
import pyvista as pv
from odc import occupancy_dual_contouring


def main():
    unit_cell_origin = np.array([0, 0, 0])
    unit_cell_dims = np.array([18, 18, 18], dtype=float)
    lattice_dims = np.array([90, 90, 90], dtype=float) + 2
    # lattice_dims = 2 * unit_cell_dims

    unit_cell_resolution = 80
    lattice_resolution = (
        np.ceil(lattice_dims / unit_cell_dims).astype(int) * unit_cell_resolution + 2
    )
    thickness = 0.8

    weights = (1 / unit_cell_dims) * 2 * np.pi
    imp_func = partial(
        shell_tpms_diamond,
        weights=th.tensor(weights),
        origin=th.tensor(unit_cell_origin),
        bounds=th.tensor(lattice_dims),
        c=0,
        pad_val=2 * thickness,
    )

    odc_filter = occupancy_dual_contouring()
    verts, faces = odc_filter.extract_mesh(
        imp_func=imp_func,
        min_coord=-(1.05 * lattice_dims / 2),
        max_coord=(1.05 * lattice_dims / 2),
        num_grid=np.max(lattice_resolution),
        isolevel=thickness,
    )
    cells = np.empty((faces.shape[0], 4), dtype=int)
    cells[:, 0] = 3
    cells[:, 1:] = np.asarray(faces)

    mesh = pv.PolyData(np.asarray(verts), faces=cells.ravel(order="C"))
    mesh.save("lattice.vtp")
    print(mesh)
    mesh.plot()


def shell_tpms_diamond(
    xyz: th.tensor,
    weights: th.tensor,
    origin: th.tensor,
    bounds: th.tensor,
    c: float = 0,
    pad_val: float = 100,
):
    # detect values outside of bounds
    min_bounds = -bounds / 2
    max_bounds = bounds / 2
    mask = th.any(th.logical_or(xyz < min_bounds, xyz > max_bounds), dim=1)

    # pad values outside of bounds
    distance = th.empty(size=mask.shape, dtype=float)
    distance[mask] = pad_val

    # compute absolute distance function inside bounds
    xyz = weights * (xyz[~mask, :] + origin)
    distance[~mask] = (1.0 / 0.115) * th.abs(
        th.sin(xyz[:, 0]) * th.cos(xyz[:, 1] - xyz[:, 2])
        + th.cos(xyz[:, 0]) * th.sin(xyz[:, 1] + xyz[:, 2])
        - c
    )  # Q

    return distance


def compute_cosines(
    w: float,
    offset: float,
    arr: th.tensor,
    mask=np.asarray([True, True, False, False]),
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Compute sin and cos of mesh grid, use mask to set which versions to compute.
    ```
    arg = w * (arr + offset)
    mask[0] -> cos(arg)
    mask[1] -> sin(arg)
    mask[2] -> cos(2*arg)
    mask[3] -> sin(2*arg)
    ```
    """
    arg = w * (arr + offset)
    result = []
    for i, val in enumerate(mask):
        if not val:
            result.append(None)
            continue
        if i == 0:
            result.append(th.cos(arg))
        elif i == 1:
            result.append(th.sin(arg))
        elif i == 2:
            result.append(th.sin(2 * arg))
        elif i == 3:
            result.append(th.sin(2 * arg))

    return result[0], result[1], result[2], result[3]


if __name__ == "__main__":
    main()
