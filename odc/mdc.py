from pathlib import Path
from typing import Tuple
from functools import partial

import numpy as np
import torch as th
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
from vtkmodules.all import (
    vtkTransform,
    vtkConstrainedSmoothingFilter,
    vtkTransformPolyDataFilter,
    vtkOrientPolyData,
    vtkPolyDataToImageStencil,
    vtkImageStencil,
    vtkSurfaceNets3D,
)
from odc import occupancy_dual_contouring


class ManifoldDualContouring:

    # this method does not seem to limit the smoothing as much since the
    # placement of vertexes is not guaranteed to be dead center. Therefore, need
    # to limit the constraint box to a smaller region. This limit was found
    # empirically, so may need fine-tuning in the future for different meshes.
    # - This was tuned by setting `constraint==1` and comparing the number of
    # invalid points generated relative to the vtkSurfaceNets3D filter and
    # selecting the limit that resulted in an equal number of invalid points.
    __CBOX_LIMIT = 0.42

    def __init__(
        self,
        background: float = 0,
        smoothing: bool = True,
        n_iters: int = 100,
        relaxation_factor: float = 0.05,
        constraint: float = 0.9,
        eps: float = 1e-6,
    ):
        self.background = background
        self.smoothing = smoothing
        self.n_iters = n_iters
        self.relaxation_factor = relaxation_factor
        self.constraint = constraint
        self.eps = eps

    def extract_surface(self, image: pv.ImageData, scalars: str, isovalue: float = 0.5):
        """Use occupancy dual contouring to extract a surface mesh from the label map
        defined by `point_data_name`.

        Parameters
        -------------

        """
        spacing = np.asarray(image.spacing)
        dims = np.asarray(image.dimensions)
        data = image.point_data[scalars].reshape(dims, order="F")

        if not np.all(dims[0] == dims):
            # pad image to use a square grid such that it produces an output that only
            # contains the segmented region if not smoothed
            max_dim = np.max(dims)
            diffs = max_dim - dims
            dims = dims + diffs

            padding = []
            for diff in diffs:
                padding.append((int(np.floor(diff / 2)), int(np.ceil(diff / 2))))

            data = np.pad(
                data,
                pad_width=padding,
                mode="constant",
                constant_values=self.background,
            )

        # odc requires the grid to be aligned with the global axes and centered about the
        # global origin
        min_coord = -spacing * (dims - 1) / 2
        max_coord = min_coord + spacing * dims
        grid_pts = tuple(
            min_coord[i] + np.arange(dims[i]) * spacing[i] for i in range(3)
        )

        num_grid = np.max(dims)
        grid_spacing = (
            max_coord - min_coord
        ) / num_grid  # this should equal image spacing
        if not np.allclose(grid_spacing, spacing):
            print("WARNING: ODC grid spacing does not match image spacing!")

        interpolator = RegularGridInterpolator(
            points=grid_pts,
            values=data,
            method="linear",
            fill_value=self.background,
            bounds_error=False,
        )

        impl_func = partial(self.__impl_dist_func, interpolator=interpolator)

        odc_filter = occupancy_dual_contouring()
        verts, faces = odc_filter.extract_mesh(
            imp_func=impl_func,
            min_coord=min_coord,
            max_coord=max_coord,
            num_grid=num_grid,
            isolevel=isovalue,
        )
        # construct mesh
        cells = np.empty((faces.shape[0], 4), dtype=int)
        cells[:, 0] = 3
        cells[:, 1:] = np.asarray(faces)
        mesh = pv.PolyData(np.asarray(verts), faces=cells.ravel(order="C"))

        # construct transform to align mesh with the original image axes
        # Note: can't use min_coord as that may be padded and result in wrong offset
        transform = vtkTransform()
        transform.PostMultiply()
        transform.Translate(spacing * (np.asarray(image.dimensions) - 1) / 2)
        transform.Scale(1 / spacing)
        transform.Concatenate(image.GetIndexToPhysicalMatrix())

        orient_filter = vtkOrientPolyData()
        orient_filter.SetInputData(mesh)
        orient_filter.ConsistencyOn()
        orient_filter.AutoOrientNormalsOn()
        orient_filter.NonManifoldTraversalOn()

        transform_filter = vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)

        if self.smoothing:
            smoother = vtkConstrainedSmoothingFilter()
            smoother.AddInputConnection(orient_filter.GetOutputPort())
            smoother.SetNumberOfIterations(self.n_iters)
            smoother.SetConstraintStrategyToConstraintBox()
            smoother.SetConstraintBox(
                self.constraint * grid_spacing * self.__CBOX_LIMIT
            )
            smoother.SetRelaxationFactor(self.relaxation_factor)
            smoother.SetConvergence(self.eps)
            transform_filter.SetInputConnection(smoother.GetOutputPort())
        else:
            transform_filter.SetInputConnection(orient_filter.GetOutputPort())
        transform_filter.Update()

        mesh: pv.PolyData = pv.wrap(transform_filter.GetOutput())
        return mesh

    @staticmethod
    def __impl_dist_func(xyz: th.tensor, interpolator: RegularGridInterpolator):
        """Compute distance function using interpolator and cast result to a tensor"""
        result = interpolator(xyz)
        return th.as_tensor(result)
