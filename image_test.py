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


def main():
    # image = make_test_image(radius=20)
    img_path = Path(r"D:\Data\SSIM-Scans\B2\AAW\AAW.vti")
    image: pv.ImageData = pv.read(img_path)
    mask = image.point_data["Mask"]

    image.GetPointData().SetActiveScalars("Mask")

    smoothing = False
    constraint = 0.5
    n_iters = 100
    relaxation_factor = 0.05
    eps = 1e-6

    surf_mesh = surface_net(
        image,
        scalars="Mask",
        smoothing=smoothing,
        constraint=constraint,
        n_iters=n_iters,
        relaxation_factor=relaxation_factor,
        eps=eps,
    )

    selection = select_enclosed_points(image, surf_mesh).point_data["SelectedPoints"]
    print(f"Surface Net Invalid Points: {np.nansum(selection != mask)}")

    mesh = odc_surface(
        image,
        scalars="Mask",
        smoothing=smoothing,
        constraint=constraint,
        n_iters=n_iters,
        relaxation_factor=relaxation_factor,
        eps=eps,
        isovalue=0.25,
        normalize_spacing=False,
    )
    mesh = mesh.compute_normals()

    edge_lengths = (
        mesh.extract_all_edges()
        .compute_cell_sizes(length=True, area=False, volume=False)
        .cell_data["Length"]
    )
    print(mesh)
    print(edge_lengths)

    selection = select_enclosed_points(image, mesh).point_data["SelectedPoints"]
    print(f"ODC Invalid Points: {np.nansum(selection != mask)}")

    plotter = pv.Plotter(shape=(1, 3))
    plotter.add_mesh(surf_mesh, color="red")

    plotter.subplot(0, 1)
    plotter.add_mesh(mesh, color="blue")

    plotter.subplot(0, 2)
    plotter.add_mesh(mesh, color="blue", opacity=0.5)
    plotter.add_mesh(surf_mesh, color="red", opacity=0.5)

    plotter.link_views()
    plotter.show()


def make_test_image(radius: float = 20, spacing=(0.25, 0.5, 1), padding: int = 4):
    spacing = np.asarray(spacing)
    ball = pv.Sphere(radius=radius)
    edge_lengths = np.array(ball.bounds_size)
    dims = np.ceil(edge_lengths / spacing).astype(int) + 2 * padding
    origin = -spacing * (dims - 1) / 2
    image = pv.ImageData(origin=origin, dimensions=dims, spacing=spacing)
    image = image.select_interior_points(ball)
    image.GetPointData().GetAbstractArray("selected_points").SetName("Mask")
    return image


def odc_surface(
    image: pv.ImageData,
    scalars: str,
    isovalue: float = 0.5,
    background: float = 0,
    smoothing: bool = True,
    n_iters: int = 100,
    relaxation_factor: float = 0.05,
    constraint: float = 0.9,
    eps: float = 1e-6,
    normalize_spacing: bool = False,
):
    """Use occupancy dual contouring to extract a surface mesh from the label map
    defined by `point_data_name`.

    Parameters
    -------------

    normalize_spacing: bool
        Whether to normalize the image spacing to (1,1,1) during isosurface extraction.
        If enabled the extracted mesh is still transformed back to the correct image
        dimensions.
    """

    if normalize_spacing:
        spacing = np.array([1, 1, 1])
    else:
        spacing = np.asarray(image.spacing)
    dims = np.asarray(image.dimensions)
    data = image.point_data[scalars].reshape(dims, order="F")

    if not np.all(dims[0] == dims):  # pad image to use a square grid
        max_dim = np.max(dims)
        diffs = max_dim - dims
        dims = dims + diffs

        padding = []
        for diff in diffs:
            padding.append((int(np.floor(diff / 2)), int(np.ceil(diff / 2))))

        data = np.pad(
            data, pad_width=padding, mode="constant", constant_values=background
        )

    # odc requires the grid to be aligned with the global axes and centered about the
    # global origin
    min_coord = -spacing * (dims - 1) / 2
    max_coord = min_coord + spacing * dims
    grid_pts = tuple(min_coord[i] + np.arange(dims[i]) * spacing[i] for i in range(3))

    interpolator = RegularGridInterpolator(
        points=grid_pts,
        values=data,
        method="linear",
        fill_value=background,
        bounds_error=False,
    )

    impl_func = partial(impl_dist_func, interpolator=interpolator)

    # num_grid = np.max(int(np.ceil(dims / 2)))
    num_grid = np.max(dims)
    grid_spacing = (max_coord - min_coord) / num_grid
    print(image.spacing)
    print(num_grid)
    print(grid_spacing)

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

    if smoothing:
        smoother = vtkConstrainedSmoothingFilter()
        smoother.AddInputConnection(orient_filter.GetOutputPort())
        smoother.SetNumberOfIterations(n_iters)
        smoother.SetConstraintStrategyToConstraintBox()
        smoother.SetConstraintBox(constraint * grid_spacing)
        smoother.SetRelaxationFactor(relaxation_factor)
        smoother.SetConvergence(eps)
        transform_filter.SetInputConnection(smoother.GetOutputPort())
    else:
        transform_filter.SetInputConnection(orient_filter.GetOutputPort())
    transform_filter.Update()

    mesh: pv.PolyData = pv.wrap(transform_filter.GetOutput())
    return mesh


def impl_dist_func(xyz: th.tensor, interpolator: RegularGridInterpolator):
    """Compute distance function using interpolator and cast result to a tensor"""
    result = interpolator(xyz)
    return th.as_tensor(result)


def select_enclosed_points(image: pv.ImageData, surface: pv.PolyData):
    """"""
    select_name = "SelectedPoints"
    transform = vtkTransform()
    transform.PostMultiply()
    transform.Scale(1 / np.array(image.spacing))
    transform.Concatenate(image.GetIndexToPhysicalMatrix())
    transform.Inverse()

    # align input image with global axes to prevent filter errors
    # apply same transform to source to maintain relationship with source
    image = image.transform(transform, inplace=False)
    surface = surface.transform(transform, inplace=False)

    #  set point data of aligned image to 1's
    image.point_data[select_name] = np.full(image.n_points, 1, dtype=np.uint8)
    image.GetPointData().SetActiveScalars(select_name)

    pol2stenc = vtkPolyDataToImageStencil()
    pol2stenc.SetInputDataObject(surface)
    pol2stenc.SetInformationInput(image)

    imgstenc = vtkImageStencil()
    imgstenc.SetInputDataObject(image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)

    imgstenc.Update()
    result: pv.ImageData = pv.wrap(imgstenc.GetOutput())

    transform.Inverse()
    return result.transform(transform, inplace=False)


def surface_net(
    image: pv.ImageData,
    scalars: str,
    smoothing: bool = True,
    n_iters: int = 100,
    relaxation_factor: float = 0.05,
    constraint: float = 0.9,
    eps: float = 1e-6,
):

    transform = vtkTransform()
    transform.PostMultiply()
    transform.Scale(1 / np.array(image.spacing))
    transform.Concatenate(image.GetIndexToPhysicalMatrix())
    transform.Inverse()

    image = image.transform(transform, inplace=False)
    image.GetPointData().SetActiveScalars(scalars)

    alg = vtkSurfaceNets3D()
    alg.SetInputData(image)
    alg.SetValue(0, 1)
    alg.SetSmoothing(smoothing)
    alg.SetConstraintStrategyToConstraintBox()
    alg.SetConstraintBox(constraint * np.array(image.spacing))
    alg.AutomaticSmoothingConstraintsOff()
    alg.SetOutputStyleToBoundary()
    alg.SetOutputMeshTypeToTriangles()
    alg.OptimizedSmoothingStencilsOn()
    alg.SetRelaxationFactor(relaxation_factor)
    alg.SetNumberOfIterations(n_iters)
    alg.GetSmoother().SetConvergence(eps)

    transform.Inverse()
    transformer = vtkTransformPolyDataFilter()
    transformer.SetTransform(transform)
    transformer.SetInputConnection(alg.GetOutputPort())
    transformer.Update()
    result: pv.PolyData = pv.wrap(transformer.GetOutput())
    return result


if __name__ == "__main__":
    main()
