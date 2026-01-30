"""
Export biventricular model to STL using marching cubes algorithm.

This script creates a volumetric representation of the biventricular geometry
and uses PyVista's marching cubes (contour) algorithm to extract a smooth
isosurface, which is then exported to STL format.

Dependencies:
    pip install pyvista numpy scipy

Run:
    python biventricular_stl_export.py
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from typing import Tuple

from biventricular_model import (
    create_lv_default_control_points,
    create_lv_mesh,
    create_rv_mesh,
)


def prepare_mesh_for_distance(mesh: pv.PolyData) -> pv.PolyData:
    """
    Prepare a mesh for distance computations by triangulating and cleaning.

    Args:
        mesh: Input PyVista mesh

    Returns:
        Cleaned, triangulated mesh with normals
    """
    # Triangulate quads to triangles
    mesh = mesh.triangulate()
    # Clean up degenerate cells and duplicate points
    mesh = mesh.clean()
    # Compute normals
    mesh = mesh.compute_normals(auto_orient_normals=True)
    return mesh


def compute_signed_distance_field(
    points: np.ndarray,
    mesh: pv.PolyData,
    invert: bool = False,
) -> np.ndarray:
    """
    Compute signed distance from points to a mesh surface.

    Uses closest point on mesh and surface normal to determine sign.
    Negative = inside (on the side opposite to normal), Positive = outside.

    Args:
        points: (n, 3) array of query points
        mesh: PyVista mesh (should have normals computed)
        invert: if True, flip the sign convention

    Returns:
        distances: (n,) array of signed distances
    """
    # Ensure mesh has normals
    if 'Normals' not in mesh.cell_data and 'Normals' not in mesh.point_data:
        mesh = mesh.compute_normals(auto_orient_normals=True)

    # Find closest cell and point for each query point
    closest_cells, closest_points = mesh.find_closest_cell(points, return_closest_point=True)

    # Compute unsigned distance
    diff = points - closest_points
    distances = np.linalg.norm(diff, axis=1)

    # Get cell normals for sign determination
    if 'Normals' in mesh.cell_data:
        cell_normals = mesh.cell_data['Normals'][closest_cells]
    else:
        # Interpolate from point normals
        cell_normals = mesh.point_data['Normals'][closest_cells]

    # Determine sign: positive if point is on the normal side, negative otherwise
    # dot product > 0 means point is on the outward normal side
    dot_products = np.sum(diff * cell_normals, axis=1)
    signs = np.sign(dot_products)
    signs[signs == 0] = 1  # Handle points exactly on surface

    signed_distances = signs * distances

    if invert:
        signed_distances = -signed_distances

    return signed_distances


def create_biventricular_distance_field(
    grid_bounds: Tuple[float, float, float, float, float, float],
    resolution: int = 100,
) -> Tuple[pv.ImageData, dict, dict]:
    """
    Create a 3D distance field for the biventricular epicardial surface.

    Args:
        grid_bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
        resolution: number of grid points along each axis

    Returns:
        grid: PyVista ImageData with scalar distance field
        lv_result: LV mesh dictionary
        rv_result: RV mesh dictionary
    """
    # Create LV and RV meshes
    lv_endo_cp = create_lv_default_control_points()
    lv_result = create_lv_mesh(lv_endo_cp)
    rv_result = create_rv_mesh(lv_endo_cp)

    # Get and prepare meshes
    lv_epi = prepare_mesh_for_distance(lv_result['epi_mesh'])
    rv_epi = prepare_mesh_for_distance(rv_result['epi_mesh'])

    # Create uniform grid
    xmin, xmax, ymin, ymax, zmin, zmax = grid_bounds

    grid = pv.ImageData(
        dimensions=(resolution, resolution, resolution),
        spacing=(
            (xmax - xmin) / (resolution - 1),
            (ymax - ymin) / (resolution - 1),
            (zmax - zmin) / (resolution - 1),
        ),
        origin=(xmin, ymin, zmin),
    )

    points = grid.points

    print("Computing distance field for LV epicardium...")
    lv_epi_dist = compute_signed_distance_field(points, lv_epi, invert=True)

    print("Computing distance field for RV epicardium...")
    rv_epi_dist = compute_signed_distance_field(points, rv_epi, invert=True)

    # Combine: use minimum distance (union of surfaces)
    # Negative inside, positive outside
    combined_dist = np.minimum(lv_epi_dist, rv_epi_dist)

    grid.point_data['distance'] = combined_dist

    return grid, lv_result, rv_result


def create_myocardium_distance_field(
    grid_bounds: Tuple[float, float, float, float, float, float],
    resolution: int = 100,
) -> Tuple[pv.ImageData, dict, dict]:
    """
    Create a 3D distance field representing the myocardial wall.

    Negative values are inside the myocardium (between endo and epi surfaces).

    Args:
        grid_bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
        resolution: number of grid points along each axis

    Returns:
        grid: PyVista ImageData with scalar distance field
        lv_result: LV mesh dictionary
        rv_result: RV mesh dictionary
    """
    # Create LV and RV meshes
    lv_endo_cp = create_lv_default_control_points()
    lv_result = create_lv_mesh(lv_endo_cp)
    rv_result = create_rv_mesh(lv_endo_cp)

    # Get and prepare meshes
    lv_endo = prepare_mesh_for_distance(lv_result['endo_mesh'])
    lv_epi = prepare_mesh_for_distance(lv_result['epi_mesh'])
    rv_endo = prepare_mesh_for_distance(rv_result['endo_mesh'])
    rv_epi = prepare_mesh_for_distance(rv_result['epi_mesh'])

    # Create uniform grid
    xmin, xmax, ymin, ymax, zmin, zmax = grid_bounds

    grid = pv.ImageData(
        dimensions=(resolution, resolution, resolution),
        spacing=(
            (xmax - xmin) / (resolution - 1),
            (ymax - ymin) / (resolution - 1),
            (zmax - zmin) / (resolution - 1),
        ),
        origin=(xmin, ymin, zmin),
    )

    points = grid.points

    print("Computing distance fields...")

    # Compute signed distances
    # For epicardium: negative inside, positive outside
    lv_epi_dist = compute_signed_distance_field(points, lv_epi, invert=True)
    rv_epi_dist = compute_signed_distance_field(points, rv_epi, invert=True)

    # For endocardium: negative inside cavity, positive outside (in wall)
    lv_endo_dist = compute_signed_distance_field(points, lv_endo, invert=True)
    rv_endo_dist = compute_signed_distance_field(points, rv_endo, invert=True)

    # LV myocardium: inside epi (dist < 0) AND outside endo (dist > 0)
    # Use max to get intersection: max(inside_epi, outside_endo)
    # inside_epi = -lv_epi_dist (negative when inside)
    # outside_endo = lv_endo_dist (positive when outside cavity)
    lv_myo_dist = np.maximum(-lv_epi_dist, lv_endo_dist)

    # RV myocardium
    rv_myo_dist = np.maximum(-rv_epi_dist, rv_endo_dist)

    # Combined biventricular myocardium (union)
    combined_dist = np.minimum(lv_myo_dist, rv_myo_dist)

    grid.point_data['distance'] = -combined_dist

    return grid, lv_result, rv_result


def extract_surface_marching_cubes(
    grid: pv.ImageData,
    isosurface_value: float = 0.0,
) -> pv.PolyData:
    """
    Extract isosurface using marching cubes algorithm.

    Args:
        grid: ImageData with scalar 'distance' field
        isosurface_value: value at which to extract surface

    Returns:
        PyVista PolyData mesh of the extracted surface
    """
    print(f"Extracting isosurface at value = {isosurface_value}...")

    # Use contour (marching cubes) to extract isosurface
    surface = grid.contour(
        isosurfaces=[isosurface_value],
        scalars='distance',
        method='marching_cubes',
    )

    # Clean up the mesh
    surface = surface.clean()
    surface = surface.compute_normals(auto_orient_normals=True)

    return surface


def export_biventricular_stl(
    output_filename: str = "biventricular.stl",
    resolution: int = 100,
    export_type: str = "epicardium",
) -> pv.PolyData:
    """
    Export biventricular model to STL using marching cubes.

    Args:
        output_filename: output STL filename
        resolution: grid resolution for marching cubes
        export_type: "epicardium" for outer surface, "myocardium" for wall

    Returns:
        The extracted surface mesh
    """
    # Define grid bounds based on typical model dimensions
    bounds = (-3.5, 3.5, -3.5, 3.5, -5.0, 2.5)

    if export_type == "myocardium":
        grid, lv_result, rv_result = create_myocardium_distance_field(bounds, resolution)
    else:
        grid, lv_result, rv_result = create_biventricular_distance_field(bounds, resolution)

    # Extract surface using marching cubes
    surface = extract_surface_marching_cubes(grid, isosurface_value=0.0)

    # Save to STL
    print(f"Saving to {output_filename}...")
    surface.save(output_filename)

    print(f"Exported {surface.n_points} vertices and {surface.n_cells} faces")

    return surface


def visualize_result(surface: pv.PolyData, title: str = "Biventricular Surface"):
    """Visualize the extracted surface."""
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.set_background("white")
    plotter.add_mesh(surface, color="salmon", opacity=0.9, smooth_shading=True)
    plotter.add_title(title)
    plotter.camera_position = 'iso'
    plotter.show()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export biventricular model to STL using marching cubes"
    )
    parser.add_argument(
        "-o", "--output",
        default="biventricular.stl",
        help="Output STL filename (default: biventricular.stl)"
    )
    parser.add_argument(
        "-r", "--resolution",
        type=int,
        default=100,
        help="Grid resolution for marching cubes (default: 100)"
    )
    parser.add_argument(
        "-t", "--type",
        choices=["epicardium", "myocardium"],
        default="epicardium",
        help="Surface type to export (default: epicardium)"
    )
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        help="Visualize the result after export"
    )

    args = parser.parse_args()

    print(f"Generating biventricular {args.type} surface...")
    print(f"Resolution: {args.resolution}^3 grid")

    surface = export_biventricular_stl(
        output_filename=args.output,
        resolution=args.resolution,
        export_type=args.type,
    )

    print(f"Done! Saved to {args.output}")

    if args.visualize:
        visualize_result(surface, f"Biventricular {args.type.title()}")


if __name__ == "__main__":
    main()
