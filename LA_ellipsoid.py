"""
LV model using B-spline surfaces for epicardial and endocardial boundaries.

The LV is modeled as a surface of revolution:
- A B-spline curve defines the endocardial profile (cross-section in r-z plane)
- The epicardial surface is a parallel offset (constant wall thickness along normals)
- Both curves are revolved around the z-axis to create 3D surfaces

Constraints:
- Apex normal is (0, 0, -1): achieved by horizontal tangent at apex
- Epi and endo surfaces are parallel: epi is offset along endo normals

Dependencies:
    pip install pyvista numpy scipy

Run:
    python lv_bspline_model.py
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from typing import Tuple, List, Optional
from scipy.interpolate import BSpline


def create_bspline_curve(
        control_points: np.ndarray,
        degree: int = 3,
        num_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray, BSpline, BSpline]:
    """
    Create a B-spline curve from control points.

    Args:
        control_points: (n, 2) array of [r, z] control points
        degree: B-spline degree (default 3 = cubic)
        num_samples: number of points to sample along curve

    Returns:
        points: (num_samples, 2) array of [r, z] points on curve
        tangents: (num_samples, 2) array of tangent vectors
        bspline_r: BSpline object for r coordinate
        bspline_z: BSpline object for z coordinate
    """
    n = len(control_points)

    # Ensure degree doesn't exceed n-1
    degree = min(degree, n - 1)

    # Clamped knot vector: repeats at ends for interpolation of endpoints
    num_knots = n + degree + 1

    # Clamped uniform knot vector
    knots = np.zeros(num_knots)
    knots[:degree + 1] = 0.0
    knots[-(degree + 1):] = 1.0

    # Interior knots uniformly spaced
    num_interior = num_knots - 2 * (degree + 1)
    if num_interior > 0:
        knots[degree + 1:degree + 1 + num_interior] = np.linspace(0, 1, num_interior + 2)[1:-1]

    # Create B-spline for each coordinate
    r_coords = control_points[:, 0]
    z_coords = control_points[:, 1]

    bspline_r = BSpline(knots, r_coords, degree)
    bspline_z = BSpline(knots, z_coords, degree)

    # Sample the curve
    t = np.linspace(0, 1, num_samples)
    r = bspline_r(t)
    z = bspline_z(t)

    # Compute tangents (derivatives)
    dr = bspline_r.derivative()(t)
    dz = bspline_z.derivative()(t)

    points = np.column_stack([r, z])
    tangents = np.column_stack([dr, dz])

    return points, tangents, bspline_r, bspline_z


def compute_curve_normals(tangents: np.ndarray) -> np.ndarray:
    """
    Compute outward-pointing normals for a profile curve.

    For a curve in the r-z plane that will be revolved around z-axis,
    the "outward" normal points in the +r direction (away from axis).

    Given tangent (dr, dz), the perpendicular is (-dz, dr) or (dz, -dr).
    We choose the one with positive r component (pointing outward).
    """
    normals = np.zeros_like(tangents)

    for i in range(len(tangents)):
        dr, dz = tangents[i]

        # Two perpendicular options: (-dz, dr) or (dz, -dr)
        # Choose the one pointing outward (positive r component when possible)
        n1 = np.array([-dz, dr])
        n2 = np.array([dz, -dr])

        # Normalize
        len1 = np.linalg.norm(n1)
        len2 = np.linalg.norm(n2)

        if len1 > 1e-10:
            n1 = n1 / len1
        if len2 > 1e-10:
            n2 = n2 / len2

        # Choose the one with positive r component (outward)
        # At apex (r=0), we want the normal pointing down (-z), so r=0, z=-1
        if n1[0] > n2[0]:
            normals[i] = n1
        elif n2[0] > n1[0]:
            normals[i] = n2
        else:
            # Equal r components (both zero at apex) - choose based on z
            # At apex with horizontal tangent, we want normal = (0, -1)
            if n1[1] < n2[1]:
                normals[i] = n1
            else:
                normals[i] = n2

    return normals


def offset_curve(
        points: np.ndarray,
        normals: np.ndarray,
        offset: float,
) -> np.ndarray:
    """
    Offset a curve along its normals by a constant distance.

    Args:
        points: (n, 2) curve points
        normals: (n, 2) unit normal vectors
        offset: offset distance (positive = outward)

    Returns:
        offset_points: (n, 2) offset curve points
    """
    offset_points = points + offset * normals

    # Ensure r >= 0 (can't go past the axis)
    offset_points[:, 0] = np.maximum(offset_points[:, 0], 0.0)

    return offset_points


def revolve_profile(
        profile_points: np.ndarray,
        num_theta: int = 64,
        theta_range: Tuple[float, float] = (0, 2 * np.pi),
) -> pv.PolyData:
    """
    Create a surface of revolution from a profile curve.

    Args:
        profile_points: (n, 2) array of [r, z] points defining the profile
        num_theta: number of angular divisions
        theta_range: (start, end) angles in radians

    Returns:
        PyVista PolyData mesh of the revolved surface
    """
    n_profile = len(profile_points)
    r = profile_points[:, 0]
    z = profile_points[:, 1]

    theta = np.linspace(theta_range[0], theta_range[1], num_theta, endpoint=False)

    # Generate 3D points
    points = np.zeros((n_profile, num_theta, 3))

    for i, t in enumerate(theta):
        points[:, i, 0] = r * np.cos(t)  # x
        points[:, i, 1] = r * np.sin(t)  # y
        points[:, i, 2] = z  # z

    # Flatten points
    points_flat = points.reshape(-1, 3)

    # Create faces (quads connecting adjacent profile points and theta steps)
    faces = []
    for i in range(n_profile - 1):
        for j in range(num_theta):
            j_next = (j + 1) % num_theta

            p0 = i * num_theta + j
            p1 = i * num_theta + j_next
            p2 = (i + 1) * num_theta + j_next
            p3 = (i + 1) * num_theta + j

            faces.extend([4, p0, p1, p2, p3])

    faces = np.array(faces)

    mesh = pv.PolyData(points_flat, faces)
    mesh = mesh.compute_normals(auto_orient_normals=True)

    return mesh


def rv_revolve_profile(
        profile_points: np.ndarray,
        num_theta: int = 32,
        theta_range: Tuple[float, float] = (0, np.pi),
        scale: float = 0.8,
) -> pv.PolyData:
    """
    Create a surface of revolution from a profile curve. Note that the radius of
    revolution varies with theta (RV)

    Args:
        profile_points: (n, 2) array of [r, z] points defining the profile
        num_theta: number of angular divisions
        theta_range: (start, end) angles in radians

    Returns:
        PyVista PolyData mesh of the revolved surface
    """
    n_profile = len(profile_points)
    r = profile_points[:, 0]
    z = profile_points[:, 1]

    theta = np.linspace(theta_range[0], theta_range[1], num_theta)

    # Generate 3D points
    points = np.zeros((n_profile, num_theta, 3))

    for i, t in enumerate(theta):
        points[:, i, 0] = r * (1 + scale * np.sin(t)) * np.cos(t)  # x
        points[:, i, 1] = r * (1 + scale * np.sin(t)) * np.sin(t)  # y
        points[:, i, 2] = z  # z

    # Flatten points
    points_flat = points.reshape(-1, 3)

    # Create faces (quads) WITHOUT wrap-around for partial revolution
    faces = []
    for i in range(n_profile - 1):
        for j in range(num_theta - 1):
            p0 = i * num_theta + j
            p1 = i * num_theta + (j + 1)
            p2 = (i + 1) * num_theta + (j + 1)
            p3 = (i + 1) * num_theta + j
            faces.extend([4, p0, p1, p2, p3])

    faces = np.array(faces)

    mesh = pv.PolyData(points_flat, faces)
    mesh = mesh.compute_normals(auto_orient_normals=True)

    return mesh


def create_lv_default_control_points() -> np.ndarray:
    """
    Create default control points for LV endocardial profile.

    Returns (n, 2) array of [r, z] control points.

    The profile is in the r-z plane where:
    - r >= 0 is the radial distance from the long axis
    - z is the long axis (apex at bottom, base at top)

    Control points go from apex (bottom) to base (top).

    IMPORTANT: First two control points have same z-value to ensure
    horizontal tangent at apex, giving surface normal (0, 0, -1).
    """
    # Endocardial surface control points
    # P0 and P1 have same z to create horizontal tangent at apex
    endo = np.array([
        [0.0, -3.5],  # P0: Apex (on axis)
        [0.8, -3.5],  # P1: Same z as P0 -> horizontal tangent at apex
        [1.4, -2.0],  # P2: Lower body
        [1.6, -0.5],  # P3: Mid body (widest)
        [1.5, 0.5],  # P4: Upper body
        [1.3, 1.0],  # P5: Near base
        [0.9, 1.3],  # P6: Base opening
    ])

    return endo

def create_lv_mesh(
        endo_control_points: np.ndarray,
        wall_thickness: float = 0.5,
        degree: int = 3,
        num_profile_samples: int = 80,
        num_theta: int = 64,
) -> dict:
    """
    Create LV meshes from endocardial control points.

    The epicardial surface is computed as a parallel offset of the
    endocardial surface along its outward normals.

    Returns dict with:
        'endo_mesh': endocardial surface mesh
        'epi_mesh': epicardial surface mesh
        'endo_profile': sampled endocardial profile curve
        'epi_profile': sampled epicardial profile curve (offset)
        'endo_normals': normal vectors along endo profile
    """
    # Create B-spline curve for endocardium
    endo_profile, endo_tangents, _, _ = create_bspline_curve(
        endo_control_points, degree=degree, num_samples=num_profile_samples
    )

    # Ensure r >= 0
    endo_profile[:, 0] = np.maximum(endo_profile[:, 0], 0.0)

    # Compute outward normals
    endo_normals = compute_curve_normals(endo_tangents)

    # Create epicardial profile by offsetting along normals
    epi_profile = offset_curve(endo_profile, endo_normals, wall_thickness)

    # Create surfaces of revolution
    endo_mesh = revolve_profile(endo_profile, num_theta=num_theta)
    epi_mesh = revolve_profile(epi_profile, num_theta=num_theta)

    return {
        'endo_mesh': endo_mesh,
        'epi_mesh': epi_mesh,
        'endo_profile': endo_profile,
        'epi_profile': epi_profile,
        'endo_normals': endo_normals,
    }


def create_rv_mesh(
        endo_control_points: np.ndarray,
        wall_thickness: float = 0.25,
        degree: int = 3,
        num_profile_samples: int = 80,
        num_theta: int = 32,
) -> dict:
    """
    Create RV meshes from endocardial control points.

    The epicardial surface is computed as a parallel offset of the
    endocardial surface along its outward normals.

    Returns dict with:
        'endo_mesh': endocardial surface mesh
        'epi_mesh': epicardial surface mesh
        'endo_profile': sampled endocardial profile curve
        'epi_profile': sampled epicardial profile curve (offset)
        'endo_normals': normal vectors along endo profile
    """
    # Create LV
    lv_endo_cp = create_lv_default_control_points()
    lv_result = create_lv_mesh(lv_endo_cp)

    # Create B-spline curve for epicardium
    '''epi_profile, epi_tangents, _, _ = create_bspline_curve(
        endo_control_points, degree=degree, num_samples=num_profile_samples
    )'''

    epi_profile = lv_result['epi_profile'].copy()
    epi_tangents = np.gradient(epi_profile, axis=0)

    # Compute outward normals
    epi_normals = compute_curve_normals(epi_tangents)

    # Create endocardium profile by offsetting along normals
    endo_profile = offset_curve(epi_profile, -epi_normals, wall_thickness)

    # Create surfaces of revolution
    endo_mesh = rv_revolve_profile(endo_profile, num_theta=num_theta)
    epi_mesh = rv_revolve_profile(epi_profile, num_theta=num_theta)

    return {
        'endo_mesh': endo_mesh,
        'epi_mesh': epi_mesh,
        'endo_profile': endo_profile,
        'epi_profile': epi_profile,
        'epi_normals': epi_normals,
    }

def build_la_mesh(
    lv_result: dict,
    rv_result: dict,
    spacing: float = 0.05,  # Voxel grid spacing
    # LA shape (ellipsoid radii):
    r_scale_xy: tuple[float, float] = (0.95, 0.75),  # (x_radius, y_radius)
    r_scale_z: float = 0.90,  # z_radius
    # LA placement relative to LV base
    center_offset_xy: tuple[float, float] = (0, 0),  # LA placement relative to LV base (x, y) [R_base]
    center_offset_z: float = 0.85,
    av_plane_offset: float = 0.05,  # Define minimum for LA (along z-direction)
    wall_thickness: float = 0.20,   # Wall thickness (for inner shell)
) -> pv.PolyData:
    """
    Build a Left Atrium (LA) as an implicit ellipsoidal shell that is positioned
    relative to pre-existing LV/RV geometry.
    """

    lv_epi = lv_result["epi_mesh"]
    rv_epi = rv_result["epi_mesh"]

    # Determine where top of LV is:
    z_base = float(lv_epi.bounds[5])  # zmax

    # Find the radius of LA to match th half-span of the LVL
    x0, x1 = lv_epi.bounds[0:2]
    R_base = 0.5 * (x1 - x0)

    # Find the center of RV mesh:
    rv_cx = float(rv_epi.center[1])
    if rv_cx > 0:
        lv_sign = -1.0
    else:
        lv_sign = 1.0

    # Define minor axes stretch:
    ax = r_scale_xy[0] * R_base
    by = r_scale_xy[1] * R_base
    cz = r_scale_z * R_base

    # Define centroid:
    cx = center_offset_xy[0] * R_base
    cy = lv_sign * center_offset_xy[1] * R_base
    cz0 = z_base + center_offset_z * R_base

    z_av = z_base + av_plane_offset  # cushion

    # Inner ellipsoid radii (wall thickness)
    ax_i = max(ax - wall_thickness, 0)
    by_i = max(by - wall_thickness, 0)
    cz_i = max(cz - wall_thickness, 0)

    # Build a voxel grid that covers LA domain with a small margin:
    margin = 2.0 * spacing + 0.35 * R_base
    xmin = cx - ax - margin
    xmax = cx + ax + margin
    ymin = cy - by - margin
    ymax = cy + by + margin
    zmin = z_av - margin
    zmax = cz0 + cz + margin

    # Specify the length of the grid in (x,y,z) respectively:
    dims = (int(np.ceil((xmax - xmin) / spacing)) + 1, int(np.ceil((ymax - ymin) / spacing)) + 1, int(np.ceil((zmax - zmin) / spacing)) + 1)

    # Create the voxel grid:
    grid = pv.ImageData(dimensions=dims, spacing=(spacing, spacing, spacing), origin=(xmin, ymin, zmin))

    # Grid point coordinates (N x 3)
    X = grid.points[:, 0]
    Y = grid.points[:, 1]
    Z = grid.points[:, 2]


    # Ellipsoid implicit functions:
    # F = (x/a)^2 + (y/b)^2 + (z/c)^2 - 1
    xo = (X - cx) / ax
    yo = (Y - cy) / by
    zo = (Z - cz0) / cz
    F_outer = xo ** 2 + yo ** 2 + zo **2 - 1.0

    xi = (X - cx) / ax_i
    yi = (Y - cy) / by_i
    zi = (Z - cz0) / cz_i
    F_inner = xi **2 + yi **2 + zi ** 2 - 1.0

    inside_outer = F_outer <= 0.0
    inside_inner = F_inner <= 0.0

    la_shell = inside_outer & (~inside_inner)

    # Store as scalar field and extract an isosurface
    # Use float scalars for contouring: 1.0 inside shell, 0.0 outside
    scalars = la_shell.astype(np.float32)
    grid.point_data["la"] = scalars

    # Extract surface with marching cubes
    # Contour at 0.5 between 0 and 1
    surf = grid.contour(isosurfaces=[0.5], scalars="la")

    # Clean and triangulate
    surf = surf.triangulate().clean(tolerance=1e-7)

    # Smoothing:
    surf = surf.smooth(n_iter=30, relaxation_factor=0.05, feature_smoothing=False, boundary_smoothing=True)

    # Recompute normals:
    surf = surf.compute_normals(auto_orient_normals=True, consistent_normals=True)

    return surf

def visualize_geo():
    """
    Create and visualize both LV and RV together.
    Shows the interventricular septum relationship.
    """
    # Create LV
    lv_endo_cp = create_lv_default_control_points()
    lv_result = create_lv_mesh(lv_endo_cp)

    # Create RV
    rv_result = create_rv_mesh(lv_endo_cp)

    # Create LA
    la_mesh = build_la_mesh(lv_result, rv_result)
    # Visualize
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.set_background("white")

    # LV surfaces
    plotter.add_mesh(lv_result['endo_mesh'], color="firebrick", opacity=0.7)
    plotter.add_mesh(lv_result['epi_mesh'], color="pink", opacity=0.4)

    # RV surfaces
    plotter.add_mesh(rv_result['endo_mesh'], color="royalblue", opacity=0.7)
    plotter.add_mesh(rv_result['epi_mesh'], color="lightblue", opacity=0.4)

    # Show profile curves in the x-z plane
    lv_endo_3d = np.column_stack([
        lv_result['endo_profile'][:, 0],
        np.zeros(len(lv_result['endo_profile'])),
        lv_result['endo_profile'][:, 1],
    ])
    plotter.add_mesh(pv.lines_from_points(lv_endo_3d), color="darkred", line_width=4)

    rv_epi_3d = np.column_stack([
        rv_result['epi_profile'][:, 0],
        np.zeros(len(rv_result['epi_profile'])),
        rv_result['epi_profile'][:, 1],
    ])
    plotter.add_mesh(pv.lines_from_points(rv_epi_3d), color="steelblue", line_width=4)

    plotter.add_mesh(la_mesh, color="goldenrod", opacity=0.6)

    # Add text annotation
    plotter.camera_position = 'iso'
    plotter.show()


if __name__ == "__main__":
    visualize_geo()
