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


def ellipsoid_surface_point(cx, cy, cz0, ax, by, cz, u: float, v: float) -> np.ndarray:
    """
    Parametric ellipsoid point.
    u in [0, 2pi): azimuth about z
    v in [-pi/2, pi/2]: elevation (like latitude)
    """
    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(v), np.sin(v)

    x = cx + ax * cv * cu
    y = cy + by * cv * su
    z = cz0 + cz * sv
    return np.array([x, y, z], dtype=np.float32)


def ellipsoid_outward_normal_at_point(cx, cy, cz0, ax, by, cz, p: np.ndarray) -> np.ndarray:
    """
    Outward normal of ellipsoid at point p on its surface using gradient of
    F=(x-cx)^2/ax^2 + (y-cy)^2/by^2 + (z-cz0)^2/cz^2 - 1.
    n ~ grad(F).
    """
    x, y, z = p
    nx = 2.0 * (x - cx) / (ax * ax)
    ny = 2.0 * (y - cy) / (by * by)
    nz = 2.0 * (z - cz0) / (cz * cz)
    n = np.array([nx, ny, nz], dtype=np.float32)
    nrm = np.linalg.norm(n)
    if nrm < 1e-12:
        return np.array([0, 0, 1], dtype=np.float32)
    return n / nrm

def add_cylinders_to_mask(
    grid: "pv.ImageData",
    base_mask: np.ndarray,
    cylinders: list[tuple[np.ndarray, np.ndarray]],
    radius: float,
    length: float,
) -> np.ndarray:
    """
    Union filled cylinders into base_mask.
    Each cylinder defined by (p0, axis) where axis must be unit length.
    """
    X = grid.points[:, 0]
    Y = grid.points[:, 1]
    Z = grid.points[:, 2]

    mask = base_mask.copy()
    r2max = radius * radius

    for p0, axis in cylinders:
        axis = np.asarray(axis, dtype=np.float32)
        axis /= max(np.linalg.norm(axis), 1e-12)
        p0 = np.asarray(p0, dtype=np.float32)

        vx = X - p0[0]
        vy = Y - p0[1]
        vz = Z - p0[2]

        # project onto axis segment [0, length]
        w_dot_a = vx * axis[0] + vy * axis[1] + vz * axis[2]
        t = np.clip(w_dot_a / length, 0.0, 1.0)

        cxp = p0[0] + t * length * axis[0]
        cyp = p0[1] + t * length * axis[1]
        czp = p0[2] + t * length * axis[2]

        dx = X - cxp
        dy = Y - cyp
        dz = Z - czp
        r2 = dx * dx + dy * dy + dz * dz

        mask |= (r2 <= r2max)

    return mask

def create_lv_default_control_points() -> np.ndarray:
    """
    Create default control points for LV endocardial profile.

    Returns (n, 2) array of [r, z] control points.

    The profile is in the r-z plane where:
    - r >= 0 is the radial distance from the long axis
    - z is the long axis (apex at bottom, base at top)

    Control points go from apex (bottom) to base (top).
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

def _make_image_grid(bounds, spacing: float) -> pv.ImageData:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    dims = (
        int(np.ceil((xmax - xmin) / spacing)) + 1,
        int(np.ceil((ymax - ymin) / spacing)) + 1,
        int(np.ceil((zmax - zmin) / spacing)) + 1,
    )
    return pv.ImageData(dimensions=dims, spacing=(spacing, spacing, spacing), origin=(xmin, ymin, zmin))

def _la_solid_mask(
    grid: pv.ImageData,
    cx, cy, cz0,
    ax, by, cz,
    R_base: float,
    clearance: float,
    x_side: float,
    y_bias: float,
    add_pulmonary_veins: bool = True,
):
    X = grid.points[:, 0]
    Y = grid.points[:, 1]
    Z = grid.points[:, 2]

    # Filled ellipsoid: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1
    xo = (X - cx) / ax
    yo = (Y - cy) / by
    zo = (Z - cz0) / cz
    inside_ellipsoid = (xo**2 + yo**2 + zo**2) <= 1.0

    if not add_pulmonary_veins:
        return inside_ellipsoid

    # --- Pulmonary veins as filled cylinders (we'll build outer & inner separately outside) ---
    pv_outer_radius = 0.18 * R_base
    pv_length = 1.5 * R_base
    posterior_tilt_deg = 35.0
    phi = posterior_tilt_deg * np.pi / 180

    axis_R = np.array([ np.cos(phi), -np.sin(phi), 0.0], dtype=np.float32)
    axis_L = np.array([-np.cos(phi), -np.sin(phi), 0.0], dtype=np.float32)
    axis_R /= np.linalg.norm(axis_R)
    axis_L /= np.linalg.norm(axis_L)

    z_attach = cz0 + 0.25 * cz
    z_sep = clearance * (2.0 * pv_outer_radius)
    z_sep = min(z_sep, 0.45 * cz)

    attach_L_sup = (cx - x_side, cy + y_bias, z_attach + 0.5 * z_sep)
    attach_L_inf = (cx - x_side, cy + y_bias, z_attach - 0.5 * z_sep)
    attach_R_sup = (cx + x_side, cy + y_bias, z_attach + 0.5 * z_sep)
    attach_R_inf = (cx + x_side, cy + y_bias, z_attach - 0.5 * z_sep)

    pv_specs = [
        (attach_L_sup, axis_L),
        (attach_L_inf, axis_L),
        (attach_R_sup, axis_R),
        (attach_R_inf, axis_R),
    ]

    pv_mask = np.zeros_like(inside_ellipsoid, dtype=bool)

    for (p0_tuple, axis) in pv_specs:
        p0 = np.array(p0_tuple, dtype=np.float32)
        p1 = p0 + pv_length * axis  # not explicitly used, but conceptually the segment end

        vx = X - p0[0]
        vy = Y - p0[1]
        vz = Z - p0[2]

        w_dot_a = vx * axis[0] + vy * axis[1] + vz * axis[2]
        t = np.clip(w_dot_a / pv_length, 0.0, 1.0)

        cxp = p0[0] + t * pv_length * axis[0]
        cyp = p0[1] + t * pv_length * axis[1]
        czp = p0[2] + t * pv_length * axis[2]

        dx = X - cxp
        dy = Y - cyp
        dz = Z - czp
        r2 = dx*dx + dy*dy + dz*dz

        pv_mask |= (r2 <= pv_outer_radius * pv_outer_radius)

    # Union: solid atrium + solid PVs
    return inside_ellipsoid | pv_mask

def build_la_mesh(
        lv_result: dict,
        rv_result: dict,
        spacing: float = 0.05,
        r_scale_xy: tuple[float, float] = (0.72, 0.62),
        r_scale_z: float = 0.58,
        center_offset_xy: tuple[float, float] = (0.28, 0.62),
        center_offset_z: float = 0.72,
        av_plane_offset: float = 0.08,
        wall_thickness: float = 0.16,
        smooth_iters: int = 30,
        # PV geometry
        pv_outer_radius_scale: float = 0.18,  # * R_base
        pv_wall_thickness_scale: float = 0.06,  # * R_base (used only if you later want a PV "shell")
        pv_length_scale: float = 0.7,  # * R_base
        # PV placement angles (slider-friendly), radians:
        pv_L_sup_uv: tuple[float, float] = (np.pi * 0.80, np.pi * 0.10),
        pv_L_inf_uv: tuple[float, float] = (np.pi * 0.80, -np.pi * 0.05),
        pv_R_sup_uv: tuple[float, float] = (np.pi * 0.20, np.pi * 0.10),
        pv_R_inf_uv: tuple[float, float] = (np.pi * 0.20, -np.pi * 0.05),
        # Small push so the tube starts just outside the surface (prevents “inward” artifacts)
        pv_start_offset: float = 0.5,  # in units of (voxel spacing)
) -> dict:
    lv_epi = lv_result["epi_mesh"]
    rv_epi = rv_result["epi_mesh"]

    z_base = float(lv_epi.bounds[5])
    x0, x1 = lv_epi.bounds[0:2]
    R_base = 0.5 * (x1 - x0)

    rv_cx = float(rv_epi.center[1])
    lv_sign = -1.0 if rv_cx > 0 else 1.0

    ax = r_scale_xy[0] * R_base
    by = r_scale_xy[1] * R_base
    cz = r_scale_z * R_base

    cx = center_offset_xy[0] * R_base
    cy = lv_sign * center_offset_xy[1] * R_base
    cz0 = z_base + center_offset_z * R_base

    ax_i = max(ax - wall_thickness, 1e-6)
    by_i = max(by - wall_thickness, 1e-6)
    cz_i = max(cz - wall_thickness, 1e-6)

    z_av = z_base + av_plane_offset

    margin = 2.0 * spacing + 0.35 * R_base
    bounds = (
        cx - ax - margin, cx + ax + margin,
        cy - by - margin, cy + by + margin,
        z_av - margin, cz0 + cz + margin
    )
    grid = _make_image_grid(bounds, spacing)

    # Build filled atrial solids:
    # outer ellipsoid filled
    X = grid.points[:, 0]
    Y = grid.points[:, 1]
    Z = grid.points[:, 2]
    xo = (X - cx) / ax
    yo = (Y - cy) / by
    zo = (Z - cz0) / cz
    outer_solid = (xo ** 2 + yo ** 2 + zo ** 2) <= 1.0

    xi = (X - cx) / ax_i
    yi = (Y - cy) / by_i
    zi = (Z - cz0) / cz_i
    inner_solid = (xi ** 2 + yi ** 2 + zi ** 2) <= 1.0

    # --- PV cylinders: choose surface points, aim along outward normal ---
    pv_outer_radius = pv_outer_radius_scale * R_base
    pv_length = pv_length_scale * R_base

    pv_uvs = [pv_L_sup_uv, pv_L_inf_uv, pv_R_sup_uv, pv_R_inf_uv]
    cylinders_outer = []
    cylinders_inner = []

    start_eps = pv_start_offset * spacing

    for (u, v) in pv_uvs:
        # On outer surface
        p_outer = ellipsoid_surface_point(cx, cy, cz0, ax, by, cz, u, v)
        n_outer = ellipsoid_outward_normal_at_point(cx, cy, cz0, ax, by, cz, p_outer)

        # Start slightly outside the surface to extrude outward
        p0_outer = p_outer + start_eps * n_outer
        cylinders_outer.append((p0_outer, n_outer))

        # On inner surface (so the endo cavity also includes PV inlets)
        p_inner = ellipsoid_surface_point(cx, cy, cz0, ax_i, by_i, cz_i, u, v)
        n_inner = ellipsoid_outward_normal_at_point(cx, cy, cz0, ax_i, by_i, cz_i, p_inner)
        p0_inner = p_inner + start_eps * n_inner
        cylinders_inner.append((p0_inner, n_inner))

    # Union PVs into both solids
    outer_solid = add_cylinders_to_mask(grid, outer_solid, cylinders_outer, pv_outer_radius, pv_length)
    inner_solid = add_cylinders_to_mask(grid, inner_solid, cylinders_inner,
                                        max(pv_outer_radius - pv_wall_thickness_scale * R_base, 1e-6), pv_length)

    # Extract meshes:
    grid.point_data["outer"] = outer_solid.astype(np.float32)
    epi = grid.contour([0.5], scalars="outer").triangulate().clean(tolerance=1e-7)

    grid.point_data["inner"] = inner_solid.astype(np.float32)
    endo = grid.contour([0.5], scalars="inner").triangulate().clean(tolerance=1e-7)

    if smooth_iters and smooth_iters > 0:
        epi = epi.smooth(n_iter=smooth_iters, relaxation_factor=0.05,
                         feature_smoothing=False, boundary_smoothing=True)
        endo = endo.smooth(n_iter=smooth_iters, relaxation_factor=0.05,
                           feature_smoothing=False, boundary_smoothing=True)

    epi = epi.compute_normals(auto_orient_normals=True, consistent_normals=True)
    endo = endo.compute_normals(auto_orient_normals=True, consistent_normals=True)

    return {
        "endo_mesh": endo,
        "epi_mesh": epi,
        "pv_specs": {
            "uvs": pv_uvs,
            "cylinders_outer": cylinders_outer,
            "cylinders_inner": cylinders_inner,
            "pv_outer_radius": pv_outer_radius,
            "pv_length": pv_length,
        },
        "params": {
            "center": (cx, cy, cz0),
            "radii_outer": (ax, by, cz),
            "radii_inner": (ax_i, by_i, cz_i),
            "R_base": R_base,
        }
    }

def build_ra_mesh(
        lv_result: dict,
        rv_result: dict,
        la_mesh: pv.PolyData | None = None,  # LA surface for clearance
        spacing: float = 0.05,
        # RA ellipsoid
        r_scale_xy: tuple[float, float] = (0.74, 0.64),
        r_scale_z: float = 0.52,
        # Placement
        center_offset_xy: tuple[float, float] = (0.18, 0.44),
        center_offset_z: float = 0.85,
        av_plane_offset: float = 0.06,
        wall_thickness: float = 0.18,
        # Prevent RA–LA intersection
        la_clearance: float = 0.10,  # in units of R_base
        # Vena cavae geometry (units of R_base via *_scale)
        vc_outer_radius_scale: float = 0.20,
        vc_wall_thickness_scale: float = 0.06,
        vc_length_sup_scale: float = 0.7,
        vc_length_inf_scale: float = 0.5,

        # Vena cavae placement on RA surface radians:
        svc_uv: tuple[float, float] = (np.pi * 0.50, np.pi * 0.30),
        ivc_uv: tuple[float, float] = (np.pi * 0.55, -np.pi * 0.25),
        # Small push so tube starts outside surface
        vc_start_offset: float = 0.5,  # in units of spacing
        smooth_iters: int = 30,
) -> dict:
    """
    Build RA as two meshes (endo/epi) from voxelized implicit solids:
      - Epicardium: outer ellipsoid + SVC/IVC (filled cylinders)
      - Endocardium: inner ellipsoid + inner SVC/IVC (filled cylinders)
    Vena cavae are oriented along the outward normal of the RA ellipsoid at the
    chosen attachment points (svc_uv, ivc_uv).

    Returns dict: {"endo_mesh": ..., "epi_mesh": ..., "params": ..., "vc_specs": ...}
    """
    lv_epi = lv_result["epi_mesh"]
    rv_epi = rv_result["epi_mesh"]

    z_base = float(lv_epi.bounds[5])
    x0, x1 = lv_epi.bounds[0:2]
    R_base = 0.5 * (x1 - x0)

    # Determine RA side from RV center y
    rv_cy = float(rv_epi.center[1])
    ra_sign_y = 1.0 if rv_cy >= 0 else -1.0

    # Ellipsoid radii
    ax = r_scale_xy[0] * R_base
    by = r_scale_xy[1] * R_base
    cz = r_scale_z * R_base

    # RA centroid
    cx = center_offset_xy[0] * R_base
    cy = ra_sign_y * center_offset_xy[1] * R_base
    cz0 = z_base + center_offset_z * R_base

    z_av = z_base + av_plane_offset

    # Inner radii
    ax_i = max(ax - wall_thickness, 1e-6)
    by_i = max(by - wall_thickness, 1e-6)
    cz_i = max(cz - wall_thickness, 1e-6)

    # Vena cavae radii/lengths
    vc_outer_radius = vc_outer_radius_scale * R_base
    vc_wall_thickness = vc_wall_thickness_scale * R_base
    vc_inner_radius = max(vc_outer_radius - vc_wall_thickness, 1e-6)
    L_sup = vc_length_sup_scale * R_base
    L_inf = vc_length_inf_scale * R_base

    # Grid bounds (include SVC/IVC extents along z; conservative)
    margin = 2.0 * spacing + 0.45 * R_base
    bounds = (
        cx - ax - margin, cx + ax + margin,
        cy - by - margin, cy + by + margin,
        (z_av - margin - L_inf), (cz0 + cz + margin + L_sup),
    )
    grid = _make_image_grid(bounds, spacing)

    # Build filled ellipsoid solids, clipped by AV plane
    X = grid.points[:, 0]
    Y = grid.points[:, 1]
    Z = grid.points[:, 2]

    xo = (X - cx) / ax
    yo = (Y - cy) / by
    zo = (Z - cz0) / cz
    outer_solid = (xo ** 2 + yo ** 2 + zo ** 2) <= 1.0
    outer_solid &= (Z >= z_av)

    xi = (X - cx) / ax_i
    yi = (Y - cy) / by_i
    zi = (Z - cz0) / cz_i
    inner_solid = (xi ** 2 + yi ** 2 + zi ** 2) <= 1.0
    inner_solid &= (Z >= z_av)

    # --- Compute SVC/IVC attachments and normal directions on the ellipsoid ---
    start_eps = vc_start_offset * spacing

    def vc_attachment_and_axis(ax_, by_, cz_, uv):
        u, v = uv
        p = ellipsoid_surface_point(cx, cy, cz0, ax_, by_, cz_, u, v)
        n = ellipsoid_outward_normal_at_point(cx, cy, cz0, ax_, by_, cz_, p)
        p0 = p + start_eps * n
        return p0, n

    # Outer (epi) attachments/axes
    svc_p0_o, svc_axis_o = vc_attachment_and_axis(ax, by, cz, svc_uv)
    ivc_p0_o, ivc_axis_o = vc_attachment_and_axis(ax, by, cz, ivc_uv)

    # Inner (endo) attachments/axes (so endo also includes inlets)
    svc_p0_i, svc_axis_i = vc_attachment_and_axis(ax_i, by_i, cz_i, svc_uv)
    ivc_p0_i, ivc_axis_i = vc_attachment_and_axis(ax_i, by_i, cz_i, ivc_uv)

    # Add filled cylinders into solids
    outer_solid = add_cylinders_to_mask(
        grid, outer_solid,
        cylinders=[(svc_p0_o, svc_axis_o), (ivc_p0_o, ivc_axis_o)],
        radius=vc_outer_radius, length=max(L_sup, L_inf)  # length handled by clamp; see note below
    )
    outer_solid = add_cylinders_to_mask(grid, outer_solid, [(svc_p0_o, svc_axis_o)], vc_outer_radius, L_sup)
    outer_solid = add_cylinders_to_mask(grid, outer_solid, [(ivc_p0_o, ivc_axis_o)], vc_outer_radius, L_inf)

    inner_solid = add_cylinders_to_mask(grid, inner_solid, [(svc_p0_i, svc_axis_i)], vc_inner_radius, L_sup)
    inner_solid = add_cylinders_to_mask(grid, inner_solid, [(ivc_p0_i, ivc_axis_i)], vc_inner_radius, L_inf)

    # Carve RA voxels too close to LA
    if la_mesh is not None and la_clearance > 0:
        pts = pv.PolyData(grid.points)
        d = pts.compute_implicit_distance(la_mesh)["implicit_distance"]
        carve = d <= (la_clearance * R_base)
        outer_solid &= (~carve)
        inner_solid &= (~carve)

    # Extract meshes
    grid.point_data["ra_outer"] = outer_solid.astype(np.float32)
    epi = grid.contour([0.5], scalars="ra_outer").triangulate().clean(tolerance=1e-7)

    grid.point_data["ra_inner"] = inner_solid.astype(np.float32)
    endo = grid.contour([0.5], scalars="ra_inner").triangulate().clean(tolerance=1e-7)

    if smooth_iters and smooth_iters > 0:
        epi = epi.smooth(n_iter=smooth_iters, relaxation_factor=0.05,
                         feature_smoothing=False, boundary_smoothing=True)
        endo = endo.smooth(n_iter=smooth_iters, relaxation_factor=0.05,
                           feature_smoothing=False, boundary_smoothing=True)

    epi = epi.compute_normals(auto_orient_normals=True, consistent_normals=True)
    endo = endo.compute_normals(auto_orient_normals=True, consistent_normals=True)

    return {
        "endo_mesh": endo,
        "epi_mesh": epi,
        "vc_specs": {
            "svc_uv": svc_uv,
            "ivc_uv": ivc_uv,
            "svc_outer": (svc_p0_o, svc_axis_o, vc_outer_radius, L_sup),
            "ivc_outer": (ivc_p0_o, ivc_axis_o, vc_outer_radius, L_inf),
            "svc_inner": (svc_p0_i, svc_axis_i, vc_inner_radius, L_sup),
            "ivc_inner": (ivc_p0_i, ivc_axis_i, vc_inner_radius, L_inf),
        },
        "params": {
            "center": (cx, cy, cz0),
            "radii_outer": (ax, by, cz),
            "radii_inner": (ax_i, by_i, cz_i),
            "R_base": R_base,
            "z_av": z_av,
        }
    }

def build_pulmonary_trunk(
    rv_result: dict,
    outer_radius: float = 0.45,
    wall_thickness: float = 0.1,
    length: float = 2.2,
    bend: float = 0.9,
    n_samples: int = 160,
    n_theta: int = 48,
    z_offset: float = -0.02,
    slab_tol: float = 0.03,
    safety: float = 0.98,
    xy_bias: tuple[float, float] = (0.0, 0.45),
) -> dict:
    rv_endo = rv_result["endo_mesh"]

    # Choose base point for centerline:
    z_top = float(rv_endo.bounds[5])
    z0 = z_top + float(z_offset)

    pts = np.asarray(rv_endo.points)
    slab = pts[np.abs(pts[:, 2] - z0) < slab_tol]
    if len(slab) < 30:
        slab = pts[np.abs(pts[:, 2] - z0) < 3 * slab_tol]

    if len(slab) < 30:
        cx, cy = rv_endo.center[0], rv_endo.center[1]
        R_avail = 0.25 * max(rv_endo.length, 1e-6)
    else:
        cx, cy = slab[:, 0].mean(), slab[:, 1].mean()
        r = np.sqrt((slab[:, 0] - cx) ** 2 + (slab[:, 1] - cy) ** 2)
        R_avail = float(np.percentile(r, 80))

    allow = max((R_avail - outer_radius) * safety, 0.0)
    b = np.array([xy_bias[0], xy_bias[1]], dtype=float)
    nb = np.linalg.norm(b)
    if nb > 1e-12:
        b = b * min(1.0, allow / nb)
    else:
        b[:] = 0.0

    p0 = np.array([cx + b[0], cy + b[1], z0], dtype=float)

    # Create centerline:
    p1 = p0 + np.array([0.0, 0.0, 0.35 * length])
    p2 = p0 + np.array([0.35 * bend * length, 0.10 * bend * length, 0.75 * length])
    p3 = p0 + np.array([0.55 * bend * length, 0.15 * bend * length, 1.00 * length])

    t = np.linspace(0.0, 1.0, n_samples)
    cl = (
            (1 - t)[:, None] ** 3 * p0
            + 3 * (1 - t)[:, None] ** 2 * t[:, None] * p1
            + 3 * (1 - t)[:, None] * t[:, None] ** 2 * p2
            + t[:, None] ** 3 * p3
    )

    def unit(v):
        n = np.linalg.norm(v)
        return v / (n if n > 1e-12 else 1.0)

    Np = len(cl)
    T = np.zeros((Np, 3))
    for i in range(Np):
        if i == 0:
            T[i] = unit(cl[1] - cl[0])
        elif i == Np - 1:
            T[i] = unit(cl[-1] - cl[-2])
        else:
            T[i] = unit(cl[i + 1] - cl[i - 1])

    up = unit(np.array([0.0, 0.0, 1.0]))
    if abs(np.dot(up, T[0])) > 0.95:
        up = unit(np.array([0.0, 1.0, 0.0]))

    B = unit(np.cross(T[0], up))
    Nvec = unit(np.cross(B, T[0]))

    N_arr = np.zeros((Np, 3))
    B_arr = np.zeros((Np, 3))
    N_arr[0], B_arr[0] = Nvec, B

    for i in range(1, Np):
        v = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(v)
        c = np.dot(T[i - 1], T[i])
        if s < 1e-10:
            N_arr[i] = N_arr[i - 1]
            B_arr[i] = B_arr[i - 1]
            continue
        v = v / s
        ang = np.arctan2(s, c)

        # Rodrigues rotate previous normal around axis v
        a = N_arr[i - 1]
        N_new = a * np.cos(ang) + np.cross(v, a) * np.sin(ang) + v * np.dot(v, a) * (1 - np.cos(ang))
        N_new = unit(N_new)
        B_new = unit(np.cross(T[i], N_new))

        N_arr[i], B_arr[i] = N_new, B_new

    # Revolve about inner and outer radii:
    r_out = float(outer_radius)
    r_in = max(r_out - float(wall_thickness), 1e-6)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    outer_pts = np.zeros((Np, n_theta, 3))
    inner_pts = np.zeros((Np, n_theta, 3))

    for i in range(Np):
        ct = np.cos(theta)
        st = np.sin(theta)
        ring_dir = ct[:, None] * N_arr[i][None, :] + st[:, None] * B_arr[i][None, :]
        outer_pts[i] = cl[i][None, :] + r_out * ring_dir
        inner_pts[i] = cl[i][None, :] + r_in * ring_dir

    def skin(pts_grid):
        pts_flat = pts_grid.reshape(-1, 3)
        faces = []
        for i in range(Np - 1):
            for j in range(n_theta):
                jn = (j + 1) % n_theta
                p00 = i * n_theta + j
                p01 = i * n_theta + jn
                p11 = (i + 1) * n_theta + jn
                p10 = (i + 1) * n_theta + j
                faces.extend([4, p00, p01, p11, p10])
        return pv.PolyData(pts_flat, np.array(faces, dtype=np.int64)).triangulate().clean()

    outer = skin(outer_pts).compute_normals(auto_orient_normals=True)
    inner = skin(inner_pts).compute_normals(auto_orient_normals=True)

    inner_rev = inner.copy(deep=True)
    inner_rev.flip_faces()

    solid = outer.merge(inner_rev, merge_points=False).compute_normals(auto_orient_normals=True)

    return {
        "centerline": cl,
        "outer": outer,
        "inner": inner,
        "mesh": solid
    }

def build_aorta(
    lv_result: dict,
    outer_radius: float = 0.48,
    wall_thickness: float = 0.10,
    straight_up: float = 1.2,  # ascending aorta length
    arch_radius: float = 1.8,  # radius of arched portion
    arch_angle_deg: float = 165.0,
    end_straight: float = 0.8,  # descending aorta length
    # orientation: arch plane directions in x/y
    arch_dir_xy: tuple[float, float] = (-1.0, -0.35),
    # placement
    z_offset: float = -0.02,
    slab_tol: float = 0.03,
    safety: float = 0.98,
    xy_bias: tuple[float, float] = (-0.15, -0.35),  # root bias inside LV top
    # meshing
    n_up: int = 40,
    n_arch: int = 140,
    n_down: int = 40,
    n_theta: int = 56,
    twist_turns: float = 0.20,  # mild helical twist along the tube
) -> dict:
    """
    Aorta as: straight-up segment + upside-down U arch + short descending segment,
    then skinned into a thick-walled tube.
    """

    lv_endo = lv_result["endo_mesh"]

    # Pick base point for centerline:
    z_top = float(lv_endo.bounds[5])
    z0 = z_top + float(z_offset)

    pts = np.asarray(lv_endo.points)
    slab = pts[np.abs(pts[:, 2] - z0) < slab_tol]
    if len(slab) < 30:
        slab = pts[np.abs(pts[:, 2] - z0) < 3 * slab_tol]

    if len(slab) < 30:
        cx, cy = lv_endo.center[0], lv_endo.center[1]
        R_avail = 0.25 * max(lv_endo.length, 1e-6)
    else:
        cx, cy = slab[:, 0].mean(), slab[:, 1].mean()
        r = np.sqrt((slab[:, 0] - cx) ** 2 + (slab[:, 1] - cy) ** 2)
        R_avail = float(np.percentile(r, 80))

    allow = max((R_avail - outer_radius) * safety, 0.0)
    b = np.array([xy_bias[0], xy_bias[1]], dtype=float)
    nb = np.linalg.norm(b)
    if nb > 1e-12:
        b = b * min(1.0, allow / nb)
    else:
        b[:] = 0.0

    p0 = np.array([cx + b[0], cy + b[1], z0], dtype=float)

    # Making centerline:
    dxy = np.array([arch_dir_xy[0], arch_dir_xy[1], 0.0], dtype=float)
    nd = np.linalg.norm(dxy[:2])
    if nd < 1e-12:
        dxy = np.array([-1.0, 0.0, 0.0], dtype=float)
    else:
        dxy[:2] /= nd  # normalize xy only
    ex = dxy                   # arch sweeps along ex
    ez = np.array([0.0, 0.0, 1.0], dtype=float)

    # Ascending aorta:
    z_up = np.linspace(0.0, straight_up, max(n_up, 2))
    up_pts = p0[None, :] + z_up[:, None] * ez[None, :]

    # Create arc over ascending portion
    phi0 = np.deg2rad(0.0 + (180.0 - arch_angle_deg) * 0.5)
    phi1 = np.deg2rad(180.0 - (180.0 - arch_angle_deg) * 0.5)
    phi = np.linspace(phi0, phi1, max(n_arch, 3))

    # Place the arch center at the end of ascending, shifted by +R in z so the arch is above it.
    p_up_end = up_pts[-1]
    c = p_up_end + arch_radius * ez

    # Arch points in local (ex, ez) plane
    arch_pts = (c[None, :]
                + arch_radius * np.cos(phi)[:, None] * ex[None, :]
                + arch_radius * np.sin(phi)[:, None] * ez[None, :])

    # Connect smoothly: shift arch so its first point equals p_up_end
    arch_pts = arch_pts + (p_up_end - arch_pts[0])

    # Descending segment: along -z plus a bit along ex to keep going outward
    phi_end = phi[-1]
    t_end = (-np.sin(phi_end) * ex + np.cos(phi_end) * ez)
    t_end = t_end / (np.linalg.norm(t_end) + 1e-12)

    down_dir = t_end - 0.65 * ez
    down_dir = down_dir / (np.linalg.norm(down_dir) + 1e-12)

    s = np.linspace(0.0, end_straight, max(n_down, 2))
    down_pts = arch_pts[-1][None, :] + s[:, None] * down_dir[None, :]

    # Concatenate (avoid duplicate endpoints)
    cl = np.vstack([up_pts, arch_pts[1:], down_pts[1:]])

    # Build frames along centerline
    def unit(v):
        n = np.linalg.norm(v)
        return v / (n if n > 1e-12 else 1.0)

    Np = len(cl)
    T = np.zeros((Np, 3))
    for i in range(Np):
        if i == 0:
            T[i] = unit(cl[1] - cl[0])
        elif i == Np - 1:
            T[i] = unit(cl[-1] - cl[-2])
        else:
            T[i] = unit(cl[i + 1] - cl[i - 1])

    up = unit(np.array([0.0, 0.0, 1.0]))
    if abs(np.dot(up, T[0])) > 0.95:
        up = unit(np.array([0.0, 1.0, 0.0]))

    B0 = unit(np.cross(T[0], up))
    N0 = unit(np.cross(B0, T[0]))

    N_arr = np.zeros((Np, 3))
    B_arr = np.zeros((Np, 3))
    N_arr[0], B_arr[0] = N0, B0

    for i in range(1, Np):
        v = np.cross(T[i - 1], T[i])
        srot = np.linalg.norm(v)
        crot = np.dot(T[i - 1], T[i])
        if srot < 1e-10:
            N_arr[i] = N_arr[i - 1]
            B_arr[i] = B_arr[i - 1]
            continue
        v = v / srot
        ang = np.arctan2(srot, crot)

        a = N_arr[i - 1]
        N_new = a * np.cos(ang) + np.cross(v, a) * np.sin(ang) + v * np.dot(v, a) * (1 - np.cos(ang))
        N_new = unit(N_new)
        B_new = unit(np.cross(T[i], N_new))
        N_arr[i], B_arr[i] = N_new, B_new

    r_out = float(outer_radius)
    r_in = max(r_out - float(wall_thickness), 1e-6)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    outer_pts = np.zeros((Np, n_theta, 3))
    inner_pts = np.zeros((Np, n_theta, 3))

    for i in range(Np):
        twist = 2 * np.pi * twist_turns * (i / max(Np - 1, 1))
        ct = np.cos(theta + twist)
        st = np.sin(theta + twist)
        ring_dir = ct[:, None] * N_arr[i][None, :] + st[:, None] * B_arr[i][None, :]
        outer_pts[i] = cl[i][None, :] + r_out * ring_dir
        inner_pts[i] = cl[i][None, :] + r_in * ring_dir

    def skin(pts_grid):
        pts_flat = pts_grid.reshape(-1, 3)
        faces = []
        for i in range(Np - 1):
            for j in range(n_theta):
                jn = (j + 1) % n_theta
                p00 = i * n_theta + j
                p01 = i * n_theta + jn
                p11 = (i + 1) * n_theta + jn
                p10 = (i + 1) * n_theta + j
                faces.extend([4, p00, p01, p11, p10])
        return pv.PolyData(pts_flat, np.array(faces, dtype=np.int64)).triangulate().clean()

    outer = skin(outer_pts).compute_normals(auto_orient_normals=True)
    inner = skin(inner_pts).compute_normals(auto_orient_normals=True)

    inner_rev = inner.copy(deep=True)
    inner_rev.flip_faces()
    solid = outer.merge(inner_rev, merge_points=False).compute_normals(auto_orient_normals=True)

    return {
        "centerline": cl,
        "outer": outer,
        "inner": inner,
        "mesh": solid
    }


def visualize_geometry():
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

    # Create RA:
    ra_mesh = build_ra_mesh(lv_result, rv_result)

    # Create Pulmonary Trunk:
    pulm = build_pulmonary_trunk(rv_result)

    # Create Aorta:
    aorta = build_aorta(lv_result)

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

    plotter.add_mesh(la_mesh['endo_mesh'], color="darkgoldenrod", opacity=0.7)
    plotter.add_mesh(la_mesh['epi_mesh'], color="goldenrod", opacity=0.4)

    plotter.add_mesh(ra_mesh['endo_mesh'], color="darkviolet", opacity=0.7)
    plotter.add_mesh(ra_mesh['epi_mesh'], color="violet", opacity=0.4)

    plotter.add_mesh(pv.lines_from_points(pulm["centerline"]), color="darkgreen", line_width=3)
    plotter.add_mesh(pulm["mesh"], color="darkgreen", line_width=3)

    plotter.add_mesh(pv.lines_from_points(aorta["centerline"]), color="firebrick", line_width=3)
    plotter.add_mesh(aorta["mesh"], color="firebrick", line_width=3)


    # Add text annotation
    plotter.camera_position = 'iso'
    plotter.show()


if __name__ == "__main__":
    visualize_geometry()
