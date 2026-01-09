from __future__ import annotations

from typing import Callable, Dict, Tuple, Optional
import numpy as np
import pyvista as pv

# -------------------------
# Helpers / CSG
# -------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def translate_field(f: ScalarField, dx: float, dy: float, dz: float) -> ScalarField:
    def g(x: float, y: float, z: float) -> float:
        return f(x - dx, y - dy, z - dz)
    return g

def union(fa: ScalarField, fb: ScalarField) -> ScalarField:
    # inside if either inside
    def f(x: float, y: float, z: float) -> float:
        return min(fa(x, y, z), fb(x, y, z))
    return f

def difference(outer: ScalarField, inner: ScalarField) -> ScalarField:
    # outer \ inner
    def f(x: float, y: float, z: float) -> float:
        return max(outer(x, y, z), -inner(x, y, z))
    return f


# -------------------------
# LV: truncated ellipsoids (endo + epi) and myocardium shell
# -------------------------

def make_truncated_ellipsoid_segment_field(
    a: float, b: float, c: float,
    z0: float, z1: float,
    center: Tuple[float, float, float],
) -> ScalarField:
    if min(a, b, c) <= 0:
        raise ValueError("a,b,c must be > 0")
    if z1 < z0:
        raise ValueError("z1 must be >= z0")

    cx, cy, cz = center

    def f(x: float, y: float, z: float) -> float:
        X = x - cx
        Y = y - cy
        Z = z - cz
        g_shape = (X / a) ** 2 + (Y / b) ** 2 + (Z / c) ** 2 - 1.0
        g_zlow = z0 - Z
        g_zhigh = Z - z1
        return max(g_shape, g_zlow, g_zhigh)

    return f

def make_lv_fields(params: Dict[str, float]) -> Dict[str, ScalarField]:
    center = (params["lv_cx"], params["lv_cy"], params["lv_cz"])
    a_endo = params["lv_a_endo"]
    b_endo = params["lv_b_endo"]
    c_endo = params["lv_c_endo"]
    wall   = params["lv_wall"]
    z0     = params["lv_z0"]
    z1     = params["lv_z1"]

    f_lv_endo = make_truncated_ellipsoid_segment_field(a_endo, b_endo, c_endo, z0, z1, center)
    f_lv_epi  = make_truncated_ellipsoid_segment_field(a_endo + wall, b_endo + wall, c_endo + wall, z0, z1, center)
    f_lv_myo  = difference(f_lv_epi, f_lv_endo)

    return {"endo": f_lv_endo, "epi": f_lv_epi, "myo": f_lv_myo}