"""
Interactive GUI for exploring the biventricular model.

This embeds a PyVista 3D viewport in a Qt application and exposes model
parameters via sliders/spinboxes so you can see the geometry update live.

Dependencies:
  - numpy, scipy
  - pyvista, pyvistaqt
  - a Qt binding (PyQt6/PyQt5 or PySide6). This file uses `qtpy` to support any.

Install (example):
  pip install numpy scipy pyvista pyvistaqt qtpy PySide6

Run:
  python biventricular_gui.py
"""

from __future__ import annotations

import sys
import os
import platform
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy import QtCore, QtWidgets

from biventricular_model import (
    compute_curve_normals,
    create_lv_default_control_points,
    create_lv_mesh,
    offset_curve,
    rv_revolve_profile,
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class FloatSliderRow(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int = 3,
        tooltip: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)
        if maximum <= minimum:
            raise ValueError("maximum must be > minimum")
        if step <= 0:
            raise ValueError("step must be > 0")

        self._min = float(minimum)
        self._max = float(maximum)
        self._step = float(step)
        self._decimals = int(decimals)
        self._updating = False

        steps = int(round((self._max - self._min) / self._step))
        steps = max(1, steps)
        self._steps = steps

        name = QtWidgets.QLabel(label)
        name.setMinimumWidth(170)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._steps)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(max(1, self._steps // 20))

        self._spin = QtWidgets.QDoubleSpinBox()
        self._spin.setDecimals(self._decimals)
        self._spin.setRange(self._min, self._max)
        self._spin.setSingleStep(self._step)
        self._spin.setKeyboardTracking(False)
        self._spin.setFixedWidth(110)

        if tooltip:
            name.setToolTip(tooltip)
            self._slider.setToolTip(tooltip)
            self._spin.setToolTip(tooltip)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(name)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._spin)
        self.setLayout(layout)

        self.setValue(value)

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spin.valueChanged.connect(self._on_spin_changed)

    def value(self) -> float:
        return float(self._spin.value())

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 - Qt API
        super().setEnabled(enabled)
        self._slider.setEnabled(enabled)
        self._spin.setEnabled(enabled)

    def setValue(self, v: float) -> None:  # noqa: N802 - Qt API
        v = _clamp(float(v), self._min, self._max)
        slider_value = int(round((v - self._min) / self._step))
        slider_value = max(0, min(self._steps, slider_value))
        v = self._min + slider_value * self._step
        v = float(np.round(v, self._decimals))

        self._updating = True
        try:
            self._spin.setValue(v)
            self._slider.setValue(slider_value)
        finally:
            self._updating = False

    def _on_slider_changed(self, slider_value: int) -> None:
        if self._updating:
            return
        v = self._min + float(slider_value) * self._step
        v = float(np.round(v, self._decimals))
        self._updating = True
        try:
            self._spin.setValue(v)
        finally:
            self._updating = False
        self.valueChanged.emit(v)

    def _on_spin_changed(self, v: float) -> None:
        if self._updating:
            return
        slider_value = int(round((float(v) - self._min) / self._step))
        slider_value = max(0, min(self._steps, slider_value))
        v2 = self._min + slider_value * self._step
        v2 = float(np.round(v2, self._decimals))
        self._updating = True
        try:
            self._slider.setValue(slider_value)
            if abs(v2 - float(v)) > (0.5 * self._step):
                self._spin.setValue(v2)
        finally:
            self._updating = False
        self.valueChanged.emit(float(self._spin.value()))


class IntSliderRow(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(int)

    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        value: int,
        step: int = 1,
        tooltip: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)
        if maximum <= minimum:
            raise ValueError("maximum must be > minimum")
        if step <= 0:
            raise ValueError("step must be > 0")

        self._updating = False

        name = QtWidgets.QLabel(label)
        name.setMinimumWidth(170)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setRange(minimum, maximum)
        self._slider.setSingleStep(step)
        self._slider.setPageStep(max(step, (maximum - minimum) // 20))

        self._spin = QtWidgets.QSpinBox()
        self._spin.setRange(minimum, maximum)
        self._spin.setSingleStep(step)
        self._spin.setKeyboardTracking(False)
        self._spin.setFixedWidth(110)

        if tooltip:
            name.setToolTip(tooltip)
            self._slider.setToolTip(tooltip)
            self._spin.setToolTip(tooltip)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(name)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._spin)
        self.setLayout(layout)

        self.setValue(value)

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spin.valueChanged.connect(self._on_spin_changed)

    def value(self) -> int:
        return int(self._spin.value())

    def setValue(self, v: int) -> None:  # noqa: N802 - Qt API
        self._updating = True
        try:
            self._slider.setValue(int(v))
            self._spin.setValue(int(v))
        finally:
            self._updating = False

    def _on_slider_changed(self, v: int) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self._spin.setValue(int(v))
        finally:
            self._updating = False
        self.valueChanged.emit(int(v))

    def _on_spin_changed(self, v: int) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self._slider.setValue(int(v))
        finally:
            self._updating = False
        self.valueChanged.emit(int(v))


@dataclass(frozen=True)
class DisplayOptions:
    show_lv_endo: bool = True
    show_lv_epi: bool = True
    show_rv_endo: bool = True
    show_rv_epi: bool = True
    show_profiles: bool = False
    show_axes: bool = True
    smooth_shading: bool = True
    wireframe: bool = False


def _create_rv_mesh_from_lv_epi(
    lv_epi_profile: np.ndarray,
    wall_thickness: float,
    num_theta: int,
    theta_end_rad: float,
    scale: float,
) -> dict:
    epi_profile = lv_epi_profile.copy()
    epi_tangents = np.gradient(epi_profile, axis=0)
    epi_normals = compute_curve_normals(epi_tangents)
    endo_profile = offset_curve(epi_profile, -epi_normals, wall_thickness)

    theta_range = (0.0, float(theta_end_rad))
    endo_mesh = rv_revolve_profile(endo_profile, num_theta=num_theta, theta_range=theta_range, scale=scale)
    epi_mesh = rv_revolve_profile(epi_profile, num_theta=num_theta, theta_range=theta_range, scale=scale)

    return {
        "endo_mesh": endo_mesh,
        "epi_mesh": epi_mesh,
        "endo_profile": endo_profile,
        "epi_profile": epi_profile,
        "epi_normals": epi_normals,
    }


class BiventricularViewer(QtWidgets.QMainWindow):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle("Biventricular Model Explorer")
        self.resize(1400, 850)

        pv.global_theme.allow_empty_mesh = True
        self._shown_once = False

        self._default_cp = create_lv_default_control_points()
        self._cp_r: list[FloatSliderRow] = []
        self._cp_z: list[FloatSliderRow] = []

        self._actors: set[str] = set()
        self._implicit_actor_name = "implicit_surface"
        self._last_lv: Optional[dict] = None
        self._last_rv: Optional[dict] = None
        self._implicit_mesh: Optional[pv.PolyData] = None

        self._mesh_update_timer = QtCore.QTimer(self)
        self._mesh_update_timer.setSingleShot(True)
        self._mesh_update_timer.timeout.connect(self._update_meshes)

        self._implicit_update_timer = QtCore.QTimer(self)
        self._implicit_update_timer.setSingleShot(True)
        self._implicit_update_timer.timeout.connect(self._update_implicit_surface)

        self._implicit_dirty = False

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(splitter)
        central.setLayout(layout)

        controls = self._build_controls()
        splitter.addWidget(controls)

        # Create the VTK/Qt widget with its final parent up-front.
        # Re-parenting a QVTKRenderWindowInteractor (which QtInteractor wraps)
        # can trigger X11 "BadWindow" errors on some setups.
        # Disable QtInteractor's internal auto-update timer. That timer can
        # trigger renders before the widget has a valid native window on some
        # X11 setups, leading to BadWindow errors.
        self.plotter = QtInteractor(splitter, auto_update=False)
        self.plotter.set_background("white")
        splitter.addWidget(self.plotter)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 940])

        self._status = QtWidgets.QLabel("Ready")
        self.statusBar().addPermanentWidget(self._status, stretch=1)

    # -------------------------
    # UI construction
    # -------------------------

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        if self._shown_once:
            return
        self._shown_once = True
        try:
            self.plotter.enable_anti_aliasing("ssaa")
        except Exception:
            try:
                self.plotter.enable_anti_aliasing("fxaa")
            except Exception:
                pass
        self._schedule_mesh_update()

    def _build_controls(self) -> QtWidgets.QWidget:
        tabs = QtWidgets.QTabWidget()

        tabs.addTab(self._build_tab_parameters(), "Parameters")
        tabs.addTab(self._build_tab_control_points(), "LV Control Points")
        tabs.addTab(self._build_tab_implicit(), "Implicit")

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(tabs)
        container.setLayout(layout)
        return container

    def _build_tab_parameters(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(10, 10, 10, 10)

        # Geometry parameters
        geom_box = QtWidgets.QGroupBox("Geometry")
        geom_layout = QtWidgets.QVBoxLayout()
        geom_box.setLayout(geom_layout)

        self.lv_wall = FloatSliderRow("LV wall thickness", 0.05, 1.25, 0.50, 0.01, tooltip="Offset from LV endo to epi")
        self.rv_wall = FloatSliderRow("RV wall thickness", 0.05, 0.90, 0.25, 0.01, tooltip="Offset from RV epi to endo")
        self.lv_num_theta = IntSliderRow("LV theta resolution", 16, 256, 64, step=1)
        self.rv_num_theta = IntSliderRow("RV theta resolution", 8, 192, 32, step=1)
        self.num_profile_samples = IntSliderRow("Profile samples", 20, 250, 80, step=1)
        self.spline_degree = IntSliderRow("Spline degree", 1, 5, 3, step=1)

        geom_layout.addWidget(self.lv_wall)
        geom_layout.addWidget(self.rv_wall)
        geom_layout.addWidget(self.lv_num_theta)
        geom_layout.addWidget(self.rv_num_theta)
        geom_layout.addWidget(self.num_profile_samples)
        geom_layout.addWidget(self.spline_degree)

        # RV shaping parameters
        rv_box = QtWidgets.QGroupBox("RV Shaping")
        rv_layout = QtWidgets.QVBoxLayout()
        rv_box.setLayout(rv_layout)

        self.rv_warp_scale = FloatSliderRow("RV warp scale", 0.0, 1.75, 0.70, 0.01, tooltip="Controls RV bulging in the free wall")
        self.rv_wrap_deg = FloatSliderRow("RV wrap angle (deg)", 90.0, 180.0, 180.0, 1.0, decimals=0, tooltip="Angular sweep for RV revolution")

        rv_layout.addWidget(self.rv_warp_scale)
        rv_layout.addWidget(self.rv_wrap_deg)

        # Display options
        display_box = QtWidgets.QGroupBox("Display")
        display_layout = QtWidgets.QVBoxLayout()
        display_box.setLayout(display_layout)

        self.cb_lv_endo = QtWidgets.QCheckBox("Show LV endocardium")
        self.cb_lv_endo.setChecked(True)
        self.cb_lv_epi = QtWidgets.QCheckBox("Show LV epicardium")
        self.cb_lv_epi.setChecked(True)
        self.cb_rv_endo = QtWidgets.QCheckBox("Show RV endocardium")
        self.cb_rv_endo.setChecked(True)
        self.cb_rv_epi = QtWidgets.QCheckBox("Show RV epicardium")
        self.cb_rv_epi.setChecked(True)
        self.cb_profiles = QtWidgets.QCheckBox("Show profile curves (x–z plane)")
        self.cb_profiles.setChecked(False)
        self.cb_axes = QtWidgets.QCheckBox("Show axes")
        self.cb_axes.setChecked(True)
        self.cb_smooth = QtWidgets.QCheckBox("Smooth shading")
        self.cb_smooth.setChecked(True)
        self.cb_wireframe = QtWidgets.QCheckBox("Wireframe")
        self.cb_wireframe.setChecked(False)

        display_layout.addWidget(self.cb_lv_endo)
        display_layout.addWidget(self.cb_lv_epi)
        display_layout.addWidget(self.cb_rv_endo)
        display_layout.addWidget(self.cb_rv_epi)
        display_layout.addWidget(self.cb_profiles)
        display_layout.addWidget(self.cb_axes)
        display_layout.addWidget(self.cb_smooth)
        display_layout.addWidget(self.cb_wireframe)

        actions_layout = QtWidgets.QHBoxLayout()
        self.btn_reset_all = QtWidgets.QPushButton("Reset all")
        self.btn_reset_camera = QtWidgets.QPushButton("Reset camera")
        actions_layout.addWidget(self.btn_reset_all)
        actions_layout.addWidget(self.btn_reset_camera)

        outer.addWidget(geom_box)
        outer.addWidget(rv_box)
        outer.addWidget(display_box)
        outer.addLayout(actions_layout)
        outer.addStretch(1)

        tab.setLayout(outer)

        # Signals
        for w in (
            self.lv_wall,
            self.rv_wall,
            self.lv_num_theta,
            self.rv_num_theta,
            self.num_profile_samples,
            self.spline_degree,
            self.rv_warp_scale,
            self.rv_wrap_deg,
        ):
            w.valueChanged.connect(self._on_any_param_changed)

        for cb in (
            self.cb_lv_endo,
            self.cb_lv_epi,
            self.cb_rv_endo,
            self.cb_rv_epi,
            self.cb_profiles,
            self.cb_axes,
            self.cb_smooth,
            self.cb_wireframe,
        ):
            cb.stateChanged.connect(self._on_any_param_changed)

        self.btn_reset_all.clicked.connect(self._reset_all)
        self.btn_reset_camera.clicked.connect(self._reset_camera)

        return tab

    def _build_tab_control_points(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(10, 10, 10, 10)

        help_text = QtWidgets.QLabel(
            "These are the LV endocardial B-spline control points in the (r, z) plane.\n"
            "Constraint: P0 is kept on the axis (r=0) and P0.z == P1.z to preserve the apex tangent."
        )
        help_text.setWordWrap(True)
        outer.addWidget(help_text)

        points_box = QtWidgets.QGroupBox("LV Endocardial Control Points")
        points_layout = QtWidgets.QVBoxLayout()
        points_box.setLayout(points_layout)

        cp_grid = QtWidgets.QGridLayout()
        cp_grid.setHorizontalSpacing(8)
        cp_grid.setVerticalSpacing(6)

        header_p = QtWidgets.QLabel("Point")
        header_r = QtWidgets.QLabel("r")
        header_z = QtWidgets.QLabel("z")
        header_p.setStyleSheet("font-weight: 600;")
        header_r.setStyleSheet("font-weight: 600;")
        header_z.setStyleSheet("font-weight: 600;")
        cp_grid.addWidget(header_p, 0, 0)
        cp_grid.addWidget(header_r, 0, 1)
        cp_grid.addWidget(header_z, 0, 2)

        self._cp_r.clear()
        self._cp_z.clear()

        for i, (r0, z0) in enumerate(self._default_cp):
            point_label = QtWidgets.QLabel(f"P{i}")

            r_row = FloatSliderRow("", 0.0, 3.5, float(r0), 0.01, decimals=2)
            z_row = FloatSliderRow("", -8.0, 4.0, float(z0), 0.01, decimals=2)

            if i == 0:
                r_row.setEnabled(False)
                r_row.setValue(0.0)

            self._cp_r.append(r_row)
            self._cp_z.append(z_row)

            cp_grid.addWidget(point_label, i + 1, 0)
            cp_grid.addWidget(r_row, i + 1, 1)
            cp_grid.addWidget(z_row, i + 1, 2)

            r_row.valueChanged.connect(self._on_any_param_changed)
            z_row.valueChanged.connect(self._on_control_point_z_changed)

        points_layout.addLayout(cp_grid)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_reset_cp = QtWidgets.QPushButton("Reset control points")
        btn_layout.addWidget(self.btn_reset_cp)
        btn_layout.addStretch(1)
        points_layout.addLayout(btn_layout)

        outer.addWidget(points_box)
        outer.addStretch(1)
        tab.setLayout(outer)

        self.btn_reset_cp.clicked.connect(self._reset_control_points)

        return tab

    def _build_tab_implicit(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(10, 10, 10, 10)

        info = QtWidgets.QLabel(
            "Optional distance-field view: sample an implicit distance function on a grid\n"
            "and extract an isosurface using marching cubes. This can be slower than the\n"
            "parametric surfaces, especially at high grid resolutions."
        )
        info.setWordWrap(True)
        outer.addWidget(info)

        box = QtWidgets.QGroupBox("Implicit Surface (Marching Cubes)")
        layout = QtWidgets.QVBoxLayout()
        box.setLayout(layout)

        self.cb_show_implicit = QtWidgets.QCheckBox("Show implicit surface")
        self.cb_show_implicit.setChecked(False)
        layout.addWidget(self.cb_show_implicit)

        type_layout = QtWidgets.QHBoxLayout()
        type_layout.addWidget(QtWidgets.QLabel("Type"))
        self.combo_implicit_type = QtWidgets.QComboBox()
        self.combo_implicit_type.addItem("Epicardium (union)", userData="epicardium")
        self.combo_implicit_type.addItem("Myocardium (wall)", userData="myocardium")
        type_layout.addWidget(self.combo_implicit_type, stretch=1)
        layout.addLayout(type_layout)

        self.implicit_resolution = IntSliderRow("Grid resolution", 24, 160, 72, step=1, tooltip="Resolution^3 samples")
        layout.addWidget(self.implicit_resolution)

        self.cb_auto_implicit = QtWidgets.QCheckBox("Auto-update implicit (slow)")
        self.cb_auto_implicit.setChecked(False)
        layout.addWidget(self.cb_auto_implicit)

        btns = QtWidgets.QHBoxLayout()
        self.btn_update_implicit = QtWidgets.QPushButton("Recompute implicit now")
        self.btn_export_stl = QtWidgets.QPushButton("Export implicit STL…")
        self.btn_screenshot = QtWidgets.QPushButton("Save screenshot…")
        btns.addWidget(self.btn_update_implicit)
        btns.addWidget(self.btn_export_stl)
        btns.addWidget(self.btn_screenshot)
        layout.addLayout(btns)

        outer.addWidget(box)
        outer.addStretch(1)
        tab.setLayout(outer)

        # Signals
        self.cb_show_implicit.stateChanged.connect(self._on_any_param_changed)
        self.combo_implicit_type.currentIndexChanged.connect(self._on_any_param_changed)
        self.implicit_resolution.valueChanged.connect(self._on_any_param_changed)
        self.cb_auto_implicit.stateChanged.connect(self._on_any_param_changed)
        self.btn_update_implicit.clicked.connect(self._force_implicit_update)
        self.btn_export_stl.clicked.connect(self._export_implicit_stl)
        self.btn_screenshot.clicked.connect(self._save_screenshot)

        return tab

    # -------------------------
    # Parameter gathering
    # -------------------------

    def _display_options(self) -> DisplayOptions:
        return DisplayOptions(
            show_lv_endo=self.cb_lv_endo.isChecked(),
            show_lv_epi=self.cb_lv_epi.isChecked(),
            show_rv_endo=self.cb_rv_endo.isChecked(),
            show_rv_epi=self.cb_rv_epi.isChecked(),
            show_profiles=self.cb_profiles.isChecked(),
            show_axes=self.cb_axes.isChecked(),
            smooth_shading=self.cb_smooth.isChecked(),
            wireframe=self.cb_wireframe.isChecked(),
        )

    def _current_control_points(self) -> np.ndarray:
        cp = np.zeros_like(self._default_cp, dtype=float)
        for i in range(len(cp)):
            cp[i, 0] = float(self._cp_r[i].value())
            cp[i, 1] = float(self._cp_z[i].value())

        # Enforce constraints for a stable apex.
        cp[0, 0] = 0.0
        cp[1, 1] = cp[0, 1]

        cp[:, 0] = np.maximum(cp[:, 0], 0.0)
        return cp

    # -------------------------
    # Update pipeline (debounced)
    # -------------------------

    def _on_control_point_z_changed(self, _: float) -> None:
        # Keep apex constraint tight: P0.z == P1.z by mirroring edits.
        if not self._cp_z:
            return
        z0 = float(self._cp_z[0].value())
        self._cp_z[1].setValue(z0)
        self._on_any_param_changed()

    def _on_any_param_changed(self, *_args) -> None:
        if self.cb_show_implicit.isChecked() and self.cb_auto_implicit.isChecked():
            self._implicit_dirty = True
        self._schedule_mesh_update()

    def _schedule_mesh_update(self) -> None:
        self._mesh_update_timer.start(60)

    def _schedule_implicit_update(self) -> None:
        self._implicit_update_timer.start(250)

    # -------------------------
    # Rendering
    # -------------------------

    def _remove_actor(self, name: str) -> None:
        try:
            self.plotter.remove_actor(name)
        except Exception:
            pass
        self._actors.discard(name)

    def _set_actor_mesh(self, name: str, mesh: pv.DataSet, **kwargs) -> None:
        self._remove_actor(name)
        self.plotter.add_mesh(mesh, name=name, reset_camera=False, **kwargs)
        self._actors.add(name)

    def _update_meshes(self) -> None:
        options = self._display_options()
        if options.show_axes:
            self.plotter.show_axes()
        else:
            self.plotter.hide_axes()

        cp = self._current_control_points()

        try:
            lv = create_lv_mesh(
                cp,
                wall_thickness=float(self.lv_wall.value()),
                degree=int(self.spline_degree.value()),
                num_profile_samples=int(self.num_profile_samples.value()),
                num_theta=int(self.lv_num_theta.value()),
            )

            rv = _create_rv_mesh_from_lv_epi(
                lv_epi_profile=lv["epi_profile"],
                wall_thickness=float(self.rv_wall.value()),
                num_theta=int(self.rv_num_theta.value()),
                theta_end_rad=np.deg2rad(float(self.rv_wrap_deg.value())),
                scale=float(self.rv_warp_scale.value()),
            )
        except Exception as exc:
            self._status.setText(f"Model error: {exc}")
            return

        self._last_lv = lv
        self._last_rv = rv

        wireframe = options.wireframe
        smooth = options.smooth_shading and not wireframe

        if options.show_lv_endo:
            self._set_actor_mesh(
                "lv_endo",
                lv["endo_mesh"],
                color="firebrick",
                opacity=0.70,
                smooth_shading=smooth,
                style="wireframe" if wireframe else "surface",
            )
        else:
            self._remove_actor("lv_endo")

        if options.show_lv_epi:
            self._set_actor_mesh(
                "lv_epi",
                lv["epi_mesh"],
                color="pink",
                opacity=0.35,
                smooth_shading=smooth,
                style="wireframe" if wireframe else "surface",
            )
        else:
            self._remove_actor("lv_epi")

        if options.show_rv_endo:
            self._set_actor_mesh(
                "rv_endo",
                rv["endo_mesh"],
                color="royalblue",
                opacity=0.70,
                smooth_shading=smooth,
                style="wireframe" if wireframe else "surface",
            )
        else:
            self._remove_actor("rv_endo")

        if options.show_rv_epi:
            self._set_actor_mesh(
                "rv_epi",
                rv["epi_mesh"],
                color="lightblue",
                opacity=0.35,
                smooth_shading=smooth,
                style="wireframe" if wireframe else "surface",
            )
        else:
            self._remove_actor("rv_epi")

        if options.show_profiles:
            lv_endo_3d = np.column_stack([lv["endo_profile"][:, 0], np.zeros(len(lv["endo_profile"])), lv["endo_profile"][:, 1]])
            rv_epi_3d = np.column_stack([rv["epi_profile"][:, 0], np.zeros(len(rv["epi_profile"])), rv["epi_profile"][:, 1]])
            self._set_actor_mesh("profile_lv", pv.lines_from_points(lv_endo_3d), color="darkred", line_width=4)
            self._set_actor_mesh("profile_rv", pv.lines_from_points(rv_epi_3d), color="steelblue", line_width=4)
        else:
            self._remove_actor("profile_lv")
            self._remove_actor("profile_rv")

        if not self.cb_show_implicit.isChecked():
            self._remove_actor(self._implicit_actor_name)
            self._implicit_mesh = None
        elif self._implicit_dirty and self.cb_auto_implicit.isChecked():
            self._implicit_dirty = False
            self._schedule_implicit_update()

        self._status.setText(
            f"LV epi: {lv['epi_mesh'].n_points} pts, RV epi: {rv['epi_mesh'].n_points} pts"
            + (" | implicit: on" if self.cb_show_implicit.isChecked() else "")
        )

        self.plotter.render()

    def _force_implicit_update(self) -> None:
        if not self.cb_show_implicit.isChecked():
            self.cb_show_implicit.setChecked(True)
        self._update_implicit_surface()

    def _implicit_bounds(self, lv_epi: pv.PolyData, rv_epi: pv.PolyData) -> Tuple[float, float, float, float, float, float]:
        b1 = lv_epi.bounds
        b2 = rv_epi.bounds
        xmin = min(b1[0], b2[0])
        xmax = max(b1[1], b2[1])
        ymin = min(b1[2], b2[2])
        ymax = max(b1[3], b2[3])
        zmin = min(b1[4], b2[4])
        zmax = max(b1[5], b2[5])

        pad = 0.6
        return (xmin - pad, xmax + pad, ymin - pad, ymax + pad, zmin - pad, zmax + pad)

    def _update_implicit_surface(self) -> None:
        if not self.cb_show_implicit.isChecked():
            self._remove_actor(self._implicit_actor_name)
            self._implicit_mesh = None
            return
        if self._last_lv is None or self._last_rv is None:
            return

        lv = self._last_lv
        rv = self._last_rv

        kind = self.combo_implicit_type.currentData()
        res = int(self.implicit_resolution.value())

        # The generated meshes are truncated (open boundaries). Cap openings so
        # implicit distance fields don't create a spurious "closure" at the grid
        # bounds during marching cubes.
        hole_size = 1000.0
        lv_epi = (
            lv["epi_mesh"]
            .triangulate()
            .clean()
            .fill_holes(hole_size)
            .clean()
            .compute_normals(auto_orient_normals=True)
        )
        rv_epi = (
            rv["epi_mesh"]
            .triangulate()
            .clean()
            .fill_holes(hole_size)
            .clean()
            .compute_normals(auto_orient_normals=True)
        )

        bounds = self._implicit_bounds(lv_epi, rv_epi)
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        spacing = (
            (xmax - xmin) / (res - 1),
            (ymax - ymin) / (res - 1),
            (zmax - zmin) / (res - 1),
        )
        grid = pv.ImageData(dimensions=(res, res, res), spacing=spacing, origin=(xmin, ymin, zmin))

        self._status.setText(f"Computing implicit surface ({kind}, {res}^3)…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            # Reuse the grid to avoid copying it repeatedly.
            grid.compute_implicit_distance(lv_epi, inplace=True)
            lv_epi_dist = np.asarray(grid.point_data["implicit_distance"]).copy()
            del grid.point_data["implicit_distance"]

            grid.compute_implicit_distance(rv_epi, inplace=True)
            rv_epi_dist = np.asarray(grid.point_data["implicit_distance"]).copy()
            del grid.point_data["implicit_distance"]

            if kind == "myocardium":
                lv_endo = (
                    lv["endo_mesh"]
                    .triangulate()
                    .clean()
                    .fill_holes(hole_size)
                    .clean()
                    .compute_normals(auto_orient_normals=True)
                )
                rv_endo = (
                    rv["endo_mesh"]
                    .triangulate()
                    .clean()
                    .fill_holes(hole_size)
                    .clean()
                    .compute_normals(auto_orient_normals=True)
                )

                grid.compute_implicit_distance(lv_endo, inplace=True)
                lv_endo_dist = np.asarray(grid.point_data["implicit_distance"]).copy()
                del grid.point_data["implicit_distance"]

                grid.compute_implicit_distance(rv_endo, inplace=True)
                rv_endo_dist = np.asarray(grid.point_data["implicit_distance"]).copy()
                del grid.point_data["implicit_distance"]

                lv_myo = np.maximum(lv_epi_dist, -lv_endo_dist)
                rv_myo = np.maximum(rv_epi_dist, -rv_endo_dist)
                dist = np.minimum(lv_myo, rv_myo)
            else:
                dist = np.minimum(lv_epi_dist, rv_epi_dist)

            grid.point_data["distance"] = dist

            surface = grid.contour(isosurfaces=[0.0], scalars="distance", method="marching_cubes")
            surface = surface.clean()
            surface = surface.compute_normals(auto_orient_normals=True)
        except Exception as exc:
            self._status.setText(f"Implicit error: {exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._implicit_mesh = surface
        self._set_actor_mesh(
            self._implicit_actor_name,
            surface,
            color="wheat",
            opacity=0.75,
            smooth_shading=True,
        )

        self._status.setText(f"Implicit surface: {surface.n_points} pts, {surface.n_cells} cells")
        self.plotter.render()

    # -------------------------
    # Actions
    # -------------------------

    def _reset_camera(self) -> None:
        try:
            self.plotter.camera_position = "iso"
        except Exception:
            self.plotter.reset_camera()
        self.plotter.render()

    def _reset_control_points(self) -> None:
        for i, (r0, z0) in enumerate(self._default_cp):
            if i == 0:
                self._cp_r[i].setValue(0.0)
            else:
                self._cp_r[i].setValue(float(r0))
            self._cp_z[i].setValue(float(z0))

        # Enforce apex constraint immediately.
        self._cp_z[1].setValue(float(self._default_cp[0, 1]))
        self._schedule_mesh_update()

    def _reset_all(self) -> None:
        self.lv_wall.setValue(0.50)
        self.rv_wall.setValue(0.25)
        self.lv_num_theta.setValue(64)
        self.rv_num_theta.setValue(32)
        self.num_profile_samples.setValue(80)
        self.spline_degree.setValue(3)
        self.rv_warp_scale.setValue(0.70)
        self.rv_wrap_deg.setValue(180.0)

        self.cb_lv_endo.setChecked(True)
        self.cb_lv_epi.setChecked(True)
        self.cb_rv_endo.setChecked(True)
        self.cb_rv_epi.setChecked(True)
        self.cb_profiles.setChecked(False)
        self.cb_axes.setChecked(True)
        self.cb_smooth.setChecked(True)
        self.cb_wireframe.setChecked(False)

        self.cb_show_implicit.setChecked(False)
        self.cb_auto_implicit.setChecked(False)
        self.combo_implicit_type.setCurrentIndex(0)
        self.implicit_resolution.setValue(72)

        self._reset_control_points()

    def _export_implicit_stl(self) -> None:
        if self._implicit_mesh is None or self._implicit_mesh.n_points == 0:
            QtWidgets.QMessageBox.information(self, "Export STL", "No implicit surface is available. Click 'Recompute implicit now' first.")
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export implicit surface to STL",
            "biventricular_implicit.stl",
            "STL Files (*.stl)",
        )
        if not filename:
            return

        try:
            self._implicit_mesh.save(filename)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export STL", f"Failed to save STL:\n{exc}")
            return
        self._status.setText(f"Saved STL: {filename}")

    def _save_screenshot(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save screenshot",
            "biventricular.png",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)",
        )
        if not filename:
            return
        try:
            self.plotter.screenshot(filename)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Screenshot", f"Failed to save screenshot:\n{exc}")
            return
        self._status.setText(f"Saved screenshot: {filename}")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        try:
            self._mesh_update_timer.stop()
            self._implicit_update_timer.stop()
        except Exception:
            pass
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)


def main() -> int:
    # VTK's Qt interactor relies on an X11 window ID. Under a Wayland session,
    # Qt may run with the "wayland" platform plugin even when DISPLAY is set,
    # which can yield X11 BadWindow errors when VTK tries to configure the
    # render window. Prefer X11 via xcb unless the user explicitly overrides.
    if platform.system() == "Linux" and os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = BiventricularViewer()
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
