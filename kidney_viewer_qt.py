import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from vascular_tree import (
    add_partition_frame_overlay,
    add_scale_bar,
    default_partition_frame,
    get_colormap,
    load_camera_position,
    load_partition_frame,
    save_camera_position,
    save_high_resolution_screenshot,
    save_partition_frame,
    transform_color_limits,
    transform_color_values,
    vascular_tree_display_mask_partition,
)


def load_text_matrix(filename, usecols=None, skiprows=0):
    try:
        return np.loadtxt(filename, delimiter=",", usecols=usecols, skiprows=skiprows, ndmin=2)
    except ValueError:
        return np.loadtxt(filename, usecols=usecols, skiprows=skiprows, ndmin=2)


class KidneyViewerQt(QMainWindow):
    def __init__(self, elements_path=None, nodes_path=None, outputs_dir=None):
        super().__init__()
        self.setWindowTitle("Kidney Vascular Tree Viewer")
        self.resize(1500, 920)

        self.nodes = None
        self.elements = None
        self.so = None
        self.scalar_data = None
        self.times = None
        self.scalar_name = "Strahler order"
        self.has_rendered = False

        self.plotter = QtInteractor(self)
        self.controls = self.build_controls()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.controls)
        splitter.addWidget(self.plotter)
        splitter.setSizes([430, 1070])
        self.setCentralWidget(splitter)

        self.set_default_paths(elements_path=elements_path, nodes_path=nodes_path, outputs_dir=outputs_dir)
        if elements_path and nodes_path and outputs_dir:
            self.load_data()

    def build_controls(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        files = QGroupBox("Files")
        file_layout = QGridLayout(files)
        self.elements_path = self.file_row(file_layout, 0, "Elements", "Elements.txt")
        self.nodes_path = self.file_row(file_layout, 1, "Nodes", "Nodes.txt")
        self.outputs_dir = self.dir_row(file_layout, 2, "OUTPUTS", "OUTPUTS")
        self.screenshot_path = self.file_save_row(file_layout, 3, "Screenshot", "OUTPUTS/Vascular_Tree.png")
        self.camera_path = self.file_row(file_layout, 4, "Camera", "OUTPUTS/Camera_View.json")
        self.partition_path = self.file_row(file_layout, 5, "Partition", "OUTPUTS/Partition_Frame.json")
        layout.addWidget(files)

        display = QGroupBox("Display")
        display_layout = QFormLayout(display)
        self.color_by = QComboBox()
        self.color_by.addItems(["Strahler order", "Pressure", "Flow rate"])
        self.color_by.currentTextChanged.connect(self.load_scalar_data)
        display_layout.addRow("Color coding", self.color_by)

        self.colormap = QComboBox()
        self.colormap.addItems(["jet", "viridis", "plasma", "turbo", "coolwarm", "inferno", "magma", "cividis", "parula"])
        self.colormap.currentTextChanged.connect(self.update_scene)
        display_layout.addRow("Colormap", self.colormap)

        self.log_color = QCheckBox()
        self.log_color.stateChanged.connect(self.update_color_limits)
        self.log_color.stateChanged.connect(self.update_scene)
        display_layout.addRow("Log color", self.log_color)

        self.show_partition = QCheckBox()
        self.show_partition.setChecked(False)
        self.show_partition.stateChanged.connect(self.update_scene)
        display_layout.addRow("Show partition planes", self.show_partition)

        self.paper_quality = QCheckBox()
        self.paper_quality.stateChanged.connect(self.update_scene)
        display_layout.addRow("Paper quality tubes", self.paper_quality)

        self.color_min = QDoubleSpinBox()
        self.color_max = QDoubleSpinBox()
        for spin in (self.color_min, self.color_max):
            spin.setDecimals(8)
            spin.setRange(-1.0e30, 1.0e30)
            spin.setKeyboardTracking(False)
            spin.editingFinished.connect(self.update_scene)
        display_layout.addRow("Color min", self.color_min)
        display_layout.addRow("Color max", self.color_max)

        self.radius_scale = QDoubleSpinBox()
        self.radius_scale.setDecimals(6)
        self.radius_scale.setRange(1.0e-9, 1.0e9)
        self.radius_scale.setValue(0.35)
        self.radius_scale.setKeyboardTracking(False)
        self.radius_scale.editingFinished.connect(self.update_scene)
        display_layout.addRow("Radius scale", self.radius_scale)

        self.tube_sides = QSpinBox()
        self.tube_sides.setRange(3, 64)
        self.tube_sides.setValue(8)
        self.tube_sides.setKeyboardTracking(False)
        self.tube_sides.editingFinished.connect(self.update_scene)
        display_layout.addRow("Tube sides", self.tube_sides)

        self.quality_sides = QSpinBox()
        self.quality_sides.setRange(6, 96)
        self.quality_sides.setValue(24)
        self.quality_sides.setKeyboardTracking(False)
        self.quality_sides.editingFinished.connect(self.update_scene)
        display_layout.addRow("Paper tube sides", self.quality_sides)
        layout.addWidget(display)

        time_group = QGroupBox("Time")
        time_layout = QVBoxLayout(time_group)
        self.time_label = QLabel("Time: -")
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.valueChanged.connect(self.update_time_label)
        self.time_slider.valueChanged.connect(self.update_scene)
        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.time_slider)
        layout.addWidget(time_group)

        partition = QGroupBox("Partition Frame")
        partition_layout = QFormLayout(partition)
        self.origin_x = self.float_spin()
        self.origin_y = self.float_spin()
        self.origin_z = self.float_spin()
        self.rot_x = self.angle_spin()
        self.rot_y = self.angle_spin()
        self.rot_z = self.angle_spin()
        for spin in (self.origin_x, self.origin_y, self.origin_z, self.rot_x, self.rot_y, self.rot_z):
            spin.setKeyboardTracking(False)
            spin.editingFinished.connect(self.update_scene)
        partition_layout.addRow("Origin X", self.origin_x)
        partition_layout.addRow("Origin Y", self.origin_y)
        partition_layout.addRow("Origin Z", self.origin_z)
        partition_layout.addRow("Rotation X", self.rot_x)
        partition_layout.addRow("Rotation Y", self.rot_y)
        partition_layout.addRow("Rotation Z", self.rot_z)

        self.min_xp_yp = self.so_spin(1)
        self.min_xm_yp = self.so_spin(3)
        self.min_xm_ym = self.so_spin(3)
        self.min_xp_ym = self.so_spin(3)
        for spin in (self.min_xp_yp, self.min_xm_yp, self.min_xm_ym, self.min_xp_ym):
            spin.setKeyboardTracking(False)
            spin.editingFinished.connect(self.update_scene)
        partition_layout.addRow("X+ Y+ min SO", self.min_xp_yp)
        partition_layout.addRow("X- Y+ min SO", self.min_xm_yp)
        partition_layout.addRow("X- Y- min SO", self.min_xm_ym)
        partition_layout.addRow("X+ Y- min SO", self.min_xp_ym)
        layout.addWidget(partition)

        buttons = QGroupBox("Actions")
        button_layout = QGridLayout(buttons)
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        self.center_button = QPushButton("Use Data Center")
        self.center_button.clicked.connect(self.use_data_center)
        self.update_button = QPushButton("Refresh View")
        self.update_button.clicked.connect(self.update_scene)
        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.clicked.connect(self.save_image)
        self.save_camera_button = QPushButton("Save Camera")
        self.save_camera_button.clicked.connect(self.save_camera)
        self.load_camera_button = QPushButton("Load Camera")
        self.load_camera_button.clicked.connect(self.load_camera)
        self.save_frame_button = QPushButton("Save Frame")
        self.save_frame_button.clicked.connect(self.save_frame)
        self.load_frame_button = QPushButton("Load Frame")
        self.load_frame_button.clicked.connect(self.load_frame)
        button_layout.addWidget(self.load_button, 0, 0)
        button_layout.addWidget(self.center_button, 0, 1)
        button_layout.addWidget(self.update_button, 1, 0)
        button_layout.addWidget(self.save_image_button, 1, 1)
        button_layout.addWidget(self.save_camera_button, 2, 0)
        button_layout.addWidget(self.load_camera_button, 2, 1)
        button_layout.addWidget(self.save_frame_button, 3, 0)
        button_layout.addWidget(self.load_frame_button, 3, 1)
        layout.addWidget(buttons)

        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(420)
        return scroll

    def file_row(self, layout, row, label, default):
        edit = QLineEdit(str(Path(default).resolve()))
        button = QPushButton("Browse")
        button.clicked.connect(lambda: self.pick_file(edit))
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(edit, row, 1)
        layout.addWidget(button, row, 2)
        return edit

    def file_save_row(self, layout, row, label, default):
        edit = QLineEdit(str(Path(default).resolve()))
        button = QPushButton("Browse")
        button.clicked.connect(lambda: self.pick_save_file(edit))
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(edit, row, 1)
        layout.addWidget(button, row, 2)
        return edit

    def dir_row(self, layout, row, label, default):
        edit = QLineEdit(str(Path(default).resolve()))
        button = QPushButton("Browse")
        button.clicked.connect(lambda: self.pick_dir(edit))
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(edit, row, 1)
        layout.addWidget(button, row, 2)
        return edit

    def pick_file(self, edit):
        filename, _ = QFileDialog.getOpenFileName(self, "Select file")
        if filename:
            edit.setText(filename)

    def pick_save_file(self, edit):
        filename, _ = QFileDialog.getSaveFileName(self, "Save file")
        if filename:
            edit.setText(filename)

    def pick_dir(self, edit):
        directory = QFileDialog.getExistingDirectory(self, "Select folder")
        if directory:
            edit.setText(directory)
            self.camera_path.setText(str((Path(directory) / "Camera_View.json").resolve()))
            self.partition_path.setText(str((Path(directory) / "Partition_Frame.json").resolve()))
            self.screenshot_path.setText(str((Path(directory) / "Vascular_Tree.png").resolve()))

    def float_spin(self):
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1.0e12, 1.0e12)
        spin.setSingleStep(100.0)
        return spin

    def angle_spin(self):
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(-360.0, 360.0)
        spin.setSingleStep(1.0)
        return spin

    def so_spin(self, value):
        spin = QSpinBox()
        spin.setRange(1, 100)
        spin.setValue(value)
        return spin

    def set_default_paths(self, elements_path=None, nodes_path=None, outputs_dir=None):
        explicit_paths = (
            (self.elements_path, elements_path),
            (self.nodes_path, nodes_path),
            (self.outputs_dir, outputs_dir),
        )
        for edit, path in explicit_paths:
            if path:
                edit.setText(str(Path(path).resolve()))

        defaults = (
            (self.elements_path, Path("Elements.txt")),
            (self.nodes_path, Path("Nodes.txt")),
            (self.outputs_dir, Path("OUTPUTS")),
        )
        for edit, path in defaults:
            if not edit.text() and path.exists():
                edit.setText(str(path.resolve()))

        outputs_path = Path(self.outputs_dir.text()) if self.outputs_dir.text() else None
        if outputs_path is not None:
            self.camera_path.setText(str((outputs_path / "Camera_View.json").resolve()))
            self.partition_path.setText(str((outputs_path / "Partition_Frame.json").resolve()))
            self.screenshot_path.setText(str((outputs_path / "Vascular_Tree.png").resolve()))

    def show_error(self, message):
        QMessageBox.critical(self, "Viewer error", str(message))

    def load_data(self):
        try:
            self.elements = load_text_matrix(self.elements_path.text(), usecols=(0, 1, 2, 3))
            self.nodes = load_text_matrix(self.nodes_path.text(), usecols=(0, 1, 2))
            self.so = np.loadtxt(Path(self.outputs_dir.text()) / "StrahlerOrder.txt", dtype=float, ndmin=1)
            partition_path = Path(self.partition_path.text())
            if partition_path.exists():
                self.load_frame()
            else:
                self.use_data_center()
            self.load_scalar_data()
            camera_path = Path(self.camera_path.text())
            if camera_path.exists():
                self.load_camera()
        except Exception as exc:
            self.show_error(exc)

    def load_scalar_data(self, *_args):
        if self.outputs_dir.text() == "":
            return
        outputs_dir = Path(self.outputs_dir.text())
        color_by = self.color_by.currentText()
        try:
            if color_by == "Strahler order":
                self.scalar_data = np.loadtxt(outputs_dir / "StrahlerOrder.txt", dtype=float, ndmin=1)[:, None]
                self.times = None
                self.scalar_name = "Strahler order"
            elif color_by == "Pressure":
                self.scalar_data = load_text_matrix(outputs_dir / "Pressure_Elements_Time.txt")
                self.times = np.loadtxt(outputs_dir / "Solution_Times.txt", ndmin=1)
                self.scalar_name = "Pressure (mmHg)"
            else:
                self.scalar_data = load_text_matrix(outputs_dir / "Flowrate_Elements_Time.txt")
                self.times = np.loadtxt(outputs_dir / "Solution_Times.txt", ndmin=1)
                self.scalar_name = "Flow rate (mL/min)"
            self.time_slider.setMaximum(max(0, self.scalar_data.shape[1] - 1))
            self.update_time_label()
            self.update_color_limits()
            self.update_scene()
        except Exception as exc:
            self.show_error(exc)

    def update_color_limits(self, *_args):
        if self.scalar_data is None:
            return
        data = self.scalar_data
        if self.log_color.isChecked():
            data = np.abs(data)
            data = data[data > 0]
        if data.size == 0:
            return
        self.color_min.blockSignals(True)
        self.color_max.blockSignals(True)
        self.color_min.setValue(float(np.nanmin(data)))
        self.color_max.setValue(float(np.nanmax(data)))
        self.color_min.blockSignals(False)
        self.color_max.blockSignals(False)

    def update_time_label(self, *_args):
        if self.times is None or len(self.times) == 0:
            self.time_label.setText("Time: -")
        else:
            index = self.time_slider.value()
            self.time_label.setText(f"Time: {self.times[index]:.6g} s ({index + 1}/{len(self.times)})")

    def get_partition_frame(self):
        return {
            "origin": [self.origin_x.value(), self.origin_y.value(), self.origin_z.value()],
            "rotation_degrees": [self.rot_x.value(), self.rot_y.value(), self.rot_z.value()],
        }

    def set_partition_frame(self, frame):
        origin = frame["origin"]
        rotation = frame["rotation_degrees"]
        self.origin_x.setValue(float(origin[0]))
        self.origin_y.setValue(float(origin[1]))
        self.origin_z.setValue(float(origin[2]))
        self.rot_x.setValue(float(rotation[0]))
        self.rot_y.setValue(float(rotation[1]))
        self.rot_z.setValue(float(rotation[2]))

    def get_region_min_strahler(self):
        return {
            "x_plus_y_plus": self.min_xp_yp.value(),
            "x_minus_y_plus": self.min_xm_yp.value(),
            "x_minus_y_minus": self.min_xm_ym.value(),
            "x_plus_y_minus": self.min_xp_ym.value(),
        }

    def use_data_center(self):
        if self.nodes is None:
            try:
                self.nodes = load_text_matrix(self.nodes_path.text(), usecols=(0, 1, 2))
            except Exception:
                return
        frame = default_partition_frame(self.nodes)
        self.set_partition_frame(frame)

    def filtered_data(self):
        mask = vascular_tree_display_mask_partition(
            self.nodes,
            self.elements,
            self.so,
            self.get_partition_frame(),
            self.get_region_min_strahler(),
        )
        return self.elements[mask], self.scalar_data[mask]

    def build_tube_mesh(self, elements, scalar_values):
        connectivity = elements[:, :2].astype(np.int64, copy=False) - 1
        radii = elements[:, 2].astype(np.float32, copy=False) * np.float32(self.radius_scale.value())
        color_values = transform_color_values(scalar_values[:, self.time_slider.value()], self.log_color.isChecked())

        if self.paper_quality.isChecked():
            lines = np.empty((elements.shape[0], 3), dtype=np.int64)
            lines[:, 0] = 2
            lines[:, 1:] = connectivity

            line_mesh = pv.PolyData(self.nodes[:, :3], lines=lines.ravel())
            line_mesh.cell_data["Radius"] = radii
            line_mesh.cell_data["Color Value"] = color_values
            point_mesh = line_mesh.cell_data_to_point_data()
            return point_mesh.tube(
                scalars="Radius",
                absolute=True,
                n_sides=self.quality_sides.value(),
                capping=True,
            )

        n_elem = elements.shape[0]
        tube_points = np.empty((2 * n_elem, 3), dtype=np.float32)
        tube_points[0::2] = self.nodes[connectivity[:, 0], :3]
        tube_points[1::2] = self.nodes[connectivity[:, 1], :3]

        lines = np.empty((n_elem, 3), dtype=np.int64)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(0, 2 * n_elem, 2, dtype=np.int64)
        lines[:, 2] = lines[:, 1] + 1

        line_mesh = pv.PolyData(tube_points, lines=lines.ravel())
        line_mesh.point_data["Radius"] = np.repeat(radii, 2)
        line_mesh.point_data["Color Value"] = np.repeat(color_values, 2)
        line_mesh.cell_data["Color Value"] = color_values
        return line_mesh.tube(
            scalars="Radius",
            absolute=True,
            n_sides=self.tube_sides.value(),
            capping=True,
        )

    def update_scene(self, *_args):
        if self.nodes is None or self.elements is None or self.scalar_data is None or self.so is None:
            return
        try:
            current_camera = self.plotter.camera_position if self.has_rendered else None
            self.plotter.clear()
            self.configure_lighting()
            elements, scalar_values = self.filtered_data()
            tube_mesh = self.build_tube_mesh(elements, scalar_values)
            clim = transform_color_limits((self.color_min.value(), self.color_max.value()), self.log_color.isChecked())
            self.plotter.add_mesh(
                tube_mesh,
                scalars="Color Value",
                cmap=get_colormap(self.colormap.currentText()),
                clim=clim,
                log_scale=self.log_color.isChecked(),
                show_scalar_bar=True,
                smooth_shading=self.paper_quality.isChecked(),
                specular=0.18 if self.paper_quality.isChecked() else 0.08,
                specular_power=18 if self.paper_quality.isChecked() else 10,
                ambient=0.38 if self.paper_quality.isChecked() else 0.45,
                diffuse=0.72,
                scalar_bar_args={
                    "title": "",
                    "vertical": False,
                    "position_x": 0.20,
                    "position_y": 0.04,
                    "width": 0.60,
                    "height": 0.08,
                },
            )
            if self.show_partition.isChecked():
                add_partition_frame_overlay(self.plotter, self.nodes, self.get_partition_frame())
            add_scale_bar(self.plotter, self.nodes)
            self.add_colorbar_caption()
            self.plotter.add_axes(labels_off=True)
            if current_camera is None:
                self.plotter.reset_camera(render=False)
            else:
                self.plotter.camera_position = current_camera
            self.has_rendered = True
            self.plotter.render()
        except Exception as exc:
            self.show_error(exc)

    def configure_lighting(self):
        try:
            self.plotter.disable_eye_dome_lighting()
        except AttributeError:
            pass
        try:
            self.plotter.remove_all_lights()
        except AttributeError:
            pass

        if self.nodes is not None:
            center = np.mean(self.nodes[:, :3], axis=0)
            span = np.ptp(self.nodes[:, :3], axis=0)
            distance = max(float(np.max(span)), 1.0) * 2.0
        else:
            center = np.zeros(3)
            distance = 1.0

        light_specs = [
            (center + distance * np.array([1.0, -1.0, 1.0]), 0.75),
            (center + distance * np.array([-0.7, 0.5, 0.8]), 0.55),
            (center + distance * np.array([0.0, 0.0, -1.0]), 0.30),
        ]
        for position, intensity in light_specs:
            light = pv.Light(position=position, focal_point=center, light_type="scene light")
            light.intensity = intensity
            self.plotter.add_light(light)

    def add_colorbar_caption(self):
        try:
            import vtk
        except Exception:
            self.plotter.add_text(self.scalar_name, position=(510, 8), font_size=11, color="black")
            return

        text = vtk.vtkTextActor()
        text.SetInput(self.scalar_name)
        text.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text.SetPosition(0.50, 0.012)
        text.GetTextProperty().SetFontSize(18)
        text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        text.GetTextProperty().SetJustificationToCentered()
        self.plotter.renderer.AddActor2D(text)

    def save_image(self):
        try:
            save_high_resolution_screenshot(self.plotter, self.screenshot_path.text(), scale=6)
        except Exception as exc:
            self.show_error(exc)

    def save_camera(self):
        try:
            save_camera_position(self.plotter.camera_position, self.camera_path.text())
        except Exception as exc:
            self.show_error(exc)

    def load_camera(self):
        try:
            self.plotter.camera_position = load_camera_position(self.camera_path.text())
            self.plotter.render()
        except Exception as exc:
            self.show_error(exc)

    def save_frame(self):
        try:
            frame = self.get_partition_frame()
            frame["region_min_strahler"] = self.get_region_min_strahler()
            save_partition_frame(frame, self.partition_path.text())
        except Exception as exc:
            self.show_error(exc)

    def load_frame(self):
        try:
            frame = load_partition_frame(self.partition_path.text())
            self.set_partition_frame(frame)
            region = frame.get("region_min_strahler", {})
            self.min_xp_yp.setValue(int(region.get("x_plus_y_plus", self.min_xp_yp.value())))
            self.min_xm_yp.setValue(int(region.get("x_minus_y_plus", self.min_xm_yp.value())))
            self.min_xm_ym.setValue(int(region.get("x_minus_y_minus", self.min_xm_ym.value())))
            self.min_xp_ym.setValue(int(region.get("x_plus_y_minus", self.min_xp_ym.value())))
            self.update_scene()
        except Exception as exc:
            self.show_error(exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kidney vascular tree Qt viewer")
    parser.add_argument("--elements", default=None, help="Path to Elements.txt")
    parser.add_argument("--nodes", default=None, help="Path to Nodes.txt")
    parser.add_argument("--outputs", default=None, help="Path to OUTPUTS folder")
    args = parser.parse_args()

    app = QApplication([])
    window = KidneyViewerQt(
        elements_path=args.elements,
        nodes_path=args.nodes,
        outputs_dir=args.outputs,
    )
    window.show()
    app.exec()
