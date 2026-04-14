# -*- coding: utf-8 -*-
"""
GemPy 3D Model Builder – main dialog.

Structural-group logic
----------------------
* Faults are auto-detected from the formation column (any name containing
  "Fault").  Each fault gets its own FAULT group.
* The "Configure Structural Groups" button opens a full editor where the user
  can create groups with any relation type and assign formations freely.
* The configured layout is stored in ``self.structural_groups_config`` and
  used verbatim by ``gempy_core_plugin``.

GemPy API used
--------------
  gp.add_structural_group(model, group_index, structural_group_name,
                          elements, structural_relation, fault_relations)

  StackRelationType        : ERODE | ONLAP | FAULT | BASEMENT
  FaultsRelationSpecialCase: OFFSET_ALL | OFFSET_NONE | OFFSET_FORMATIONS
"""

import os
import tempfile

import numpy as np

from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import (
    QDialog, QMessageBox, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QSlider, QLabel,
    QSplitter, QListWidget, QLineEdit, QComboBox,
    QDialogButtonBox,
)
from qgis.PyQt.QtCore import Qt
from qgis.core import (
    QgsApplication, QgsProject, QgsMapLayer,
    QgsMapLayerProxyModel, QgsMessageLog, Qgis, QgsProviderRegistry,
)
from qgis.gui import QgsMapLayerComboBox
from qgis.utils import iface

from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import Qt

from pyvistaqt import QtInteractor
import pyvista as pv

from scipy.spatial import Delaunay
from stl import mesh

# ---------------------------------------------------------------------------
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'test_module_dialog_base.ui'))

RELATION_TYPES   = ['ERODE', 'ONLAP', 'FAULT', 'BASEMENT']
FAULT_REL_TYPES  = ['OFFSET_ALL', 'OFFSET_NONE', 'OFFSET_FORMATIONS']

# Known coordinate column names to skip when searching for the formation column
_COORD_COLS = {'x', 'y', 'z', 'gx', 'gy', 'gz'}


def _find_formation_field(layer):
    """Return the field name that holds formation/surface names in a GemPy CSV layer.

    Strategy (in order):
    1. Any field whose name is exactly 'surface' or 'formation' (case-insensitive).
    2. The last field whose name is not a known coordinate name.
    3. fields[3] if the layer has ≥ 4 fields (standard X,Y,Z,surface layout).
    4. None (caller should skip processing).
    """
    if layer is None:
        return None
    fields = layer.fields()
    if fields.count() == 0:
        return None

    # 1 — exact match on common GemPy column names
    for fname in [f.name() for f in fields]:
        if fname.lower() in ('surface', 'formation', 'litho', 'lithology', 'unit'):
            return fname

    # 2 — last field that is not a coordinate column
    non_coord = [f.name() for f in fields if f.name().lower() not in _COORD_COLS]
    if non_coord:
        return non_coord[-1]

    # 3 — positional fallback: standard 4-column CSV (X,Y,Z,surface)
    if fields.count() >= 4:
        return fields[3].name()

    return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _clean_layer_path(layer):
    """Return the plain file-system path for a QGIS layer (strips URI params)."""
    try:
        decoded = QgsProviderRegistry.instance().decodeUri(
            layer.providerType(), layer.source())
        path = decoded.get('path', '')
        if path:
            return path
    except Exception:
        pass
    uri = layer.source()
    if '?' in uri:
        uri = uri.split('?')[0]
    if uri.startswith('file:///'):
        uri = uri[8:]
    elif uri.startswith('file://'):
        uri = uri[7:]
    return uri


# ---------------------------------------------------------------------------
# Structural-groups editor dialog
# ---------------------------------------------------------------------------
class StructuralGroupsDialog(QDialog):
    """
    Full editor for GemPy structural groups.

    Data model
    ----------
    self.groups  – list of dicts (ordered youngest → oldest):
        {
            'name'          : str,
            'relation'      : 'ERODE' | 'ONLAP' | 'FAULT' | 'BASEMENT',
            'fault_relation': 'OFFSET_ALL' | 'OFFSET_NONE' | 'OFFSET_FORMATIONS',
            'formations'    : [str, ...]
        }

    Notes
    -----
    * BASEMENT is automatically applied by GemPy to the last group regardless
      of what relation you set; the option is included for clarity.
    * fault_relation is only meaningful when relation == 'FAULT'.
    """

    def __init__(self, geology_layer, existing_groups=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Structural Groups")
        self.setMinimumSize(750, 520)

        self.geology_layer = geology_layer
        self.all_formations = self._load_formations()
        self.groups = []
        self._current_row = -1
        self._updating = False          # block recursive signal handling

        self._build_ui()

        if existing_groups:
            self.groups = [dict(g) for g in existing_groups]
            # deep-copy formation lists
            for g in self.groups:
                g['formations'] = list(g['formations'])
            self._refresh_groups_list()
        else:
            self._auto_populate()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_formations(self):
        names = []
        layer = self.geology_layer
        if layer and layer.type() == QgsMapLayer.VectorLayer:
            fname = _find_formation_field(layer)
            if fname:
                seen = set()
                for feat in layer.getFeatures():
                    v = feat[fname]
                    if v is not None and str(v) not in seen:
                        seen.add(str(v))
                        names.append(str(v))
                names.sort()
        return names

    def _auto_populate(self):
        faults     = sorted([n for n in self.all_formations if 'Fault' in n])
        non_faults = sorted([n for n in self.all_formations if 'Fault' not in n])
        for i, fn in enumerate(faults):
            self.groups.append({
                'name'          : f'fault_series_{i + 1}',
                'relation'      : 'FAULT',
                'fault_relation': 'OFFSET_ALL',
                'formations'    : [fn],
            })
        if non_faults:
            self.groups.append({
                'name'          : 'default_formation',
                'relation'      : 'ERODE',
                'fault_relation': 'OFFSET_ALL',
                'formations'    : non_faults,
            })
        self._refresh_groups_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        main = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left: group list ──────────────────────────────────────────
        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.addWidget(QLabel("<b>Groups</b>  (youngest → oldest / top → bottom)"))

        self.groups_list = QListWidget()
        self.groups_list.currentRowChanged.connect(self._on_group_selected)
        ll.addWidget(self.groups_list)

        btn_row = QHBoxLayout()
        for label, slot, tip in [
            ('+ Append',       self._add_group,
             'Add a new group at the bottom (oldest position)'),
            ('↑ Insert above', self._insert_group_above,
             'Insert a new group above the selected one\n'
             'Use this to add an unconformity surface between two sequences'),
            ('− Remove',       self._remove_group,
             'Remove the selected group (its formations become unassigned)'),
            ('↑ Up',           self._move_up,   'Move group one position younger (upward)'),
            ('↓ Down',         self._move_down, 'Move group one position older (downward)'),
        ]:
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        ll.addLayout(btn_row)
        splitter.addWidget(lw)

        # ── Right: group properties ───────────────────────────────────
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.addWidget(QLabel("<b>Group properties</b>"))

        # Name
        row = QHBoxLayout()
        row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.textEdited.connect(self._on_name_changed)
        row.addWidget(self.name_edit)
        rl.addLayout(row)

        # Relation type
        row = QHBoxLayout()
        row.addWidget(QLabel("Relation type:"))
        self.relation_combo = QComboBox()
        self.relation_combo.addItems(RELATION_TYPES)
        self.relation_combo.setToolTip(
            "ERODE    – Angular unconformity: this group's top surface truncates\n"
            "           (erodes) all older groups below it. Use for erosional\n"
            "           unconformities between stratigraphic sequences.\n\n"
            "ONLAP    – Onlap unconformity: younger layers lap onto an older\n"
            "           surface without eroding it. Use for sedimentary onlap\n"
            "           onto a basin margin or existing topography.\n\n"
            "FAULT    – Cross-cutting body. Use for faults (set Fault offsets to\n"
            "           OFFSET_ALL / OFFSET_FORMATIONS) or for intrusions / dikes\n"
            "           (set Fault offsets to OFFSET_NONE — the body cuts through\n"
            "           existing formations without displacing them).\n\n"
            "BASEMENT – The bottommost unit. GemPy always applies this to the\n"
            "           last group regardless of what you set here.")
        self.relation_combo.currentTextChanged.connect(self._on_relation_changed)
        row.addWidget(self.relation_combo)
        rl.addLayout(row)

        # Fault-relation (visible only when FAULT is selected)
        self.fault_rel_row = QWidget()
        fr = QHBoxLayout(self.fault_rel_row)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.addWidget(QLabel("Fault offsets:"))
        self.fault_rel_combo = QComboBox()
        self.fault_rel_combo.addItems(FAULT_REL_TYPES)
        self.fault_rel_combo.setToolTip(
            "OFFSET_ALL         – fault offsets all other groups (standard fault)\n"
            "OFFSET_FORMATIONS  – fault offsets formations only, not other faults\n"
            "OFFSET_NONE        – cross-cuts formations without displacing them;\n"
            "                     use this for dikes and other intrusive bodies")
        self.fault_rel_combo.currentTextChanged.connect(self._on_fault_rel_changed)
        fr.addWidget(self.fault_rel_combo)
        rl.addWidget(self.fault_rel_row)
        self.fault_rel_row.setVisible(False)

        rl.addWidget(QLabel("<b>Formations in this group:</b>"))

        self.formations_list = QListWidget()
        rl.addWidget(self.formations_list)

        # Add / remove formation row
        af = QHBoxLayout()
        self.add_formation_combo = QComboBox()
        self.add_formation_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        af.addWidget(self.add_formation_combo, stretch=1)
        btn_add_f = QPushButton("Add →")
        btn_add_f.clicked.connect(self._add_formation)
        af.addWidget(btn_add_f)
        btn_rem_f = QPushButton("← Remove")
        btn_rem_f.clicked.connect(self._remove_formation)
        af.addWidget(btn_rem_f)
        rl.addLayout(af)

        splitter.addWidget(rw)
        splitter.setSizes([320, 430])
        main.addWidget(splitter)

        # ── Info label ────────────────────────────────────────────────
        note = QLabel(
            "<b>How to model unconformities:</b><br>"
            "Create two or more formation groups and set the relation type to define "
            "the contact between them:<br>"
            "&nbsp;&nbsp;<b>ERODE</b> — the younger group's surface cuts (erodes) the older group below "
            "(angular unconformity).<br>"
            "&nbsp;&nbsp;<b>ONLAP</b> — the younger group laps onto the older surface without truncating it "
            "(onlap unconformity).<br>"
            "Use <i>↑ Insert above</i> to add a new sequence at any position. "
            "FAULT groups must always sit above (younger than) the groups they offset. "
            "GemPy treats the bottom group as BASEMENT automatically.")
        note.setWordWrap(True)
        main.addWidget(note)

        # ── OK / Cancel ───────────────────────────────────────────────
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self._on_accept)
        bbox.rejected.connect(self.reject)
        main.addWidget(bbox)

    # ------------------------------------------------------------------
    # List refresh helpers
    # ------------------------------------------------------------------
    def _group_label(self, g):
        lbl = f"{g['name']}  [{g['relation']}]"
        if g['relation'] == 'FAULT':
            lbl += f"  – {g['fault_relation']}"
        lbl += f"  ({len(g['formations'])} formation(s))"
        return lbl

    def _refresh_groups_list(self):
        prev = self.groups_list.currentRow()
        self.groups_list.blockSignals(True)
        self.groups_list.clear()
        for g in self.groups:
            self.groups_list.addItem(self._group_label(g))
        self.groups_list.blockSignals(False)
        row = max(0, min(prev, len(self.groups) - 1)) if self.groups else -1
        self.groups_list.setCurrentRow(row)
        if row != prev:
            self._on_group_selected(row)

    def _refresh_current_item(self):
        row = self._current_row
        if 0 <= row < self.groups_list.count():
            self.groups_list.item(row).setText(
                self._group_label(self.groups[row]))

    # ------------------------------------------------------------------
    # Group selection
    # ------------------------------------------------------------------
    def _on_group_selected(self, row):
        self._current_row = row
        if row < 0 or row >= len(self.groups):
            self.name_edit.clear()
            self.formations_list.clear()
            self.add_formation_combo.clear()
            return

        self._updating = True
        g = self.groups[row]
        self.name_edit.setText(g['name'])

        idx = self.relation_combo.findText(g['relation'])
        self.relation_combo.setCurrentIndex(max(0, idx))
        self.fault_rel_row.setVisible(g['relation'] == 'FAULT')

        idx = self.fault_rel_combo.findText(g['fault_relation'])
        self.fault_rel_combo.setCurrentIndex(max(0, idx))

        self.formations_list.clear()
        for fn in g['formations']:
            self.formations_list.addItem(fn)

        self._updating = False
        self._refresh_available_combo()

    def _refresh_available_combo(self):
        assigned = {fn for g in self.groups for fn in g['formations']}
        available = [fn for fn in self.all_formations if fn not in assigned]
        self.add_formation_combo.clear()
        for fn in available:
            self.add_formation_combo.addItem(fn)

    # ------------------------------------------------------------------
    # Property change slots
    # ------------------------------------------------------------------
    def _on_name_changed(self, text):
        if self._updating:
            return
        row = self._current_row
        if 0 <= row < len(self.groups):
            self.groups[row]['name'] = text
            self._refresh_current_item()

    def _on_relation_changed(self, text):
        if self._updating:
            return
        row = self._current_row
        if 0 <= row < len(self.groups):
            self.groups[row]['relation'] = text
            self.fault_rel_row.setVisible(text == 'FAULT')
            self._refresh_current_item()

    def _on_fault_rel_changed(self, text):
        if self._updating:
            return
        row = self._current_row
        if 0 <= row < len(self.groups):
            self.groups[row]['fault_relation'] = text
            self._refresh_current_item()

    # ------------------------------------------------------------------
    # Group CRUD / reorder
    # ------------------------------------------------------------------
    def _add_group(self):
        self.groups.append({
            'name'          : f'group_{len(self.groups) + 1}',
            'relation'      : 'ERODE',
            'fault_relation': 'OFFSET_ALL',
            'formations'    : [],
        })
        self._refresh_groups_list()
        self.groups_list.setCurrentRow(len(self.groups) - 1)

    def _insert_group_above(self):
        """Insert a new empty group above (younger than) the currently selected group.

        This is the primary way to add an unconformity: place the new group above
        an existing sequence, assign it ERODE or ONLAP, then assign formations.
        """
        row = self.groups_list.currentRow()
        insert_at = max(row, 0)   # if nothing selected, insert at top
        new_group = {
            'name'          : f'group_{len(self.groups) + 1}',
            'relation'      : 'ERODE',
            'fault_relation': 'OFFSET_ALL',
            'formations'    : [],
        }
        self.groups.insert(insert_at, new_group)
        self._refresh_groups_list()
        self.groups_list.setCurrentRow(insert_at)

    def _remove_group(self):
        row = self.groups_list.currentRow()
        if row < 0:
            return
        if len(self.groups) <= 1:
            QMessageBox.warning(self, "Warning", "At least one group is required.")
            return
        del self.groups[row]
        self._refresh_groups_list()
        self._refresh_available_combo()

    def _move_up(self):
        row = self.groups_list.currentRow()
        if row <= 0:
            return
        self.groups[row], self.groups[row - 1] = self.groups[row - 1], self.groups[row]
        self._refresh_groups_list()
        self.groups_list.setCurrentRow(row - 1)

    def _move_down(self):
        row = self.groups_list.currentRow()
        if row < 0 or row >= len(self.groups) - 1:
            return
        self.groups[row], self.groups[row + 1] = self.groups[row + 1], self.groups[row]
        self._refresh_groups_list()
        self.groups_list.setCurrentRow(row + 1)

    # ------------------------------------------------------------------
    # Formation CRUD
    # ------------------------------------------------------------------
    def _add_formation(self):
        fn = self.add_formation_combo.currentText()
        if not fn:
            return
        row = self._current_row
        if row < 0 or row >= len(self.groups):
            return
        if fn not in self.groups[row]['formations']:
            self.groups[row]['formations'].append(fn)
            self.formations_list.addItem(fn)
            self._refresh_available_combo()
            self._refresh_current_item()

    def _remove_formation(self):
        row = self._current_row
        if row < 0 or row >= len(self.groups):
            return
        item = self.formations_list.currentItem()
        if not item:
            return
        fn = item.text()
        self.groups[row]['formations'].remove(fn)
        self.formations_list.takeItem(self.formations_list.currentRow())
        self._refresh_available_combo()
        self._refresh_current_item()

    # ------------------------------------------------------------------
    # Accept / validate
    # ------------------------------------------------------------------
    def _on_accept(self):
        # Check every formation is assigned
        assigned = {fn for g in self.groups for fn in g['formations']}
        unassigned = [fn for fn in self.all_formations if fn not in assigned]
        if unassigned:
            ans = QMessageBox.question(
                self, "Unassigned formations",
                f"The following formations are not assigned to any group:\n"
                f"{', '.join(unassigned)}\n\nContinue anyway?",
                QMessageBox.Yes | QMessageBox.No)
            if ans == QMessageBox.No:
                return
        # Check no group is empty
        empty = [g['name'] for g in self.groups if not g['formations']]
        if empty:
            QMessageBox.warning(
                self, "Empty groups",
                f"These groups have no formations and will be skipped:\n"
                f"{', '.join(empty)}")
        self.accept()

    def get_groups(self):
        return self.groups


# ---------------------------------------------------------------------------
# PyVista viewer dialog
# ---------------------------------------------------------------------------
class PyVistaDialog(QDialog):
    """Standalone dialog for PyVista 3D visualisation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GemPy 3D Model Viewer")
        self.resize(800, 600)
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self.surface_actors = {}
        self.export_folder = None
        self.plotter_added = False
        self.plotter = None
        self.plot_3d = False
        self.elevation_data = None
        self.transform = None
        self._create_controls()
        self.setWindowModality(Qt.NonModal)

    def _create_controls(self):
        cw = QWidget()
        cl = QHBoxLayout(cw)
        for label, slot in [
            ("Reset View",  self.reset_camera),
            ("Clear",       self.clear_plot),
            ("Export STL",  self.export_stl_files),
        ]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            cl.addWidget(b)
        cl.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._update_opacity)
        cl.addWidget(self.opacity_slider)
        cl.addStretch()
        self._layout.addWidget(cw)

    def display_gempy_model(self, geo_model, export_folder=None,
                            plot_3d=False, elevation_data=None, transform=None):
        try:
            self.export_folder = export_folder
            self.plot_3d = plot_3d
            self.elevation_data = elevation_data
            self.transform = transform
            pv.set_plot_theme("document")

            if self.plotter is not None:
                try:
                    self.plotter.clear()
                except Exception:
                    try:
                        self.plotter.close()
                    except Exception:
                        pass
                    self.plotter = None

            if self.plotter is None:
                self.plotter = QtInteractor(self)
                if not self.plotter_added:
                    self._layout.addWidget(self.plotter.interactor)
                    self.plotter_added = True

            self.surface_actors = {}
            legend_entries = []

            # Build surfaces directly from GemPy structural elements
            # (element.vertices / element.edges are populated by gp.compute_model)
            elements = geo_model.structural_frame.structural_elements
            QgsMessageLog.logMessage(
                f"Elements: {[(e.name, 'ok' if e.vertices is not None and len(e.vertices) > 0 else 'no vertices') for e in elements]}",
                "TestPlugin", Qgis.Info)

            for element in elements:
                if element.vertices is None or element.vertices.shape[0] == 0:
                    QgsMessageLog.logMessage(
                        f"Skipping '{element.name}' — no vertices after compute",
                        "TestPlugin", Qgis.Warning)
                    continue
                if element.edges is None or element.edges.shape[0] == 0:
                    QgsMessageLog.logMessage(
                        f"Skipping '{element.name}' — no edges after compute",
                        "TestPlugin", Qgis.Warning)
                    continue
                try:
                    surf = pv.PolyData(
                        element.vertices,
                        np.insert(element.edges, 0, 3, axis=1).ravel())
                    color = element.color
                    opacity = self.opacity_slider.value() / 100.0
                    actor_new = self.plotter.add_mesh(
                        surf, name=element.name, color=color,
                        opacity=opacity, show_edges=False)
                    self.surface_actors[element.name] = actor_new
                    legend_entries.append(
                        [element.name, list(pv.Color(color).float_rgb)])
                except Exception as e:
                    QgsMessageLog.logMessage(
                        f"Error adding surface '{element.name}': {e}",
                        "TestPlugin", Qgis.Warning)

            # Add topography surface if enabled
            if plot_3d and geo_model.grid.topography is not None:
                try:
                    import gempy_viewer as gpv
                    from gempy_viewer.modules.plot_3d.drawer_topography_3d import plot_topography_3d
                    from gempy_viewer.core.scalar_data_type import TopographyDataType
                    from gempy_viewer.modules.plot_3d.vista import GemPyToVista

                    tmp = GemPyToVista(
                        extent=geo_model.grid.extent,
                        plotter_type='basic',
                        off_screen=True)
                    plot_topography_3d(
                        gempy_vista=tmp,
                        topography=geo_model.grid.topography,
                        solution=geo_model.solutions.raw_arrays
                        if geo_model.solutions else None,
                        topography_scalar_type=TopographyDataType.GEOMAP,
                        elements_colors=geo_model.structural_frame.elements_colors[::-1])
                    for name, surf in tmp.surface_poly.items():
                        actor_t = self.plotter.add_mesh(
                            surf, name=name, opacity=0.5, show_edges=False)
                        self.surface_actors[name] = actor_t
                    try:
                        tmp.p.close()
                    except Exception:
                        pass
                except Exception as e:
                    QgsMessageLog.logMessage(
                        f"Topography display error: {e}",
                        "TestPlugin", Qgis.Warning)

            if legend_entries:
                try:
                    self.plotter.add_legend(
                        legend_entries, bcolor="white",
                        size=(0.15, 0.15), loc="upper right")
                except Exception:
                    pass

            self.plotter.enable_trackball_style()
            self.plotter.reset_camera()
            self.plotter.render()
            QgsMessageLog.logMessage(
                f"Displayed {len(self.surface_actors)} surfaces",
                "TestPlugin", Qgis.Info)

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(
                f"Display error: {e}\n{traceback.format_exc()}",
                "TestPlugin", Qgis.Critical)

    def clear_plot(self):
        if self.plotter:
            self.plotter.clear()
            self.surface_actors = {}
            self.plotter.render()

    def reset_camera(self):
        if self.plotter:
            self.plotter.reset_camera()

    def _update_opacity(self, value):
        if not self.plotter:
            return
        opacity = value / 100.0
        for actor in self.plotter.renderer.actors.values():
            if hasattr(actor, 'GetProperty'):
                actor.GetProperty().SetOpacity(opacity)
        self.plotter.render()

    def export_stl_files(self):
        if not self.export_folder or not self.surface_actors:
            return
        for name, actor in self.surface_actors.items():
            if name not in ("topography", "topography_cont"):
                actor.mapper.dataset.save(
                    os.path.join(self.export_folder, f"{name}.stl"))
                QgsMessageLog.logMessage(
                    f"Exported: {name}.stl", "TestPlugin", Qgis.Info)

        if self.plot_3d and self.elevation_data is not None:
            ed = self.elevation_data
            tr = self.transform
            rows, cols = ed.shape
            x = np.arange(cols) * tr[0] + tr[2]
            y = np.arange(rows) * tr[4] + tr[5]
            X, Y = np.meshgrid(x, y)
            Xf, Yf, Zf = X.flatten(), Y.flatten(), ed.flatten()
            if np.ma.is_masked(ed):
                valid = ~ed.mask.flatten()
                Xf, Yf, Zf = Xf[valid], Yf[valid], Zf[valid]
            pts3 = np.column_stack([Xf, Yf, Zf.astype(float)])
            tri = Delaunay(np.column_stack([Xf, Yf]))
            n = len(tri.simplices)
            data = np.zeros(n, dtype=mesh.Mesh.dtype)
            for i, t in enumerate(tri.simplices):
                data['vectors'][i] = pts3[t]
            stl_mesh = mesh.Mesh(data, remove_empty_areas=True)
            stl_path = os.path.join(self.export_folder, "Topography.stl")
            stl_mesh.save(stl_path)
            QgsMessageLog.logMessage(
                f"Exported: Topography.stl", "TestPlugin", Qgis.Info)

    def closeEvent(self, event):
        try:
            if self.plotter:
                self.plotter.clear()
                self.plotter.close()
                self.plotter = None
                self.plotter_added = False
            self.surface_actors = {}
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Plotter cleanup: {e}", "TestPlugin", Qgis.Warning)
            self.plotter = None
            self.plotter_added = False
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Main plugin dialog
# ---------------------------------------------------------------------------
class testDialog(QDialog, FORM_CLASS):

    def __init__(self, parent=None):
        super(testDialog, self).__init__(parent)
        self.setupUi(self)

        self.selected_geology_layer     = None
        self.selected_orientation_layer = None
        self.selected_DTM_layer         = None
        self.selected_aoi_layer         = None

        self.geology_layer_path     = None
        self.orientation_layer_path = None
        self.DTM_layer_path         = None

        self.x_value         = None
        self.y_value         = None
        self.z_value         = None
        self.model_deep_value= None
        self.buffer_pct      = 6.0
        self.plot_3d         = False

        self.structural_groups_config = None   # None → auto-detect faults
        self.progress_dialog          = None
        self.pyvista_panel            = None

        self.setup_layer_combos()
        self.connect_signals()
        self._sync_initial_layers()

        self.input_x.setText("10")
        self.input_y.setText("10")
        self.input_z.setText("10")
        self.model_deep.setText("500")
        self.input_buffer.setText("6")
        self.plot_checkbox.setChecked(False)
        self.output_folder.setText("Select output folder...")

        # Try to auto-select layers named layer_i / layer_o
        for layer in QgsProject.instance().mapLayers().values():
            if _clean_layer_path(layer).lower().endswith('.csv'):
                if layer.name() == "layer_i":
                    self.GeologyComboBox.setLayer(layer)
                elif layer.name() == "layer_o":
                    self.OrientationComboBox.setLayer(layer)

        self._update_fault_status()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup_layer_combos(self):
        if isinstance(getattr(self, 'GeologyComboBox', None), QgsMapLayerComboBox):
            self.GeologyComboBox.setFilters(QgsMapLayerProxyModel.VectorLayer)
        if isinstance(getattr(self, 'OrientationComboBox', None), QgsMapLayerComboBox):
            self.OrientationComboBox.setFilters(QgsMapLayerProxyModel.VectorLayer)
        if isinstance(getattr(self, 'DTMComboBox', None), QgsMapLayerComboBox):
            self.DTMComboBox.setFilters(QgsMapLayerProxyModel.RasterLayer)
        if isinstance(getattr(self, 'AOIComboBox', None), QgsMapLayerComboBox):
            self.AOIComboBox.setFilters(QgsMapLayerProxyModel.PolygonLayer)
            self.AOIComboBox.setAllowEmptyLayer(True)
            self.AOIComboBox.setCurrentIndex(0)

    def connect_signals(self):
        if isinstance(getattr(self, 'GeologyComboBox', None), QgsMapLayerComboBox):
            self.GeologyComboBox.layerChanged.connect(self.on_geology_layer_changed)
        if isinstance(getattr(self, 'OrientationComboBox', None), QgsMapLayerComboBox):
            self.OrientationComboBox.layerChanged.connect(self.on_orientation_layer_changed)
        if isinstance(getattr(self, 'DTMComboBox', None), QgsMapLayerComboBox):
            self.DTMComboBox.layerChanged.connect(self.on_DTM_layer_changed)
        if isinstance(getattr(self, 'AOIComboBox', None), QgsMapLayerComboBox):
            self.AOIComboBox.layerChanged.connect(self.on_aoi_layer_changed)
        if hasattr(self, 'button_box'):
            self.button_box.accepted.connect(self.accept_selection)
            self.button_box.rejected.connect(self.reject)
        self.browseButton.clicked.connect(self.browse_output_folder)
        if hasattr(self, 'btnConfigureGroups'):
            self.btnConfigureGroups.clicked.connect(self._open_groups_editor)

    def _sync_initial_layers(self):
        if isinstance(getattr(self, 'GeologyComboBox', None), QgsMapLayerComboBox):
            layer = self.GeologyComboBox.currentLayer()
            if layer:
                self.selected_geology_layer = layer
                self.geology_layer_path = _clean_layer_path(layer)
        if isinstance(getattr(self, 'OrientationComboBox', None), QgsMapLayerComboBox):
            layer = self.OrientationComboBox.currentLayer()
            if layer:
                self.selected_orientation_layer = layer
                self.orientation_layer_path = _clean_layer_path(layer)
        if isinstance(getattr(self, 'DTMComboBox', None), QgsMapLayerComboBox):
            layer = self.DTMComboBox.currentLayer()
            if layer:
                self.selected_DTM_layer = layer
                self.DTM_layer_path = _clean_layer_path(layer)
        if isinstance(getattr(self, 'AOIComboBox', None), QgsMapLayerComboBox):
            self.selected_aoi_layer = self.AOIComboBox.currentLayer()

    # ------------------------------------------------------------------
    # Layer-change slots
    # ------------------------------------------------------------------
    def on_geology_layer_changed(self, layer):
        self.selected_geology_layer = layer
        if layer:
            self.geology_layer_path = _clean_layer_path(layer)
        self.structural_groups_config = None   # reset manual config on layer change
        self._update_fault_status()

    def on_orientation_layer_changed(self, layer):
        self.selected_orientation_layer = layer
        if layer:
            self.orientation_layer_path = _clean_layer_path(layer)

    def on_DTM_layer_changed(self, layer):
        self.selected_DTM_layer = layer
        if layer:
            self.DTM_layer_path = _clean_layer_path(layer)

    def on_aoi_layer_changed(self, layer):
        self.selected_aoi_layer = layer

    # ------------------------------------------------------------------
    # Fault auto-detection
    # ------------------------------------------------------------------
    def get_fault_list(self):
        """Return sorted unique formation names that contain 'Fault'."""
        names = []
        layer = self.selected_geology_layer
        if not layer or layer.type() != QgsMapLayer.VectorLayer:
            return names
        fname = _find_formation_field(layer)
        if not fname:
            return names
        seen = set()
        for feat in layer.getFeatures():
            v = feat[fname]
            if v and 'Fault' in str(v) and str(v) not in seen:
                seen.add(str(v))
                names.append(str(v))
        return sorted(names)

    def _update_fault_status(self):
        if not hasattr(self, 'lblGroupsStatus'):
            return
        if self.structural_groups_config:
            summary = ', '.join(
                f"{g['name']} [{g['relation']}]"
                for g in self.structural_groups_config)
            self.lblGroupsStatus.setText(f"Manual: {summary}")
        else:
            faults = self.get_fault_list()
            if faults:
                self.lblGroupsStatus.setText(
                    f"Auto – faults: {', '.join(faults)}")
            else:
                self.lblGroupsStatus.setText(
                    "Auto – no faults detected (single ERODE group)")

    # ------------------------------------------------------------------
    # Groups editor
    # ------------------------------------------------------------------
    def _open_groups_editor(self):
        if not self.selected_geology_layer:
            QMessageBox.information(
                self, "Structural Groups",
                "Please select a geology layer first.")
            return

        dlg = StructuralGroupsDialog(
            geology_layer=self.selected_geology_layer,
            existing_groups=self.structural_groups_config,
            parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            self.structural_groups_config = dlg.get_groups()
            QgsMessageLog.logMessage(
                f"User configured {len(self.structural_groups_config)} "
                f"structural group(s)", "TestPlugin", Qgis.Info)
        self._update_fault_status()

    # ------------------------------------------------------------------
    # Folder browser
    # ------------------------------------------------------------------
    def browse_output_folder(self):
        current = self.output_folder.text().strip()
        start = current if current and os.path.exists(current) \
            else os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(
            None, "Select Output Folder", start)
        if path:
            self.output_folder.setText(path)
            self.folder_path = path

    # ------------------------------------------------------------------
    # OK handler
    # ------------------------------------------------------------------
    def accept_selection(self):
        QgsMessageLog.logMessage("=== START NEW MODEL ===", "TestPlugin", Qgis.Info)

        if not self.selected_geology_layer:
            QMessageBox.warning(self, "Warning", "Please select a geology layer.")
            return
        if not self.selected_orientation_layer:
            QMessageBox.warning(self, "Warning", "Please select an orientation layer.")
            return
        if self.selected_geology_layer == self.selected_orientation_layer:
            QMessageBox.warning(
                self, "Warning",
                "Geology and Orientation layers must be different.")
            return

        try:
            self.x_value          = int(self.input_x.text().strip())
            self.y_value          = int(self.input_y.text().strip())
            self.z_value          = int(self.input_z.text().strip())
            self.model_deep_value = float(self.model_deep.text().strip())
            self.buffer_pct       = float(self.input_buffer.text().strip())
        except ValueError as e:
            QMessageBox.warning(self, "Warning", f"Invalid numeric input: {e}")
            return

        self.plot_3d     = self.plot_checkbox.isChecked()
        self.fault_names = self.get_fault_list()
        self.folder_path = self.output_folder.text().strip()

        QgsMessageLog.logMessage(
            f"Geology:     {self.geology_layer_path}", "TestPlugin", Qgis.Info)
        QgsMessageLog.logMessage(
            f"Orientation: {self.orientation_layer_path}", "TestPlugin", Qgis.Info)
        QgsMessageLog.logMessage(
            f"DTM:         {self.DTM_layer_path}", "TestPlugin", Qgis.Info)
        QgsMessageLog.logMessage(
            f"Resolution {self.x_value}x{self.y_value}x{self.z_value}  "
            f"Deep {self.model_deep_value}  Topo {self.plot_3d}",
            "TestPlugin", Qgis.Info)

        self.accept()
        self.gempy_core_plugin()

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------
    def show_qgis_progress(self, message="Processing..."):
        self.progress_dialog = QProgressDialog(message, "Cancel", 0, 100, None)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        QgsApplication.processEvents()

    def update_qgis_progress(self, value, message=None):
        if self.progress_dialog:
            self.progress_dialog.setValue(value)
            if message:
                self.progress_dialog.setLabelText(message)
            QgsApplication.processEvents()
            if self.progress_dialog.wasCanceled():
                return False
        return True

    def hide_qgis_progress(self, msg="Done"):
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        iface.messageBar().pushMessage("Success", msg, level=Qgis.Success, duration=3)

    # ------------------------------------------------------------------
    # AOI helpers
    # ------------------------------------------------------------------
    def _get_aoi_union(self):
        if not self.selected_aoi_layer:
            return None
        try:
            from shapely import wkt as shapely_wkt
            from shapely.ops import unary_union
            shapes = []
            for feat in self.selected_aoi_layer.getFeatures():
                geom = feat.geometry()
                if geom and not geom.isEmpty():
                    shapes.append(shapely_wkt.loads(geom.asWkt()))
            if shapes:
                return unary_union(shapes)
        except Exception as e:
            QgsMessageLog.logMessage(
                f"AOI geometry error: {e}", "TestPlugin", Qgis.Warning)
        return None

    def _filter_csv_to_aoi(self, csv_path, aoi_union, suffix, tmp_list):
        import pandas as pd
        from shapely.geometry import Point
        from shapely.prepared import prep

        df = pd.read_csv(csv_path)
        x_col = next((c for c in df.columns if c.upper() == 'X'), None)
        y_col = next((c for c in df.columns if c.upper() == 'Y'), None)
        if x_col and y_col:
            prepared = prep(aoi_union)
            mask = [
                prepared.contains(Point(float(r[x_col]), float(r[y_col])))
                for _, r in df.iterrows()
            ]
            df = df[mask]
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        df.to_csv(tmp_path, index=False)
        tmp_list.append(tmp_path)
        return tmp_path, len(df)

    # ------------------------------------------------------------------
    # GemPy core
    # ------------------------------------------------------------------
    def gempy_core_plugin(self):
        import gempy as gp
        import gempy_viewer as gpv
        import rasterio
        from gempy_engine.core.data.stack_relation_type import StackRelationType
        from gempy.core.data.structural_group import FaultsRelationSpecialCase

        # Map string keys → GemPy enum values
        relation_map = {
            'ERODE'   : StackRelationType.ERODE,
            'ONLAP'   : StackRelationType.ONLAP,
            'FAULT'   : StackRelationType.FAULT,
            'BASEMENT': StackRelationType.BASEMENT,
        }
        fault_rel_map = {
            'OFFSET_ALL'        : FaultsRelationSpecialCase.OFFSET_ALL,
            'OFFSET_NONE'       : FaultsRelationSpecialCase.OFFSET_NONE,
            'OFFSET_FORMATIONS' : FaultsRelationSpecialCase.OFFSET_FORMATIONS,
        }

        self.show_qgis_progress("Initializing GemPy model...")
        tmp_files = []

        try:
            # ── 1. Read DTM and determine model extent ─────────────────
            aoi_union = self._get_aoi_union()
            from rasterio.transform import array_bounds

            with rasterio.open(self.DTM_layer_path) as src:
                elevation_data = src.read(1)
                if src.nodata is not None:
                    elevation_data = np.ma.masked_equal(
                        elevation_data, src.nodata)
                transform = src.transform

                if aoi_union is not None:
                    # Clip DTM to AOI polygon
                    try:
                        from rasterio.mask import mask as rasterio_mask
                        from shapely.geometry import mapping
                        nodata_val = src.nodata if src.nodata is not None else np.nan
                        out_image, out_transform = rasterio_mask(
                            src, [mapping(aoi_union)],
                            crop=True, nodata=nodata_val)
                        elevation_data = out_image[0]
                        if src.nodata is not None:
                            elevation_data = np.ma.masked_equal(
                                elevation_data, src.nodata)
                        else:
                            elevation_data = np.ma.masked_invalid(elevation_data)
                        transform = out_transform
                    except Exception as e:
                        QgsMessageLog.logMessage(
                            f"AOI DTM clipping failed, using full DTM: {e}",
                            "TestPlugin", Qgis.Warning)
                        aoi_union = None  # fall back to DTM bounds

            # Raster pixel-aligned bounds (for topography coordinates)
            rows, cols = elevation_data.shape
            rast_min_x, rast_min_y, rast_max_x, rast_max_y = array_bounds(
                rows, cols, transform)

            if aoi_union is not None:
                # GemPy model extent from AOI polygon
                min_x, min_y, max_x, max_y = aoi_union.bounds
            else:
                # GemPy model extent from raster bounds
                min_x, min_y, max_x, max_y = (
                    rast_min_x, rast_min_y, rast_max_x, rast_max_y)

            # Z extent from valid (non-nodata) elevation only
            if isinstance(elevation_data, np.ma.MaskedArray):
                valid_elev = elevation_data.compressed()
            else:
                valid_elev = elevation_data[np.isfinite(elevation_data)]

            if valid_elev.size == 0:
                raise RuntimeError(
                    "DTM has no valid elevation data within the model area.\n"
                    "Check that the DTM and AOI overlap and the DTM is not "
                    "entirely nodata.")
            min_elev = float(valid_elev.min())
            max_elev = float(valid_elev.max())
            QgsMessageLog.logMessage(
                f"Model extent: X[{min_x:.1f}, {max_x:.1f}] "
                f"Y[{min_y:.1f}, {max_y:.1f}] "
                f"Z[{min_elev:.1f}, {max_elev:.1f}]",
                "TestPlugin", Qgis.Info)

            self.update_qgis_progress(5, "Loading geological data...")

            # ── 2. AOI filter of CSV files ─────────────────────────────
            sp_path = self.geology_layer_path
            or_path = self.orientation_layer_path

            if aoi_union is not None:
                n_sp = n_or = None
                try:
                    sp_path, n_sp = self._filter_csv_to_aoi(
                        sp_path, aoi_union, '_sp_aoi.csv', tmp_files)
                    or_path, n_or = self._filter_csv_to_aoi(
                        or_path, aoi_union, '_or_aoi.csv', tmp_files)
                    QgsMessageLog.logMessage(
                        f"AOI filtering: {n_sp} surface points, "
                        f"{n_or} orientations kept",
                        "TestPlugin", Qgis.Info)
                except Exception as e:
                    QgsMessageLog.logMessage(
                        f"AOI CSV filtering failed: {e}",
                        "TestPlugin", Qgis.Warning)
                if n_sp == 0:
                    raise RuntimeError(
                        "The AOI polygon contains no surface points from the geology layer.\n\n"
                        "Your geology data and AOI polygon do not overlap geographically.\n"
                        "Remove the AOI layer or use one that covers your geology data.")
                if n_or == 0:
                    raise RuntimeError(
                        "The AOI polygon contains no orientation points.\n\n"
                        "Your orientation data and AOI polygon do not overlap geographically.\n"
                        "Remove the AOI layer or use one that covers your orientation data.")

            # ── 3. Buffer / extent ─────────────────────────────────────
            xr = max_x - min_x
            yr = max_y - min_y
            zr = max_elev - min_elev
            block = max(xr / self.x_value,
                        yr / self.y_value,
                        zr / self.z_value)
            frac = self.buffer_pct / 100.0
            buf = max(frac * xr, frac * yr, frac * zr, block)
            if self.plot_3d:
                buf = 0

            # ── 4. Create GemPy model ──────────────────────────────────
            geo_model: gp.data.GeoModel = gp.create_geomodel(
                project_name='Model_1',
                extent=[
                    min_x - buf, max_x + buf,
                    min_y - buf, max_y + buf,
                    min_elev - self.model_deep_value - buf,
                    max_elev + buf,
                ],
                resolution=[self.x_value, self.y_value, self.z_value],
                importer_helper=gp.data.ImporterHelper(
                    path_to_orientations=or_path,
                    path_to_surface_points=sp_path,
                )
            )

            self.update_qgis_progress(10, "Configuring structural groups...")

            # ── 5. Structural groups ───────────────────────────────────
            default_group = geo_model.structural_frame.get_group_by_name(
                "default_formation")

            # Snapshot of all elements BEFORE any removal
            all_elements = {
                el.name: el
                for grp in geo_model.structural_frame.structural_groups
                for el in grp.elements
            }
            QgsMessageLog.logMessage(
                f"Formations in model: {sorted(all_elements.keys())}",
                "TestPlugin", Qgis.Info)

            # Decide which config to use
            if self.structural_groups_config:
                groups_cfg = self.structural_groups_config
            else:
                # Auto-detect: one FAULT group per fault, one ERODE for the rest
                fault_names   = [fn for fn in self.fault_names
                                 if fn in all_elements]
                non_fault     = [fn for fn in all_elements
                                 if fn not in set(fault_names)]
                groups_cfg = []
                for i, fn in enumerate(fault_names):
                    groups_cfg.append({
                        'name'          : f'fault_series_{i + 1}',
                        'relation'      : 'FAULT',
                        'fault_relation': 'OFFSET_ALL',
                        'formations'    : [fn],
                    })
                if non_fault:
                    groups_cfg.append({
                        'name'          : 'default_formation',
                        'relation'      : 'ERODE',
                        'fault_relation': 'OFFSET_ALL',
                        'formations'    : non_fault,
                    })

            # Remove ALL elements from default group
            for el in list(default_group.elements):
                default_group.elements.remove(el)

            # Create structural groups from config
            groups_created = 0
            for i, cfg in enumerate(groups_cfg):
                elements = [all_elements[fn]
                            for fn in cfg['formations']
                            if fn in all_elements]
                missing  = [fn for fn in cfg['formations']
                            if fn not in all_elements]
                for fn in missing:
                    QgsMessageLog.logMessage(
                        f"Formation '{fn}' not found in model data — skipped",
                        "TestPlugin", Qgis.Warning)

                if not elements:
                    QgsMessageLog.logMessage(
                        f"Group '{cfg['name']}' has no valid formations — skipped",
                        "TestPlugin", Qgis.Warning)
                    continue

                rel      = relation_map[cfg['relation']]
                flt_rel  = fault_rel_map[cfg['fault_relation']]

                gp.add_structural_group(
                    model                  = geo_model,
                    group_index            = i,
                    structural_group_name  = cfg['name'],
                    elements               = elements,
                    structural_relation    = rel,
                    fault_relations        = flt_rel,
                )
                groups_created += 1
                QgsMessageLog.logMessage(
                    f"Group '{cfg['name']}' [{cfg['relation']}]"
                    f"  fault_rel={cfg['fault_relation']}"
                    f"  formations={[e.name for e in elements]}",
                    "TestPlugin", Qgis.Info)

            # Remove the now-empty default group (or restore if nothing worked)
            if not default_group.elements and groups_created > 0:
                try:
                    geo_model.structural_frame.structural_groups.remove(
                        default_group)
                except ValueError:
                    pass
            elif groups_created == 0:
                # Fallback: put everything back so GemPy has at least one group
                for el in all_elements.values():
                    default_group.elements.append(el)
                QgsMessageLog.logMessage(
                    "No groups were created — using single default group",
                    "TestPlugin", Qgis.Warning)

            self.update_qgis_progress(15, "Setting topography...")

            # ── 6. Optional topography ─────────────────────────────────
            if self.plot_3d:
                xc = np.linspace(rast_min_x, rast_max_x, elevation_data.shape[1])
                yc = np.linspace(rast_min_y, rast_max_y, elevation_data.shape[0])
                XX, YY = np.meshgrid(xc, np.flipud(yc))

                # Flatten and filter out nodata pixels
                x_flat = XX.ravel()
                y_flat = YY.ravel()
                if isinstance(elevation_data, np.ma.MaskedArray):
                    z_flat = np.ma.filled(elevation_data, np.nan).ravel()
                else:
                    z_flat = elevation_data.ravel().astype(float)
                valid = np.isfinite(z_flat)
                topo = np.column_stack([x_flat[valid],
                                        y_flat[valid],
                                        z_flat[valid]])
                gp.set_topography_from_arrays(
                    grid=geo_model.grid, xyz_vertices=topo)

            self.update_qgis_progress(20, "Computing model...")

            # ── 7. Compute ─────────────────────────────────────────────
            geo_model.input_transform.apply_anisotropy(
                gp.data.GlobalAnisotropy.NONE)
            gp.compute_model(geo_model)

            # ── 8. Display ─────────────────────────────────────────────
            self.update_qgis_progress(90, "Displaying 3D model...")

            if (getattr(self, 'pyvista_panel', None)
                    and self.pyvista_panel.isVisible()):
                self.pyvista_panel.display_gempy_model(
                    geo_model, self.folder_path,
                    self.plot_3d, elevation_data, transform)
                self.pyvista_panel.raise_()
            else:
                if getattr(self, 'pyvista_panel', None):
                    try:
                        self.pyvista_panel.close()
                        self.pyvista_panel.deleteLater()
                    except Exception:
                        pass
                    self.pyvista_panel = None

                import gc
                gc.collect()
                QgsApplication.processEvents()

                self.pyvista_panel = PyVistaDialog(iface.mainWindow())
                self.pyvista_panel.setWindowModality(Qt.NonModal)
                self.pyvista_panel.show()
                self.pyvista_panel.raise_()
                QgsApplication.processEvents()

                self.pyvista_panel.display_gempy_model(
                    geo_model, self.folder_path,
                    self.plot_3d, elevation_data, transform)

            QgsApplication.processEvents()
            self.hide_qgis_progress("3D model displayed successfully!")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            QgsMessageLog.logMessage(
                f"gempy_core_plugin error:\n{tb}", "TestPlugin", Qgis.Critical)
            QMessageBox.critical(
                None, "GemPy Error",
                f"Model computation failed:\n\n{e}\n\n"
                "See QGIS Log Messages (TestPlugin) for details.")
        finally:
            for f in tmp_files:
                try:
                    os.remove(f)
                except Exception:
                    pass
