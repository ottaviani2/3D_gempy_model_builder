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
    QDialogButtonBox, QSpinBox,
)
from qgis.PyQt.QtCore import Qt
from qgis.core import (
    QgsApplication, QgsProject, QgsMapLayer,
    QgsMapLayerProxyModel, QgsMessageLog, Qgis, QgsProviderRegistry,
    QgsTask,
)
from qgis.gui import QgsMapLayerComboBox
from qgis.utils import iface

from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import Qt

from pyvistaqt import QtInteractor
import pyvista as pv

from scipy.spatial import Delaunay
from stl import mesh
try:
    from stl import Mode as _StlMode
except ImportError:  # older numpy-stl layouts
    _StlMode = None

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


def _vector_layer_to_csv(layer, tmp_list=None):
    """Return a filesystem CSV path for a QGIS vector layer.

    Layers already backed by a ``.csv`` file are returned unchanged. Any
    other vector format (e.g. an ESRI Shapefile, GeoPackage, GeoJSON…) is
    converted to a temporary CSV in the GemPy layout the rest of the plugin
    expects: every attribute field is preserved as a column, and ``X`` / ``Y``
    / ``Z`` columns are synthesised from each feature's geometry when they are
    not already present as attributes (case-insensitive). The temp path is
    appended to ``tmp_list`` (when supplied) so the caller can delete it later.
    """
    path = _clean_layer_path(layer)
    if path.lower().endswith('.csv'):
        return path

    import math
    import pandas as pd

    field_names = [f.name() for f in layer.fields()]
    lower_names = {n.lower() for n in field_names}
    add_x = 'x' not in lower_names
    add_y = 'y' not in lower_names
    add_z = 'z' not in lower_names

    rows = []
    for feat in layer.getFeatures():
        row = {n: feat[n] for n in field_names}
        geom = feat.geometry()
        if geom is not None and not geom.isEmpty():
            try:
                # First vertex as a QgsPoint (the point itself for point
                # layers; a representative node for multi-part geometries).
                v = geom.vertexAt(0)
                if add_x:
                    row['X'] = v.x()
                if add_y:
                    row['Y'] = v.y()
                if add_z:
                    z = v.z()
                    if not math.isnan(z):
                        row['Z'] = z
            except Exception:
                pass
        rows.append(row)

    df = pd.DataFrame(rows)
    fd, tmp_path = tempfile.mkstemp(suffix='_from_vector.csv')
    os.close(fd)
    df.to_csv(tmp_path, index=False)
    if tmp_list is not None:
        tmp_list.append(tmp_path)
    return tmp_path


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
        self.clip_surfaces = True
        self.topo_clip_mesh = None
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
            ("Export Plot", self.export_plot_image),
        ]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            cl.addWidget(b)

        cl.addWidget(QLabel("Topo cells/side:"))
        self.stl_cells_spin = QSpinBox()
        self.stl_cells_spin.setRange(50, 5000)
        self.stl_cells_spin.setSingleStep(50)
        self.stl_cells_spin.setValue(500)
        self.stl_cells_spin.setToolTip(
            "Max DTM cells per side for Topography.stl.\n"
            "Final size and estimated file weight update live below.")
        self.stl_cells_spin.valueChanged.connect(self._update_stl_preview)
        cl.addWidget(self.stl_cells_spin)

        cl.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._update_opacity)
        cl.addWidget(self.opacity_slider)
        cl.addStretch()
        self._layout.addWidget(cw)

        # Status row showing the last topography export size.
        self.stl_status_label = QLabel("")
        self.stl_status_label.setStyleSheet("color: #555;")
        self._layout.addWidget(self.stl_status_label)

    def _update_stl_preview(self, *_):
        """Refresh the status label with the projected decimated DTM size,
        triangle count and estimated binary STL file size for the current
        spinbox value. Called on spinbox change and when the viewer opens."""
        if not hasattr(self, 'stl_status_label'):
            return
        ed = getattr(self, 'elevation_data', None)
        if not getattr(self, 'plot_3d', False) or ed is None:
            self.stl_status_label.setText(
                "Topography export disabled — enable 3D topography before "
                "opening the viewer.")
            return
        rows, cols = ed.shape
        max_cells = int(self.stl_cells_spin.value())
        stride_y = max(1, rows // max_cells)
        stride_x = max(1, cols // max_cells)
        new_cols = cols // stride_x
        new_rows = rows // stride_y
        # Delaunay of a regular grid = 2 triangles per cell on top + bottom,
        # plus 2 triangles per boundary edge on the side walls.
        cx = max(1, new_cols - 1)
        cy = max(1, new_rows - 1)
        n_top = 2 * cx * cy
        n_sides = 4 * (cx + cy)
        total = 2 * n_top + n_sides
        size_mb = (total * 50 + 84) / (1024 * 1024)
        self.stl_status_label.setText(
            f"Preview: DTM {cols}x{rows} → {new_cols}x{new_rows} "
            f"(stride {stride_x},{stride_y}), ~{total:,} triangles, "
            f"~{size_mb:.1f} MB"
        )

    @staticmethod
    def _mesh_quality_report(poly, name):
        """Return a one-line mesh-quality summary for a PyVista surface.

        Reports vertex / triangle counts, the triangle aspect-ratio range,
        the minimum scaled Jacobian (a degeneracy / sliver indicator) and
        the number of non-manifold edges (a proxy for self-intersections,
        holes or T-junctions).  Every metric is computed defensively so a
        missing PyVista filter never breaks the export.
        """
        try:
            n_pts = int(poly.n_points)
            n_cells = int(poly.n_cells)
        except Exception:
            return f"{name}: mesh quality unavailable"

        if n_cells == 0:
            return f"{name}: empty mesh (0 triangles)"

        msg = f"{name}: {n_pts:,} vertices, {n_cells:,} triangles"

        # Aspect ratio (1 = equilateral; large = sliver triangle)
        try:
            qa = poly.compute_cell_quality(quality_measure='aspect_ratio')
            ar = np.asarray(qa.cell_data['CellQuality'])
            ar = ar[np.isfinite(ar)]
            if ar.size:
                msg += (f"; aspect ratio min/mean/max = "
                        f"{ar.min():.2f}/{ar.mean():.2f}/{ar.max():.2f}")
                n_sliver = int(np.sum(ar > 20.0))
                if n_sliver:
                    msg += f" ({n_sliver} slivers AR>20)"
        except Exception:
            pass

        # Minimum scaled Jacobian — values near 0 flag degenerate triangles
        try:
            qj = poly.compute_cell_quality(
                quality_measure='scaled_jacobian')
            sj = np.asarray(qj.cell_data['CellQuality'])
            sj = sj[np.isfinite(sj)]
            if sj.size:
                msg += f"; min scaled Jacobian = {sj.min():.3f}"
                n_deg = int(np.sum(np.abs(sj) <= 1e-6))
                if n_deg:
                    msg += f" ({n_deg} near-degenerate)"
        except Exception:
            pass

        # Non-manifold edges — proxy for self-intersections / holes
        try:
            nm = poly.extract_feature_edges(
                boundary_edges=False, feature_edges=False,
                manifold_edges=False, non_manifold_edges=True)
            n_nm = int(nm.n_cells)
            msg += f"; non-manifold edges = {n_nm}"
            if n_nm > 0:
                msg += " (check for self-intersections / T-junctions)"
        except Exception:
            pass

        return msg

    def display_gempy_model(self, geo_model, export_folder=None,
                            plot_3d=False, elevation_data=None, transform=None,
                            model_deep_value=0.0, clip_surfaces=True):
        # ``plot_3d`` here means "a topography was built" (so it is rendered
        # and exported); ``clip_surfaces`` independently controls whether the
        # structural surfaces are cut against the topography. Decoupling the
        # two lets a single run export the topography *and* the full,
        # un-clipped surfaces in one consistent coordinate frame.
        try:
            self.export_folder = export_folder
            self.plot_3d = plot_3d
            self.clip_surfaces = clip_surfaces
            self.elevation_data = elevation_data
            self.transform = transform
            self.model_deep_value = model_deep_value
            self._update_stl_preview()
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

            # Build the topography mesh first so structural surfaces can be
            # clipped against it (same approach as gempy_viewer's plot_surfaces).
            topo_clip_mesh = None
            topo_surface_polys = {}
            # Topography mesh used to clip the structural surfaces in the 3D
            # view (only when clip_surfaces is on). The exported Topography.stl
            # is built separately, straight from the source DTM, in
            # export_stl_files.
            self.topo_clip_mesh = None

            if plot_3d and geo_model.grid.topography is not None:
                try:
                    import warnings
                    from gempy_viewer.modules.plot_3d.drawer_topography_3d import plot_topography_3d
                    from gempy_viewer.core.scalar_data_type import TopographyDataType
                    from gempy_viewer.modules.plot_3d.vista import GemPyToVista

                    # gempy_viewer casts the geological_map raster to int via
                    # np.round(...).astype(int); NaN cells (outside AOI / model
                    # domain) trigger a "invalid value encountered in cast"
                    # RuntimeWarning and produce garbage indices. Sanitize NaN
                    # → 1 (first valid formation id) before the call, then
                    # restore.
                    sol_raw = (geo_model.solutions.raw_arrays
                               if geo_model.solutions else None)
                    orig_geomap = None
                    if sol_raw is not None:
                        gm = getattr(sol_raw, 'geological_map', None)
                        if (gm is not None and gm.size > 0
                                and np.any(~np.isfinite(gm))):
                            orig_geomap = gm
                            sol_raw.geological_map = np.where(
                                np.isfinite(gm), gm, 1)

                    tmp = GemPyToVista(
                        extent=geo_model.grid.extent,
                        plotter_type='basic',
                        off_screen=True)
                    try:
                        with warnings.catch_warnings(), \
                                np.errstate(invalid='ignore'):
                            warnings.simplefilter('ignore', RuntimeWarning)
                            plot_topography_3d(
                                gempy_vista=tmp,
                                topography=geo_model.grid.topography,
                                solution=sol_raw,
                                topography_scalar_type=TopographyDataType.GEOMAP,
                                elements_colors=geo_model.structural_frame.elements_colors[::-1])
                    finally:
                        if orig_geomap is not None:
                            sol_raw.geological_map = orig_geomap

                    topo_surface_polys = dict(tmp.surface_poly)
                    topo_clip_mesh = topo_surface_polys.get('topography', None)
                    self.topo_clip_mesh = topo_clip_mesh
                    try:
                        tmp.p.close()
                    except Exception:
                        pass
                except Exception as e:
                    QgsMessageLog.logMessage(
                        f"Topography mesh build error: {e}",
                        "TestPlugin", Qgis.Warning)

            # Build surfaces directly from GemPy structural elements
            # (element.vertices / element.edges are populated by gp.compute_model)
            elements = geo_model.structural_frame.structural_elements
            QgsMessageLog.logMessage(
                f"Elements: {[(e.name, 'ok' if e.vertices is not None and len(e.vertices) > 0 else 'no vertices') for e in elements]}",
                "TestPlugin", Qgis.Info)

            # Frame check: combined bounds of all surface vertices. Compare
            # with the "gempy topography (export source)" bounds above — they
            # share the model frame, so they must overlap. If they don't, the
            # geology layer and the DTM are georeferenced to different places.
            try:
                _allv = np.vstack([
                    np.asarray(e.vertices) for e in elements
                    if e.vertices is not None and len(e.vertices) > 0])
                QgsMessageLog.logMessage(
                    "FRAME CHECK — surfaces (element.vertices) bounds "
                    f"X[{_allv[:, 0].min():.1f}, {_allv[:, 0].max():.1f}] "
                    f"Y[{_allv[:, 1].min():.1f}, {_allv[:, 1].max():.1f}] "
                    f"Z[{_allv[:, 2].min():.1f}, {_allv[:, 2].max():.1f}]",
                    "TestPlugin", Qgis.Info)
            except Exception:
                pass

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
                    if clip_surfaces and topo_clip_mesh is not None:
                        try:
                            surf = surf.clip_surface(
                                topo_clip_mesh, invert=True)
                        except Exception as ce:
                            QgsMessageLog.logMessage(
                                f"Topography clip failed for '{element.name}': {ce}",
                                "TestPlugin", Qgis.Warning)
                    # Mesh-quality check on the computed surface
                    QgsMessageLog.logMessage(
                        "Mesh quality — "
                        + self._mesh_quality_report(surf, element.name),
                        "TestPlugin", Qgis.Info)
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

            # Add topography surface(s) for rendering.
            # show_scalar_bar=False suppresses the 'id' colour-bar that the
            # topography mesh otherwise triggers (it carries an 'id' scalar
            # array populated by gempy_viewer).
            for name, surf in topo_surface_polys.items():
                try:
                    actor_t = self.plotter.add_mesh(
                        surf, name=name, opacity=0.5, show_edges=False,
                        show_scalar_bar=False)
                    self.surface_actors[name] = actor_t
                except Exception as e:
                    QgsMessageLog.logMessage(
                        f"Topography display error for '{name}': {e}",
                        "TestPlugin", Qgis.Warning)

            # Belt-and-suspenders: remove any scalar bars that slipped through
            # (e.g. from meshes added earlier in this display cycle).
            try:
                self.plotter.remove_scalar_bar()
            except Exception:
                pass

            if legend_entries:
                try:
                    self.plotter.add_legend(
                        legend_entries, bcolor="white",
                        size=(0.15, 0.15), loc="upper right")
                except Exception:
                    pass

            # XYZ orientation widget (bottom-left corner)
            try:
                self.plotter.add_axes(
                    xlabel='X', ylabel='Y', zlabel='Z',
                    line_width=3, labels_off=False)
            except Exception:
                try:
                    self.plotter.show_axes()
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
        QgsApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for name, actor in self.surface_actors.items():
                if name not in ("topography", "topography_cont"):
                    ds = actor.mapper.dataset
                    # Export the surface exactly as generated by GemPy /
                    # clip_surface — no Taubin smoothing and no folded /
                    # overlapping-face repair. Those passes rewrote the
                    # geometry and produced the distorted surfaces the user
                    # reported, so the raw mesh is saved as-is. Only a
                    # triangulation pass is kept because the STL writer
                    # requires triangles; it splits any polygon into
                    # triangles without moving a single vertex.
                    surf = ds
                    try:
                        surf = surf.extract_geometry()
                    except Exception:
                        pass
                    try:
                        if not surf.is_all_triangles:
                            surf = surf.triangulate()
                    except Exception:
                        pass
                    QgsMessageLog.logMessage(
                        "Mesh quality (pre-export) — "
                        + self._mesh_quality_report(surf, name),
                        "TestPlugin", Qgis.Info)
                    surf.save(
                        os.path.join(self.export_folder, f"{name}.stl"))
                    sb = surf.bounds
                    QgsMessageLog.logMessage(
                        f"Exported: {name}.stl — bounds "
                        f"X[{sb[0]:.1f}, {sb[1]:.1f}] "
                        f"Y[{sb[2]:.1f}, {sb[3]:.1f}] "
                        f"Z[{sb[4]:.1f}, {sb[5]:.1f}]",
                        "TestPlugin", Qgis.Info)
                    QgsApplication.processEvents()

            # Topography export — re-triangulate the DTM (self.elevation_data
            # + self.transform) with scipy Delaunay and extrude into a solid
            # box. This is the original, known-good path: it uses the very
            # elevation grid the model was built from, so the Topography.stl
            # lines up with the structural surfaces. NEW additions on top of
            # the original: (1) a Z-only smoothing of the DTM heights before
            # triangulation, and (2) a pymeshfix mesh-correction pass on the
            # final box.
            if self.plot_3d and self.elevation_data is not None:
                ed = self.elevation_data
                tr = self.transform
                rows, cols = ed.shape

                # Decimate the DTM before triangulation. Without this the raw
                # raster resolution drives the mesh: a 5000×5000 DTM produces
                # ~50 M triangles and a ~2 GB binary STL. The user picks the
                # cap via the "Topo cells/side" spinbox.
                max_cells = int(self.stl_cells_spin.value()) \
                    if hasattr(self, 'stl_cells_spin') else 500
                orig_rows, orig_cols = rows, cols
                stride_y = max(1, rows // max_cells)
                stride_x = max(1, cols // max_cells)
                if stride_x > 1 or stride_y > 1:
                    ed = ed[::stride_y, ::stride_x]
                    rows, cols = ed.shape
                    QgsMessageLog.logMessage(
                        f"Topography STL: decimated DTM {orig_cols}x{orig_rows} "
                        f"→ {cols}x{rows} (stride {stride_x},{stride_y})",
                        "TestPlugin", Qgis.Info)

                px_x = tr[0] * stride_x
                px_y = tr[4] * stride_y
                x = np.arange(cols) * px_x + tr[2]
                y = np.arange(rows) * px_y + tr[5]
                X, Y = np.meshgrid(x, y)
                Xf, Yf, Zf = X.flatten(), Y.flatten(), ed.flatten()
                if np.ma.is_masked(ed):
                    valid = ~ed.mask.flatten()
                    Xf, Yf, Zf = Xf[valid], Yf[valid], Zf[valid]
                pts3 = np.column_stack([Xf, Yf, Zf.astype(float)])
                QgsMessageLog.logMessage(
                    f"Topography STL: triangulating {len(Xf):,} points...",
                    "TestPlugin", Qgis.Info)
                QgsApplication.processEvents()
                tri = Delaunay(np.column_stack([Xf, Yf]))
                simplices = tri.simplices
                neighbors = tri.neighbors

                # Create a 3D box by extruding downwards
                bottom_z = np.min(Zf) - self.model_deep_value
                n_top = len(simplices)
                n_bottom = n_top

                # Vectorised boundary-edge detection: a triangle edge is on the
                # hull iff its opposite neighbour is -1.
                ii, jj = np.nonzero(neighbors == -1)
                bnd_v1 = simplices[ii, (jj + 1) % 3]
                bnd_v2 = simplices[ii, (jj + 2) % 3]
                n_sides = 2 * bnd_v1.size
                total_triangles = n_top + n_bottom + n_sides

                bottom_pts3 = np.column_stack(
                    [Xf, Yf, np.full(len(Zf), bottom_z, dtype=float)])

                data = np.zeros(total_triangles, dtype=mesh.Mesh.dtype)
                vectors = data['vectors']

                # Top surface — fancy-indexed in one shot
                vectors[:n_top] = pts3[simplices]

                # Bottom surface — reverse winding to flip normals downwards
                vectors[n_top:n_top + n_bottom] = bottom_pts3[simplices[:, [0, 2, 1]]]

                # Side walls: two triangles per boundary edge
                side_start = n_top + n_bottom
                pts_v1 = pts3[bnd_v1]
                pts_v2 = pts3[bnd_v2]
                bot_v1 = bottom_pts3[bnd_v1]
                bot_v2 = bottom_pts3[bnd_v2]
                vectors[side_start::2, 0] = pts_v1
                vectors[side_start::2, 1] = bot_v1
                vectors[side_start::2, 2] = bot_v2
                vectors[side_start + 1::2, 0] = pts_v1
                vectors[side_start + 1::2, 1] = bot_v2
                vectors[side_start + 1::2, 2] = pts_v2

                QgsApplication.processEvents()
                stl_mesh = mesh.Mesh(data, remove_empty_areas=True)
                stl_path = os.path.join(self.export_folder, "Topography.stl")

                # Save the raw extruded box directly — no smoothing and no
                # mesh-correction pass (those degraded the topography). The
                # box is watertight by construction.
                if _StlMode is not None:
                    stl_mesh.save(stl_path, mode=_StlMode.BINARY)
                else:
                    stl_mesh.save(stl_path)

                try:
                    size_bytes = os.path.getsize(stl_path)
                except OSError:
                    size_bytes = total_triangles * 50 + 84
                size_mb = size_bytes / (1024 * 1024)
                summary = (
                    f"Topography.stl: DTM {cols}x{rows} "
                    f"(stride {stride_x},{stride_y}), "
                    f"{total_triangles:,} triangles, {size_mb:.1f} MB "
                    f"(depth={self.model_deep_value}); "
                    f"bounds X[{Xf.min():.1f}, {Xf.max():.1f}] "
                    f"Y[{Yf.min():.1f}, {Yf.max():.1f}] "
                    f"Z[{bottom_z:.1f}, {Zf.max():.1f}]"
                )
                QgsMessageLog.logMessage("Exported: " + summary,
                                         "TestPlugin", Qgis.Info)
                if hasattr(self, 'stl_status_label'):
                    self.stl_status_label.setText(summary)
        finally:
            QgsApplication.restoreOverrideCursor()

    def export_plot_image(self):
        """Save the current 3D view as a raster image (PNG/JPG/TIFF).

        Captures exactly what is on screen via the plotter's framebuffer, so
        the exported picture matches the current camera, opacity and legend.
        """
        if not self.plotter:
            QMessageBox.warning(self, "Export Plot",
                                "Nothing to export — the 3D view is empty.")
            return
        default = os.path.join(self.export_folder or os.path.expanduser("~"),
                               "gempy_model.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export plot image", default,
            "PNG image (*.png);;JPEG image (*.jpg);;TIFF image (*.tif)")
        if not path:
            return
        QgsApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.plotter.render()
            self.plotter.screenshot(path)
            QgsMessageLog.logMessage(
                f"Exported plot image: {path}", "TestPlugin", Qgis.Info)
            if hasattr(self, 'stl_status_label'):
                self.stl_status_label.setText(f"Plot image saved: {path}")
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Plot image export error: {e}", "TestPlugin", Qgis.Warning)
            QMessageBox.critical(self, "Export Plot",
                                 f"Failed to save image:\n{e}")
        finally:
            QgsApplication.restoreOverrideCursor()

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


class ProbabilitySectionDialog(QDialog):
    """Interactive cross-section viewer for the probabilistic ensemble.

    Left panel  : the selected field as a 3-D volume cut by a draggable
                  plane widget (you see the model sliced away in real time).
    Right panel : the 2-D section through that same plane, viewed head-on
                  and refreshed instantly as the plane is moved.

    A single ``pv.ImageData`` carries all three ensemble fields as cell
    data, so switching field never rebuilds geometry.
    """

    # (combo label, cell-data key) — keys match _run_ensemble's arrays.
    FIELDS = [
        ("Modal probability", "probability"),
        ("Shannon entropy",   "entropy"),
        ("Modal lithology",   "modal"),
    ]

    def __init__(self, fields, extent, resolution, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Probability section viewer")
        self.resize(1100, 600)
        self.grid = self._build_grid(fields, extent, resolution)
        self.active = self.FIELDS[0][1]
        self._normal = (1.0, 0.0, 0.0)
        self._origin = self.grid.center
        self._build_ui()
        self._render()

    @staticmethod
    def _build_grid(fields, extent, resolution):
        """Wrap the (nx,ny,nz) ensemble arrays in a single cell-centred
        ImageData spanning the model extent."""
        nx, ny, nz = (int(r) for r in resolution)
        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.origin = (extent[0], extent[2], extent[4])
        grid.spacing = ((extent[1] - extent[0]) / nx,
                        (extent[3] - extent[2]) / ny,
                        (extent[5] - extent[4]) / nz)
        for _, key in ProbabilitySectionDialog.FIELDS:
            arr = fields.get(key)
            if arr is not None:
                grid.cell_data[key] = np.asarray(arr).reshape(
                    (nx, ny, nz)).flatten(order='F')
        return grid

    def _build_ui(self):
        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        for label, key in self.FIELDS:
            if key in self.grid.cell_data:
                self.field_combo.addItem(label, key)
        self.field_combo.currentIndexChanged.connect(self._on_field)
        bar.addWidget(self.field_combo)
        reset = QPushButton("Reset view")
        reset.clicked.connect(self._render)
        bar.addWidget(reset)
        export = QPushButton("Export plot")
        export.setToolTip("Save the 3D clip and the 2D section as images")
        export.clicked.connect(self._export_plot)
        bar.addWidget(export)
        bar.addStretch()
        root.addLayout(bar)

        split = QSplitter(Qt.Horizontal)
        self.left = QtInteractor(split)
        self.right = QtInteractor(split)
        split.addWidget(self.left)
        split.addWidget(self.right)
        split.setSizes([550, 550])
        root.addWidget(split, 1)

    def _cmap(self):
        return "tab20" if self.active == "modal" else "viridis"

    def _on_field(self, _idx):
        self.active = self.field_combo.currentData()
        self._render()

    def _render(self):
        """(Re)build both panels for the current field and replant the
        plane widget at its last position."""
        self.left.clear()
        self.right.clear()
        self.grid.set_active_scalars(self.active)
        self.left.add_mesh(self.grid.outline(), color="black")
        self.left.add_axes()
        self.right.enable_parallel_projection()
        self.left.add_plane_widget(
            self._on_plane, normal=self._normal, origin=self._origin)
        self._on_plane(self._normal, self._origin)
        self.left.reset_camera()

    def _on_plane(self, normal, origin):
        """Plane-widget callback: clip the volume on the left and draw the
        matching section on the right. Re-used name= actors update in place
        so dragging is flicker-free."""
        self._normal, self._origin = normal, origin
        cmap = self._cmap()
        clip = self.grid.clip(normal=normal, origin=origin)
        self.left.add_mesh(clip, name="clip", scalars=self.active,
                           cmap=cmap, show_scalar_bar=False)
        section = self.grid.slice(normal=normal, origin=origin)
        self.right.add_mesh(section, name="section", scalars=self.active,
                            cmap=cmap,
                            scalar_bar_args={"title": self.active})
        self.right.view_vector(normal)
        self.right.reset_camera()

    def _export_plot(self):
        """Save both panels (3D clip + 2D section) as raster images.

        The user picks a base path; the two views are written next to it with
        ``_3d`` and ``_section`` suffixes so the current field is captured for
        both the volume clip and the head-on section.
        """
        label = self.field_combo.currentText().replace(" ", "_").lower() \
            if hasattr(self, 'field_combo') else self.active
        default = os.path.join(os.path.expanduser("~"),
                               f"probability_{label}.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export probability plot", default,
            "PNG image (*.png);;JPEG image (*.jpg);;TIFF image (*.tif)")
        if not path:
            return
        base, ext = os.path.splitext(path)
        if not ext:
            ext = ".png"
        QgsApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.left.render()
            self.right.render()
            self.left.screenshot(f"{base}_3d{ext}")
            self.right.screenshot(f"{base}_section{ext}")
            QgsMessageLog.logMessage(
                f"Exported probability plots: {base}_3d{ext}, "
                f"{base}_section{ext}", "TestPlugin", Qgis.Info)
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Probability plot export error: {e}",
                "TestPlugin", Qgis.Warning)
            QMessageBox.critical(self, "Export plot",
                                 f"Failed to save images:\n{e}")
        finally:
            QgsApplication.restoreOverrideCursor()

    def closeEvent(self, event):
        for p in (getattr(self, "left", None), getattr(self, "right", None)):
            try:
                if p is not None:
                    p.clear()
                    p.close()
            except Exception:
                pass
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
        self.build_topo      = False
        self.clip_surfaces   = False
        self.rotate_align    = False
        self.translate_to_origin = False

        # Probabilistic / ensemble mode — perturb orientation dip & azimuth
        # by a Gaussian sigma, recompute the model N times and aggregate a
        # lithology-probability and information-entropy field.
        self.prob_enabled = False
        self.dip_sigma    = 5.0
        self.az_sigma     = 10.0
        self.n_real       = 20
        self.n_workers    = max(1, (os.cpu_count() or 2) - 1)
        self.ensemble_task = None   # keeps the running QgsTask alive

        self.structural_groups_config = None   # None → auto-detect faults
        self.progress_dialog          = None
        self.pyvista_panel            = None
        self.section_panel            = None

        # Temp CSVs produced when a non-CSV vector layer (e.g. a shapefile)
        # is selected — cleared and rebuilt on every run, deleted on close.
        self._shp_tmp_csvs            = []

        self.setup_layer_combos()
        self.connect_signals()
        self._sync_initial_layers()

        self.input_x.setText("10")
        self.input_y.setText("10")
        self.input_z.setText("10")
        self.model_deep.setText("500")
        self.input_buffer.setText("6")
        self.plot_checkbox.setChecked(False)
        self.rotate_checkbox.setChecked(False)
        self.translate_origin_checkbox.setChecked(False)
        self.output_folder.setText("Select output folder...")

        # Try to auto-select layers named layer_i / layer_o
        for layer in QgsProject.instance().mapLayers().values():
            if _clean_layer_path(layer).lower().endswith('.csv'):
                if layer.name() == "layer_i":
                    self.GeologyComboBox.setLayer(layer)
                elif layer.name() == "layer_o":
                    self.OrientationComboBox.setLayer(layer)

        self._build_probabilistic_controls()
        self._update_fault_status()

    # ------------------------------------------------------------------
    # Probabilistic / ensemble controls (added programmatically — the
    # .ui file uses absolute geometry, so we extend the dialog and place
    # the new widgets below the existing output-folder row).
    # ------------------------------------------------------------------
    def _build_probabilistic_controls(self):
        from PyQt5.QtWidgets import QCheckBox

        # Grow the dialog and push the OK/Cancel box down to make room.
        self.resize(620, 470)
        if hasattr(self, 'button_box'):
            g = self.button_box.geometry()
            self.button_box.setGeometry(g.x(), 430, g.width(), g.height())

        self.prob_checkbox = QCheckBox(
            "Probabilistic mode — orientation ensemble "
            "(perturb dip/azimuth, recompute N times)", self)
        self.prob_checkbox.setGeometry(20, 350, 580, 21)
        self.prob_checkbox.setToolTip(
            "When checked, after the deterministic model is built the "
            "orientation dip and azimuth are perturbed by a Gaussian "
            "sigma and the model is recomputed N times. A per-cell "
            "lithology probability and Shannon-entropy field are written "
            "to the output folder as .npy arrays plus a summary text "
            "file.")
        self.prob_checkbox.setChecked(False)

        QLabel("Dip σ°", self).setGeometry(20, 378, 45, 20)
        self.input_dip_sigma = QLineEdit("5", self)
        self.input_dip_sigma.setGeometry(65, 378, 40, 20)
        self.input_dip_sigma.setToolTip(
            "Standard deviation (degrees) of the Gaussian noise added to "
            "each orientation's dip.")

        QLabel("Azimuth σ°", self).setGeometry(120, 378, 65, 20)
        self.input_az_sigma = QLineEdit("10", self)
        self.input_az_sigma.setGeometry(185, 378, 40, 20)
        self.input_az_sigma.setToolTip(
            "Standard deviation (degrees) of the Gaussian noise added to "
            "each orientation's azimuth.")

        QLabel("Realizations", self).setGeometry(245, 378, 75, 20)
        self.input_n_real = QLineEdit("20", self)
        self.input_n_real.setGeometry(320, 378, 45, 20)
        self.input_n_real.setToolTip(
            "Number of perturbed models to compute and aggregate. "
            "Each realization rebuilds and recomputes the GemPy model, "
            "so large N can be slow.")

        # Worker processes — realizations are independent, so they are spread
        # across a process pool. Default to all-but-one physical core so the
        # UI thread stays responsive.
        import os as _os
        default_workers = max(1, (_os.cpu_count() or 2) - 1)
        QLabel("Workers", self).setGeometry(20, 402, 55, 20)
        self.input_n_workers = QLineEdit(str(default_workers), self)
        self.input_n_workers.setGeometry(75, 402, 40, 20)
        self.input_n_workers.setToolTip(
            "Number of parallel worker processes used to compute the "
            "realizations. 1 = sequential (no parallelism). Set this to "
            "around the number of physical CPU cores; values above the core "
            "count rarely help. The ensemble runs in the background, so QGIS "
            "stays usable while it computes.")

        self.open_section_button = QPushButton(
            "Open saved section…", self)
        self.open_section_button.setGeometry(380, 377, 220, 22)
        self.open_section_button.setToolTip(
            "Open the interactive probability cross-section viewer on the "
            "ensemble_*.npy arrays of a previous run, without recomputing.")
        self.open_section_button.clicked.connect(self._load_section_viewer)

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
    def closeEvent(self, event):
        """Remove any temp CSVs converted from non-CSV vector layers."""
        for p in getattr(self, '_shp_tmp_csvs', []):
            try:
                os.remove(p)
            except OSError:
                pass
        self._shp_tmp_csvs = []
        super().closeEvent(event)

    def _resolve_layer_csv_paths(self):
        """Ensure geology/orientation paths point at CSV files.

        CSV-backed layers are used as-is; any other vector format is
        converted to a temporary CSV (see ``_vector_layer_to_csv``).
        ``self.geology_layer_path`` / ``self.orientation_layer_path`` are
        updated in place. Returns ``True`` on success, or ``False`` (after
        warning the user) if a conversion failed.
        """
        # Drop temp CSVs left over from a previous run before rebuilding.
        for p in self._shp_tmp_csvs:
            try:
                os.remove(p)
            except OSError:
                pass
        self._shp_tmp_csvs = []
        try:
            self.geology_layer_path = _vector_layer_to_csv(
                self.selected_geology_layer, self._shp_tmp_csvs)
            self.orientation_layer_path = _vector_layer_to_csv(
                self.selected_orientation_layer, self._shp_tmp_csvs)
        except Exception as e:
            QMessageBox.warning(
                self, "Conversion failed",
                f"Could not convert the selected layer to CSV:\n{e}")
            return False
        return True

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
            self.dip_sigma        = float(self.input_dip_sigma.text().strip())
            self.az_sigma         = float(self.input_az_sigma.text().strip())
            self.n_real           = int(self.input_n_real.text().strip())
            self.n_workers        = max(1, int(self.input_n_workers.text().strip()))
        except ValueError as e:
            QMessageBox.warning(self, "Warning", f"Invalid numeric input: {e}")
            return

        # The "Cut with Topography" checkbox gates the topography entirely:
        # when it is UNCHECKED no topography is built, rendered or exported
        # (and the surfaces are not clipped); when CHECKED, the topography is
        # built from the loaded DTM, the surfaces are clipped against it, and
        # Topography.stl is exported.
        self.build_topo   = (bool(self.DTM_layer_path)
                             and self.plot_checkbox.isChecked())
        self.clip_surfaces = self.plot_checkbox.isChecked()
        # Kept for backward compatibility across the pipeline: "is a
        # topography available" (rendered / exported).
        self.plot_3d      = self.build_topo
        self.rotate_align = self.rotate_checkbox.isChecked()
        self.translate_to_origin = self.translate_origin_checkbox.isChecked()
        self.prob_enabled = self.prob_checkbox.isChecked()
        self.fault_names  = self.get_fault_list()
        self.folder_path  = self.output_folder.text().strip()

        if self.prob_enabled and self.n_real < 2:
            QMessageBox.warning(
                self, "Warning",
                "Probabilistic mode needs at least 2 realizations.")
            return
        if self.prob_enabled and not os.path.isdir(self.folder_path):
            QMessageBox.warning(
                self, "Warning",
                "Probabilistic mode writes ensemble .npy files — please "
                "select a valid output folder first.")
            return

        # Non-CSV vector inputs (shapefiles, GeoPackage, …) are converted to
        # temporary CSVs here so the rest of the pipeline can keep reading
        # plain CSV paths via pandas.
        if not self._resolve_layer_csv_paths():
            return

        QgsMessageLog.logMessage(
            f"Geology:     {self.geology_layer_path}", "TestPlugin", Qgis.Info)
        QgsMessageLog.logMessage(
            f"Orientation: {self.orientation_layer_path}", "TestPlugin", Qgis.Info)
        QgsMessageLog.logMessage(
            f"DTM:         {self.DTM_layer_path}", "TestPlugin", Qgis.Info)

        # CRS CHECK — the plugin reads each layer's RAW stored coordinates and
        # never reprojects, so geology, orientation and DTM must all be in the
        # same CRS. If they differ, layers can still look overlapped in QGIS
        # (on-the-fly reprojection) while their stored coordinates — and thus
        # the gempy surfaces vs. the topography — end up far apart.
        def _crs_id(layer):
            try:
                c = layer.crs()
                return c.authid() or c.description() or "unknown"
            except Exception:
                return "unknown"
        g_crs = _crs_id(self.selected_geology_layer)
        o_crs = _crs_id(self.selected_orientation_layer)
        d_crs = (_crs_id(self.selected_DTM_layer)
                 if getattr(self, 'selected_DTM_layer', None) else "none")
        QgsMessageLog.logMessage(
            f"CRS CHECK — geology={g_crs}  orientation={o_crs}  DTM={d_crs}",
            "TestPlugin", Qgis.Info)
        if d_crs not in ("none", "unknown") and g_crs not in ("unknown",) \
                and g_crs != d_crs:
            QgsMessageLog.logMessage(
                f"CRS MISMATCH — geology CRS ({g_crs}) differs from DTM CRS "
                f"({d_crs}). They may overlap in the QGIS canvas via on-the-fly "
                "reprojection, but their stored coordinates differ, so the "
                "exported surfaces and topography will NOT line up. Reproject "
                "the layers to a single common CRS (Layer ▸ Export ▸ Save As… "
                "with the same target CRS) and re-run.",
                "TestPlugin", Qgis.Warning)

        QgsMessageLog.logMessage(
            f"Resolution {self.x_value}x{self.y_value}x{self.z_value}  "
            f"Deep {self.model_deep_value}  BuildTopo {self.build_topo}  "
            f"ClipSurfaces {self.clip_surfaces}",
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
    # Axis-alignment rotation helpers
    # ------------------------------------------------------------------
    def _compute_alignment_rotation(self, aoi_union, dtm_bounds=None):
        """Return (theta_rad, pivot_xy, offset_xy) for the alignment transform.

        Forward formula applied everywhere downstream::

            p' = R(-theta) · (p - pivot) + offset

        Two modes:

        - ``self.translate_to_origin`` is False (legacy axis-alignment):
            ``theta`` is the angle of the longest side of the AOI's
            ``minimum_rotated_rectangle`` (or the combined geology +
            orientation points if no AOI). ``pivot`` is the real
            feature point closest to the source geometry's centroid.
            ``offset = pivot`` so the pivot is a fixed point of the
            transform — coordinate magnitudes stay the same.

        - ``self.translate_to_origin`` is True (new mode):
            ``theta`` is the angle of the longest side of the source
            DTM's bounding rectangle (0 when the DTM is wider than
            tall, π/2 when taller than wide). ``pivot`` is the DTM
            bounding-box corner with smallest x²+y² (i.e. closest to
            the CRS origin). ``offset = (0, 0)`` so that pivot lands
            at exactly (0, 0) in the rotated frame; downstream
            coordinates are shifted to small numbers near the origin.
            ``dtm_bounds`` is required and must be the source DTM's
            ``(xmin, ymin, xmax, ymax)``.
        """
        import pandas as pd
        from shapely.geometry import MultiPoint

        # ── Translate-to-origin mode: theta from longest DTM side, ──
        # pivot at the DTM corner closest to (0, 0), offset = (0, 0).
        if self.translate_to_origin:
            if dtm_bounds is None:
                raise RuntimeError(
                    "translate_to_origin mode requires the source DTM "
                    "bounds (xmin, ymin, xmax, ymax).")
            xmin, ymin, xmax, ymax = (float(b) for b in dtm_bounds)
            width = xmax - xmin
            height = ymax - ymin
            # 0 when the longest side is along X, π/2 when along Y.
            theta = 0.0 if width >= height else float(np.pi / 2.0)
            corners = np.array([
                [xmin, ymin], [xmin, ymax],
                [xmax, ymin], [xmax, ymax],
            ], dtype=float)
            d2 = corners[:, 0] ** 2 + corners[:, 1] ** 2
            pivot = corners[int(np.argmin(d2))].copy()
            offset = np.array([0.0, 0.0])
            return theta, pivot, offset

        # Collect feature points from both CSVs. Used for the
        # closest-to-centroid pivot below, and as the angle source
        # when no AOI is provided.
        feature_xs = []
        feature_ys = []
        try:
            df_g = pd.read_csv(self.geology_layer_path)
            gx = next((c for c in df_g.columns if c.upper() == 'X'), None)
            gy = next((c for c in df_g.columns if c.upper() == 'Y'), None)
            if gx and gy:
                feature_xs.extend(df_g[gx].astype(float).tolist())
                feature_ys.extend(df_g[gy].astype(float).tolist())
        except Exception:
            pass
        try:
            df_o = pd.read_csv(self.orientation_layer_path)
            ox = next((c for c in df_o.columns if c.upper() == 'X'), None)
            oy = next((c for c in df_o.columns if c.upper() == 'Y'), None)
            if ox and oy:
                feature_xs.extend(df_o[ox].astype(float).tolist())
                feature_ys.extend(df_o[oy].astype(float).tolist())
        except Exception:
            pass

        if aoi_union is not None and not aoi_union.is_empty:
            source_geom = aoi_union
        elif feature_xs:
            source_geom = MultiPoint(list(zip(feature_xs, feature_ys)))
        else:
            raise RuntimeError(
                "Cannot compute alignment rotation: no AOI and no "
                "geology/orientation CSV X/Y columns.")

        rect = source_geom.minimum_rotated_rectangle
        if rect.is_empty or not hasattr(rect, 'exterior'):
            return 0.0, np.array([0.0, 0.0])

        coords = list(rect.exterior.coords)[:4]
        edge_a = np.array(coords[1]) - np.array(coords[0])
        edge_b = np.array(coords[2]) - np.array(coords[1])
        longest = (edge_a if np.linalg.norm(edge_a) >= np.linalg.norm(edge_b)
                   else edge_b)
        theta = float(np.arctan2(longest[1], longest[0]))
        # Rotating by 180° gives the same alignment, so collapse to
        # (-pi/2, pi/2] for a minimal rotation.
        while theta > np.pi / 2:
            theta -= np.pi
        while theta <= -np.pi / 2:
            theta += np.pi

        # Rotation center: the feature point closest to the source
        # geometry's centroid. Falls back to the centroid itself only
        # when no feature points were loaded.
        centroid = source_geom.centroid
        cx, cy = float(centroid.x), float(centroid.y)
        if feature_xs:
            xs = np.asarray(feature_xs, dtype=float)
            ys = np.asarray(feature_ys, dtype=float)
            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            i = int(np.argmin(d2))
            pivot = np.array([float(xs[i]), float(ys[i])])
        else:
            pivot = np.array([cx, cy])

        # Legacy axis-alignment leaves the pivot in place.
        offset = pivot.copy()
        return theta, pivot, offset

    def _rotate_xy(self, x, y, theta, pivot, offset=None):
        """Apply forward alignment transform to (x, y) arrays.

        ``p' = R(-theta) · (p - pivot) + offset``

        When ``offset`` is None it defaults to ``pivot`` (classical
        pivot rotation: ``pivot`` is a fixed point). Pass
        ``offset = (0, 0)`` to translate the pivot to the origin.
        """
        if offset is None:
            offset = pivot
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        dx = np.asarray(x, dtype=float) - pivot[0]
        dy = np.asarray(y, dtype=float) - pivot[1]
        xr = cos_t * dx - sin_t * dy + offset[0]
        yr = sin_t * dx + cos_t * dy + offset[1]
        return xr, yr

    def _unrotate_xy(self, x, y, theta, pivot, offset=None):
        """Apply the inverse alignment transform: rotated -> original.

        Inverse of ``_rotate_xy`` with the same ``theta``, ``pivot``,
        ``offset``::

            p = R(+theta) · (p' - offset) + pivot
        """
        if offset is None:
            offset = pivot
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        dx = np.asarray(x, dtype=float) - offset[0]
        dy = np.asarray(y, dtype=float) - offset[1]
        xr = cos_t * dx - sin_t * dy + pivot[0]
        yr = sin_t * dx + cos_t * dy + pivot[1]
        return xr, yr

    def _rotate_csv_to_temp(self, src_csv_path, theta, pivot,
                            suffix, tmp_list, offset=None):
        """Write a rotated copy of ``src_csv_path`` to a temp file.

        The ORIGINAL file on disk is never modified — this reads it,
        rotates all geometric columns in memory, and writes the result
        to a fresh ``tempfile.mkstemp`` path (registered in
        ``tmp_list`` so ``gempy_core_plugin``'s ``finally`` block
        deletes it after the run).

        Transformations applied:

        1. Position columns ``X`` / ``Y`` — rotated by ``-theta``
           around ``center``. ``Z`` is preserved.
        2. If the CSV is in polar form (``azimuth`` + ``dip``, with
           or without ``polarity``), convert to the gradient vector
           using GemPy's formula::

               G_x = sin(dip) * sin(azimuth) * polarity
               G_y = sin(dip) * cos(azimuth) * polarity
               G_z = cos(dip) * polarity

           then rotate ``(G_x, G_y)`` by ``-theta`` around the origin
           (directions translate trivially). The polar columns
           (``azimuth``, ``dip``, ``polarity``) are dropped and
           replaced by the three rotated gradient columns. This
           bypasses any azimuth convention ambiguity — the rotation
           is a plain 3D math operation on a unit vector.
        3. Else if the CSV already carries gradient columns
           ``G_x`` / ``G_y``, rotate those around the origin in
           place.
        4. A surface-points CSV (only X/Y/Z/formation) passes
           through with just X/Y rotated.

        Formation name and any other columns are preserved. Returns
        the temp path (or the original path when there are no X/Y
        columns to rotate).
        """
        import pandas as pd
        if not src_csv_path or not os.path.isfile(src_csv_path):
            return src_csv_path
        df = pd.read_csv(src_csv_path)
        cols_by_lower = {c.lower(): c for c in df.columns}

        x_col = cols_by_lower.get('x')
        y_col = cols_by_lower.get('y')
        if not (x_col and y_col):
            return src_csv_path

        # 1) Positions: forward alignment transform
        xr, yr = self._rotate_xy(df[x_col].values, df[y_col].values,
                                 theta, pivot, offset=offset)
        df[x_col] = xr
        df[y_col] = yr

        # 2) Orientation directions
        az_col = (cols_by_lower.get('azimuth')
                  or cols_by_lower.get('dip_azimuth')
                  or cols_by_lower.get('dipazimuth'))
        dip_col = cols_by_lower.get('dip')
        pol_col = cols_by_lower.get('polarity')
        gx_col = cols_by_lower.get('g_x') or cols_by_lower.get('gx')
        gy_col = cols_by_lower.get('g_y') or cols_by_lower.get('gy')

        cos_t = float(np.cos(-theta))
        sin_t = float(np.sin(-theta))

        if az_col and dip_col and not (gx_col and gy_col):
            # Polar form → convert to gradient vector, rotate, drop
            # polar columns. GemPy's polar→gradient formula:
            az_rad = np.deg2rad(
                np.asarray(df[az_col].values, dtype=float))
            dip_rad = np.deg2rad(
                np.asarray(df[dip_col].values, dtype=float))
            if pol_col is not None:
                pol = np.asarray(df[pol_col].values, dtype=float)
            else:
                pol = np.ones_like(az_rad)
            gx = np.sin(dip_rad) * np.sin(az_rad) * pol
            gy = np.sin(dip_rad) * np.cos(az_rad) * pol
            gz = np.cos(dip_rad) * pol
            gx_rot = cos_t * gx - sin_t * gy
            gy_rot = sin_t * gx + cos_t * gy
            drop_cols = [c for c in (az_col, dip_col, pol_col)
                         if c is not None]
            df = df.drop(columns=drop_cols)
            df['G_x'] = gx_rot
            df['G_y'] = gy_rot
            df['G_z'] = gz
        elif gx_col and gy_col:
            # Already in gradient form → rotate G_x, G_y in place
            gx = np.asarray(df[gx_col].values, dtype=float)
            gy = np.asarray(df[gy_col].values, dtype=float)
            df[gx_col] = cos_t * gx - sin_t * gy
            df[gy_col] = sin_t * gx + cos_t * gy
        # else: surface-points CSV — no direction data to transform.

        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        df.to_csv(tmp_path, index=False)
        tmp_list.append(tmp_path)
        return tmp_path

    def _resample_dtm_to_rotated_grid(self, elev, transform, theta, pivot,
                                      rot_min_x, rot_max_x,
                                      rot_min_y, rot_max_y, cell_size,
                                      offset=None):
        """Inverse-warp a source DTM onto an axis-aligned grid in the rotated frame.

        Given the source raster (``elev`` + ``transform`` in the original
        frame) and the applied rotation (rotate by ``-theta`` around
        ``center``), build a new grid aligned with X and Y axes in the
        rotated frame covering ``[rot_min_x, rot_max_x] x [rot_min_y,
        rot_max_y]`` at ``cell_size`` step. Each output cell center is
        unrotated back to the original frame and bilinearly sampled
        from ``elev``.

        Cells whose pre-image falls outside the source DTM, or whose
        bilinear stencil hits a nodata pixel, are filled with the
        **nearest valid resampled cell** (Euclidean nearest-neighbour
        extrapolation via ``scipy.ndimage.distance_transform_edt``).
        This guarantees the output grid is fully populated, so the
        topography mesh's convex hull is an exact rectangle — no
        sawtooth on the boundary when the rotated extent pokes past
        the source DTM or when the source has nodata near the edge.

        Returns (new_elev, new_transform, new_rast_min_x, new_rast_max_x,
        new_rast_min_y, new_rast_max_y, n_filled).
        """
        from rasterio.transform import Affine, array_bounds
        from scipy.ndimage import distance_transform_edt, gaussian_filter

        # Axis-aligned grid in rotated frame. Row 0 = upper (max y),
        # last row = lower (min y) to match standard raster orientation.
        xs = np.arange(rot_min_x + cell_size / 2.0,
                       rot_max_x, cell_size)
        ys = np.arange(rot_max_y - cell_size / 2.0,
                       rot_min_y, -cell_size)
        if xs.size == 0 or ys.size == 0:
            raise RuntimeError(
                "Resampled DTM grid is empty — model extent or cell size "
                "may be degenerate.")
        nx = xs.size
        ny = ys.size

        XX, YY = np.meshgrid(xs, ys)

        # Inverse alignment transform: rotated frame -> original frame.
        # Forward applied: p' = R(-theta) · (p - pivot) + offset.
        # Inverse:          p  = R(+theta) · (p' - offset) + pivot.
        X_orig, Y_orig = self._unrotate_xy(XX, YY, theta, pivot,
                                           offset=offset)

        # Bilinear sample from the source raster.
        nrows, ncols = elev.shape
        # rasterio affine: x = a*col + b*row + c, y = d*col + e*row + f
        # For north-up DTMs b = d = 0, a = pixel width, e = -pixel height.
        c_frac = (X_orig - transform[2]) / transform[0]
        r_frac = (Y_orig - transform[5]) / transform[4]

        r0 = np.floor(r_frac).astype(int)
        c0 = np.floor(c_frac).astype(int)
        in_bounds = ((r0 >= 0) & (r0 < nrows - 1)
                     & (c0 >= 0) & (c0 < ncols - 1))
        r0c = np.clip(r0, 0, nrows - 2)
        c0c = np.clip(c0, 0, ncols - 2)
        dr = r_frac - r0
        dc = c_frac - c0

        if isinstance(elev, np.ma.MaskedArray):
            e = np.ma.filled(elev, np.nan).astype(float)
        else:
            e = np.asarray(elev, dtype=float)

        v00 = e[r0c, c0c]
        v01 = e[r0c, c0c + 1]
        v10 = e[r0c + 1, c0c]
        v11 = e[r0c + 1, c0c + 1]
        all_fin = (np.isfinite(v00) & np.isfinite(v01)
                   & np.isfinite(v10) & np.isfinite(v11))
        interp = (v00 * (1.0 - dr) * (1.0 - dc)
                  + v01 * (1.0 - dr) * dc
                  + v10 * dr * (1.0 - dc)
                  + v11 * dr * dc)

        out = np.full((ny, nx), np.nan, dtype=float)
        good = in_bounds & all_fin
        out[good] = interp[good]

        # Fill any remaining NaN cells with the nearest valid cell so
        # the resampled grid is a fully populated rectangle. This is
        # what makes the topography boundary come out straight after
        # Delaunay meshing — no sawtooth from missing cells at the
        # rotated-AABB corners or at source nodata pixels.
        n_filled = 0
        if not good.any():
            raise RuntimeError(
                "Resampled DTM has no valid cells to extrapolate from "
                "— the rotated model extent does not overlap the source "
                "DTM at all.")
        nan_mask = ~good
        if nan_mask.any():
            idx = distance_transform_edt(
                nan_mask, return_distances=False, return_indices=True)
            out = out[idx[0], idx[1]]
            n_filled = int(nan_mask.sum())

        # Smooth the full grid so the step between real-sampled cells
        # and nearest-valid-extrapolated cells at the rotated AABB
        # corners becomes a gentle gradient. A one-cell Gaussian
        # (sigma = 1.0) barely perturbs interior terrain but removes
        # the ridge the nearest-neighbour fill would otherwise leave
        # at the boundary.
        out = gaussian_filter(out, sigma=1.0, mode='nearest')

        # Affine for the new north-up grid. Origin = upper-left *corner*
        # of the upper-left *pixel*; xs[0]/ys[0] are pixel centers so we
        # offset by half a cell.
        new_transform = Affine(
            cell_size, 0.0, float(xs[0] - cell_size / 2.0),
            0.0, -cell_size, float(ys[0] + cell_size / 2.0))
        new_min_x, new_min_y, new_max_x, new_max_y = array_bounds(
            ny, nx, new_transform)
        return (out, new_transform,
                float(new_min_x), float(new_max_x),
                float(new_min_y), float(new_max_y),
                n_filled)

    def _save_rotation_matrix_csv(self, theta, pivot, folder, offset=None):
        """Write applied alignment transform to ``rotation_matrix.csv``.

        Forward formula::

            p' = R · (p - center) + offset

        where ``R`` rotates by ``-theta``. ``offset`` defaults to
        ``center`` (legacy pivot-rotation behaviour); pass an explicit
        ``offset`` (e.g. (0, 0) for translate-to-origin mode) to
        record the post-rotation translation. Readers that only know
        the legacy fields will still get a correct transform when
        ``offset == center`` because ``offset_x``/``offset_y`` falls
        back to ``center_x``/``center_y`` in that case.
        """
        import pandas as pd
        if offset is None:
            offset = pivot
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        path = os.path.join(folder, 'rotation_matrix.csv')
        df = pd.DataFrame({
            'parameter': [
                'applied_angle_deg', 'applied_angle_rad',
                'center_x', 'center_y',
                'offset_x', 'offset_y',
                'R_00', 'R_01',
                'R_10', 'R_11',
            ],
            'value': [
                float(np.degrees(-theta)), float(-theta),
                float(pivot[0]), float(pivot[1]),
                float(offset[0]), float(offset[1]),
                float(R[0, 0]), float(R[0, 1]),
                float(R[1, 0]), float(R[1, 1]),
            ],
        })
        df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # GemPy core
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Talkative diagnostics
    # ------------------------------------------------------------------
    def _diag_reset(self):
        """Start a fresh diagnostic file in the output folder."""
        self._diag_path = None
        try:
            if self.folder_path and os.path.isdir(self.folder_path):
                self._diag_path = os.path.join(
                    self.folder_path, 'diagnostic_log.txt')
                import datetime
                with open(self._diag_path, 'w', encoding='utf-8') as fh:
                    fh.write("GemPy 3D Model Builder — diagnostic log\n")
                    fh.write("Generated: "
                             + datetime.datetime.now().isoformat() + "\n")
                    fh.write("=" * 70 + "\n")
        except Exception:
            self._diag_path = None

    def _diag(self, msg):
        """Log one diagnostic line to the QGIS log (tag DIAG) AND append it
        to diagnostic_log.txt in the output folder, so the whole trace can be
        shared as a single file."""
        line = str(msg)
        try:
            QgsMessageLog.logMessage("DIAG " + line, "TestPlugin", Qgis.Info)
        except Exception:
            pass
        try:
            if getattr(self, '_diag_path', None):
                with open(self._diag_path, 'a', encoding='utf-8') as fh:
                    fh.write(line + "\n")
        except Exception:
            pass

    @staticmethod
    def _xy_bounds(arr_xy, label):
        """Format XY(Z) bounds of an (N,2|3) array for the diagnostic log."""
        try:
            a = np.asarray(arr_xy, dtype=float)
            if a.ndim != 2 or a.shape[0] == 0:
                return f"{label}: <empty>"
            x, y = a[:, 0], a[:, 1]
            msg = (f"{label}: n={a.shape[0]} "
                   f"X[{np.nanmin(x):.1f}, {np.nanmax(x):.1f}] "
                   f"Y[{np.nanmin(y):.1f}, {np.nanmax(y):.1f}]")
            if a.shape[1] >= 3:
                z = a[:, 2]
                msg += f" Z[{np.nanmin(z):.1f}, {np.nanmax(z):.1f}]"
            return msg
        except Exception as e:
            return f"{label}: <bounds error: {e}>"

    def gempy_core_plugin(self):
        import pandas as pd
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
            # ── 0. Talkative diagnostics ───────────────────────────────
            self._diag_reset()
            self._diag("=== INPUTS ===")
            def _crs_of(layer):
                try:
                    return layer.crs().authid() or layer.crs().description()
                except Exception:
                    return "?"
            self._diag(f"geology layer CRS    : "
                       f"{_crs_of(self.selected_geology_layer)}")
            self._diag(f"orientation layer CRS: "
                       f"{_crs_of(self.selected_orientation_layer)}")
            if getattr(self, 'selected_DTM_layer', None):
                self._diag(f"DTM layer CRS        : "
                           f"{_crs_of(self.selected_DTM_layer)}")
            self._diag(f"geology CSV path     : {self.geology_layer_path}")
            self._diag(f"DTM path             : {self.DTM_layer_path}")
            self._diag(f"options: rotate_align={self.rotate_align} "
                       f"translate_to_origin={self.translate_to_origin} "
                       f"build_topo={self.build_topo} "
                       f"clip_surfaces={self.clip_surfaces}")
            # Raw geology / orientation point bounds (as the plugin reads them)
            for tag, path in (("geology CSV", self.geology_layer_path),
                              ("orientation CSV", self.orientation_layer_path)):
                try:
                    _df = pd.read_csv(path)
                    _cl = {c.lower(): c for c in _df.columns}
                    if 'x' in _cl and 'y' in _cl:
                        self._diag(self._xy_bounds(
                            _df[[_cl['x'], _cl['y']]].to_numpy(), tag))
                    else:
                        self._diag(f"{tag}: no x/y columns "
                                   f"(columns={list(_df.columns)})")
                except Exception as e:
                    self._diag(f"{tag}: read error {e}")

            # ── 1. Read DTM and determine model extent ─────────────────
            # ``aoi_union_original`` is kept in DTM/original coordinates for
            # raster clipping; ``aoi_union`` is what drives model extent and
            # CSV filtering and is rotated below when alignment is enabled.
            aoi_union_original = self._get_aoi_union()
            aoi_union = aoi_union_original
            from rasterio.transform import array_bounds

            with rasterio.open(self.DTM_layer_path) as src:
                elevation_data = src.read(1)
                self._diag(
                    f"DTM raster: shape={elevation_data.shape} "
                    f"crs={src.crs} nodata={src.nodata} "
                    f"bounds X[{src.bounds.left:.1f}, {src.bounds.right:.1f}] "
                    f"Y[{src.bounds.bottom:.1f}, {src.bounds.top:.1f}]")
                if aoi_union_original is not None:
                    self._diag(
                        f"AOI bounds: X[{aoi_union_original.bounds[0]:.1f}, "
                        f"{aoi_union_original.bounds[2]:.1f}] "
                        f"Y[{aoi_union_original.bounds[1]:.1f}, "
                        f"{aoi_union_original.bounds[3]:.1f}]")
                else:
                    self._diag("AOI: none")
                if src.nodata is not None:
                    elevation_data = np.ma.masked_equal(
                        elevation_data, src.nodata)
                transform = src.transform

                # Skip the polygon clip when axis-alignment is on — the
                # DTM will be resampled onto a straight rectangular grid
                # in the rotated frame below, and we need the full
                # unclipped source to bilinear-sample from.
                if (aoi_union_original is not None
                        and not (self.rotate_align
                                 or self.translate_to_origin)):
                    # Clip DTM to AOI polygon (in original DTM coordinates)
                    try:
                        from rasterio.mask import mask as rasterio_mask
                        from shapely.geometry import mapping
                        nodata_val = src.nodata if src.nodata is not None else np.nan
                        out_image, out_transform = rasterio_mask(
                            src, [mapping(aoi_union_original)],
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
                        aoi_union_original = None  # fall back to DTM bounds
                        aoi_union = None

            # Raster pixel-aligned bounds (for topography coordinates)
            rows, cols = elevation_data.shape
            rast_min_x, rast_min_y, rast_max_x, rast_max_y = array_bounds(
                rows, cols, transform)

            # Z extent from valid (non-nodata) elevation only
            if isinstance(elevation_data, np.ma.MaskedArray):
                valid_elev = elevation_data.compressed()
            else:
                valid_elev = np.asarray(elevation_data)
            # Drop NaN/inf too — masked_equal only handles the raster's nodata
            # sentinel, not float NaN/inf that can appear in some DEMs.
            valid_elev = valid_elev[np.isfinite(valid_elev)]

            if valid_elev.size == 0:
                raise RuntimeError(
                    "DTM has no valid elevation data within the model area.\n"
                    "Check that the DTM and AOI overlap and the DTM is not "
                    "entirely nodata.")
            min_elev = float(valid_elev.min())
            max_elev = float(valid_elev.max())
            if not (np.isfinite(min_elev) and np.isfinite(max_elev)
                    and max_elev > min_elev):
                raise RuntimeError(
                    "DTM elevation range is invalid (NaN or zero-thickness).\n"
                    "The DTM may be entirely nodata or contain only a single "
                    "elevation value within the model area.")

            # Sanity check (in ORIGINAL coordinates, before any rotation):
            # geology CSV bounds vs DTM bounds. If they don't overlap the CRS
            # is almost certainly mismatched — fail early with a useful
            # message instead of letting GemPy build disjoint, misaligned
            # meshes. The X/Y columns are matched case-insensitively (the
            # source layer may use 'x'/'y', 'X'/'Y', etc.); if no coordinate
            # columns are found we log a warning rather than silently skipping.
            try:
                sp_head = pd.read_csv(self.geology_layer_path)
                cols_lower = {c.lower(): c for c in sp_head.columns}
                gx_col = cols_lower.get('x')
                gy_col = cols_lower.get('y')
                if gx_col and gy_col:
                    g_minx = float(sp_head[gx_col].min())
                    g_maxx = float(sp_head[gx_col].max())
                    g_miny = float(sp_head[gy_col].min())
                    g_maxy = float(sp_head[gy_col].max())
                    if (g_maxx < rast_min_x or g_minx > rast_max_x
                            or g_maxy < rast_min_y or g_miny > rast_max_y):
                        raise RuntimeError(
                            "Geology data and DTM do not overlap — likely a "
                            "CRS / georeferencing mismatch. The structural "
                            "surfaces and the topography would be built in "
                            "different places.\n\n"
                            f"Geology bounds: X[{g_minx:.1f}, {g_maxx:.1f}] "
                            f"Y[{g_miny:.1f}, {g_maxy:.1f}]\n"
                            f"DTM bounds:     X[{rast_min_x:.1f}, "
                            f"{rast_max_x:.1f}] "
                            f"Y[{rast_min_y:.1f}, {rast_max_y:.1f}]\n\n"
                            "Load both layers in QGIS, confirm they overlap on "
                            "the map, and reproject one so both share the same "
                            "CRS before running the plugin.")
                else:
                    QgsMessageLog.logMessage(
                        "Geology/DTM overlap check skipped: no X/Y columns "
                        f"found in {self.geology_layer_path} "
                        f"(columns: {list(sp_head.columns)}).",
                        "TestPlugin", Qgis.Warning)
            except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
                QgsMessageLog.logMessage(
                    f"Geology/DTM overlap check skipped: {e}",
                    "TestPlugin", Qgis.Warning)

            # ── 1b. Optional axis-alignment rotation ──────────────────
            # When enabled, apply the alignment rotation VIRTUALLY:
            #   - Geology + orientation CSVs are rotated into temp files
            #     (the originals on disk are never modified).
            #   - The AOI polygon is rotated in memory.
            #   - The DTM is inverse-warped in step 1c below onto an
            #     axis-aligned grid in the rotated frame.
            # Only ``rotation_matrix.csv`` is written to the output
            # folder so the rotation can be reproduced / reversed.
            self._rotation_theta = 0.0
            self._rotation_center = np.array([0.0, 0.0])
            self._rotation_offset = np.array([0.0, 0.0])
            sp_rotated_path = None
            or_rotated_path = None
            if self.rotate_align or self.translate_to_origin:
                dtm_bounds = (rast_min_x, rast_min_y,
                              rast_max_x, rast_max_y)
                theta, pivot, offset = self._compute_alignment_rotation(
                    aoi_union_original, dtm_bounds=dtm_bounds)
                self._rotation_theta = theta
                self._rotation_center = pivot
                self._rotation_offset = offset

                sp_rotated_path = self._rotate_csv_to_temp(
                    self.geology_layer_path, theta, pivot,
                    '_sp_rot.csv', tmp_files, offset=offset)
                or_rotated_path = self._rotate_csv_to_temp(
                    self.orientation_layer_path, theta, pivot,
                    '_or_rot.csv', tmp_files, offset=offset)

                if self.folder_path and os.path.isdir(self.folder_path):
                    try:
                        rot_path = self._save_rotation_matrix_csv(
                            theta, pivot, self.folder_path,
                            offset=offset)
                        QgsMessageLog.logMessage(
                            f"Rotation matrix saved to: {rot_path}",
                            "TestPlugin", Qgis.Info)
                    except Exception as e:
                        QgsMessageLog.logMessage(
                            f"Failed to save rotation_matrix.csv: {e}",
                            "TestPlugin", Qgis.Warning)
                else:
                    QgsMessageLog.logMessage(
                        "Output folder not set — rotation_matrix.csv "
                        "not written.", "TestPlugin", Qgis.Warning)

                if aoi_union_original is not None:
                    from shapely.affinity import rotate as shapely_rotate
                    from shapely.affinity import (
                        translate as shapely_translate)
                    aoi_union = shapely_rotate(
                        aoi_union_original,
                        -float(np.degrees(theta)),
                        origin=(float(pivot[0]), float(pivot[1])))
                    dx = float(offset[0]) - float(pivot[0])
                    dy = float(offset[1]) - float(pivot[1])
                    if dx != 0.0 or dy != 0.0:
                        aoi_union = shapely_translate(
                            aoi_union, xoff=dx, yoff=dy)

                mode = ("translate-to-origin"
                        if self.translate_to_origin
                        else "axis-alignment")
                QgsMessageLog.logMessage(
                    f"{mode}: rotated by "
                    f"{float(np.degrees(-theta)):.3f}° around "
                    f"pivot "
                    f"({float(pivot[0]):.1f}, {float(pivot[1]):.1f}); "
                    f"post-rotation offset "
                    f"({float(offset[0]):.1f}, "
                    f"{float(offset[1]):.1f}). "
                    f"Rotation applied to temp copies; original CSVs "
                    f"untouched.",
                    "TestPlugin", Qgis.Info)

            # Compute X/Y extent in the (possibly rotated) working frame
            if aoi_union is not None:
                min_x, min_y, max_x, max_y = aoi_union.bounds
            elif self.rotate_align or self.translate_to_origin:
                # No AOI: use bounding box of rotated raster corners
                cx = np.array([rast_min_x, rast_max_x, rast_max_x, rast_min_x])
                cy = np.array([rast_min_y, rast_min_y, rast_max_y, rast_max_y])
                cx_r, cy_r = self._rotate_xy(
                    cx, cy, self._rotation_theta, self._rotation_center,
                    offset=self._rotation_offset)
                min_x, max_x = float(cx_r.min()), float(cx_r.max())
                min_y, max_y = float(cy_r.min()), float(cy_r.max())
            else:
                min_x, min_y, max_x, max_y = (
                    rast_min_x, rast_min_y, rast_max_x, rast_max_y)

            # ── 1c. Resample DTM onto an axis-aligned grid covering the ──
            # working extent so the topography mesh has four straight sides
            # aligned with X and Y. Runs whenever an AOI is set or axis
            # alignment is on; otherwise the source DTM is already a clean
            # rectangle and is used as-is. With AOI + no rotation, theta=0
            # so the resample is a pure rectangular re-grid that fills any
            # cells outside the AOI shape via nearest-valid extrapolation.
            if (self.rotate_align or self.translate_to_origin
                    or aoi_union is not None):
                cell_size = float(abs(transform[0]))
                if cell_size <= 0.0:
                    raise RuntimeError(
                        "DTM has an invalid cell size — cannot resample "
                        "to a rectangular grid.")
                (elevation_data, transform,
                 rast_min_x, rast_max_x, rast_min_y, rast_max_y,
                 n_filled) = self._resample_dtm_to_rotated_grid(
                    elevation_data, transform,
                    self._rotation_theta, self._rotation_center,
                    min_x, max_x, min_y, max_y, cell_size,
                    offset=self._rotation_offset)
                # The new raster covers the model AABB exactly.
                min_x, max_x = rast_min_x, rast_max_x
                min_y, max_y = rast_min_y, rast_max_y

                # Recompute Z extent from the resampled grid.
                valid_elev = elevation_data[np.isfinite(elevation_data)]
                if valid_elev.size == 0:
                    raise RuntimeError(
                        "Resampled DTM has no valid elevation cells — "
                        "the model extent does not overlap the DTM. "
                        "Provide a DTM that fully covers the AOI, or "
                        "remove the AOI / axis-alignment options.")
                min_elev = float(valid_elev.min())
                max_elev = float(valid_elev.max())
                if not (np.isfinite(min_elev) and np.isfinite(max_elev)
                        and max_elev > min_elev):
                    raise RuntimeError(
                        "Resampled DTM has a degenerate elevation range "
                        "(all equal or non-finite).")

                total_cells = elevation_data.size
                QgsMessageLog.logMessage(
                    f"DTM resampled to axis-aligned grid "
                    f"{elevation_data.shape[1]} x {elevation_data.shape[0]} "
                    f"@ {cell_size:.2f} m "
                    f"({n_filled} of {total_cells} cells filled via "
                    f"nearest-valid extrapolation for a straight "
                    f"rectangular boundary).",
                    "TestPlugin", Qgis.Info)

            QgsMessageLog.logMessage(
                f"Model extent: X[{min_x:.1f}, {max_x:.1f}] "
                f"Y[{min_y:.1f}, {max_y:.1f}] "
                f"Z[{min_elev:.1f}, {max_elev:.1f}]",
                "TestPlugin", Qgis.Info)

            self.update_qgis_progress(5, "Loading geological data...")

            # ── 2. AOI filter of CSV files ─────────────────────────────
            # When alignment is on, feed the rotated temp copies into
            # the rest of the pipeline so the originals are never read
            # again after this point.
            if ((self.rotate_align or self.translate_to_origin)
                    and sp_rotated_path is not None):
                sp_path = sp_rotated_path
            else:
                sp_path = self.geology_layer_path
            if ((self.rotate_align or self.translate_to_origin)
                    and or_rotated_path is not None):
                or_path = or_rotated_path
            else:
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
            # With a topography the model extent must match the DTM/AOI
            # bounds exactly (no buffer) so the topography grid stays in the
            # source CRS and the surfaces line up with it.
            if self.build_topo:
                buf = 0

            # ── 4. Create GemPy model ──────────────────────────────────
            # Pandas 3.x uses Arrow-backed strings by default, but GemPy
            # expects plain object-dtype strings.  Re-write the CSVs with
            # string columns cast to object so GemPy can read them safely.
            def _csv_without_arrow_strings(src_path, suffix, tmp_list):
                """Read a CSV, convert Arrow string cols → object, write temp."""
                df = pd.read_csv(src_path)
                for col in df.columns:
                    if pd.api.types.is_string_dtype(df[col]):
                        df[col] = df[col].astype(object)
                fd, tmp = tempfile.mkstemp(suffix=suffix)
                os.close(fd)
                df.to_csv(tmp, index=False)
                tmp_list.append(tmp)
                return tmp

            sp_compat = _csv_without_arrow_strings(
                sp_path, '_sp_compat.csv', tmp_files)
            or_compat = _csv_without_arrow_strings(
                or_path, '_or_compat.csv', tmp_files)

            # GemPy re-reads the CSVs internally; without this context pandas
            # would re-infer Arrow-backed strings and break GemPy's name handling.
            pd_opts = []
            try:
                pd_opts.append(('future.infer_string',
                                pd.get_option('future.infer_string')))
                pd.set_option('future.infer_string', False)
            except (KeyError, pd.errors.OptionError):
                pass
            try:
                pd_opts.append(('mode.string_storage',
                                pd.get_option('mode.string_storage')))
                pd.set_option('mode.string_storage', 'python')
            except (KeyError, pd.errors.OptionError):
                pass

            # Model extent and resolution — captured so the probabilistic
            # ensemble can rebuild identical-geometry models.
            model_extent = [
                min_x - buf, max_x + buf,
                min_y - buf, max_y + buf,
                min_elev - self.model_deep_value - buf,
                max_elev + buf,
            ]
            model_resolution = [self.x_value, self.y_value, self.z_value]

            try:
                geo_model: gp.data.GeoModel = gp.create_geomodel(
                    project_name='Model_1',
                    extent=model_extent,
                    resolution=model_resolution,
                    importer_helper=gp.data.ImporterHelper(
                        path_to_orientations=or_compat,
                        path_to_surface_points=sp_compat,
                    )
                )
            finally:
                for key, val in pd_opts:
                    try:
                        pd.set_option(key, val)
                    except (KeyError, pd.errors.OptionError):
                        pass

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
            # When axis-alignment is on the DTM was already resampled to
            # an axis-aligned grid in the rotated frame (step 1c above),
            # so no on-the-fly rotation is needed here — we just read
            # pixel centers from rast_* bounds.
            topo_xyz = None   # captured for the probabilistic ensemble
            if self.build_topo:
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
                topo_xyz = topo
                self._diag(self._xy_bounds(
                    topo, "topography fed to gempy (set_topography)"))
                gp.set_topography_from_arrays(
                    grid=geo_model.grid, xyz_vertices=topo)

            self._diag(f"model_extent passed to gempy = "
                       f"[{model_extent[0]:.1f}, {model_extent[1]:.1f}, "
                       f"{model_extent[2]:.1f}, {model_extent[3]:.1f}, "
                       f"{model_extent[4]:.1f}, {model_extent[5]:.1f}]")
            self._diag(self._xy_bounds(
                np.column_stack([
                    np.arange(elevation_data.shape[1]) * transform[0]
                    + transform[2],
                    np.zeros(elevation_data.shape[1])]),
                "elevation_data X axis (export source)").replace(
                    " Y[0.0, 0.0]", ""))

            self.update_qgis_progress(20, "Computing model...")

            # ── 7. Compute ─────────────────────────────────────────────
            geo_model.input_transform.apply_anisotropy(
                gp.data.GlobalAnisotropy.NONE)
            try:
                gp.compute_model(geo_model)
            except np.linalg.LinAlgError as lae:
                raise RuntimeError(
                    f"GemPy solver failed: {lae}\n\n"
                    "A singular kernel matrix usually means the interpolation "
                    "problem is degenerate. Common causes:\n"
                    "  • Duplicate or near-duplicate surface points\n"
                    "  • Collinear orientations or too few orientations per "
                    "formation (need at least 1 per formation, ideally more)\n"
                    "  • Conflicting fault relations (e.g. OFFSET_FORMATIONS "
                    "on faults that don't intersect the offset formations)\n\n"
                    "Try: set all fault relations to OFFSET_ALL, add more "
                    "orientation points, or remove duplicate surface points.")

            # Post-compute diagnostics: where gempy actually placed things.
            try:
                _allv = np.vstack([
                    np.asarray(e.vertices)
                    for e in geo_model.structural_frame.structural_elements
                    if e.vertices is not None and len(e.vertices) > 0])
                self._diag(self._xy_bounds(
                    _allv, "element.vertices (surfaces, in memory)"))
            except Exception as e:
                self._diag(f"element.vertices bounds: <error {e}>")
            try:
                _v2 = np.asarray(
                    geo_model.grid.topography.values_2d, dtype=float
                ).reshape(-1, 3)
                self._diag(self._xy_bounds(
                    _v2, "gempy grid.topography.values_2d"))
            except Exception as e:
                self._diag(f"grid.topography bounds: <error {e}>")
            self._diag("NOTE: the exported surface STLs should match "
                       "'element.vertices' above; the exported Topography.stl "
                       "should match 'elevation_data X axis'. If a surface STL "
                       "differs, the smoothing drifted it (now fixed by "
                       "local-frame smoothing).")

            # ── 8. Display ─────────────────────────────────────────────
            self.update_qgis_progress(90, "Displaying 3D model...")

            if (getattr(self, 'pyvista_panel', None)
                    and self.pyvista_panel.isVisible()):
                self.pyvista_panel.display_gempy_model(
                    geo_model, self.folder_path,
                    self.build_topo, elevation_data, transform,
                    self.model_deep_value, self.clip_surfaces)
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
                    self.build_topo, elevation_data, transform,
                    self.model_deep_value, self.clip_surfaces)

            QgsApplication.processEvents()

            # ── 9. Optional probabilistic ensemble ─────────────────────
            # Runs after the deterministic model is shown. A failure here
            # never discards the deterministic result — it is logged and
            # the deterministic model stays on screen.
            if getattr(self, 'prob_enabled', False):
                try:
                    self._run_ensemble(
                        sp_compat, or_compat, model_extent,
                        model_resolution, groups_cfg, topo_xyz)
                except Exception as ee:
                    import traceback
                    QgsMessageLog.logMessage(
                        f"Probabilistic ensemble failed:\n"
                        f"{traceback.format_exc()}",
                        "TestPlugin", Qgis.Critical)
                    QMessageBox.warning(
                        None, "Probabilistic ensemble",
                        f"The deterministic model was built successfully, "
                        f"but the probabilistic ensemble failed:\n\n{ee}\n\n"
                        "See QGIS Log Messages (TestPlugin) for details.")

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

    # ------------------------------------------------------------------
    # Probabilistic / ensemble helpers
    # ------------------------------------------------------------------
    def _run_ensemble(self, sp_compat, or_compat, extent, resolution,
                      groups_cfg, topo_xyz):
        """Launch the probabilistic ensemble as a background ``QgsTask``.

        Each realization perturbs the orientation dip/azimuth and recomputes
        the GemPy model.  The realizations are independent, so they are spread
        across a pool of worker processes (``self.n_workers``); the heavy
        compute, aggregation and .npy writes all happen off the GUI thread.

        This returns immediately: progress and a Cancel button appear in the
        QGIS task-manager bar, and the section viewer is opened from
        :meth:`_on_ensemble_finished` once the task completes.  The actual
        per-realization work lives in the qgis-free ``ensemble_worker`` module
        so the worker processes can import it without dragging in Qt.
        """
        n_real = int(self.n_real)
        n_workers = max(1, int(getattr(self, 'n_workers', 1)))

        params = {
            'sp_path'   : sp_compat,
            'or_path'   : or_compat,
            'extent'    : list(extent),
            'resolution': list(resolution),
            'groups_cfg': groups_cfg,
            'topo_xyz'  : topo_xyz,
            'n_real'    : n_real,
            'dip_sigma' : float(self.dip_sigma),
            'az_sigma'  : float(self.az_sigma),
            'n_workers' : n_workers,
            'out_folder': self.folder_path,
        }

        task = EnsembleTask(params, on_finished=self._on_ensemble_finished)
        self.ensemble_task = task   # keep a reference so it is not GC'd
        QgsApplication.taskManager().addTask(task)

        QgsMessageLog.logMessage(
            f"Probabilistic ensemble queued: {n_real} realizations on "
            f"{n_workers} worker process(es), dip sigma={params['dip_sigma']} "
            f"deg, azimuth sigma={params['az_sigma']} deg.",
            "TestPlugin", Qgis.Info)
        iface.messageBar().pushMessage(
            "Ensemble started",
            f"Computing {n_real} realizations in the background "
            f"({n_workers} worker process(es)) — watch the task bar for "
            "progress. QGIS stays usable meanwhile.",
            level=Qgis.Info, duration=5)

    def _on_ensemble_finished(self, result):
        """Main-thread callback fired when :class:`EnsembleTask` completes.

        ``result`` is the dict produced by the task, or None when it was
        cancelled.  Opening the section viewer must happen here (on the GUI
        thread), never inside the task's background ``run()``.
        """
        self.ensemble_task = None
        if not result:
            return
        if result.get('error'):
            QMessageBox.warning(
                None, "Probabilistic ensemble",
                "The deterministic model was built successfully, but the "
                f"probabilistic ensemble failed:\n\n{result['error']}\n\n"
                "See QGIS Log Messages (TestPlugin) for details.")
            return

        iface.messageBar().pushMessage(
            "Ensemble complete",
            f"{result['n_used']}/{result['n_real']} realizations — results "
            f"saved to {result['out_folder']}",
            level=Qgis.Success, duration=6)

        if result.get('shaped') and result.get('fields') is not None:
            try:
                self._open_section_viewer(
                    result['fields'], result['extent'], result['resolution'])
            except Exception as ve:
                QgsMessageLog.logMessage(
                    f"Probability section viewer failed to open: {ve}",
                    "TestPlugin", Qgis.Warning)

    def _open_section_viewer(self, fields, extent, resolution):
        """Open (non-modally) the interactive probability cross-section
        viewer on the just-computed ensemble fields."""
        self.section_panel = ProbabilitySectionDialog(
            fields, extent, resolution, iface.mainWindow())
        self.section_panel.setWindowModality(Qt.NonModal)
        self.section_panel.show()
        self.section_panel.raise_()

    def _load_section_viewer(self):
        """Load the ensemble_*.npy arrays of a previous run from a folder
        and open the interactive section viewer on them."""
        start = self.folder_path if (
            getattr(self, 'folder_path', None)
            and os.path.isdir(self.folder_path)) else ""
        folder = QFileDialog.getExistingDirectory(
            self, "Select a folder with ensemble_*.npy results", start)
        if not folder:
            return

        files = {
            'probability': 'ensemble_probability.npy',
            'entropy':     'ensemble_entropy.npy',
            'modal':       'ensemble_modal_lithology.npy',
        }
        fields = {}
        for key, fname in files.items():
            path = os.path.join(folder, fname)
            if os.path.isfile(path):
                fields[key] = np.load(path)

        prob = fields.get('probability')
        if prob is None or prob.ndim != 3:
            QMessageBox.warning(
                self, "Open saved section",
                "No grid-shaped ensemble_probability.npy was found in this "
                "folder.\nThe viewer needs ensemble outputs that were saved "
                "with [nx,ny,nz] arrays.")
            return

        resolution = list(prob.shape)
        ext_path = os.path.join(folder, 'ensemble_extent.npy')
        if os.path.isfile(ext_path):
            extent = np.load(ext_path).tolist()
        else:   # older run without saved extent — use cell-index coords
            extent = [0, resolution[0], 0, resolution[1], 0, resolution[2]]

        self._open_section_viewer(fields, extent, resolution)


# ---------------------------------------------------------------------------
# Probabilistic ensemble — background task + parallel compute
# ---------------------------------------------------------------------------
def _qgis_python_executable():
    """Best-effort path to a real ``python`` executable for spawning worker
    processes.

    Inside a QGIS plugin ``sys.executable`` points at the ``qgis-bin`` launcher,
    which cannot host a ``multiprocessing`` pool. We look next to the
    interpreter's prefix for the actual interpreter instead. Returns None when
    nothing usable is found (the caller then stays sequential).
    """
    import sys
    exe = sys.executable or ""
    if os.path.basename(exe).lower().startswith('python'):
        return exe
    candidates = [
        os.path.join(sys.exec_prefix, 'python.exe'),
        os.path.join(sys.exec_prefix, 'pythonw.exe'),
        os.path.join(sys.exec_prefix, 'python3.exe'),
        os.path.join(sys.exec_prefix, 'bin', 'python3'),
        os.path.join(sys.exec_prefix, 'bin', 'python'),
        os.path.join(sys.base_exec_prefix, 'python.exe'),
        os.path.join(sys.base_exec_prefix, 'bin', 'python3'),
    ]
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return None


def aggregate_ensemble(blocks, params):
    """Aggregate a list of int32 lithology blocks into per-cell modal
    probability, Shannon entropy and modal-lithology fields, write the .npy
    outputs plus a summary, and return a result dict for the GUI thread.

    Pure numpy + file IO — safe to call from a background thread. Raises on
    inconsistent block sizes; the caller turns that into a user-facing error.
    """
    extent = params['extent']
    resolution = params['resolution']
    n_real = params['n_real']
    dip_sigma = params['dip_sigma']
    az_sigma = params['az_sigma']
    out = params['out_folder']

    sizes = {b.size for b in blocks}
    if len(sizes) != 1:
        raise RuntimeError(
            f"Realizations produced lithology blocks of different sizes "
            f"{sorted(sizes)} — cannot aggregate.")

    stack = np.vstack(blocks)            # (n_used, ncells)
    n_used, ncells = stack.shape

    ids = np.unique(stack)
    best_count = np.zeros(ncells, dtype=np.int32)
    modal = np.zeros(ncells, dtype=np.int32)
    entropy = np.zeros(ncells, dtype=float)
    for lid in ids:
        cnt = np.sum(stack == lid, axis=0).astype(np.int32)
        p = cnt / n_used
        sel = cnt > best_count
        best_count[sel] = cnt[sel]
        modal[sel] = lid
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy += np.where(p > 0, -p * np.log(p), 0.0)
    probability = best_count.astype(float) / n_used

    # Reshape onto the regular grid when possible; otherwise stay flat.
    try:
        shape = (resolution[0], resolution[1], resolution[2])
        prob_out = probability.reshape(shape)
        ent_out = entropy.reshape(shape)
        modal_out = modal.reshape(shape)
        shaped = True
    except Exception:
        prob_out, ent_out, modal_out = probability, entropy, modal
        shaped = False

    np.save(os.path.join(out, 'ensemble_probability.npy'), prob_out)
    np.save(os.path.join(out, 'ensemble_entropy.npy'), ent_out)
    np.save(os.path.join(out, 'ensemble_modal_lithology.npy'), modal_out)
    if shaped:   # extent lets the section viewer reload in map coords
        np.save(os.path.join(out, 'ensemble_extent.npy'),
                np.asarray(extent, dtype=float))

    mean_prob = float(np.mean(probability))
    mean_ent = float(np.mean(entropy))
    frac_uncertain = float(np.mean(probability < 0.8))
    summary = (
        "GemPy probabilistic ensemble\n"
        "============================\n"
        f"Realizations requested : {n_real}\n"
        f"Realizations used      : {n_used}\n"
        f"Dip sigma (deg)        : {dip_sigma}\n"
        f"Azimuth sigma (deg)    : {az_sigma}\n"
        f"Grid resolution        : {resolution}\n"
        f"Cells per realization  : {ncells}\n"
        f"Arrays shaped [nx,ny,nz]: {shaped}\n"
        f"Mean modal probability : {mean_prob:.3f}\n"
        f"Mean Shannon entropy   : {mean_ent:.3f}\n"
        f"Cells with prob < 0.8  : {100.0 * frac_uncertain:.1f}%\n"
        "\nFiles written to the output folder:\n"
        "  ensemble_probability.npy      per-cell probability of the "
        "modal lithology (1 = all realizations agree)\n"
        "  ensemble_entropy.npy          per-cell Shannon entropy "
        "(higher = more uncertain)\n"
        "  ensemble_modal_lithology.npy  per-cell most-frequent "
        "lithology id\n"
    )
    with open(os.path.join(out, 'ensemble_summary.txt'), 'w',
              encoding='utf-8') as fh:
        fh.write(summary)
    QgsMessageLog.logMessage(
        "Probabilistic ensemble complete:\n" + summary,
        "TestPlugin", Qgis.Info)

    return {
        'error'     : None,
        'shaped'    : shaped,
        'n_used'    : n_used,
        'n_real'    : n_real,
        'out_folder': out,
        'extent'    : extent,
        'resolution': resolution,
        'fields'    : ({'probability': prob_out,
                        'entropy': ent_out,
                        'modal': modal_out} if shaped else None),
    }


class EnsembleTask(QgsTask):
    """Run the probabilistic ensemble off the GUI thread.

    ``run()`` (worker thread) computes the N realizations — in parallel across
    a ``ProcessPoolExecutor`` when ``n_workers > 1``, otherwise sequentially —
    then aggregates the probability / entropy / modal fields and writes the
    .npy outputs.  ``finished()`` (main thread) hands the result dict back to
    the dialog so the section viewer can be opened safely on the GUI thread.

    The pool always has a sequential fallback: if it cannot start (a common
    situation in embedded interpreters) or produces no usable realization, the
    task recomputes serially so the feature never silently fails.
    """

    def __init__(self, params, on_finished=None):
        super().__init__("GemPy probabilistic ensemble", QgsTask.CanCancel)
        self.params = params
        self._on_finished_cb = on_finished
        self.result = None

    # -- background thread --------------------------------------------------
    def run(self):
        try:
            blocks = self._compute_blocks()
            if self.isCanceled():
                return False
            if len(blocks) < 2:
                self.result = {
                    'error': (
                        f"Only {len(blocks)} of {self.params['n_real']} "
                        "realizations produced a usable lithology block — "
                        "cannot build ensemble statistics.")}
                return False
            self.result = aggregate_ensemble(blocks, self.params)
            return True
        except Exception:
            import traceback
            tb = traceback.format_exc()
            self.result = {'error': tb}
            QgsMessageLog.logMessage(
                "Ensemble task crashed:\n" + tb, "TestPlugin", Qgis.Critical)
            return False

    def finished(self, ok):
        if self._on_finished_cb is not None:
            try:
                self._on_finished_cb(self.result)
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Ensemble finished-callback error: {e}",
                    "TestPlugin", Qgis.Warning)

    # -- compute orchestration ---------------------------------------------
    def _build_tasks(self):
        p = self.params
        n_real = p['n_real']
        # One independent seed per realization → the ensemble is reproducible
        # regardless of the order in which workers finish.
        seeds = np.random.SeedSequence(42).spawn(n_real)
        return [
            (k, p['sp_path'], p['or_path'], p['extent'], p['resolution'],
             p['groups_cfg'], p['topo_xyz'], p['dip_sigma'], p['az_sigma'],
             seeds[k])
            for k in range(n_real)
        ]

    def _compute_blocks(self):
        tasks = self._build_tasks()
        n_workers = max(1, int(self.params.get('n_workers', 1)))

        if n_workers > 1:
            try:
                blocks = self._compute_pool(tasks, n_workers)
                if blocks or self.isCanceled():
                    return blocks
                QgsMessageLog.logMessage(
                    "Process pool produced no usable realizations; retrying "
                    "sequentially.", "TestPlugin", Qgis.Warning)
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Process pool unavailable ({e}); falling back to a "
                    "sequential ensemble.", "TestPlugin", Qgis.Warning)
        return self._compute_sequential(tasks)

    def _record(self, blocks, idx, block, err):
        if block is not None:
            blocks.append(block)
        else:
            QgsMessageLog.logMessage(
                f"Realization {idx + 1} failed/skipped: {err}",
                "TestPlugin", Qgis.Warning)

    def _compute_sequential(self, tasks):
        from . import ensemble_worker
        blocks = []
        n_real = len(tasks)
        for done, t in enumerate(tasks):
            if self.isCanceled():
                break
            idx, block, err = ensemble_worker.run_realization(t)
            self._record(blocks, idx, block, err)
            self.setProgress(100.0 * (done + 1) / n_real)
        return blocks

    def _compute_pool(self, tasks, n_workers):
        import sys
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # The worker module must be importable from a bare child interpreter
        # by a plain top-level name, so the pickled function reference resolves
        # there. Import it under its own name (not the package-relative path)
        # and make its directory discoverable to the children via PYTHONPATH.
        worker_dir = os.path.dirname(os.path.abspath(__file__))
        if worker_dir not in sys.path:
            sys.path.insert(0, worker_dir)
        import ensemble_worker  # noqa: E402  (top-level → picklable by children)

        exe = _qgis_python_executable()
        if exe:
            mp.set_executable(exe)

        old_pythonpath = os.environ.get('PYTHONPATH', '')
        parts = [worker_dir] + ([old_pythonpath] if old_pythonpath else [])
        os.environ['PYTHONPATH'] = os.pathsep.join(parts)

        ctx = mp.get_context('spawn')
        blocks = []
        n_real = len(tasks)
        done = 0
        try:
            with ProcessPoolExecutor(max_workers=n_workers,
                                     mp_context=ctx) as ex:
                futures = {ex.submit(ensemble_worker.run_realization, t): t[0]
                           for t in tasks}
                for fut in as_completed(futures):
                    if self.isCanceled():
                        ex.shutdown(wait=False, cancel_futures=True)
                        break
                    idx, block, err = fut.result()
                    self._record(blocks, idx, block, err)
                    done += 1
                    self.setProgress(100.0 * done / n_real)
        finally:
            os.environ['PYTHONPATH'] = old_pythonpath
        return blocks
