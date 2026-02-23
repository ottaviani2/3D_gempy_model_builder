# -*- coding: utf-8 -*-
"""
PS-InSAR Analysis Dialog
Standalone dialog for PS-InSAR moving area detection and boundary extraction.
"""

import os

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox,
    QPushButton, QLineEdit, QFileDialog, QProgressBar,
    QMessageBox,
)
from qgis.PyQt.QtCore import Qt
from qgis.core import (
    QgsProject, QgsMapLayerProxyModel, QgsVectorLayer,
    QgsMessageLog, Qgis,
)
from qgis.gui import QgsMapLayerComboBox

from .ps_insar_module import PSInSARProcessor, SurfaceEvolutionProcessor, HAS_TORCH, HAS_PANDAS
from .ps_insar_decomposition import DEPTH_FORMULAS


class PSInSARDialog(QDialog):
    """Dialog for PS-InSAR moving area detection."""

    def __init__(self, output_folder=None, dtm_layer=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PS-InSAR Analysis")
        self.resize(540, 920)
        self.setWindowModality(Qt.ApplicationModal)

        self.output_folder = output_folder or ""
        self.dtm_layer = dtm_layer

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        root = QVBoxLayout()
        self.setLayout(root)

        # --- PS Data section ---
        grp_data = QGroupBox("PS Data")
        lay_data = QVBoxLayout()
        grp_data.setLayout(lay_data)

        row = QHBoxLayout()
        row.addWidget(QLabel("PS Layer:"))
        self.cmbPSLayer = QgsMapLayerComboBox()
        self.cmbPSLayer.setFilters(QgsMapLayerProxyModel.VectorLayer)
        row.addWidget(self.cmbPSLayer)
        lay_data.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Velocity field:"))
        self.cmbVelocityField = QComboBox()
        row.addWidget(self.cmbVelocityField)
        lay_data.addLayout(row)

        root.addWidget(grp_data)

        # --- Classification Parameters ---
        grp_class = QGroupBox("Classification Parameters")
        lay_class = QVBoxLayout()
        grp_class.setLayout(lay_class)

        row = QHBoxLayout()
        row.addWidget(QLabel("Velocity threshold (mm/yr):"))
        self.spnThreshold = QDoubleSpinBox()
        self.spnThreshold.setRange(0.1, 100.0)
        self.spnThreshold.setValue(2.0)
        self.spnThreshold.setSingleStep(0.5)
        row.addWidget(self.spnThreshold)
        lay_class.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("K neighbors:"))
        self.spnKNeighbors = QSpinBox()
        self.spnKNeighbors.setRange(3, 100)
        self.spnKNeighbors.setValue(20)
        row.addWidget(self.spnKNeighbors)
        lay_class.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Epochs:"))
        self.spnEpochs = QSpinBox()
        self.spnEpochs.setRange(5, 500)
        self.spnEpochs.setValue(50)
        row.addWidget(self.spnEpochs)
        lay_class.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Refinement iterations:"))
        self.spnRefinement = QSpinBox()
        self.spnRefinement.setRange(0, 10)
        self.spnRefinement.setValue(1)
        row.addWidget(self.spnRefinement)
        lay_class.addLayout(row)

        self.chkUseNN = QCheckBox("Use NN spatial smoothing (requires PyTorch)")
        self.chkUseNN.setChecked(HAS_TORCH)
        if not HAS_TORCH:
            self.chkUseNN.setEnabled(False)
            self.chkUseNN.setToolTip("PyTorch is not installed. Using threshold fallback.")
        lay_class.addWidget(self.chkUseNN)

        root.addWidget(grp_class)

        # --- Boundary Extraction ---
        grp_boundary = QGroupBox("Boundary Extraction")
        lay_boundary = QVBoxLayout()
        grp_boundary.setLayout(lay_boundary)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min cluster size:"))
        self.spnMinCluster = QSpinBox()
        self.spnMinCluster.setRange(3, 1000)
        self.spnMinCluster.setValue(10)
        row.addWidget(self.spnMinCluster)
        lay_boundary.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Alpha (0=auto):"))
        self.spnAlpha = QDoubleSpinBox()
        self.spnAlpha.setRange(0.0, 100000.0)
        self.spnAlpha.setValue(0.0)
        self.spnAlpha.setSingleStep(10.0)
        row.addWidget(self.spnAlpha)
        lay_boundary.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Z source:"))
        self.cmbZSource = QComboBox()
        self.cmbZSource.addItems(["Constant", "From DTM", "From PS data"])
        row.addWidget(self.cmbZSource)
        lay_boundary.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Z constant value:"))
        self.spnZConstant = QDoubleSpinBox()
        self.spnZConstant.setRange(-10000.0, 100000.0)
        self.spnZConstant.setValue(0.0)
        row.addWidget(self.spnZConstant)
        lay_boundary.addLayout(row)

        root.addWidget(grp_boundary)

        # --- Surface Evolution (LOS Decomposition) ---
        grp_evolution = QGroupBox("Surface Evolution (Multi-orbit Decomposition)")
        lay_evo = QVBoxLayout()
        grp_evolution.setLayout(lay_evo)

        self.chkEnableEvolution = QCheckBox("Enable surface evolution analysis")
        self.chkEnableEvolution.setChecked(False)
        if not HAS_PANDAS:
            self.chkEnableEvolution.setEnabled(False)
            self.chkEnableEvolution.setToolTip("pandas is required for CSV processing")
        lay_evo.addWidget(self.chkEnableEvolution)

        # CSV file list
        row = QHBoxLayout()
        row.addWidget(QLabel("InSAR CSV files:"))
        self.lstCSVFiles = QComboBox()
        self.lstCSVFiles.setEditable(False)
        self.lstCSVFiles.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        row.addWidget(self.lstCSVFiles)
        self.btnAddCSV = QPushButton("+")
        self.btnAddCSV.setFixedWidth(30)
        self.btnAddCSV.setToolTip("Add InSAR CSV file (need >= 2 orbits)")
        row.addWidget(self.btnAddCSV)
        self.btnRemoveCSV = QPushButton("-")
        self.btnRemoveCSV.setFixedWidth(30)
        self.btnRemoveCSV.setToolTip("Remove selected CSV file")
        row.addWidget(self.btnRemoveCSV)
        lay_evo.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Slope azimuth (deg):"))
        self.spnSlopeAzimuth = QDoubleSpinBox()
        self.spnSlopeAzimuth.setRange(0.0, 360.0)
        self.spnSlopeAzimuth.setValue(270.0)
        row.addWidget(self.spnSlopeAzimuth)
        row.addWidget(QLabel("Slope angle (deg):"))
        self.spnSlopeAngle = QDoubleSpinBox()
        self.spnSlopeAngle.setRange(0.0, 90.0)
        self.spnSlopeAngle.setValue(50.0)
        row.addWidget(self.spnSlopeAngle)
        lay_evo.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Friction coeff f:"))
        self.spnFriction = QDoubleSpinBox()
        self.spnFriction.setRange(0.001, 1.0)
        self.spnFriction.setValue(0.01)
        self.spnFriction.setSingleStep(0.01)
        self.spnFriction.setDecimals(3)
        row.addWidget(self.spnFriction)
        row.addWidget(QLabel("Max distance (m):"))
        self.spnMaxDist = QDoubleSpinBox()
        self.spnMaxDist.setRange(1.0, 10000.0)
        self.spnMaxDist.setValue(100.0)
        row.addWidget(self.spnMaxDist)
        lay_evo.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Grid NX:"))
        self.spnGridNX = QSpinBox()
        self.spnGridNX.setRange(10, 1000)
        self.spnGridNX.setValue(100)
        row.addWidget(self.spnGridNX)
        row.addWidget(QLabel("Grid NY:"))
        self.spnGridNY = QSpinBox()
        self.spnGridNY.setRange(10, 1000)
        self.spnGridNY.setValue(100)
        row.addWidget(self.spnGridNY)
        lay_evo.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Date column offset:"))
        self.spnDateOffset = QSpinBox()
        self.spnDateOffset.setRange(1, 200)
        self.spnDateOffset.setValue(28)
        self.spnDateOffset.setToolTip(
            "Column index where time-series displacement data starts in the CSV")
        row.addWidget(self.spnDateOffset)
        lay_evo.addLayout(row)

        self.btnRunEvolution = QPushButton("Run Surface Evolution")
        self.btnRunEvolution.setMinimumHeight(28)
        lay_evo.addWidget(self.btnRunEvolution)

        root.addWidget(grp_evolution)

        # --- Depth Estimation Formula ---
        grp_depth = QGroupBox("Sliding Surface Depth Estimation")
        lay_depth = QVBoxLayout()
        grp_depth.setLayout(lay_depth)

        row = QHBoxLayout()
        row.addWidget(QLabel("Formula:"))
        self.cmbDepthFormula = QComboBox()
        for key, info in DEPTH_FORMULAS.items():
            self.cmbDepthFormula.addItem(info['label'], userData=key)
        row.addWidget(self.cmbDepthFormula)
        lay_depth.addLayout(row)

        self.lblFormulaDesc = QLabel()
        self.lblFormulaDesc.setWordWrap(True)
        self.lblFormulaDesc.setStyleSheet("color: #555555; font-style: italic;")
        lay_depth.addWidget(self.lblFormulaDesc)

        row = QHBoxLayout()
        row.addWidget(QLabel("Reference:"))
        self.lblFormulaRef = QLabel()
        self.lblFormulaRef.setStyleSheet("font-weight: bold;")
        row.addWidget(self.lblFormulaRef)
        row.addStretch()
        lay_depth.addLayout(row)

        root.addWidget(grp_depth)

        # --- Output ---
        grp_output = QGroupBox("Output")
        lay_output = QVBoxLayout()
        grp_output.setLayout(lay_output)

        row = QHBoxLayout()
        row.addWidget(QLabel("Output shapefile:"))
        self.txtOutputPath = QLineEdit()
        if self.output_folder:
            self.txtOutputPath.setText(
                os.path.join(self.output_folder, "ps_moving_boundary.shp"))
        row.addWidget(self.txtOutputPath)
        self.btnBrowseOutput = QPushButton("...")
        self.btnBrowseOutput.setFixedWidth(30)
        row.addWidget(self.btnBrowseOutput)
        lay_output.addLayout(row)

        self.chkAddToQGIS = QCheckBox("Add result to QGIS layers")
        self.chkAddToQGIS.setChecked(True)
        lay_output.addWidget(self.chkAddToQGIS)

        root.addWidget(grp_output)

        # --- Run / Progress ---
        self.btnRun = QPushButton("Run")
        self.btnRun.setMinimumHeight(30)
        root.addWidget(self.btnRun)

        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)
        root.addWidget(self.progressBar)

        self.lblStatus = QLabel("Ready")
        root.addWidget(self.lblStatus)

    def _connect_signals(self):
        self.cmbPSLayer.layerChanged.connect(self._on_layer_changed)
        self.btnBrowseOutput.clicked.connect(self._browse_output)
        self.btnRun.clicked.connect(self._run_analysis)
        self.cmbZSource.currentTextChanged.connect(self._on_z_source_changed)

        # Surface evolution signals
        self.btnAddCSV.clicked.connect(self._add_csv_file)
        self.btnRemoveCSV.clicked.connect(self._remove_csv_file)
        self.btnRunEvolution.clicked.connect(self._run_surface_evolution)
        self.chkEnableEvolution.toggled.connect(self._on_evolution_toggled)
        self.cmbDepthFormula.currentIndexChanged.connect(self._on_formula_changed)
        self._csv_paths = []

        # Set initial state of evolution controls (disabled until checkbox is checked)
        self._on_evolution_toggled(False)

        # Populate depth formula description for the default selection
        self._on_formula_changed(0)

        # Populate velocity field for initial layer
        self._on_layer_changed(self.cmbPSLayer.currentLayer())

    def _on_formula_changed(self, index):
        """Update formula description/reference labels when selection changes."""
        key = self.cmbDepthFormula.itemData(index)
        info = DEPTH_FORMULAS.get(key, {})
        self.lblFormulaDesc.setText(info.get('desc', ''))
        self.lblFormulaRef.setText(info.get('ref', ''))

    def _on_layer_changed(self, layer):
        """Populate velocity field combo when layer changes."""
        self.cmbVelocityField.clear()
        if layer is None or not isinstance(layer, QgsVectorLayer):
            return

        fields = layer.fields()
        for f in fields:
            # Include numeric fields as velocity candidates
            type_name = f.typeName().lower()
            if any(t in type_name for t in ['int', 'real', 'double', 'float', 'numeric', 'decimal']):
                self.cmbVelocityField.addItem(f.name())
            elif f.type() in (2, 6):  # QVariant.Int, QVariant.Double
                self.cmbVelocityField.addItem(f.name())

        # Try to auto-select a velocity-like field
        for i in range(self.cmbVelocityField.count()):
            name = self.cmbVelocityField.itemText(i).lower()
            if any(kw in name for kw in ['vel', 'v_', 'defo', 'disp', 'mm', 'rate']):
                self.cmbVelocityField.setCurrentIndex(i)
                break

    def _on_z_source_changed(self, text):
        """Enable/disable Z constant based on source."""
        self.spnZConstant.setEnabled(text == "Constant")

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Boundary Shapefile",
            self.txtOutputPath.text() or "",
            "Shapefile (*.shp)")
        if path:
            if not path.lower().endswith('.shp'):
                path += '.shp'
            self.txtOutputPath.setText(path)

    def _run_analysis(self):
        """Execute the PS-InSAR analysis pipeline."""
        layer = self.cmbPSLayer.currentLayer()
        if layer is None:
            QMessageBox.warning(self, "Warning", "Please select a PS layer.")
            return

        vel_field = self.cmbVelocityField.currentText()
        if not vel_field:
            QMessageBox.warning(self, "Warning", "Please select a velocity field.")
            return

        output_path = self.txtOutputPath.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Warning", "Please specify an output path.")
            return

        # Determine DTM path
        dtm_path = None
        z_source_text = self.cmbZSource.currentText()
        if z_source_text == "From DTM":
            if self.dtm_layer and hasattr(self.dtm_layer, 'source'):
                dtm_path = self.dtm_layer.source()
            else:
                QMessageBox.warning(
                    self, "Warning",
                    "No DTM layer available. Please select a DTM in the main dialog first.")
                return

        z_source_map = {
            "Constant": "constant",
            "From DTM": "dtm",
            "From PS data": "ps_data",
        }

        processor = PSInSARProcessor(
            ps_layer=layer,
            velocity_field=vel_field,
            output_path=output_path,
            vel_threshold=self.spnThreshold.value(),
            k_neighbors=self.spnKNeighbors.value(),
            epochs=self.spnEpochs.value(),
            refinement_iters=self.spnRefinement.value(),
            min_cluster_size=self.spnMinCluster.value(),
            alpha=self.spnAlpha.value(),
            z_source=z_source_map.get(z_source_text, "constant"),
            z_constant=self.spnZConstant.value(),
            dtm_path=dtm_path,
            use_nn=self.chkUseNN.isChecked(),
        )

        self.btnRun.setEnabled(False)
        self.progressBar.setValue(0)
        self.lblStatus.setText("Running...")

        try:
            from qgis.core import QgsApplication

            def progress_callback(value, message=""):
                self.progressBar.setValue(value)
                if message:
                    self.lblStatus.setText(message)
                QgsApplication.processEvents()

            boundary_points, labels, xy = processor.run(
                progress_callback=progress_callback)

            # Add result layer to QGIS
            if self.chkAddToQGIS.isChecked() and os.path.exists(output_path):
                result_layer = QgsVectorLayer(
                    output_path, "PS_moving_boundary", "ogr")
                if result_layer.isValid():
                    QgsProject.instance().addMapLayer(result_layer)

            n_moving = int(labels.sum())
            n_boundary = len(boundary_points)
            self.lblStatus.setText(
                f"Done: {n_moving} moving points, {n_boundary} boundary points")
            self.progressBar.setValue(100)

            QMessageBox.information(
                self, "PS-InSAR Analysis Complete",
                f"Classification complete:\n"
                f"  - {len(labels)} total points\n"
                f"  - {n_moving} moving points\n"
                f"  - {n_boundary} boundary points extracted\n\n"
                f"Output: {output_path}")

        except Exception as e:
            self.lblStatus.setText(f"Error: {e}")
            QgsMessageLog.logMessage(
                f"PS-InSAR error: {e}", "TestPlugin", Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(
                traceback.format_exc(), "TestPlugin", Qgis.Critical)
            QMessageBox.critical(
                self, "Error", f"Analysis failed:\n{e}")
        finally:
            self.btnRun.setEnabled(True)

    # --- Surface Evolution methods ---

    def _on_evolution_toggled(self, checked):
        """Enable/disable evolution controls."""
        self.btnAddCSV.setEnabled(checked)
        self.btnRemoveCSV.setEnabled(checked)
        self.btnRunEvolution.setEnabled(checked)
        self.spnSlopeAzimuth.setEnabled(checked)
        self.spnSlopeAngle.setEnabled(checked)
        self.spnFriction.setEnabled(checked)
        self.spnMaxDist.setEnabled(checked)
        self.spnGridNX.setEnabled(checked)
        self.spnGridNY.setEnabled(checked)
        self.spnDateOffset.setEnabled(checked)

    def _add_csv_file(self):
        """Add an InSAR CSV file to the list."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select InSAR CSV Files", "",
            "CSV Files (*.csv);;All Files (*)")
        for path in paths:
            if path and path not in self._csv_paths:
                self._csv_paths.append(path)
                self.lstCSVFiles.addItem(os.path.basename(path))

    def _remove_csv_file(self):
        """Remove the selected CSV file from the list."""
        idx = self.lstCSVFiles.currentIndex()
        if idx >= 0 and idx < len(self._csv_paths):
            self._csv_paths.pop(idx)
            self.lstCSVFiles.removeItem(idx)

    def _run_surface_evolution(self):
        """Execute the surface evolution pipeline."""
        if len(self._csv_paths) < 2:
            QMessageBox.warning(
                self, "Warning",
                "At least 2 InSAR CSV files (different orbits) are required.")
            return

        # Need DTM
        dtm_path = None
        if self.dtm_layer and hasattr(self.dtm_layer, 'source'):
            dtm_path = self.dtm_layer.source()
        else:
            QMessageBox.warning(
                self, "Warning",
                "A DTM layer is required for surface evolution.\n"
                "Please select a DTM in the main dialog first.")
            return

        output_path = self.txtOutputPath.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Warning", "Please specify an output path.")
            return
        # Change extension to .csv for evolution output
        base, _ = os.path.splitext(output_path)
        evo_output = base + "_evolution.csv"

        self.btnRunEvolution.setEnabled(False)
        self.progressBar.setValue(0)
        self.lblStatus.setText("Running surface evolution...")

        try:
            from qgis.core import QgsApplication

            def progress_callback(value, message=""):
                self.progressBar.setValue(value)
                if message:
                    self.lblStatus.setText(message)
                QgsApplication.processEvents()

            depth_formula_key = self.cmbDepthFormula.currentData()

            processor = SurfaceEvolutionProcessor(
                csv_paths=self._csv_paths,
                dtm_path=dtm_path,
                output_path=evo_output,
                slope_azimuth_deg=self.spnSlopeAzimuth.value(),
                slope_angle_deg=self.spnSlopeAngle.value(),
                friction_coeff=self.spnFriction.value(),
                grid_nx=self.spnGridNX.value(),
                grid_ny=self.spnGridNY.value(),
                max_distance=self.spnMaxDist.value(),
                date_col_offset=self.spnDateOffset.value(),
                depth_formula=depth_formula_key,
            )

            results = processor.run(progress_callback=progress_callback)

            self.progressBar.setValue(100)

            formula_info = results.get('depth_formula_info', {})
            depth_est = results.get('depth_estimate', 0.0)
            formula_label = formula_info.get('label', depth_formula_key)
            formula_ref = formula_info.get('ref', '')
            alpha = formula_info.get('alpha', '?')
            beta = formula_info.get('beta', '?')

            msg = (
                f"Surface evolution complete:\n"
                f"  - {results['n_timesteps']} time steps processed\n"
                f"  - Moving area: {results['moving_area_km2']:.3f} km²\n\n"
                f"Sliding Surface Depth Estimate:\n"
                f"  - Formula: {formula_label}\n"
                f"  - Equation: D = {alpha} × A^{beta}\n"
                f"  - Reference: {formula_ref}\n"
                f"  - Estimated depth: {depth_est:.2f} m\n\n"
                f"Output: {evo_output}"
            )
            self.lblStatus.setText(
                f"Done: {results['n_timesteps']} steps, "
                f"area={results['moving_area_km2']:.3f} km², "
                f"depth={depth_est:.2f} m")

            QMessageBox.information(
                self, "Surface Evolution Complete", msg)

        except Exception as e:
            self.lblStatus.setText(f"Error: {e}")
            QgsMessageLog.logMessage(
                f"Surface evolution error: {e}", "TestPlugin", Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(
                traceback.format_exc(), "TestPlugin", Qgis.Critical)
            QMessageBox.critical(
                self, "Error", f"Surface evolution failed:\n{e}")
        finally:
            self.btnRunEvolution.setEnabled(True)
