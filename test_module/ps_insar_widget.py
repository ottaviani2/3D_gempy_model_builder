# -*- coding: utf-8 -*-
"""
PS-InSAR Analysis Dialog
Standalone dialog for PS-InSAR moving area detection and boundary extraction.
"""

import os

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QRadioButton,
    QPushButton, QLineEdit, QFileDialog, QProgressBar,
    QMessageBox,
)
from qgis.PyQt.QtCore import Qt
from qgis.core import (
    QgsProject, QgsMapLayerProxyModel, QgsVectorLayer,
    QgsMessageLog, Qgis,
)
from qgis.gui import QgsMapLayerComboBox

from .ps_insar_module import (
    PSInSARProcessor, SurfaceEvolutionProcessor,
    GempyExporter, HAS_TORCH, HAS_PANDAS,
    LOSDecompositionProcessor,
)
from .ps_insar_decomposition import DEPTH_FORMULAS


class PSInSARDialog(QDialog):
    """Dialog for PS-InSAR moving area detection."""

    def __init__(self, output_folder=None, dtm_layer=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PS-InSAR Analysis")
        self.resize(800, 620)
        self.setWindowModality(Qt.ApplicationModal)

        self.output_folder = output_folder or ""
        self.dtm_layer = dtm_layer

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        root = QVBoxLayout()
        self.setLayout(root)

        # --- PS Data (2 controls per row) ---
        grp_data = QGroupBox("PS Data")
        lay_data = QGridLayout()
        lay_data.setColumnStretch(1, 1)
        lay_data.setColumnStretch(3, 1)
        grp_data.setLayout(lay_data)

        lay_data.addWidget(QLabel("PS Layer:"), 0, 0)
        self.cmbPSLayer = QgsMapLayerComboBox()
        self.cmbPSLayer.setFilters(QgsMapLayerProxyModel.VectorLayer)
        lay_data.addWidget(self.cmbPSLayer, 0, 1)

        lay_data.addWidget(QLabel("Velocity field:"), 0, 2)
        self.cmbVelocityField = QComboBox()
        lay_data.addWidget(self.cmbVelocityField, 0, 3)

        root.addWidget(grp_data)

        # --- Classification Parameters (2 controls per row) ---
        grp_class = QGroupBox("Classification Parameters")
        lay_class = QGridLayout()
        lay_class.setColumnStretch(1, 1)
        lay_class.setColumnStretch(3, 1)
        grp_class.setLayout(lay_class)

        lay_class.addWidget(QLabel("Velocity threshold:"), 0, 0)
        self.spnThreshold = QDoubleSpinBox()
        self.spnThreshold.setRange(0.1, 100.0)
        self.spnThreshold.setValue(2.0)
        self.spnThreshold.setSingleStep(0.5)
        self.spnThreshold.setToolTip(
            "Threshold applied to the velocity field selected for classification.\n"
            "Must be in the same units as the velocity field (typically mm/yr for\n"
            "EGMS/PS-InSAR data).  Points whose |velocity| exceeds this value are\n"
            "initially labelled as 'moving'.")
        lay_class.addWidget(self.spnThreshold, 0, 1)

        lay_class.addWidget(QLabel("K neighbors:"), 0, 2)
        self.spnKNeighbors = QSpinBox()
        self.spnKNeighbors.setRange(3, 100)
        self.spnKNeighbors.setValue(20)
        lay_class.addWidget(self.spnKNeighbors, 0, 3)

        lay_class.addWidget(QLabel("Epochs:"), 1, 0)
        self.spnEpochs = QSpinBox()
        self.spnEpochs.setRange(5, 500)
        self.spnEpochs.setValue(50)
        lay_class.addWidget(self.spnEpochs, 1, 1)

        lay_class.addWidget(QLabel("Refinement iterations:"), 1, 2)
        self.spnRefinement = QSpinBox()
        self.spnRefinement.setRange(0, 10)
        self.spnRefinement.setValue(1)
        lay_class.addWidget(self.spnRefinement, 1, 3)

        self.chkUseNN = QCheckBox("Use NN spatial smoothing (requires PyTorch)")
        self.chkUseNN.setChecked(HAS_TORCH)
        if not HAS_TORCH:
            self.chkUseNN.setEnabled(False)
            self.chkUseNN.setToolTip("PyTorch is not installed. Using threshold fallback.")
        lay_class.addWidget(self.chkUseNN, 2, 0, 1, 4)

        root.addWidget(grp_class)

        # --- Boundary Extraction (2 controls per row) ---
        grp_boundary = QGroupBox("Boundary Extraction")
        lay_boundary = QGridLayout()
        lay_boundary.setColumnStretch(1, 1)
        lay_boundary.setColumnStretch(3, 1)
        grp_boundary.setLayout(lay_boundary)

        lay_boundary.addWidget(QLabel("Min cluster size:"), 0, 0)
        self.spnMinCluster = QSpinBox()
        self.spnMinCluster.setRange(3, 1000)
        self.spnMinCluster.setValue(10)
        lay_boundary.addWidget(self.spnMinCluster, 0, 1)

        lay_boundary.addWidget(QLabel("Alpha (0=auto):"), 0, 2)
        self.spnAlpha = QDoubleSpinBox()
        self.spnAlpha.setRange(0.0, 100000.0)
        self.spnAlpha.setValue(0.0)
        self.spnAlpha.setSingleStep(10.0)
        lay_boundary.addWidget(self.spnAlpha, 0, 3)

        lay_boundary.addWidget(QLabel("Z source:"), 1, 0)
        self.cmbZSource = QComboBox()
        self.cmbZSource.addItems(["Constant", "From DTM", "From PS data"])
        lay_boundary.addWidget(self.cmbZSource, 1, 1)

        lay_boundary.addWidget(QLabel("Z constant value:"), 1, 2)
        self.spnZConstant = QDoubleSpinBox()
        self.spnZConstant.setRange(-10000.0, 100000.0)
        self.spnZConstant.setValue(0.0)
        lay_boundary.addWidget(self.spnZConstant, 1, 3)

        root.addWidget(grp_boundary)

        # --- Surface Evolution (2 controls per row) ---
        grp_evolution = QGroupBox("Surface Evolution (Multi-orbit Decomposition)")
        lay_evo = QGridLayout()
        lay_evo.setColumnStretch(1, 1)
        lay_evo.setColumnStretch(3, 1)
        grp_evolution.setLayout(lay_evo)

        self.chkEnableEvolution = QCheckBox("Enable surface evolution analysis")
        self.chkEnableEvolution.setChecked(False)
        if not HAS_PANDAS:
            self.chkEnableEvolution.setEnabled(False)
            self.chkEnableEvolution.setToolTip("pandas is required for CSV processing")
        lay_evo.addWidget(self.chkEnableEvolution, 0, 0, 1, 4)

        lay_evo.addWidget(QLabel("InSAR CSV files:"), 1, 0)
        self.lstCSVFiles = QComboBox()
        self.lstCSVFiles.setEditable(False)
        self.lstCSVFiles.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        lay_evo.addWidget(self.lstCSVFiles, 1, 1)
        self.btnAddCSV = QPushButton("+")
        self.btnAddCSV.setFixedWidth(30)
        self.btnAddCSV.setToolTip("Add InSAR CSV file (need >= 2 orbits)")
        lay_evo.addWidget(self.btnAddCSV, 1, 2)
        self.btnRemoveCSV = QPushButton("-")
        self.btnRemoveCSV.setFixedWidth(30)
        self.btnRemoveCSV.setToolTip("Remove selected CSV file")
        lay_evo.addWidget(self.btnRemoveCSV, 1, 3)

        lay_evo.addWidget(QLabel("Slope azimuth (deg):"), 2, 0)
        self.spnSlopeAzimuth = QDoubleSpinBox()
        self.spnSlopeAzimuth.setRange(0.0, 360.0)
        self.spnSlopeAzimuth.setValue(270.0)
        lay_evo.addWidget(self.spnSlopeAzimuth, 2, 1)
        lay_evo.addWidget(QLabel("Slope angle (deg):"), 2, 2)
        self.spnSlopeAngle = QDoubleSpinBox()
        self.spnSlopeAngle.setRange(0.0, 90.0)
        self.spnSlopeAngle.setValue(50.0)
        lay_evo.addWidget(self.spnSlopeAngle, 2, 3)

        lay_evo.addWidget(QLabel("Friction coeff f:"), 3, 0)
        self.spnFriction = QDoubleSpinBox()
        self.spnFriction.setRange(0.001, 1.0)
        self.spnFriction.setValue(0.01)
        self.spnFriction.setSingleStep(0.01)
        self.spnFriction.setDecimals(3)
        lay_evo.addWidget(self.spnFriction, 3, 1)
        lay_evo.addWidget(QLabel("Max distance (m):"), 3, 2)
        self.spnMaxDist = QDoubleSpinBox()
        self.spnMaxDist.setRange(1.0, 10000.0)
        self.spnMaxDist.setValue(100.0)
        lay_evo.addWidget(self.spnMaxDist, 3, 3)

        lay_evo.addWidget(QLabel("Grid NX:"), 4, 0)
        self.spnGridNX = QSpinBox()
        self.spnGridNX.setRange(10, 1000)
        self.spnGridNX.setValue(100)
        lay_evo.addWidget(self.spnGridNX, 4, 1)
        lay_evo.addWidget(QLabel("Grid NY:"), 4, 2)
        self.spnGridNY = QSpinBox()
        self.spnGridNY.setRange(10, 1000)
        self.spnGridNY.setValue(100)
        lay_evo.addWidget(self.spnGridNY, 4, 3)

        lay_evo.addWidget(QLabel("Date column offset:"), 5, 0)
        self.spnDateOffset = QSpinBox()
        self.spnDateOffset.setRange(1, 200)
        self.spnDateOffset.setValue(28)
        self.spnDateOffset.setToolTip(
            "Column index where time-series displacement data starts in the CSV")
        lay_evo.addWidget(self.spnDateOffset, 5, 1)

        self.btnRunEvolution = QPushButton("Run Surface Evolution")
        self.btnRunEvolution.setMinimumHeight(28)
        lay_evo.addWidget(self.btnRunEvolution, 5, 2, 1, 2)

        root.addWidget(grp_evolution)

        # --- Depth Estimation Formula (formula + ref on one row) ---
        grp_depth = QGroupBox("Sliding Surface Depth Estimation")
        lay_depth = QGridLayout()
        lay_depth.setColumnStretch(1, 1)
        lay_depth.setColumnStretch(3, 1)
        grp_depth.setLayout(lay_depth)

        lay_depth.addWidget(QLabel("Formula:"), 0, 0)
        self.cmbDepthFormula = QComboBox()
        for key, info in DEPTH_FORMULAS.items():
            self.cmbDepthFormula.addItem(info['label'], userData=key)
        lay_depth.addWidget(self.cmbDepthFormula, 0, 1)

        lay_depth.addWidget(QLabel("Reference:"), 0, 2)
        self.lblFormulaRef = QLabel()
        self.lblFormulaRef.setStyleSheet("font-weight: bold;")
        lay_depth.addWidget(self.lblFormulaRef, 0, 3)

        self.lblFormulaDesc = QLabel()
        self.lblFormulaDesc.setWordWrap(True)
        self.lblFormulaDesc.setStyleSheet("color: #555555; font-style: italic;")
        lay_depth.addWidget(self.lblFormulaDesc, 1, 0, 1, 4)

        root.addWidget(grp_depth)

        # --- Output ---
        grp_output = QGroupBox("Output")
        lay_output = QGridLayout()
        lay_output.setColumnStretch(1, 1)
        grp_output.setLayout(lay_output)

        lay_output.addWidget(QLabel("Output shapefile:"), 0, 0)
        self.txtOutputPath = QLineEdit()
        if self.output_folder:
            self.txtOutputPath.setText(
                os.path.join(self.output_folder, "ps_moving_boundary.shp"))
        lay_output.addWidget(self.txtOutputPath, 0, 1)
        self.btnBrowseOutput = QPushButton("...")
        self.btnBrowseOutput.setFixedWidth(30)
        lay_output.addWidget(self.btnBrowseOutput, 0, 2)

        self.chkAddToQGIS = QCheckBox("Add result to QGIS layers")
        self.chkAddToQGIS.setChecked(True)
        lay_output.addWidget(self.chkAddToQGIS, 1, 0, 1, 2)

        self.btnExportGemPy = QPushButton("Export to GemPy CSV...")
        self.btnExportGemPy.setMinimumHeight(28)
        self.btnExportGemPy.setToolTip(
            "Export boundary points and sliding surface to GemPy-compatible CSV files.\n"
            "Run analysis first.")
        self.btnExportGemPy.setEnabled(False)
        lay_output.addWidget(self.btnExportGemPy, 1, 2)

        root.addWidget(grp_output)

        # --- Run / Progress ---
        run_row = QHBoxLayout()
        self.btnRun = QPushButton("Run")
        self.btnRun.setMinimumHeight(30)
        run_row.addWidget(self.btnRun)
        self.lblStatus = QLabel("Ready")
        run_row.addWidget(self.lblStatus)
        root.addLayout(run_row)

        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)
        root.addWidget(self.progressBar)

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
        self.btnExportGemPy.clicked.connect(self._export_gempy)
        self._csv_paths = []
        self._last_boundary_points = None   # populated after successful _run_analysis

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

            # Store for GemPy export
            self._last_boundary_points = boundary_points

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

            if boundary_points:
                self.btnExportGemPy.setEnabled(True)

            QMessageBox.information(
                self, "PS-InSAR Analysis Complete",
                f"Classification complete:\n"
                f"  - {len(labels)} total points\n"
                f"  - {n_moving} moving points\n"
                f"  - {n_boundary} boundary points extracted\n\n"
                f"Output: {output_path}\n\n"
                f"Use 'Export to GemPy CSV...' to generate GemPy-ready files.")

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

    # --- GemPy export ---

    def _export_gempy(self):
        """Export boundary + sliding-surface points to GemPy-compatible CSV files."""
        if not self._last_boundary_points:
            QMessageBox.warning(
                self, "Warning",
                "No analysis results available. Please run the analysis first.")
            return

        # Ask the user for the output prefix (folder + base name)
        default_dir = self.txtOutputPath.text().strip()
        if default_dir:
            default_prefix = os.path.splitext(default_dir)[0]
        else:
            default_prefix = ""

        prefix, _ = QFileDialog.getSaveFileName(
            self,
            "Save GemPy CSV files (enter base name without extension)",
            default_prefix,
            "CSV prefix (*)")

        if not prefix:
            return

        # Strip any extension the user may have typed
        prefix = os.path.splitext(prefix)[0]

        depth_formula_key = self.cmbDepthFormula.currentData()

        try:
            exporter = GempyExporter(
                boundary_points=self._last_boundary_points,
                depth_formula_key=depth_formula_key,
            )
            sp_path, or_path = exporter.export(prefix)

            QMessageBox.information(
                self, "GemPy Export Complete",
                f"Two GemPy-compatible CSV files written:\n\n"
                f"Surface points ({len(self._last_boundary_points) * 2} rows):\n"
                f"  {sp_path}\n\n"
                f"Orientations ({len(self._last_boundary_points) * 2} rows):\n"
                f"  {or_path}\n\n"
                f"Formations:\n"
                f"  landslide_boundary_N  — PS-InSAR boundary on topography\n"
                f"  sliding_surface_N     — estimated sliding surface at depth\n\n"
                f"Depth formula: {depth_formula_key}")

            self.lblStatus.setText(
                f"GemPy CSVs exported: {os.path.basename(sp_path)}, "
                f"{os.path.basename(or_path)}")

        except Exception as e:
            QgsMessageLog.logMessage(
                f"GemPy export error: {e}", "TestPlugin", Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(
                traceback.format_exc(), "TestPlugin", Qgis.Critical)
            QMessageBox.critical(self, "Export Error", f"GemPy export failed:\n{e}")


# ---------------------------------------------------------------------------
# Helper: single-geometry input panel
# ---------------------------------------------------------------------------

_DEFAULT_INC = [38.0, 38.0, 38.0]
_DEFAULT_HDG = [350.0, 190.0, 350.0]  # asc, desc, third (asc second track)


class _GeometryPanel(QGroupBox):
    """
    Compact input panel for one InSAR geometry (satellite pass).
    Contains layer selector, LOS velocity field, incidence angle
    and satellite heading (each can be read from a field or given as a constant).
    """

    def __init__(self, number, parent=None):
        super().__init__(f"Geometry {number}", parent)
        self._number = number
        self._build_ui()
        self._connect()

    # ------------------------------------------------------------------
    def _build_ui(self):
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        self.setLayout(grid)

        # --- Row 0: PS layer ---
        grid.addWidget(QLabel("PS layer:"), 0, 0)
        self.cmbLayer = QgsMapLayerComboBox()
        self.cmbLayer.setFilters(QgsMapLayerProxyModel.PointLayer)
        self.cmbLayer.setAllowEmptyLayer(True)
        self.cmbLayer.setCurrentIndex(0)
        grid.addWidget(self.cmbLayer, 0, 1, 1, 3)

        # --- Row 1: LOS velocity field ---
        grid.addWidget(QLabel("LOS velocity:"), 1, 0)
        self.cmbVelField = QComboBox()
        grid.addWidget(self.cmbVelField, 1, 1, 1, 3)

        # --- Row 2: Incidence angle ---
        grid.addWidget(QLabel("Incidence (°):"), 2, 0)
        self.chkIncField = QCheckBox("From field:")
        self.cmbIncField = QComboBox()
        self.cmbIncField.setEnabled(False)
        self.spnIncConst = QDoubleSpinBox()
        self.spnIncConst.setRange(0.0, 90.0)
        self.spnIncConst.setValue(_DEFAULT_INC[self._number - 1])
        self.spnIncConst.setDecimals(1)
        self.spnIncConst.setSuffix(" °")
        self.spnIncConst.setToolTip("Incidence angle from vertical (nadir).")
        grid.addWidget(self.chkIncField, 2, 1)
        grid.addWidget(self.cmbIncField, 2, 2)
        grid.addWidget(self.spnIncConst, 2, 3)

        # --- Row 3: Satellite heading ---
        grid.addWidget(QLabel("Sat. heading (°):"), 3, 0)
        self.chkHdgField = QCheckBox("From field:")
        self.cmbHdgField = QComboBox()
        self.cmbHdgField.setEnabled(False)
        self.spnHdgConst = QDoubleSpinBox()
        self.spnHdgConst.setRange(-180.0, 360.0)
        self.spnHdgConst.setValue(_DEFAULT_HDG[self._number - 1])
        self.spnHdgConst.setDecimals(1)
        self.spnHdgConst.setSuffix(" °")
        self.spnHdgConst.setToolTip(
            "Satellite flight azimuth clockwise from North.\n"
            "Ascending Sentinel-1 ≈ 350°,  Descending ≈ 190°.")
        grid.addWidget(self.chkHdgField, 3, 1)
        grid.addWidget(self.cmbHdgField, 3, 2)
        grid.addWidget(self.spnHdgConst, 3, 3)

    def _connect(self):
        self.cmbLayer.layerChanged.connect(self._on_layer_changed)
        self.chkIncField.toggled.connect(self.cmbIncField.setEnabled)
        self.chkIncField.toggled.connect(
            lambda checked: self.spnIncConst.setEnabled(not checked))
        self.chkHdgField.toggled.connect(self.cmbHdgField.setEnabled)
        self.chkHdgField.toggled.connect(
            lambda checked: self.spnHdgConst.setEnabled(not checked))
        self._on_layer_changed(self.cmbLayer.currentLayer())

    def _on_layer_changed(self, layer):
        for cmb in (self.cmbVelField, self.cmbIncField, self.cmbHdgField):
            cmb.clear()
        if layer is None or not hasattr(layer, 'fields'):
            return
        for f in layer.fields():
            tn = f.typeName().lower()
            is_numeric = (
                any(t in tn for t in ['int', 'real', 'double',
                                       'float', 'numeric', 'decimal'])
                or f.type() in (2, 6))  # QVariant.Int, QVariant.Double
            if is_numeric:
                for cmb in (self.cmbVelField, self.cmbIncField,
                            self.cmbHdgField):
                    cmb.addItem(f.name())
        # Auto-select a velocity-like field
        for i in range(self.cmbVelField.count()):
            n = self.cmbVelField.itemText(i).lower()
            if any(kw in n for kw in
                   ['vel', 'los', 'v_', 'disp', 'mm', 'rate', 'defo']):
                self.cmbVelField.setCurrentIndex(i)
                break

    # ------------------------------------------------------------------
    def get_config(self):
        """Return a dict suitable for LOSDecompositionProcessor."""
        layer = self.cmbLayer.currentLayer()
        if layer is None:
            return None
        return {
            'layer':     layer,
            'vel_field': self.cmbVelField.currentText(),
            'inc_field': (self.cmbIncField.currentText()
                          if self.chkIncField.isChecked() else None),
            'inc_const': self.spnIncConst.value(),
            'hdg_field': (self.cmbHdgField.currentText()
                          if self.chkHdgField.isChecked() else None),
            'hdg_const': self.spnHdgConst.value(),
        }


# ---------------------------------------------------------------------------
# Main 3-geometry LOS decomposition dialog
# ---------------------------------------------------------------------------

class LOSDecompositionDialog(QDialog):
    """
    Dialog for 3-geometry InSAR LOS decomposition.

    Accepts three PS-InSAR point layers (one per satellite geometry) and
    decomposes their Line-of-Sight velocities into:
      • East-West  displacement rate  (dE)
      • Up-Down    displacement rate  (dU)
      • North-South displacement rate (dN)  — optional

    The decomposition uses the right-looking SAR look-vector convention:
        look_azimuth = satellite_heading + 90°
        eE = sin(θ) · sin(look_az)
        eN = sin(θ) · cos(look_az)
        eU = cos(θ)
    where θ is the incidence angle from vertical.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("InSAR 3-Geometry Decomposition  —  E/W · U/D")
        self.resize(720, 900)
        self.setWindowModality(Qt.ApplicationModal)
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout()
        self.setLayout(root)

        # Three geometry panels
        self._panels = []
        for n in range(1, 4):
            panel = _GeometryPanel(n, self)
            root.addWidget(panel)
            self._panels.append(panel)

        # Grid & interpolation parameters
        grp_grid = QGroupBox("Grid & Interpolation Parameters")
        lay_grid = QGridLayout()
        lay_grid.setColumnStretch(1, 1)
        lay_grid.setColumnStretch(3, 1)
        grp_grid.setLayout(lay_grid)

        lay_grid.addWidget(QLabel("Grid NX:"), 0, 0)
        self.spnNX = QSpinBox()
        self.spnNX.setRange(10, 2000)
        self.spnNX.setValue(100)
        lay_grid.addWidget(self.spnNX, 0, 1)

        lay_grid.addWidget(QLabel("Grid NY:"), 0, 2)
        self.spnNY = QSpinBox()
        self.spnNY.setRange(10, 2000)
        self.spnNY.setValue(100)
        lay_grid.addWidget(self.spnNY, 0, 3)

        lay_grid.addWidget(QLabel("Max interp. distance:"), 1, 0)
        self.spnMaxDist = QDoubleSpinBox()
        self.spnMaxDist.setRange(0.0, 1e7)
        self.spnMaxDist.setValue(1000.0)
        self.spnMaxDist.setDecimals(0)
        self.spnMaxDist.setToolTip(
            "Grid cells farther than this from the nearest input point\n"
            "are masked as invalid (same unit as the layer CRS).")
        lay_grid.addWidget(self.spnMaxDist, 1, 1)

        lay_grid.addWidget(QLabel("Smoothing σ (px):"), 1, 2)
        self.spnSigma = QDoubleSpinBox()
        self.spnSigma.setRange(0.0, 20.0)
        self.spnSigma.setValue(1.5)
        self.spnSigma.setSingleStep(0.5)
        self.spnSigma.setToolTip("Gaussian smoothing applied to interpolated grids.")
        lay_grid.addWidget(self.spnSigma, 1, 3)

        lay_grid.addWidget(QLabel("Solve for:"), 2, 0)
        self.chkSolveNorth = QCheckBox(
            "Include North component  (solve for E + N + U — exact 3×3 system)")
        self.chkSolveNorth.setChecked(True)
        self.chkSolveNorth.setToolTip(
            "Checked: solve the exact 3×3 system for dE, dN, dU.\n"
            "Unchecked: solve overdetermined 3×2 system for dE and dU only\n"
            "(assumes North displacement ≈ 0, which is common for InSAR).")
        lay_grid.addWidget(self.chkSolveNorth, 2, 1, 1, 3)

        lay_grid.addWidget(QLabel("Condition threshold (10^n):"), 3, 0)
        self.spnCondExp = QSpinBox()
        self.spnCondExp.setRange(2, 12)
        self.spnCondExp.setValue(4)
        self.spnCondExp.setToolTip(
            "Grid cells whose look-vector matrix condition number exceeds\n"
            "10^n are treated as geometrically degenerate and set to NaN.\n"
            "This happens when two geometries are nearly parallel.\n"
            "  n=2  → 100   (strict — rejects mildly poor geometry)\n"
            "  n=4  → 10 000 (default)\n"
            "  n=6  → 1 000 000 (lenient — keeps more data)\n"
            "A diagnostic layer 'LOS_condition_number' is always added\n"
            "to QGIS so you can inspect where masking occurred.")
        lay_grid.addWidget(self.spnCondExp, 3, 1)
        self._lblCondValue = QLabel("= 10 000")
        self.spnCondExp.valueChanged.connect(
            lambda n: self._lblCondValue.setText(f"= {10**n:,.0f}"))
        lay_grid.addWidget(self._lblCondValue, 3, 2, 1, 2)

        root.addWidget(grp_grid)

        # Moving Area Detection
        grp_ma = QGroupBox("Moving Area Detection")
        grp_ma.setCheckable(True)
        grp_ma.setChecked(False)
        grp_ma.setToolTip(
            "When enabled, the selected decomposed component is classified\n"
            "into moving / stable pixels using a velocity threshold and an\n"
            "optional neural-network refinement step.")
        self.grpMA = grp_ma
        lay_ma = QGridLayout()
        lay_ma.setColumnStretch(1, 1)
        lay_ma.setColumnStretch(3, 1)
        grp_ma.setLayout(lay_ma)

        # Row 0: component + threshold
        lay_ma.addWidget(QLabel("Component:"), 0, 0)
        self.cmbMAComponent = QComboBox()
        self.cmbMAComponent.addItems(["Magnitude (|dE|²+|dU|²)½", "East-West (dE)",
                                      "Up-Down (dU)", "North-South (dN)"])
        self.cmbMAComponent.setToolTip(
            "Decomposed displacement component on which moving area\n"
            "detection is applied.")
        lay_ma.addWidget(self.cmbMAComponent, 0, 1)

        lay_ma.addWidget(QLabel("Velocity threshold:"), 0, 2)
        self.spnMAThreshold = QDoubleSpinBox()
        self.spnMAThreshold.setRange(0.01, 1000.0)
        self.spnMAThreshold.setValue(2.0)
        self.spnMAThreshold.setDecimals(2)
        self.spnMAThreshold.setToolTip(
            "Threshold applied to the decomposed velocity component selected above.\n"
            "Must be in the same units as the input LOS field (typically mm/yr for\n"
            "EGMS/PS-InSAR data).  Points whose |velocity| exceeds this value are\n"
            "initially labelled as 'moving'.")
        lay_ma.addWidget(self.spnMAThreshold, 0, 3)

        # Row 1: k-neighbors + epochs
        lay_ma.addWidget(QLabel("k neighbors:"), 1, 0)
        self.spnMAK = QSpinBox()
        self.spnMAK.setRange(3, 200)
        self.spnMAK.setValue(20)
        self.spnMAK.setToolTip("Number of nearest neighbors used for feature extraction.")
        lay_ma.addWidget(self.spnMAK, 1, 1)

        lay_ma.addWidget(QLabel("NN epochs:"), 1, 2)
        self.spnMAEpochs = QSpinBox()
        self.spnMAEpochs.setRange(1, 500)
        self.spnMAEpochs.setValue(50)
        self.spnMAEpochs.setToolTip(
            "Training epochs per neural-network refinement iteration.\n"
            "Only used when 'Use neural network' is checked.")
        lay_ma.addWidget(self.spnMAEpochs, 1, 3)

        # Row 2: min cluster size + use NN
        lay_ma.addWidget(QLabel("Min cluster size:"), 2, 0)
        self.spnMAMinCluster = QSpinBox()
        self.spnMAMinCluster.setRange(1, 10000)
        self.spnMAMinCluster.setValue(10)
        self.spnMAMinCluster.setToolTip(
            "Clusters with fewer than this many moving points are discarded.")
        lay_ma.addWidget(self.spnMAMinCluster, 2, 1)

        self.chkMAUseNN = QCheckBox("Use neural network refinement")
        self.chkMAUseNN.setChecked(True)
        self.chkMAUseNN.setToolTip(
            "Refine the initial threshold labels using a small MLP classifier\n"
            "(requires PyTorch). Uncheck for threshold-only classification.")
        lay_ma.addWidget(self.chkMAUseNN, 2, 2, 1, 2)

        # Row 3: boundary shapefile output
        lay_ma.addWidget(QLabel("Boundary shapefile:"), 3, 0)
        self.txtMAOutput = QLineEdit()
        self.txtMAOutput.setPlaceholderText("Leave blank to skip shapefile export")
        lay_ma.addWidget(self.txtMAOutput, 3, 1, 1, 2)
        self.btnMABrowse = QPushButton("...")
        self.btnMABrowse.setFixedWidth(30)
        self.btnMABrowse.clicked.connect(self._browse_ma_output)
        lay_ma.addWidget(self.btnMABrowse, 3, 3)

        root.addWidget(grp_ma)

        # Output
        grp_out = QGroupBox("Output")
        lay_out = QGridLayout()
        lay_out.setColumnStretch(1, 1)
        grp_out.setLayout(lay_out)

        lay_out.addWidget(QLabel("Output CSV (optional):"), 0, 0)
        self.txtOutput = QLineEdit()
        self.txtOutput.setPlaceholderText("Leave blank to skip CSV export")
        lay_out.addWidget(self.txtOutput, 0, 1)
        self.btnBrowse = QPushButton("...")
        self.btnBrowse.setFixedWidth(30)
        lay_out.addWidget(self.btnBrowse, 0, 2)

        root.addWidget(grp_out)

        # Run / status row
        run_row = QHBoxLayout()
        self.btnRun = QPushButton("Run Decomposition")
        self.btnRun.setMinimumHeight(30)
        run_row.addWidget(self.btnRun)
        self.lblStatus = QLabel("Ready")
        run_row.addWidget(self.lblStatus, 1)
        root.addLayout(run_row)

        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)
        root.addWidget(self.progressBar)

    # ------------------------------------------------------------------
    def _connect_signals(self):
        self.btnRun.clicked.connect(self._run)
        self.btnBrowse.clicked.connect(self._browse_output)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Decomposition CSV", "", "CSV Files (*.csv)")
        if path:
            if not path.lower().endswith('.csv'):
                path += '.csv'
            self.txtOutput.setText(path)

    def _browse_ma_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Moving Area Boundary Shapefile", "",
            "Shapefiles (*.shp)")
        if path:
            if not path.lower().endswith('.shp'):
                path += '.shp'
            self.txtMAOutput.setText(path)

    # ------------------------------------------------------------------
    def _run(self):
        import numpy as np

        # Collect and validate geometry configs
        configs = []
        for i, panel in enumerate(self._panels):
            cfg = panel.get_config()
            if cfg is None:
                QMessageBox.warning(
                    self, "Missing layer",
                    f"Please select a PS-InSAR layer for Geometry {i + 1}.")
                return
            if not cfg['vel_field']:
                QMessageBox.warning(
                    self, "Missing field",
                    f"Please select a LOS velocity field for Geometry {i + 1}.")
                return
            configs.append(cfg)

        output_path = self.txtOutput.text().strip() or None

        # Moving area parameters
        ma_enabled   = self.grpMA.isChecked()
        _comp_map    = {0: 'magnitude', 1: 'ew', 2: 'ud', 3: 'ns'}
        ma_component = _comp_map.get(self.cmbMAComponent.currentIndex(), 'magnitude')
        ma_output    = self.txtMAOutput.text().strip() or None

        processor = LOSDecompositionProcessor(
            layer_configs=configs,
            grid_nx=self.spnNX.value(),
            grid_ny=self.spnNY.value(),
            max_distance=self.spnMaxDist.value(),
            smoothing_sigma=self.spnSigma.value(),
            solve_north=self.chkSolveNorth.isChecked(),
            output_path=output_path,
            cond_threshold=10 ** self.spnCondExp.value(),
            ma_enabled=ma_enabled,
            ma_component=ma_component,
            ma_vel_threshold=self.spnMAThreshold.value(),
            ma_k_neighbors=self.spnMAK.value(),
            ma_epochs=self.spnMAEpochs.value(),
            ma_refinement_iters=1,
            ma_min_cluster_size=self.spnMAMinCluster.value(),
            ma_alpha=0.0,
            ma_use_nn=self.chkMAUseNN.isChecked(),
            ma_output_path=ma_output,
        )

        self.btnRun.setEnabled(False)
        self.progressBar.setValue(0)
        self.lblStatus.setText("Running…")

        try:
            from qgis.core import QgsApplication

            def cb(val, msg=''):
                self.progressBar.setValue(val)
                if msg:
                    self.lblStatus.setText(msg)
                QgsApplication.processEvents()

            results = processor.run(progress_callback=cb)

            dE, dU = results['dE'], results['dU']
            n_valid = int(np.isfinite(dE).sum())

            summary = (
                f"Decomposition complete.\n\n"
                f"Valid grid cells : {n_valid}\n"
                f"East-West  range : {np.nanmin(dE):.2f} – {np.nanmax(dE):.2f} mm/yr\n"
                f"Up-Down    range : {np.nanmin(dU):.2f} – {np.nanmax(dU):.2f} mm/yr"
            )
            if self.chkSolveNorth.isChecked():
                dN = results['dN']
                summary += (
                    f"\nNorth-South range: "
                    f"{np.nanmin(dN):.2f} – {np.nanmax(dN):.2f} mm/yr"
                )
            if output_path:
                summary += f"\n\nCSV written to:\n{output_path}"

            self.lblStatus.setText(
                f"Done — {n_valid} cells, "
                f"dE [{np.nanmin(dE):.1f}, {np.nanmax(dE):.1f}], "
                f"dU [{np.nanmin(dU):.1f}, {np.nanmax(dU):.1f}]")
            self.progressBar.setValue(100)

            QMessageBox.information(
                self, "InSAR LOS Decomposition Complete", summary)

        except Exception as e:
            self.lblStatus.setText(f"Error: {e}")
            QgsMessageLog.logMessage(
                f"LOS decomposition error: {e}", "TestPlugin", Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(
                traceback.format_exc(), "TestPlugin", Qgis.Critical)
            QMessageBox.critical(
                self, "Decomposition Error",
                f"LOS decomposition failed:\n{e}")
        finally:
            self.btnRun.setEnabled(True)
