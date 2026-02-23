# -*- coding: utf-8 -*-
"""
PS-InSAR Moving Area Detection Module
Detects moving areas from PS-InSAR point data using neural network
classification and geometric boundary extraction.
"""

import os
import numpy as np
from scipy.spatial import KDTree, Delaunay

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsField, QgsFeature,
    QgsGeometry, QgsPointXY, QgsVectorFileWriter,
    QgsWkbTypes, QgsCoordinateReferenceSystem,
    QgsMessageLog, Qgis, QgsApplication,
    QgsSymbol, QgsRendererCategory, QgsCategorizedSymbolRenderer,
    QgsCoordinateTransformContext
)
from qgis.PyQt.QtCore import QVariant
from qgis.utils import iface

# Optional imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

class PSFeatureExtractor:
    """Extract spatial and velocity features from PS-InSAR points."""

    def __init__(self, xy, velocity, k_neighbors=20):
        self.xy = np.array(xy, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.k = k_neighbors
        self.tree = KDTree(self.xy)

        # Auto-compute search radius as median of k-th NN distance
        dists, _ = self.tree.query(self.xy, k=min(self.k + 1, len(self.xy)))
        self.radius = float(np.median(dists[:, -1])) if dists.shape[1] > 1 else 100.0

    def extract(self):
        """Return (N, 7) feature array."""
        n = len(self.xy)

        # Normalize XY
        xy_min = self.xy.min(axis=0)
        xy_max = self.xy.max(axis=0)
        xy_range = xy_max - xy_min
        xy_range[xy_range == 0] = 1.0
        xy_norm = (self.xy - xy_min) / xy_range

        # Standardize velocity
        vel_mean = self.velocity.mean()
        vel_std = self.velocity.std()
        if vel_std == 0:
            vel_std = 1.0
        vel_standardized = (self.velocity - vel_mean) / vel_std
        abs_vel = np.abs(vel_standardized)

        # Local features from k nearest neighbors
        local_density = np.zeros(n)
        local_mean_vel = np.zeros(n)
        local_std_vel = np.zeros(n)

        k_query = min(self.k + 1, n)
        dists_k, idx_k = self.tree.query(self.xy, k=k_query)

        for i in range(n):
            neighbors_in_radius = self.tree.query_ball_point(
                self.xy[i], self.radius)
            local_density[i] = len(neighbors_in_radius)

            neighbor_indices = idx_k[i, 1:]  # exclude self
            if len(neighbor_indices) > 0:
                neighbor_vels = self.velocity[neighbor_indices]
                local_mean_vel[i] = neighbor_vels.mean()
                local_std_vel[i] = neighbor_vels.std() if len(neighbor_vels) > 1 else 0.0

        # Normalize local features
        ld_max = local_density.max()
        if ld_max > 0:
            local_density = local_density / ld_max

        lm_std = local_mean_vel.std()
        if lm_std > 0:
            local_mean_vel = (local_mean_vel - local_mean_vel.mean()) / lm_std

        ls_max = local_std_vel.max()
        if ls_max > 0:
            local_std_vel = local_std_vel / ls_max

        features = np.column_stack([
            xy_norm[:, 0],       # x_norm
            xy_norm[:, 1],       # y_norm
            vel_standardized,    # velocity (standardized)
            abs_vel,             # abs_velocity
            local_density,       # local_density
            local_mean_vel,      # local_mean_vel
            local_std_vel,       # local_std_vel
        ])
        return features.astype(np.float32)


# ---------------------------------------------------------------------------
# Neural Network Classifier
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class PSClassifierMLP(nn.Module):
        """Simple MLP for binary PS-InSAR point classification."""

        def __init__(self, input_dim=7):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Trainer (pseudo-label self-training)
# ---------------------------------------------------------------------------

class PSTrainer:
    """Train classifier using pseudo-labels from velocity threshold."""

    def __init__(self, velocity_threshold=2.0, epochs=50, batch_size=256,
                 lr=0.001, refinement_iters=1, use_nn=True):
        self.velocity_threshold = velocity_threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.refinement_iters = refinement_iters
        self.use_nn = use_nn and HAS_TORCH

    def classify(self, features, velocity, progress_callback=None):
        """
        Classify points as moving (1) or stable (0).
        Returns labels array of shape (N,).
        """
        # Initial pseudo-labels from velocity threshold
        labels = (np.abs(velocity) > self.velocity_threshold).astype(np.float32)

        if not self.use_nn:
            return labels

        # Train MLP
        device = torch.device('cpu')
        model = PSClassifierMLP(input_dim=features.shape[1]).to(device)

        total_iters = 1 + self.refinement_iters
        for iteration in range(total_iters):
            model = self._train_model(model, features, labels, device,
                                       progress_callback, iteration, total_iters)

            # Predict new labels
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(features, dtype=torch.float32).to(device)
                logits = model(X_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                labels = (probs > 0.5).astype(np.float32)

        return labels

    def _train_model(self, model, features, labels, device,
                     progress_callback, iteration, total_iters):
        """Train model for one round."""
        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)

        # Class weights for imbalance
        n_pos = y.sum().item()
        n_neg = len(y) - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos])
        else:
            pos_weight = torch.tensor([1.0])

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

            if progress_callback:
                overall = (iteration * self.epochs + epoch + 1) / (total_iters * self.epochs)
                progress_callback(int(overall * 100))

        return model


# ---------------------------------------------------------------------------
# Moving Area Boundary Extraction
# ---------------------------------------------------------------------------

class MovingAreaBoundaryExtractor:
    """Extract boundary points from classified moving areas."""

    def __init__(self, min_cluster_size=10, alpha=0.0):
        self.min_cluster_size = min_cluster_size
        self.alpha = alpha  # 0 = auto

    def extract_boundaries(self, xy, labels, z_values=None):
        """
        From classified moving points, extract boundary points.
        Returns list of dicts: {x, y, z, cluster_id}
        """
        moving_mask = labels.astype(bool)
        moving_xy = xy[moving_mask]

        if len(moving_xy) < self.min_cluster_size:
            QgsMessageLog.logMessage(
                f"Too few moving points ({len(moving_xy)}) for boundary extraction",
                "TestPlugin", Qgis.Warning)
            return []

        # DBSCAN clustering
        if not HAS_SKLEARN:
            QgsMessageLog.logMessage(
                "scikit-learn not available, treating all moving points as one cluster",
                "TestPlugin", Qgis.Warning)
            cluster_labels = np.zeros(len(moving_xy), dtype=int)
        else:
            # Auto eps from median nearest-neighbor distance
            tree = KDTree(moving_xy)
            dists, _ = tree.query(moving_xy, k=2)
            eps = float(np.median(dists[:, 1])) * 2.0

            db = DBSCAN(eps=eps, min_samples=5)
            cluster_labels = db.fit_predict(moving_xy)

        # Z values for moving points
        if z_values is not None:
            moving_z = z_values[moving_mask]
        else:
            moving_z = np.zeros(len(moving_xy))

        boundary_points = []
        unique_clusters = set(cluster_labels)
        unique_clusters.discard(-1)  # remove noise label

        for cid in unique_clusters:
            cmask = cluster_labels == cid
            cluster_xy = moving_xy[cmask]
            cluster_z = moving_z[cmask]

            if len(cluster_xy) < self.min_cluster_size:
                continue

            if len(cluster_xy) < 4:
                # Too few for Delaunay, use all points
                for i in range(len(cluster_xy)):
                    boundary_points.append({
                        'x': float(cluster_xy[i, 0]),
                        'y': float(cluster_xy[i, 1]),
                        'z': float(cluster_z[i]),
                        'cluster_id': int(cid),
                    })
                continue

            # Alpha shape boundary extraction
            bp = self._alpha_shape_boundary(cluster_xy, cluster_z, int(cid))
            boundary_points.extend(bp)

        return boundary_points

    def _alpha_shape_boundary(self, xy, z, cluster_id):
        """Compute alpha shape boundary and densify."""
        try:
            tri = Delaunay(xy)
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Delaunay failed for cluster {cluster_id}: {e}",
                "TestPlugin", Qgis.Warning)
            return []

        # Compute alpha value if auto
        if self.alpha <= 0:
            # Use mean edge length as alpha
            edges_set = set()
            for simplex in tri.simplices:
                for i in range(3):
                    edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                    edges_set.add(edge)
            edge_lengths = []
            for e in edges_set:
                d = np.linalg.norm(xy[e[0]] - xy[e[1]])
                edge_lengths.append(d)
            alpha = float(np.mean(edge_lengths)) * 2.0 if edge_lengths else 100.0
        else:
            alpha = self.alpha

        # Find boundary edges (edges belonging to only one triangle)
        edge_count = {}
        for simplex in tri.simplices:
            # Check if triangle edges are within alpha
            edges = []
            for i in range(3):
                a_idx, b_idx = simplex[i], simplex[(i + 1) % 3]
                d = np.linalg.norm(xy[a_idx] - xy[b_idx])
                edges.append((a_idx, b_idx, d))

            # Skip triangles with edges longer than alpha
            max_edge = max(e[2] for e in edges)
            if max_edge > alpha:
                continue

            for a_idx, b_idx, _ in edges:
                edge = tuple(sorted([a_idx, b_idx]))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # Boundary edges appear exactly once
        boundary_edges = [e for e, c in edge_count.items() if c == 1]

        if not boundary_edges:
            # Fallback: use convex hull boundary indices
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(xy)
                boundary_indices = set(hull.vertices)
            except Exception:
                boundary_indices = set(range(len(xy)))
        else:
            boundary_indices = set()
            for e in boundary_edges:
                boundary_indices.add(e[0])
                boundary_indices.add(e[1])

        # Densify boundary
        boundary_points = self._densify_boundary(
            xy, z, boundary_edges if boundary_edges else None,
            boundary_indices, cluster_id)

        return boundary_points

    def _densify_boundary(self, xy, z, boundary_edges, boundary_indices, cluster_id):
        """Interpolate additional points along boundary edges."""
        # Compute median NN distance for max spacing
        tree = KDTree(xy)
        dists, _ = tree.query(xy, k=min(2, len(xy)))
        max_spacing = float(np.median(dists[:, -1])) if dists.shape[1] > 1 else 50.0

        points = []

        if boundary_edges:
            for a_idx, b_idx in boundary_edges:
                pa = xy[a_idx]
                pb = xy[b_idx]
                za = z[a_idx]
                zb = z[b_idx]
                edge_len = np.linalg.norm(pb - pa)
                n_segments = max(1, int(np.ceil(edge_len / max_spacing)))

                for s in range(n_segments + 1):
                    t = s / n_segments
                    px = pa[0] + t * (pb[0] - pa[0])
                    py = pa[1] + t * (pb[1] - pa[1])
                    pz = za + t * (zb - za)
                    points.append({
                        'x': float(px),
                        'y': float(py),
                        'z': float(pz),
                        'cluster_id': cluster_id,
                    })
        else:
            # No edges, just output boundary vertex points
            for idx in boundary_indices:
                points.append({
                    'x': float(xy[idx, 0]),
                    'y': float(xy[idx, 1]),
                    'z': float(z[idx]),
                    'cluster_id': cluster_id,
                })

        # Remove near-duplicate points
        if points:
            unique = [points[0]]
            for p in points[1:]:
                is_dup = False
                for u in unique:
                    if (abs(p['x'] - u['x']) < max_spacing * 0.1 and
                            abs(p['y'] - u['y']) < max_spacing * 0.1):
                        is_dup = True
                        break
                if not is_dup:
                    unique.append(p)
            points = unique

        return points


# ---------------------------------------------------------------------------
# Z Assignment
# ---------------------------------------------------------------------------

def assign_z_from_dtm(boundary_points, dtm_path):
    """Assign Z values from a DTM raster."""
    try:
        import rasterio
    except ImportError:
        QgsMessageLog.logMessage(
            "rasterio not available for DTM Z assignment",
            "TestPlugin", Qgis.Warning)
        return boundary_points

    try:
        with rasterio.open(dtm_path) as src:
            for pt in boundary_points:
                row, col = src.index(pt['x'], pt['y'])
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = src.read(1, window=rasterio.windows.Window(col, row, 1, 1))
                    pt['z'] = float(val[0, 0])
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Error reading DTM for Z assignment: {e}",
            "TestPlugin", Qgis.Warning)

    return boundary_points


def assign_z_from_ps_data(boundary_points, xy, z_values):
    """Assign Z by interpolating from PS point Z values."""
    if z_values is None or len(z_values) == 0:
        return boundary_points

    tree = KDTree(xy)
    for pt in boundary_points:
        _, idx = tree.query([pt['x'], pt['y']], k=3)
        if np.isscalar(idx):
            idx = [idx]
        pt['z'] = float(np.mean(z_values[idx]))

    return boundary_points


# ---------------------------------------------------------------------------
# Main Processor (orchestrator)
# ---------------------------------------------------------------------------

class PSInSARProcessor:
    """Orchestrate the full PS-InSAR moving area detection pipeline."""

    def __init__(self, ps_layer, velocity_field, output_path,
                 vel_threshold=2.0, k_neighbors=20, epochs=50,
                 refinement_iters=1, min_cluster_size=10, alpha=0.0,
                 z_source='constant', z_constant=0.0, dtm_path=None,
                 use_nn=True, crs=None):
        self.ps_layer = ps_layer
        self.velocity_field = velocity_field
        self.output_path = output_path
        self.vel_threshold = vel_threshold
        self.k_neighbors = k_neighbors
        self.epochs = epochs
        self.refinement_iters = refinement_iters
        self.min_cluster_size = min_cluster_size
        self.alpha = alpha
        self.z_source = z_source
        self.z_constant = z_constant
        self.dtm_path = dtm_path
        self.use_nn = use_nn
        self.crs = crs

    def run(self, progress_callback=None):
        """
        Execute the full pipeline.
        Returns (boundary_points, labels, xy) or raises on error.
        """
        # 1. Read data from layer
        if progress_callback:
            progress_callback(5, "Reading PS data...")

        xy, velocity, z_raw = self._read_layer_data()

        if len(xy) == 0:
            raise ValueError("No valid PS points found in the layer.")

        QgsMessageLog.logMessage(
            f"Read {len(xy)} PS points", "TestPlugin", Qgis.Info)

        # 2. Feature extraction
        if progress_callback:
            progress_callback(10, "Extracting features...")

        extractor = PSFeatureExtractor(xy, velocity, k_neighbors=self.k_neighbors)
        features = extractor.extract()

        # 3. Classification
        if progress_callback:
            progress_callback(15, "Classifying points...")

        def train_progress(pct):
            if progress_callback:
                # Map training progress (0-100) to overall (15-60)
                overall = 15 + int(pct * 0.45)
                progress_callback(overall, f"Training classifier... {pct}%")

        trainer = PSTrainer(
            velocity_threshold=self.vel_threshold,
            epochs=self.epochs,
            refinement_iters=self.refinement_iters,
            use_nn=self.use_nn,
        )
        labels = trainer.classify(features, velocity, progress_callback=train_progress)

        n_moving = int(labels.sum())
        QgsMessageLog.logMessage(
            f"Classification: {n_moving} moving, {len(labels) - n_moving} stable",
            "TestPlugin", Qgis.Info)

        # 4. Boundary extraction
        if progress_callback:
            progress_callback(65, "Extracting boundaries...")

        # Determine Z values for moving points
        if self.z_source == 'ps_data' and z_raw is not None:
            z_for_boundary = z_raw
        elif self.z_source == 'constant':
            z_for_boundary = np.full(len(xy), self.z_constant)
        else:
            z_for_boundary = None  # will assign from DTM later

        boundary_extractor = MovingAreaBoundaryExtractor(
            min_cluster_size=self.min_cluster_size,
            alpha=self.alpha,
        )

        if z_for_boundary is not None:
            boundary_points = boundary_extractor.extract_boundaries(
                xy, labels, z_values=z_for_boundary)
        else:
            # Temporary zero Z, will replace from DTM
            boundary_points = boundary_extractor.extract_boundaries(
                xy, labels, z_values=np.zeros(len(xy)))

        # 5. Z assignment from DTM if requested
        if self.z_source == 'dtm' and self.dtm_path:
            if progress_callback:
                progress_callback(75, "Assigning Z from DTM...")
            boundary_points = assign_z_from_dtm(boundary_points, self.dtm_path)
        elif self.z_source == 'ps_data' and z_raw is not None:
            boundary_points = assign_z_from_ps_data(boundary_points, xy, z_raw)

        QgsMessageLog.logMessage(
            f"Extracted {len(boundary_points)} boundary points",
            "TestPlugin", Qgis.Info)

        # 6. Write output shapefile
        if progress_callback:
            progress_callback(80, "Writing output shapefile...")

        if boundary_points:
            self._write_shapefile(boundary_points)

        # 7. Add classification layer to QGIS
        if progress_callback:
            progress_callback(90, "Adding classification layer...")

        self._add_classification_layer(xy, labels)

        if progress_callback:
            progress_callback(100, "Done!")

        return boundary_points, labels, xy

    def _read_layer_data(self):
        """Read XY, velocity, and optionally Z from the QGIS vector layer."""
        xy_list = []
        vel_list = []
        z_list = []

        has_z_field = False
        fields = self.ps_layer.fields()
        z_field_idx = fields.indexFromName('Z')
        if z_field_idx < 0:
            z_field_idx = fields.indexFromName('z')
        if z_field_idx < 0:
            z_field_idx = fields.indexFromName('height')
        has_z_field = z_field_idx >= 0

        vel_field_idx = fields.indexFromName(self.velocity_field)
        if vel_field_idx < 0:
            raise ValueError(f"Velocity field '{self.velocity_field}' not found in layer.")

        for feature in self.ps_layer.getFeatures():
            geom = feature.geometry()
            if geom is None or geom.isEmpty():
                continue

            point = geom.asPoint()
            x, y = point.x(), point.y()

            vel_val = feature[self.velocity_field]
            if vel_val is None:
                continue

            try:
                vel = float(vel_val)
            except (ValueError, TypeError):
                continue

            xy_list.append([x, y])
            vel_list.append(vel)

            if has_z_field:
                try:
                    z_list.append(float(feature[z_field_idx]))
                except (ValueError, TypeError):
                    z_list.append(0.0)

        xy = np.array(xy_list, dtype=np.float64)
        velocity = np.array(vel_list, dtype=np.float64)
        z_raw = np.array(z_list, dtype=np.float64) if z_list else None

        return xy, velocity, z_raw

    def _write_shapefile(self, boundary_points):
        """Write boundary points to a shapefile."""
        crs = self.crs
        if crs is None:
            crs = self.ps_layer.crs()

        fields_list = [
            QgsField('X', QVariant.Double),
            QgsField('Y', QVariant.Double),
            QgsField('Z', QVariant.Double),
            QgsField('formation', QVariant.String),
            QgsField('cluster_id', QVariant.Int),
        ]

        transform_context = QgsCoordinateTransformContext()
        save_options = QgsVectorFileWriter.SaveVectorOptions()
        save_options.driverName = "ESRI Shapefile"

        # Create writer
        writer = QgsVectorFileWriter.create(
            self.output_path,
            fields_list,
            QgsWkbTypes.Point,
            crs,
            transform_context,
            save_options,
        )

        if writer.hasError() != QgsVectorFileWriter.NoError:
            raise RuntimeError(f"Error creating shapefile: {writer.errorMessage()}")

        for pt in boundary_points:
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(
                QgsPointXY(pt['x'], pt['y'])))
            feat.setAttributes([
                pt['x'],
                pt['y'],
                pt['z'],
                'PS_moving_boundary',
                pt['cluster_id'],
            ])
            writer.addFeature(feat)

        del writer

        QgsMessageLog.logMessage(
            f"Shapefile written: {self.output_path}", "TestPlugin", Qgis.Info)

    def _add_classification_layer(self, xy, labels):
        """Add a colored classification layer to QGIS for verification."""
        # Create a memory layer
        crs = self.ps_layer.crs()
        uri = f"Point?crs={crs.authid()}&field=class:integer&field=velocity:double"
        mem_layer = QgsVectorLayer(uri, "PS_classification", "memory")

        if not mem_layer.isValid():
            QgsMessageLog.logMessage(
                "Failed to create classification memory layer",
                "TestPlugin", Qgis.Warning)
            return

        provider = mem_layer.dataProvider()
        features = []
        for i in range(len(xy)):
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(
                QgsPointXY(float(xy[i, 0]), float(xy[i, 1]))))
            feat.setAttributes([int(labels[i]), 0.0])
            features.append(feat)

        provider.addFeatures(features)

        # Categorized renderer: red = moving (1), green = stable (0)
        categories = []

        symbol_stable = QgsSymbol.defaultSymbol(QgsWkbTypes.PointGeometry)
        symbol_stable.setColor(QVariant('#00AA00'))  # noqa - used as color
        try:
            from qgis.PyQt.QtGui import QColor
            symbol_stable.setColor(QColor(0, 170, 0))
        except Exception:
            pass
        categories.append(QgsRendererCategory(0, symbol_stable, "Stable"))

        symbol_moving = QgsSymbol.defaultSymbol(QgsWkbTypes.PointGeometry)
        try:
            from qgis.PyQt.QtGui import QColor
            symbol_moving.setColor(QColor(220, 0, 0))
        except Exception:
            pass
        categories.append(QgsRendererCategory(1, symbol_moving, "Moving"))

        renderer = QgsCategorizedSymbolRenderer('class', categories)
        mem_layer.setRenderer(renderer)

        QgsProject.instance().addMapLayer(mem_layer)
        QgsMessageLog.logMessage(
            "Added PS classification layer to QGIS", "TestPlugin", Qgis.Info)


# ---------------------------------------------------------------------------
# Surface Evolution Processor (multi-orbit decomposition + PDE)
# ---------------------------------------------------------------------------

class SurfaceEvolutionProcessor:
    """
    Orchestrate InSAR LOS decomposition and surface evolution.

    Reads multi-orbit PS-InSAR CSV files, decomposes LOS velocities into
    slope-parallel/vertical/normal components, then solves the mass
    conservation PDE to compute surface thickness evolution over time.

    Based on: ottaviani2/InSAR-Decomposition-and-Landslide-movement-estimation
    """

    def __init__(self, csv_paths, dtm_path, output_path,
                 slope_azimuth_deg=270.0, slope_angle_deg=50.0,
                 friction_coeff=0.01, grid_nx=100, grid_ny=100,
                 max_distance=100.0, date_col_offset=28,
                 smoothing_sigma=1.5, crs=None):
        self.csv_paths = csv_paths
        self.dtm_path = dtm_path
        self.output_path = output_path
        self.slope_azimuth_deg = slope_azimuth_deg
        self.slope_angle_deg = slope_angle_deg
        self.friction_coeff = friction_coeff
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.max_distance = max_distance
        self.date_col_offset = date_col_offset
        self.smoothing_sigma = smoothing_sigma
        self.crs = crs

    def run(self, progress_callback=None):
        """
        Execute the full decomposition + surface evolution pipeline.

        Returns
        -------
        results : dict
            Contains h_initial, h_final, thickness_change, moving_area_km2,
            thickness estimates, and grid coordinates.
        """
        from .ps_insar_decomposition import InSARDecompositionProcessor

        if not HAS_PANDAS:
            raise ImportError("pandas is required for CSV-based InSAR processing")

        processor = InSARDecompositionProcessor(
            csv_paths=self.csv_paths,
            dtm_path=self.dtm_path,
            slope_azimuth_deg=self.slope_azimuth_deg,
            slope_angle_deg=self.slope_angle_deg,
            friction_coeff=self.friction_coeff,
            grid_nx=self.grid_nx,
            grid_ny=self.grid_ny,
            max_distance=self.max_distance,
            date_col_offset=self.date_col_offset,
            smoothing_sigma=self.smoothing_sigma,
        )

        results = processor.run(progress_callback=progress_callback)

        # Write output CSV (compatible with RS3 / GemPy import)
        if self.output_path:
            self._write_output_csv(results)

        # Add thickness change layer to QGIS
        self._add_thickness_layer(results)

        return results

    def _write_output_csv(self, results):
        """Write decomposition results to CSV."""
        import pandas as pd

        x_grid = results['x']
        y_grid = results['y']
        h_final = results['h_final']
        h_change = results['thickness_change']

        X, Y = np.meshgrid(x_grid, y_grid)
        Y_flipped = np.flipud(Y)

        x_flat = X.ravel()
        y_flat = Y_flipped.ravel()
        z_flat = h_final.ravel()
        h_disp_flat = h_change.ravel()

        # Filter out invalid points
        valid = np.isfinite(z_flat) & np.isfinite(h_disp_flat) & (z_flat > 0)
        ids = np.arange(1, len(x_flat) + 1)

        df = pd.DataFrame({
            'ID': ids[valid],
            'X': x_flat[valid],
            'Y': y_flat[valid],
            'Z': z_flat[valid],
            'H_disp': h_disp_flat[valid],
        })
        df.to_csv(self.output_path, sep=';', index=False)

        QgsMessageLog.logMessage(
            f"Surface evolution CSV written: {self.output_path} "
            f"({valid.sum()} points)",
            "TestPlugin", Qgis.Info)

    def _add_thickness_layer(self, results):
        """Add a thickness change point layer to QGIS."""
        x_grid = results['x']
        y_grid = results['y']
        h_change = results['thickness_change']
        h_final = results['h_final']

        X, Y = np.meshgrid(x_grid, y_grid)
        Y_flipped = np.flipud(Y)

        crs_str = "EPSG:4326"
        if self.crs:
            crs_str = self.crs.authid() if hasattr(self.crs, 'authid') else str(self.crs)

        uri = (f"Point?crs={crs_str}"
               f"&field=Z:double&field=H_disp:double")
        mem_layer = QgsVectorLayer(uri, "Surface_Evolution", "memory")

        if not mem_layer.isValid():
            QgsMessageLog.logMessage(
                "Failed to create surface evolution layer",
                "TestPlugin", Qgis.Warning)
            return

        provider = mem_layer.dataProvider()
        features = []

        for i in range(h_change.shape[0]):
            for j in range(h_change.shape[1]):
                z_val = h_final[i, j]
                h_val = h_change[i, j]
                if not np.isfinite(z_val) or z_val <= 0:
                    continue

                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPointXY(
                    QgsPointXY(float(X[i, j]), float(Y_flipped[i, j]))))
                feat.setAttributes([float(z_val), float(h_val)])
                features.append(feat)

        provider.addFeatures(features)

        # Graduated renderer by thickness change
        try:
            from qgis.core import (
                QgsGraduatedSymbolRenderer, QgsRendererRange,
                QgsSymbol, QgsClassificationEqualInterval
            )
            from qgis.PyQt.QtGui import QColor

            symbol_neg = QgsSymbol.defaultSymbol(QgsWkbTypes.PointGeometry)
            symbol_neg.setColor(QColor(255, 0, 0))

            symbol_pos = QgsSymbol.defaultSymbol(QgsWkbTypes.PointGeometry)
            symbol_pos.setColor(QColor(0, 0, 255))

            h_min = float(np.nanmin(h_change[h_change != 0])) if np.any(h_change != 0) else -1
            h_max = float(np.nanmax(h_change[h_change != 0])) if np.any(h_change != 0) else 1

            ranges = [
                QgsRendererRange(h_min, 0, symbol_neg, f"Thinning ({h_min:.2f} to 0)"),
                QgsRendererRange(0, h_max, symbol_pos, f"Thickening (0 to {h_max:.2f})"),
            ]
            renderer = QgsGraduatedSymbolRenderer('H_disp', ranges)
            mem_layer.setRenderer(renderer)
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Could not set graduated renderer: {e}",
                "TestPlugin", Qgis.Warning)

        QgsProject.instance().addMapLayer(mem_layer)
        QgsMessageLog.logMessage(
            "Added surface evolution layer to QGIS", "TestPlugin", Qgis.Info)
