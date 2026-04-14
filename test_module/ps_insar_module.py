# -*- coding: utf-8 -*-
"""
PS-InSAR Moving Area Detection Module
Detects moving areas from PS-InSAR point data using neural network
classification and geometric boundary extraction.
"""

import os
import math
import csv
import numpy as np
from scipy.spatial import KDTree, Delaunay

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsField, QgsFields, QgsFeature,
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

        fields_list = QgsFields()
        for f in [
            QgsField('X', QVariant.Double),
            QgsField('Y', QVariant.Double),
            QgsField('Z', QVariant.Double),
            QgsField('formation', QVariant.String),
            QgsField('cluster_id', QVariant.Int),
        ]:
            fields_list.append(f)

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
                 smoothing_sigma=1.5, depth_formula='guzzetti_2009',
                 include_vertical_velocity=False,
                 movement_threshold_mmday=0.1,
                 crs=None):
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
        self.depth_formula = depth_formula
        self.include_vertical_velocity = include_vertical_velocity
        self.movement_threshold_mmday = movement_threshold_mmday
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
            depth_formula=self.depth_formula,
            include_vertical_velocity=self.include_vertical_velocity,
            movement_threshold_mmday=self.movement_threshold_mmday,
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


# ---------------------------------------------------------------------------
# GemPy CSV Exporter
# ---------------------------------------------------------------------------

class GempyExporter:
    """
    Export PS-InSAR boundary points and computed sliding surface points
    to GemPy-compatible CSV files.

    Two files are written:
      {prefix}_surface_points.csv  — columns: X,Y,Z,formation
      {prefix}_orientations.csv    — columns: X,Y,Z,formation,azimuth,dip,polarity

    For each cluster the algorithm:
      1. Computes the centroid (cx, cy) and the elevation range (z_min … z_max).
      2. Estimates the sliding-surface depth D from the cluster's convex-hull area
         using the chosen empirical formula.
      3. For every boundary point (x, y, z):
           - azimuth = atan2(cx-x, cy-y) in degrees [0-360, clockwise from North]
             → always faces INWARD (toward centroid)
           - dip     = t * 90°  where  t = (z - z_min) / z_range  clamped [0,1]
             → ≈0° at the toe (low elevation), ≈90° at the head scarp (high elevation)
           - sliding surface point projected D metres inward + downward:
               x_deep = x + D·sin(dip)·sin(az)
               y_deep = y + D·sin(dip)·cos(az)
               z_deep = z - D·cos(dip)
    """

    FORMATION_SURFACE = "landslide_boundary"
    FORMATION_DEEP    = "sliding_surface"

    def __init__(self, boundary_points, depth_formula_key='guzzetti_2009'):
        """
        Parameters
        ----------
        boundary_points : list of dict
            Each dict must contain keys: x, y, z, cluster_id.
        depth_formula_key : str
            Key in DEPTH_FORMULAS (ps_insar_decomposition).
        """
        self.boundary_points   = boundary_points
        self.depth_formula_key = depth_formula_key

    # ------------------------------------------------------------------
    def export(self, output_prefix):
        """
        Write the two CSVs and return (surface_points_path, orientations_path).
        """
        from .ps_insar_decomposition import estimate_depth_from_area

        if not self.boundary_points:
            raise ValueError("No boundary points to export.")

        # Group points by cluster
        clusters = {}
        for pt in self.boundary_points:
            clusters.setdefault(pt['cluster_id'], []).append(pt)

        surface_rows     = []   # X, Y, Z, formation
        orientation_rows = []   # X, Y, Z, formation, azimuth, dip, polarity

        for cid, pts in clusters.items():
            xs = [p['x'] for p in pts]
            ys = [p['y'] for p in pts]
            zs = [p['z'] for p in pts]

            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            z_min   = min(zs)
            z_max   = max(zs)
            z_range = (z_max - z_min) if z_max != z_min else 1.0

            area_m2 = self._convex_hull_area(xs, ys)
            depth, _formula_info = estimate_depth_from_area(
                area_m2, self.depth_formula_key)

            f_surface = f"{self.FORMATION_SURFACE}_{cid}"
            f_deep    = f"{self.FORMATION_DEEP}_{cid}"

            for pt in pts:
                x, y, z = pt['x'], pt['y'], pt['z']

                # --- inward azimuth (clockwise from North) ---
                dx, dy = cx - x, cy - y
                if dx == 0.0 and dy == 0.0:
                    az_deg = 0.0
                else:
                    az_deg = math.degrees(math.atan2(dx, dy)) % 360.0

                # --- dip: 0° at toe, 90° at head scarp ---
                t       = max(0.0, min(1.0, (z - z_min) / z_range))
                dip_deg = t * 90.0

                # surface boundary point
                surface_rows.append(
                    {'X': x, 'Y': y, 'Z': z, 'formation': f_surface})

                # orientation at boundary point
                orientation_rows.append({
                    'X': x, 'Y': y, 'Z': z,
                    'formation': f_surface,
                    'azimuth': az_deg, 'dip': dip_deg, 'polarity': 1,
                })

                # sliding-surface point
                az_rad  = math.radians(az_deg)
                dip_rad = math.radians(dip_deg)
                x_deep  = x + depth * math.sin(dip_rad) * math.sin(az_rad)
                y_deep  = y + depth * math.sin(dip_rad) * math.cos(az_rad)
                z_deep  = z - depth * math.cos(dip_rad)

                surface_rows.append(
                    {'X': x_deep, 'Y': y_deep, 'Z': z_deep,
                     'formation': f_deep})

                # orientation at deep point (same attitude as at surface boundary)
                orientation_rows.append({
                    'X': x_deep, 'Y': y_deep, 'Z': z_deep,
                    'formation': f_deep,
                    'azimuth': az_deg, 'dip': dip_deg, 'polarity': 1,
                })

        sp_path = output_prefix + '_surface_points.csv'
        or_path = output_prefix + '_orientations.csv'

        self._write_csv(sp_path, surface_rows,
                        ['X', 'Y', 'Z', 'formation'])
        self._write_csv(or_path, orientation_rows,
                        ['X', 'Y', 'Z', 'formation', 'azimuth', 'dip', 'polarity'])

        QgsMessageLog.logMessage(
            f"GemPy surface points: {sp_path} ({len(surface_rows)} rows)\n"
            f"GemPy orientations:   {or_path} ({len(orientation_rows)} rows)",
            "TestPlugin", Qgis.Info)

        return sp_path, or_path

    # ------------------------------------------------------------------
    @staticmethod
    def _convex_hull_area(xs, ys):
        """Return the 2-D convex-hull area in m² for the given point set.

        Parameters
        ----------
        xs, ys : sequences of float
            Coordinates in a **projected** (metric) CRS such as UTM or
            a national grid.  If the layer CRS is geographic (decimal
            degrees), the returned value will be in degrees² — which is
            meaningless as m² and will produce a completely wrong depth
            estimate.  Always reproject to a metric CRS before calling.
        """
        # Sanity check: if coordinate magnitudes look like decimal degrees
        # (|x| ≤ 180, |y| ≤ 90) warn but do not abort — the caller is
        # responsible for reprojection.
        x_max = max(abs(v) for v in xs)
        y_max = max(abs(v) for v in ys)
        if x_max <= 180.0 and y_max <= 90.0:
            QgsMessageLog.logMessage(
                "GempyExporter._convex_hull_area: coordinates appear to be in "
                "geographic (degree) units.  Area will be in degrees², NOT m².  "
                "Depth estimates will be wrong.  Reproject the boundary layer to "
                "a projected metric CRS before exporting.",
                "TestPlugin", Qgis.Warning)

        if len(xs) < 3:
            return 0.0
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(np.column_stack([xs, ys]))
            return float(hull.volume)   # hull.volume == area in 2-D (scipy convention)
        except Exception:
            # Fallback: half the bounding-box area
            return (max(xs) - min(xs)) * (max(ys) - min(ys)) * 0.5

    @staticmethod
    def _write_csv(path, rows, fieldnames):
        """Write a list of dicts to a CSV file with the given column order."""
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in row.items()
                })


# ---------------------------------------------------------------------------
# 3-Geometry LOS Decomposition
# ---------------------------------------------------------------------------

def _sym(color):
    """Return a default point symbol with the given QColor."""
    from qgis.core import QgsSymbol, QgsWkbTypes
    s = QgsSymbol.defaultSymbol(QgsWkbTypes.PointGeometry)
    s.setColor(color)
    return s


class LOSDecompositionProcessor:
    """
    Decompose 3-geometry InSAR LOS velocities into East-West, Up-Down
    (and optionally North-South) displacement rate components.

    Look-vector convention (right-looking SAR):
        look_azimuth = satellite_heading + 90°   (clockwise from North)
        eE =  sin(θ) · sin(look_az)
        eN =  sin(θ) · cos(look_az)
        eU =  cos(θ)

    where θ is the incidence angle measured from the vertical (nadir).
    Satellite heading H is the flight azimuth clockwise from North
    (ascending Sentinel-1 ≈ 350°, descending ≈ 190°).
    For right-looking geometry the radar looks 90° to the right of the
    flight direction, hence look_az = heading + 90°.

    Reference:
        Hanssen, R.F. (2001): Radar Interferometry — Data Interpretation and
        Error Analysis. Kluwer Academic Publishers, Dordrecht.
        Rosen, P.A., et al. (2000): Synthetic aperture radar interferometry.
        Proc. IEEE, 88(3), 333–382. doi:10.1109/5.838084

    The LOS measurement is modelled as:
        d_LOS_i = eE_i · dE + eN_i · dN + eU_i · dU

    With 3 geometries and 3 unknowns the system is exactly determined;
    with solve_north=False only E and U are solved (overdetermined, solved
    in least-squares sense via normal equations).
    """

    def __init__(self, layer_configs, grid_nx=100, grid_ny=100,
                 max_distance=1000.0, smoothing_sigma=1.5,
                 solve_north=True, output_path=None,
                 cond_threshold=1e4,
                 # Moving area detection
                 ma_enabled=False,
                 ma_component='magnitude',
                 ma_vel_threshold=2.0,
                 ma_k_neighbors=20,
                 ma_epochs=50,
                 ma_refinement_iters=1,
                 ma_min_cluster_size=10,
                 ma_alpha=0.0,
                 ma_use_nn=True,
                 ma_output_path=None):
        """
        Parameters
        ----------
        layer_configs : list of exactly 3 dicts, each containing:
            'layer'     : QgsVectorLayer  (point layer)
            'vel_field' : str             (LOS velocity attribute name)
            'inc_field' : str or None     (incidence-angle field; None → inc_const)
            'inc_const' : float           (constant incidence angle in degrees)
            'hdg_field' : str or None     (satellite heading field; None → hdg_const)
            'hdg_const' : float           (constant satellite heading, degrees from N)
        grid_nx, grid_ny : int
            Output grid dimensions.
        max_distance : float
            Maximum distance from the nearest input point for a grid cell
            to be considered valid (same units as the layer CRS).
        smoothing_sigma : float
            Gaussian smoothing applied to the interpolated grids (pixels).
        solve_north : bool
            True  → solve for 3 unknowns (dE, dN, dU) — exact 3×3 system.
            False → solve for 2 unknowns (dE, dU), assuming dN ≈ 0 —
                    overdetermined 3×2 system solved by least squares.
        output_path : str or None
            If given, decomposition results are written to this CSV file.
        cond_threshold : float
            Maximum acceptable condition number of the look-vector matrix A.
            Grid cells whose condition number exceeds this value are considered
            geometrically degenerate (two or more geometries nearly parallel)
            and their solution is set to NaN.  A diagnostic layer named
            'LOS_condition_number' is added to QGIS so the user can inspect
            the spatial pattern of ill-conditioning.  Typical values:
              <  100  : excellent geometry
              < 1 000 : good
              < 1e4   : acceptable (default)
              > 1e4   : poor — treat results with caution
        ma_enabled : bool
            If True, run moving area detection on the decomposed data after
            the LOS inversion.
        ma_component : str
            Which decomposed component to use as the "velocity" signal for
            moving area detection:
              'magnitude' – √(dE² + dU²)  [default, total displacement rate]
              'ud'        – vertical (Up-Down) component dU
              'ew'        – horizontal East-West component dE
              'ns'        – North-South component dN (requires solve_north=True)
        ma_vel_threshold : float
            Velocity magnitude threshold (same units as the layer velocity)
            used to generate initial pseudo-labels for the classifier.
        ma_k_neighbors, ma_epochs, ma_refinement_iters : int
            PSTrainer hyper-parameters.
        ma_min_cluster_size : int
            Minimum number of points to form a moving cluster.
        ma_alpha : float
            Alpha-shape parameter for boundary extraction (0 = auto).
        ma_use_nn : bool
            Whether to use the neural-network refinement step (requires PyTorch).
        ma_output_path : str or None
            If given, the moving area boundary is written to this shapefile path.
        """
        if len(layer_configs) != 3:
            raise ValueError("Exactly 3 layer configurations are required.")
        self.layer_configs = layer_configs
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.max_distance = max_distance
        self.smoothing_sigma = smoothing_sigma
        self.solve_north = solve_north
        self.output_path = output_path
        self.cond_threshold = float(cond_threshold)
        self.ma_enabled = ma_enabled
        self.ma_component = ma_component
        self.ma_vel_threshold = ma_vel_threshold
        self.ma_k_neighbors = ma_k_neighbors
        self.ma_epochs = ma_epochs
        self.ma_refinement_iters = ma_refinement_iters
        self.ma_min_cluster_size = ma_min_cluster_size
        self.ma_alpha = ma_alpha
        self.ma_use_nn = ma_use_nn
        self.ma_output_path = ma_output_path

    # ------------------------------------------------------------------
    def run(self, progress_callback=None):
        """
        Execute the decomposition.

        Returns
        -------
        dict with keys: dE, dN, dU (2-D ndarrays), X, Y (meshgrid), x, y (1-D).
        """
        def _cb(pct, msg=''):
            if progress_callback:
                progress_callback(pct, msg)

        _cb(5, "Reading layer data…")
        datasets = []
        for i, cfg in enumerate(self.layer_configs):
            datasets.append(self._read_layer(cfg))
            _cb(5 + 10 * (i + 1), f"Read geometry {i + 1}/3")

        # Build common regular grid from the union of all point extents
        all_x = np.concatenate([d['x'] for d in datasets])
        all_y = np.concatenate([d['y'] for d in datasets])
        x = np.linspace(all_x.min(), all_x.max(), self.grid_nx)
        y = np.linspace(all_y.min(), all_y.max(), self.grid_ny)
        X, Y = np.meshgrid(x, y)

        _cb(40, "Interpolating LOS data to grid…")
        grids = []
        for i, ds in enumerate(datasets):
            grids.append(self._interpolate(ds, X, Y))
            _cb(40 + 10 * (i + 1), f"Interpolated geometry {i + 1}/3")

        _cb(75, "Inverting LOS system…")
        dE, dN, dU, cond_grid = self._invert(grids)

        results = dict(dE=dE, dN=dN, dU=dU, X=X, Y=Y, x=x, y=y,
                       cond_grid=cond_grid)

        _cb(85, "Adding layers to QGIS…")
        self._add_to_qgis(results)

        if self.output_path:
            self._write_output_csv(results)

        if self.ma_enabled:
            _cb(90, "Running moving area detection on decomposed data…")
            try:
                self._run_moving_area(results, _cb)
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Moving area detection failed: {e}",
                    'TestPlugin', Qgis.Warning)
                import traceback
                QgsMessageLog.logMessage(
                    traceback.format_exc(), 'TestPlugin', Qgis.Warning)

        _cb(100, "Done!")
        return results

    # ------------------------------------------------------------------
    def _read_layer(self, cfg):
        """Extract (x, y, los, inc, hdg) arrays from a QGIS vector layer."""
        layer     = cfg['layer']
        vel_field = cfg['vel_field']
        inc_field = cfg.get('inc_field')
        inc_const = float(cfg.get('inc_const', 38.0))
        hdg_field = cfg.get('hdg_field')
        hdg_const = float(cfg.get('hdg_const', 350.0))

        fields    = layer.fields()
        vel_idx   = fields.indexFromName(vel_field)
        inc_idx   = fields.indexFromName(inc_field) if inc_field else -1
        hdg_idx   = fields.indexFromName(hdg_field) if hdg_field else -1

        if vel_idx < 0:
            raise ValueError(
                f"LOS velocity field '{vel_field}' not found in "
                f"layer '{layer.name()}'.")

        xs, ys, los_vals, inc_vals, hdg_vals = [], [], [], [], []

        for feat in layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            try:
                v = float(feat[vel_field])
            except (ValueError, TypeError):
                continue

            pt = geom.asPoint()
            xs.append(pt.x())
            ys.append(pt.y())
            los_vals.append(v)
            inc_vals.append(float(feat[inc_field]) if inc_idx >= 0 else inc_const)
            hdg_vals.append(float(feat[hdg_field]) if hdg_idx >= 0 else hdg_const)

        if not xs:
            raise ValueError(
                f"No valid features found in layer '{layer.name()}'.")

        return dict(
            x=np.array(xs),
            y=np.array(ys),
            los=np.array(los_vals),
            inc=np.array(inc_vals),
            hdg=np.array(hdg_vals),
        )

    # ------------------------------------------------------------------
    def _interpolate(self, ds, X, Y):
        """
        Interpolate LOS / incidence / heading onto the regular grid and
        compute the unit look-vector components (eE, eN, eU).
        """
        from scipy.interpolate import griddata, NearestNDInterpolator
        from scipy.ndimage import gaussian_filter
        from scipy.spatial import KDTree

        pts = np.column_stack([ds['x'], ds['y']])

        def _interp(vals):
            g = griddata(pts, vals, (X, Y), method='linear')
            nm = np.isnan(g)
            if nm.any():
                nn = NearestNDInterpolator(pts, vals)
                g[nm] = nn(X[nm], Y[nm])
            return g

        los_g = _interp(ds['los'])
        inc_g = _interp(ds['inc'])
        hdg_g = _interp(ds['hdg'])

        if self.smoothing_sigma > 0:
            s = self.smoothing_sigma
            los_g = gaussian_filter(los_g, sigma=s, mode='nearest')
            inc_g = gaussian_filter(inc_g, sigma=s, mode='nearest')
            hdg_g = gaussian_filter(hdg_g, sigma=s, mode='nearest')

        # Apply distance-based validity mask
        if self.max_distance > 0:
            tree = KDTree(pts)
            dists, _ = tree.query(
                np.column_stack([X.ravel(), Y.ravel()]), k=1)
            dist_g = dists.reshape(X.shape)
            los_g[dist_g > self.max_distance] = np.nan

        # Look-vector components (right-looking convention):
        #   look_azimuth = heading + 90°  (radar looks right of flight path)
        look_az = np.deg2rad(hdg_g + 90.0)
        theta   = np.deg2rad(inc_g)
        eE = np.sin(theta) * np.sin(look_az)
        eN = np.sin(theta) * np.cos(look_az)
        eU = np.cos(theta)

        return dict(eE=eE, eN=eN, eU=eU, los=los_g)

    # ------------------------------------------------------------------
    def _invert(self, grids):
        """
        Solve the LOS decomposition at every grid cell.

        Uses vectorised batched solve for speed with a per-pixel lstsq
        fallback for exactly-singular cells.  After solving, cells whose
        look-vector matrix condition number exceeds ``self.cond_threshold``
        are set to NaN — their geometry is too degenerate for a reliable
        solution.

        Returns
        -------
        dE, dN, dU : 2-D ndarrays  (NaN where masked or unresolvable)
        cond_grid  : 2-D ndarray   condition numbers (NaN outside valid mask)
        """
        if self.solve_north:
            # Exactly-determined 3×3 system: A x = b
            # A : (ny, nx, 3, 3)   b : (ny, nx, 3)
            A = np.stack(
                [np.stack([g['eE'], g['eN'], g['eU']], axis=-1)
                 for g in grids],
                axis=-2)
            b = np.stack([g['los'] for g in grids], axis=-1)

            valid = (np.all(np.isfinite(A), axis=(-2, -1))
                     & np.all(np.isfinite(b), axis=-1))

            sol = np.full(A.shape[:-2] + (3,), np.nan)
            cond_grid = np.full(A.shape[:-2], np.nan)

            if valid.any():
                # Compute condition numbers (vectorised SVD)
                cond_vals = np.linalg.cond(A[valid])          # (n_valid,)
                cond_grid[valid] = cond_vals

                # Separate well-conditioned and ill-conditioned cells
                well = valid.copy()
                well[valid] = cond_vals <= self.cond_threshold
                ill  = valid.copy()
                ill[valid]  = cond_vals >  self.cond_threshold

                n_ill = int(ill.sum())
                if n_ill > 0:
                    pct = n_ill / int(valid.sum()) * 100
                    QgsMessageLog.logMessage(
                        f"LOS inversion: {n_ill} cells ({pct:.1f}%) masked "
                        f"— condition number > {self.cond_threshold:.0e} "
                        f"(near-parallel geometries).",
                        'TestPlugin', Qgis.Warning)

                if well.any():
                    try:
                        sol[well] = np.linalg.solve(A[well], b[well])
                    except np.linalg.LinAlgError:
                        pass
                    # Per-pixel lstsq fallback for exactly-singular cells
                    bad = well & ~np.all(np.isfinite(sol), axis=-1)
                    for idx in zip(*np.where(bad)):
                        try:
                            s, _, _, _ = np.linalg.lstsq(
                                A[idx], b[idx], rcond=None)
                            sol[idx] = s
                        except Exception:
                            pass

            dE = sol[..., 0]
            dN = sol[..., 1]
            dU = sol[..., 2]

        else:
            # Overdetermined 3×2 system (dN assumed = 0)
            # A : (ny, nx, 3, 2)   b : (ny, nx, 3)
            A = np.stack(
                [np.stack([g['eE'], g['eU']], axis=-1) for g in grids],
                axis=-2)
            b = np.stack([g['los'] for g in grids], axis=-1)

            valid = (np.all(np.isfinite(A), axis=(-2, -1))
                     & np.all(np.isfinite(b), axis=-1))

            # Condition number of the rectangular A (via SVD)
            cond_grid = np.full(A.shape[:-2], np.nan)

            # Normal equations: (A^T A) x = A^T b
            ATA = np.einsum('...ij,...ik->...jk', A, A)   # (..., 2, 2)
            ATb = np.einsum('...ij,...i->...j',   A, b)   # (..., 2)

            sol = np.full(A.shape[:-2] + (2,), np.nan)

            if valid.any():
                cond_vals = np.linalg.cond(A[valid])
                cond_grid[valid] = cond_vals

                well = valid.copy()
                well[valid] = cond_vals <= self.cond_threshold
                ill  = valid.copy()
                ill[valid]  = cond_vals >  self.cond_threshold

                n_ill = int(ill.sum())
                if n_ill > 0:
                    pct = n_ill / int(valid.sum()) * 100
                    QgsMessageLog.logMessage(
                        f"LOS inversion: {n_ill} cells ({pct:.1f}%) masked "
                        f"— condition number > {self.cond_threshold:.0e} "
                        f"(near-parallel geometries).",
                        'TestPlugin', Qgis.Warning)

                if well.any():
                    try:
                        sol[well] = np.linalg.solve(
                            ATA[well], ATb[well])
                    except np.linalg.LinAlgError:
                        pass
                    bad = well & ~np.all(np.isfinite(sol), axis=-1)
                    for idx in zip(*np.where(bad)):
                        try:
                            s, _, _, _ = np.linalg.lstsq(
                                A[idx], b[idx], rcond=None)
                            sol[idx] = s
                        except Exception:
                            pass

            dE = sol[..., 0]
            dU = sol[..., 1]
            dN = np.full(dE.shape, np.nan)

        return dE, dN, dU, cond_grid

    # ------------------------------------------------------------------
    def _add_to_qgis(self, results):
        """Add E-W, U-D (and optionally N-S) grids as QGIS point memory layers."""
        X, Y   = results['X'], results['Y']
        dE, dN, dU = results['dE'], results['dN'], results['dU']

        crs     = self.layer_configs[0]['layer'].crs()
        crs_str = crs.authid() if (crs and crs.isValid()) else 'EPSG:4326'

        outputs = [('EW_velocity_mm_yr', dE), ('UD_velocity_mm_yr', dU)]
        if self.solve_north:
            outputs.append(('NS_velocity_mm_yr', dN))

        for layer_name, grid in outputs:
            if np.all(~np.isfinite(grid)):
                continue

            uri = f"Point?crs={crs_str}&field=velocity:double"
            lyr = QgsVectorLayer(uri, layer_name, 'memory')
            provider = lyr.dataProvider()

            feats = []
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    v = float(grid[i, j])
                    if not np.isfinite(v):
                        continue
                    feat = QgsFeature()
                    feat.setGeometry(QgsGeometry.fromPointXY(
                        QgsPointXY(float(X[i, j]), float(Y[i, j]))))
                    feat.setAttributes([v])
                    feats.append(feat)

            provider.addFeatures(feats)
            self._apply_diverging_renderer(lyr)
            QgsProject.instance().addMapLayer(lyr)
            QgsMessageLog.logMessage(
                f"Added '{layer_name}' ({len(feats)} pts)",
                'TestPlugin', Qgis.Info)

        # Always add the condition-number diagnostic layer
        self._add_cond_layer(results)

    # ------------------------------------------------------------------
    def _add_cond_layer(self, results):
        """
        Add a 'LOS_condition_number' point layer coloured from green (well-
        conditioned) through yellow to red (ill-conditioned / masked).
        Points above ``self.cond_threshold`` are included but shown in red
        so the user can see exactly where the geometry is degenerate.
        """
        X, Y      = results['X'], results['Y']
        cond_grid = results.get('cond_grid')
        if cond_grid is None or np.all(~np.isfinite(cond_grid)):
            return

        crs     = self.layer_configs[0]['layer'].crs()
        crs_str = crs.authid() if (crs and crs.isValid()) else 'EPSG:4326'

        uri = (f"Point?crs={crs_str}"
               f"&field=cond:double&field=masked:integer")
        lyr = QgsVectorLayer(uri, "LOS_condition_number", "memory")
        if not lyr.isValid():
            return

        provider = lyr.dataProvider()
        feats = []
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                c = float(cond_grid[i, j])
                if not np.isfinite(c):
                    continue
                masked = int(c > self.cond_threshold)
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPointXY(
                    QgsPointXY(float(X[i, j]), float(Y[i, j]))))
                feat.setAttributes([c, masked])
                feats.append(feat)

        provider.addFeatures(feats)

        # Graduated renderer: green → yellow → red over [1, cond_threshold * 10]
        try:
            from qgis.core import QgsGraduatedSymbolRenderer, QgsRendererRange
            from qgis.PyQt.QtGui import QColor
            thr = self.cond_threshold
            ranges = [
                QgsRendererRange(1,       thr * 0.1,
                                 _sym(QColor(0, 200, 0)),   f'< {thr*0.1:.0e}'),
                QgsRendererRange(thr * 0.1, thr,
                                 _sym(QColor(255, 200, 0)), f'{thr*0.1:.0e} – {thr:.0e}'),
                QgsRendererRange(thr,     thr * 10,
                                 _sym(QColor(220, 20, 20)), f'> {thr:.0e}  (masked)'),
            ]
            lyr.setRenderer(QgsGraduatedSymbolRenderer('cond', ranges))
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Cond renderer error: {e}", 'TestPlugin', Qgis.Warning)

        QgsProject.instance().addMapLayer(lyr)
        QgsMessageLog.logMessage(
            f"Added 'LOS_condition_number' ({len(feats)} pts, "
            f"threshold {self.cond_threshold:.0e}).",
            'TestPlugin', Qgis.Info)

    @staticmethod
    def _apply_diverging_renderer(layer):
        """Blue–white–red diverging renderer centred on zero."""
        try:
            from qgis.core import QgsGraduatedSymbolRenderer, QgsRendererRange
            from qgis.PyQt.QtGui import QColor

            vals = [f['velocity'] for f in layer.getFeatures()
                    if f['velocity'] is not None]
            if not vals:
                return
            v_min, v_max = min(vals), max(vals)
            mid = 0.0
            sym_neg = QgsSymbol.defaultSymbol(QgsWkbTypes.PointGeometry)
            sym_neg.setColor(QColor(220, 20, 20))
            sym_pos = QgsSymbol.defaultSymbol(QgsWkbTypes.PointGeometry)
            sym_pos.setColor(QColor(20, 80, 220))
            sym_zero = QgsSymbol.defaultSymbol(QgsWkbTypes.PointGeometry)
            sym_zero.setColor(QColor(200, 200, 200))

            ranges = []
            if v_min < mid:
                ranges.append(QgsRendererRange(
                    v_min, mid, sym_neg, f'{v_min:.2f} – 0'))
            ranges.append(QgsRendererRange(
                max(mid, v_min), v_max, sym_pos, f'0 – {v_max:.2f}'))

            layer.setRenderer(
                QgsGraduatedSymbolRenderer('velocity', ranges))
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Renderer error: {e}", 'TestPlugin', Qgis.Warning)

    def _write_output_csv(self, results):
        """Write the decomposed grids to a CSV file."""
        X, Y   = results['X'], results['Y']
        dE, dN, dU = results['dE'], results['dN'], results['dU']

        cols = ['X', 'Y', 'dE_mm_yr', 'dU_mm_yr']
        if self.solve_north:
            cols.append('dN_mm_yr')

        with open(self.output_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    de, du = float(dE[i, j]), float(dU[i, j])
                    if not (np.isfinite(de) and np.isfinite(du)):
                        continue
                    row = {
                        'X': round(float(X[i, j]), 2),
                        'Y': round(float(Y[i, j]), 2),
                        'dE_mm_yr': round(de, 4),
                        'dU_mm_yr': round(du, 4),
                    }
                    if self.solve_north:
                        row['dN_mm_yr'] = round(float(dN[i, j]), 4)
                    writer.writerow(row)

        QgsMessageLog.logMessage(
            f"Decomposition CSV written: {self.output_path}",
            'TestPlugin', Qgis.Info)

    # ------------------------------------------------------------------
    def _run_moving_area(self, results, progress_callback=None):
        """
        Detect moving areas on the decomposed displacement data.

        Selects the grid component specified by ``self.ma_component``
        ('ew', 'ud', 'ns', or 'magnitude'), flattens valid grid cells
        to a point cloud, runs PSFeatureExtractor → PSTrainer →
        MovingAreaBoundaryExtractor, and adds results to QGIS.
        """
        def _cb(pct, msg=''):
            if progress_callback:
                progress_callback(pct, msg)

        X, Y   = results['X'], results['Y']
        dE, dN, dU = results['dE'], results['dN'], results['dU']

        # Choose velocity component
        comp = (self.ma_component or 'magnitude').lower()
        if comp == 'ew':
            grid = dE
        elif comp == 'ud':
            grid = dU
        elif comp == 'ns':
            grid = dN
        else:  # magnitude
            # Use sqrt(dE² + dU²) — include dN only when solved
            if self.solve_north and np.any(np.isfinite(dN)):
                grid = np.sqrt(np.where(np.isfinite(dE), dE**2, 0)
                               + np.where(np.isfinite(dU), dU**2, 0)
                               + np.where(np.isfinite(dN), dN**2, 0))
            else:
                grid = np.sqrt(np.where(np.isfinite(dE), dE**2, 0)
                               + np.where(np.isfinite(dU), dU**2, 0))
            # Propagate NaN where both dE and dU are NaN
            both_nan = ~np.isfinite(dE) & ~np.isfinite(dU)
            grid = np.where(both_nan, np.nan, grid)

        # Flatten to valid points
        flat_x   = X.ravel()
        flat_y   = Y.ravel()
        flat_vel = grid.ravel()
        valid    = np.isfinite(flat_vel)

        if valid.sum() < self.ma_k_neighbors + 1:
            QgsMessageLog.logMessage(
                "Moving area detection: not enough valid grid cells "
                f"({valid.sum()}) for analysis.",
                'TestPlugin', Qgis.Warning)
            return

        xy  = np.column_stack([flat_x[valid], flat_y[valid]])
        vel = flat_vel[valid]

        _cb(91, "Extracting spatial features…")
        extractor = PSFeatureExtractor(xy, vel, k_neighbors=self.ma_k_neighbors)
        features  = extractor.extract()

        _cb(93, "Classifying moving vs stable…")
        trainer = PSTrainer(
            velocity_threshold=self.ma_vel_threshold,
            epochs=self.ma_epochs,
            refinement_iters=self.ma_refinement_iters,
            use_nn=self.ma_use_nn,
        )
        labels = trainer.classify(features, vel)

        n_moving = int(labels.sum())
        QgsMessageLog.logMessage(
            f"Moving area detection: {n_moving}/{len(labels)} "
            f"grid cells classified as moving.",
            'TestPlugin', Qgis.Info)

        _cb(96, "Adding classification layer to QGIS…")
        self._add_ma_classification_layer(xy, vel, labels)

        _cb(98, "Extracting moving area boundary…")
        extractor_boundary = MovingAreaBoundaryExtractor(
            min_cluster_size=self.ma_min_cluster_size,
            alpha=self.ma_alpha,
        )
        boundary_pts = extractor_boundary.extract_boundaries(xy, labels)

        if boundary_pts and self.ma_output_path:
            self._write_ma_shapefile(boundary_pts)

        _cb(99, "Moving area detection complete.")

    # ------------------------------------------------------------------
    def _add_ma_classification_layer(self, xy, velocity, labels):
        """Add a red/green point layer showing moving / stable classification."""
        crs     = self.layer_configs[0]['layer'].crs()
        crs_str = crs.authid() if (crs and crs.isValid()) else 'EPSG:4326'

        uri = (f"Point?crs={crs_str}"
               f"&field=class:integer&field=velocity:double")
        lyr = QgsVectorLayer(uri, "MA_classification_decomposed", "memory")

        if not lyr.isValid():
            QgsMessageLog.logMessage(
                "Could not create moving area classification layer.",
                'TestPlugin', Qgis.Warning)
            return

        provider = lyr.dataProvider()
        feats = []
        for i in range(len(xy)):
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(
                QgsPointXY(float(xy[i, 0]), float(xy[i, 1]))))
            feat.setAttributes([int(labels[i]), float(velocity[i])])
            feats.append(feat)
        provider.addFeatures(feats)

        # Categorised renderer: 0 = stable (green), 1 = moving (red)
        from qgis.PyQt.QtGui import QColor
        cats = []
        for val, color, label in [
            (0, QColor(0, 180, 0),   'Stable'),
            (1, QColor(220, 20, 20), 'Moving'),
        ]:
            sym = QgsSymbol.defaultSymbol(QgsWkbTypes.PointGeometry)
            sym.setColor(color)
            cats.append(QgsRendererCategory(val, sym, label))
        lyr.setRenderer(QgsCategorizedSymbolRenderer('class', cats))

        QgsProject.instance().addMapLayer(lyr)
        QgsMessageLog.logMessage(
            f"Added 'MA_classification_decomposed' ({len(feats)} pts).",
            'TestPlugin', Qgis.Info)

    # ------------------------------------------------------------------
    def _write_ma_shapefile(self, boundary_points):
        """Write moving area boundary points to a shapefile."""
        crs = self.layer_configs[0]['layer'].crs()

        fields_list = QgsFields()
        for f in [
            QgsField('X',          QVariant.Double),
            QgsField('Y',          QVariant.Double),
            QgsField('cluster_id', QVariant.Int),
            QgsField('component',  QVariant.String),
        ]:
            fields_list.append(f)

        transform_context = QgsCoordinateTransformContext()
        save_options = QgsVectorFileWriter.SaveVectorOptions()
        save_options.driverName = "ESRI Shapefile"

        writer = QgsVectorFileWriter.create(
            self.ma_output_path,
            fields_list,
            QgsWkbTypes.Point,
            crs,
            transform_context,
            save_options,
        )

        if writer.hasError() != QgsVectorFileWriter.NoError:
            raise RuntimeError(
                f"Error creating MA shapefile: {writer.errorMessage()}")

        comp_label = self.ma_component or 'magnitude'
        for pt in boundary_points:
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(
                QgsPointXY(pt['x'], pt['y'])))
            feat.setAttributes([
                pt['x'],
                pt['y'],
                pt.get('cluster_id', 0),
                comp_label,
            ])
            writer.addFeature(feat)

        del writer

        QgsMessageLog.logMessage(
            f"MA boundary shapefile written: {self.ma_output_path}",
            'TestPlugin', Qgis.Info)
