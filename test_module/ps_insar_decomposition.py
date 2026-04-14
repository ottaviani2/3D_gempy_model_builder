# -*- coding: utf-8 -*-
"""
InSAR LOS Decomposition and Surface Evolution Module

Python conversion of the MATLAB toolkit:
  GitHub: ottaviani2/InSAR-Decomposition-and-Landslide-movement-estimation
  (No peer-reviewed journal publication identified for this toolkit;
   cite as unpublished software if used in academic work.)

Provides:
  - LOS velocity decomposition into slope-parallel, vertical, and normal components
  - Moving area detection with morphological filtering
  - Surface evolution via mass conservation PDE: dh/dt = -div(h * f * v)
  - Point cloud rotation utilities
  - Thickness estimation from empirical area-depth relationships

References
----------
Surface evolution PDE (mass conservation, depth-averaged thin-layer flow):
  Hungr, O. (1995): A model for the runout analysis of rapid flow slides,
  debris flows and avalanches. Can. Geotech. J., 32(4), 610–623.
  doi:10.1139/t95-063

LOS decomposition into slope-referenced components:
  Colesanti, C. & Wasowski, J. (2006): Investigating landslides with
  space-borne Synthetic Aperture Radar (SAR) interferometry.
  Eng. Geol., 88(3–4), 173–199. doi:10.1016/j.enggeo.2006.09.013

Depth–area empirical scaling — see DEPTH_FORMULAS dict for per-formula refs.
"""

import numpy as np
from datetime import datetime
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter, median_filter, convolve, label
from scipy.spatial import KDTree

from qgis.core import QgsMessageLog, Qgis


# ---------------------------------------------------------------------------
# Grid interpolation utilities
# ---------------------------------------------------------------------------

def interpolate_scattered_to_grid(data_x, data_y, data_values, X, Y,
                                  method='linear'):
    """
    Interpolate scattered data onto a regular grid.
    Equivalent to MATLAB scatteredInterpolant with 'linear','nearest' fallback.
    """
    points = np.column_stack([data_x, data_y])
    grid = griddata(points, data_values, (X, Y), method=method)

    # Fill remaining NaN with nearest-neighbor
    nan_mask = np.isnan(grid)
    if nan_mask.any():
        nn_interp = NearestNDInterpolator(points, data_values)
        grid[nan_mask] = nn_interp(X[nan_mask], Y[nan_mask])

    return grid


def fill_nan_values(grid, X, Y):
    """
    Fill NaN values using scattered interpolation from valid points.
    Equivalent to MATLAB fill_nan_values.
    """
    valid_mask = ~np.isnan(grid)
    n_valid = valid_mask.sum()

    if n_valid < 3:
        mean_val = np.nanmean(grid)
        if np.isnan(mean_val):
            mean_val = 0.0
        filled = grid.copy()
        filled[np.isnan(filled)] = mean_val
        return filled

    valid_x = X[valid_mask]
    valid_y = Y[valid_mask]
    valid_values = grid[valid_mask]

    filled = grid.copy()
    nan_mask = np.isnan(filled)
    if nan_mask.any():
        # Linear interpolation with nearest fallback
        interp_lin = griddata(
            np.column_stack([valid_x, valid_y]),
            valid_values,
            (X[nan_mask], Y[nan_mask]),
            method='linear'
        )
        filled[nan_mask] = interp_lin

        # Fill any remaining NaN with nearest
        still_nan = np.isnan(filled)
        if still_nan.any():
            nn_interp = NearestNDInterpolator(
                np.column_stack([valid_x, valid_y]), valid_values)
            filled[still_nan] = nn_interp(X[still_nan], Y[still_nan])

    return filled


def interpolate_grid_smooth(grid, X, Y, sigma=1.5):
    """
    Fill NaN values and apply Gaussian smoothing.
    Equivalent to MATLAB interpolate_grid.
    """
    filled = fill_nan_values(grid, X, Y)
    smoothed = gaussian_filter(filled, sigma=sigma, mode='nearest')
    return smoothed


# ---------------------------------------------------------------------------
# SAR data reading and projection coefficients
# ---------------------------------------------------------------------------

def read_sar(data_easting, data_northing, data_inc, data_hdg, data_los,
             X, Y, max_distance=20.0):
    """
    Interpolate SAR measurements onto a regular grid with distance filtering.

    Parameters
    ----------
    data_easting, data_northing : array_like
        Scattered point coordinates.
    data_inc, data_hdg, data_los : array_like
        Incidence angle (deg), track/heading angle (deg), LOS velocity.
    X, Y : ndarray
        2D meshgrid arrays.
    max_distance : float
        Maximum distance from nearest data point for valid interpolation.

    Returns
    -------
    inc, hdg, los : ndarray
        Interpolated grids (incidence in radians, heading in radians, LOS in original units).
    valid_mask : ndarray (bool)
        Points within max_distance of a data point.
    """
    # Remove NaN data points
    valid = (~np.isnan(data_inc) & ~np.isnan(data_hdg) & ~np.isnan(data_los))
    dx = data_easting[valid]
    dy = data_northing[valid]
    d_inc = data_inc[valid]
    d_hdg = data_hdg[valid]
    d_los = data_los[valid]

    # Interpolate onto grid
    inc = interpolate_scattered_to_grid(dx, dy, d_inc, X, Y)
    hdg = interpolate_scattered_to_grid(dx, dy, d_hdg, X, Y)
    los = interpolate_scattered_to_grid(dx, dy, d_los, X, Y)

    # Convert to radians
    inc = np.deg2rad(inc)
    hdg = np.deg2rad(hdg)

    # Valid mask based on interpolation success
    valid_mask = ~np.isnan(inc) & ~np.isnan(hdg) & ~np.isnan(los)

    # Distance-based filtering
    if max_distance > 0:
        tree = KDTree(np.column_stack([dx, dy]))
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        dists, _ = tree.query(grid_points, k=1)
        dist_grid = dists.reshape(X.shape)
        valid_mask = valid_mask & (dist_grid <= max_distance)

    return inc, hdg, los, valid_mask


def los_projection_coefficients(theta, phi, alpha, beta):
    """
    Compute projection coefficients (a, b, c) for LOS decomposition.

    DLOS = a * DK + b * DT + c * DI

    where:
      DK = slope-parallel motion
      DT = vertical motion (transverse to slope)
      DI = normal motion (perpendicular to slope)

    Parameters
    ----------
    theta : float
        SAR incidence angle (radians).
    phi : float
        Satellite heading angle (radians).
    alpha : float
        Slope inclination angle (radians).
    beta : float
        Slope azimuth angle (radians).

    Returns
    -------
    a, b, c : float
        Projection coefficients.
    """
    a = (np.cos(theta) * np.cos(alpha) * np.sin(beta)
         + np.sin(theta) * np.sin(phi) * np.cos(alpha) * np.cos(beta)
         + np.sin(theta) * np.cos(phi) * np.sin(alpha))

    b = (-np.cos(theta) * np.sin(beta)
         + np.sin(theta) * np.sin(phi) * np.cos(beta))

    c = (np.cos(theta) * np.sin(alpha) * np.sin(beta)
         + np.sin(theta) * np.sin(phi) * np.sin(alpha) * np.cos(beta)
         - np.sin(theta) * np.cos(phi) * np.cos(alpha))

    return a, b, c


# ---------------------------------------------------------------------------
# LOS Velocity Decomposition
# ---------------------------------------------------------------------------

class LOSDecomposer:
    """
    Decompose multi-orbit SAR LOS velocities into 3D slope-referenced components.

    Solves the overdetermined system:
        DLOS_i = a_i * DK + b_i * DT + c_i * DI
    for each grid cell, using least-squares pseudo-inverse.
    """

    def __init__(self, slope_azimuth_deg, slope_angle_deg,
                 grid_nx=100, grid_ny=100, max_distance=100.0,
                 smoothing_sigma=1.5):
        self.slope_azimuth_deg = slope_azimuth_deg
        self.slope_angle_deg = slope_angle_deg
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.max_distance = max_distance
        self.smoothing_sigma = smoothing_sigma

    def decompose(self, datasets, velocity_col_indices=None,
                  progress_callback=None):
        """
        Decompose LOS velocities from multiple SAR datasets.

        Parameters
        ----------
        datasets : list of dict
            Each dict must have keys:
              'easting', 'northing' : array_like (point coordinates)
              'incidence_angle', 'track_angle' : array_like (degrees)
              'mean_velocity' or velocity column : array_like (LOS velocity)
        velocity_col_indices : list of str or None
            Field names for velocity data in each dataset.
            If None, uses 'mean_velocity'.
        progress_callback : callable or None
            Called with (percent, message).

        Returns
        -------
        result : dict with keys:
            'DK' : ndarray - slope-parallel (down-dip) velocity grid
            'DT' : ndarray - transverse (across-slope) velocity grid
            'DI' : ndarray - slope-normal velocity grid
            'X', 'Y' : ndarray - meshgrid coordinates
            'x', 'y' : 1D coordinate arrays
        """
        alpha = np.deg2rad(self.slope_angle_deg)
        beta = np.deg2rad(self.slope_azimuth_deg)
        n_datasets = len(datasets)

        if progress_callback:
            progress_callback(5, "Computing grid extents...")

        # Compute grid extents from all datasets
        all_x = np.concatenate([np.asarray(d['easting']) for d in datasets])
        all_y = np.concatenate([np.asarray(d['northing']) for d in datasets])
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        x = np.linspace(x_min, x_max, self.grid_nx)
        y = np.linspace(y_min, y_max, self.grid_ny)
        X, Y = np.meshgrid(x, y)
        Y_flipped = np.flipud(Y)

        if progress_callback:
            progress_callback(10, "Reading and interpolating SAR data...")

        # Read and interpolate all datasets
        inc_list = []
        hdg_list = []
        los_list = []
        valid_list = []

        for i, ds in enumerate(datasets):
            vel_key = 'mean_velocity'
            if velocity_col_indices and i < len(velocity_col_indices):
                vel_key = velocity_col_indices[i]

            inc, hdg, los, valid_mask = read_sar(
                np.asarray(ds['easting']),
                np.asarray(ds['northing']),
                np.asarray(ds['incidence_angle']),
                np.asarray(ds['track_angle']),
                np.asarray(ds[vel_key]),
                X, Y_flipped,
                max_distance=self.max_distance,
            )

            # Apply distance filter, then interpolate across entire surface
            los[~valid_mask] = np.nan
            inc[~valid_mask] = np.nan
            hdg[~valid_mask] = np.nan

            los = interpolate_grid_smooth(los, X, Y_flipped,
                                          sigma=self.smoothing_sigma)
            inc = interpolate_grid_smooth(inc, X, Y_flipped,
                                          sigma=self.smoothing_sigma)
            hdg = interpolate_grid_smooth(hdg, X, Y_flipped,
                                          sigma=self.smoothing_sigma)

            inc_list.append(inc)
            hdg_list.append(hdg)
            los_list.append(los)
            valid_list.append(valid_mask)

            if progress_callback:
                pct = 10 + int(30 * (i + 1) / n_datasets)
                progress_callback(pct, f"Interpolated dataset {i + 1}/{n_datasets}")

        if progress_callback:
            progress_callback(45, "Decomposing LOS velocities...")

        # Solve decomposition at each grid point
        nrows, ncols = X.shape
        DK = np.full((nrows, ncols), np.nan)
        DT = np.full((nrows, ncols), np.nan)
        DI = np.full((nrows, ncols), np.nan)

        for i in range(nrows):
            for j in range(ncols):
                # Collect LOS values
                los_values = np.zeros(n_datasets)
                valid_data = True

                for n in range(n_datasets):
                    val = los_list[n][i, j]
                    if np.isnan(val):
                        valid_data = False
                        break
                    los_values[n] = val

                if not valid_data:
                    continue

                # Build coefficient matrix
                A = np.zeros((n_datasets, 3))
                for n in range(n_datasets):
                    theta = inc_list[n][i, j]
                    phi = hdg_list[n][i, j]
                    a, b, c = los_projection_coefficients(theta, phi, alpha, beta)
                    A[n, :] = [a, b, c]

                if np.any(np.isnan(A)):
                    continue

                # Least-squares solution via pseudo-inverse
                try:
                    S = np.linalg.pinv(A) @ los_values
                    DK[i, j] = S[0]  # slope-parallel (down-dip)
                    DT[i, j] = S[1]  # transverse (across-slope)
                    DI[i, j] = S[2]  # slope-normal
                except np.linalg.LinAlgError:
                    continue

            if progress_callback and i % 10 == 0:
                pct = 45 + int(45 * i / nrows)
                progress_callback(pct, f"Decomposing row {i}/{nrows}...")

        if progress_callback:
            progress_callback(95, "Decomposition complete")

        return {
            'DK': DK, 'DT': DT, 'DI': DI,
            'X': X, 'Y': Y_flipped,
            'x': x, 'y': y,
        }


# ---------------------------------------------------------------------------
# Binary mask operations (converted from MATLAB)
# ---------------------------------------------------------------------------

def clean_binary_mask(mask):
    """
    Simple morphological opening + closing without image processing libraries.
    Removes isolated pixels, then fills small gaps.
    """
    rows, cols = mask.shape
    # Opening: remove isolated pixels (need >= 2 neighbors)
    temp = np.zeros_like(mask, dtype=bool)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if mask[i, j]:
                neighbors = mask[i-1:i+2, j-1:j+2]
                if neighbors.sum() >= 3:  # including center
                    temp[i, j] = True

    # Closing: fill small gaps (surrounded by >= 6 true pixels)
    cleaned = temp.copy()
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if not cleaned[i, j]:
                neighbors = temp[i-1:i+2, j-1:j+2]
                if neighbors.sum() >= 6:
                    cleaned[i, j] = True

    return cleaned


def flood_fill(mask, visited, start_i, start_j):
    """
    Stack-based flood fill. Returns (component_mask, component_size).
    Uses 4-connectivity.
    """
    rows, cols = mask.shape
    component = np.zeros_like(mask, dtype=bool)
    stack = [(start_i, start_j)]
    comp_size = 0

    while stack:
        i, j = stack.pop()
        if (i < 0 or i >= rows or j < 0 or j >= cols
                or visited[i, j] or not mask[i, j] or component[i, j]):
            continue

        component[i, j] = True
        comp_size += 1
        stack.extend([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])

    return component, comp_size


def largest_connected_component(mask):
    """Find the largest connected component in a binary mask."""
    # Use scipy.ndimage.label for efficiency (equivalent to MATLAB bwlabel)
    labeled, n_features = label(mask)
    if n_features == 0:
        return np.zeros_like(mask, dtype=bool)

    # Find largest component
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    largest_label = sizes.argmax()
    return labeled == largest_label


def get_perimeter(mask):
    """
    Get perimeter pixels of a binary mask (4-connected boundary detection).
    """
    rows, cols = mask.shape
    perim = np.zeros_like(mask, dtype=bool)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if mask[i, j]:
                neighbors = [mask[i-1, j], mask[i+1, j],
                             mask[i, j-1], mask[i, j+1]]
                if not all(neighbors):
                    perim[i, j] = True

    # Handle borders
    perim[0, :] |= mask[0, :]
    perim[-1, :] |= mask[-1, :]
    perim[:, 0] |= mask[:, 0]
    perim[:, -1] |= mask[:, -1]

    return perim


# ---------------------------------------------------------------------------
# Moving area detection
# ---------------------------------------------------------------------------

def detect_moving_area(vx, vy, boundary_buffer=2, threshold_mmday=0.1):
    """
    Detect moving area from velocity components.
    Returns (interior_mask, moving_mask).

    Parameters
    ----------
    vx, vy : ndarray
        Velocity component grids in mm/day (slope-parallel, vertical).
    boundary_buffer : int
        Number of perimeter layers to erode from the moving mask.
    threshold_mmday : float
        Minimum velocity magnitude [mm/day] to classify a cell as moving.
        Default 0.1 mm/day ≈ 36 mm/yr, typical for slow PS-InSAR landslides.

    Returns
    -------
    interior_mask : ndarray (bool)
        Interior calculation domain (moving area minus boundary buffer).
        Same orientation as the input vx/vy grids (no flip applied).
    moving_mask : ndarray (bool)
        Full moving area mask.
    """
    velocity_magnitude = np.sqrt(
        np.where(np.isnan(vx), 0, vx)**2
        + np.where(np.isnan(vy), 0, vy)**2
    )

    moving_mask = (velocity_magnitude > threshold_mmday) & ~np.isnan(velocity_magnitude)

    # Clean up: morphological filtering
    moving_mask = clean_binary_mask(moving_mask)

    # Keep only largest connected component
    moving_mask = largest_connected_component(moving_mask)

    # Interior mask: erode boundary (same orientation as vx/vy — no flip)
    interior_mask = moving_mask.copy()

    for _ in range(boundary_buffer):
        perim = get_perimeter(interior_mask)
        interior_mask = interior_mask & ~perim

    # Exclude domain boundaries
    interior_mask[:2, :] = False
    interior_mask[-2:, :] = False
    interior_mask[:, :2] = False
    interior_mask[:, -2:] = False

    return interior_mask, moving_mask


# ---------------------------------------------------------------------------
# Outlier detection and smoothing
# ---------------------------------------------------------------------------

def smooth_with_outlier_detection(grid, threshold=3.0):
    """
    Replace outliers detected via MAD (median absolute deviation)
    with local median values.

    Parameters
    ----------
    grid : ndarray
        Input 2D grid.
    threshold : float
        Number of MAD units for outlier detection.

    Returns
    -------
    smoothed : ndarray
    """
    med_filtered = median_filter(grid, size=3, mode='reflect')
    deviation = np.abs(grid - med_filtered)
    mad = median_filter(np.abs(grid - med_filtered), size=3, mode='reflect')
    mad[mad == 0] = np.finfo(float).eps

    is_outlier = deviation > threshold * mad
    smoothed = grid.copy()
    smoothed[is_outlier] = med_filtered[is_outlier]
    return smoothed


# ---------------------------------------------------------------------------
# Point cloud rotation
# ---------------------------------------------------------------------------

def rotate_point_cloud(x, y, z, angle_deg=0.0, match_y=True):
    """
    Rotate a point cloud around the point with minimum X.

    Parameters
    ----------
    x, y, z : array_like
        Point coordinates.
    angle_deg : float
        Additional rotation angle in degrees.
    match_y : bool
        If True, first align the direction (min_x -> max_y) with the Y-axis,
        then apply the additional angle.

    Returns
    -------
    x_rot, y_rot, z : ndarray
        Rotated point coordinates (z unchanged).
    angle_to_y : float
        Rotation angle applied (radians).
    pivot : tuple
        Pivot point (P1) used for rotation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    idx_min_x = np.argmin(x)
    idx_max_y = np.argmax(y)

    P1 = np.array([x[idx_min_x], y[idx_min_x]])
    P2 = np.array([x[idx_max_y], y[idx_max_y]])

    if match_y:
        direction = P2 - P1
        angle_to_y = np.arctan2(direction[0], direction[1])
        angle = angle_to_y + np.deg2rad(angle_deg)
    else:
        angle = np.deg2rad(angle_deg)
        angle_to_y = angle

    # Shift to pivot
    x_shifted = x - P1[0]
    y_shifted = y - P1[1]

    # Rotation matrix
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_rot = cos_a * x_shifted - sin_a * y_shifted + P1[0]
    y_rot = sin_a * x_shifted + cos_a * y_shifted + P1[1]

    return x_rot, y_rot, z, angle_to_y, tuple(P1)


# ---------------------------------------------------------------------------
# DTM initialization for surface evolution
# ---------------------------------------------------------------------------

def initialize_height_from_dtm(dtm_path, x_grid, y_grid):
    """
    Sample a DTM raster onto the computation grid.

    Parameters
    ----------
    dtm_path : str
        Path to the DTM GeoTIFF.
    x_grid, y_grid : array_like
        1D coordinate arrays for the computation grid.

    Returns
    -------
    h : ndarray (ny, nx)
        Height grid. Nodata values set to -1.
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required for DTM loading")

    x_num = len(x_grid)
    y_num = len(y_grid)

    with rasterio.open(dtm_path) as src:
        A = src.read(1).astype(np.float64)
        bounds = src.bounds

    h = np.zeros((y_num, x_num))
    s_x = max(1, A.shape[1] // x_num)
    s_y = max(1, A.shape[0] // y_num)

    for j in range(x_num):
        for i in range(y_num):
            index_y = min(i * s_y, A.shape[0] - 1)
            index_x = min(j * s_x, A.shape[1] - 1)
            val = A[index_y, index_x]
            if val <= 0 or np.isnan(val):
                h[i, j] = -1
            else:
                h[i, j] = val

    return h


# ---------------------------------------------------------------------------
# Time-series date utilities
# ---------------------------------------------------------------------------

def parse_egms_dates(column_names, date_col_offset=28):
    """
    Parse date strings from EGMS-style column names (xYYYYMMDD format).

    Parameters
    ----------
    column_names : list of str
        Column names from the CSV data.
    date_col_offset : int
        Index where displacement time-series columns start.

    Returns
    -------
    dates : list of datetime
        Parsed dates.
    """
    dates = []
    for name in column_names[date_col_offset:]:
        clean = name.replace('x', '').replace('X', '')
        try:
            dt = datetime.strptime(clean, '%Y%m%d')
            dates.append(dt)
        except ValueError:
            continue
    return dates


def find_closest_date_index(target_date, date_array):
    """Find the index of the closest date in date_array to target_date."""
    diffs = [abs((d - target_date).total_seconds()) for d in date_array]
    return int(np.argmin(diffs))


# ---------------------------------------------------------------------------
# Surface Evolution Solver
# ---------------------------------------------------------------------------

class SurfaceEvolutionSolver:
    """
    Solve the mass conservation PDE for landslide surface evolution:

        dh/dt = -div(h * f * v)

    Using explicit (Forward Euler) finite differences on a regular grid.

    The PDE is the depth-averaged mass conservation equation for a
    thin granular layer, where f is a basal friction/rheological factor
    that modulates the effective velocity field.

    Reference:
        Hungr, O. (1995): A model for the runout analysis of rapid flow
        slides, debris flows and avalanches.
        Can. Geotech. J., 32(4), 610–623. doi:10.1139/t95-063

    Parameters
    ----------
    f : float
        Basal friction / rheological coefficient [0, 1].
    grid_nx, grid_ny : int
        Grid dimensions.
    """

    def __init__(self, f=0.01, grid_nx=100, grid_ny=100):
        self.f = f
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny

    def run(self, h_initial, vx_series, vy_series, dt_series,
            dx=1.0, dy=1.0, interior_masks=None, progress_callback=None):
        """
        Run the surface evolution over a time-series of velocity fields.

        Parameters
        ----------
        h_initial : ndarray (ny, nx)
            Initial elevation/thickness grid.
        vx_series : list of ndarray
            Slope-parallel velocity grids for each time step.
        vy_series : list of ndarray
            Vertical velocity grids for each time step.
        dt_series : list of float
            Time increment (days) for each step.
        dx : float
            Grid spacing in the x-direction (metres). Defaults to 1.0 (pixel
            units); pass real-world spacing to obtain physically correct
            thickness-change magnitudes.
        dy : float
            Grid spacing in the y-direction (metres). Defaults to 1.0 (pixel
            units).
        interior_masks : list of ndarray (bool) or None
            Calculation masks for each step. If None, computed automatically.
        progress_callback : callable or None
            Called with (percent, message).

        Returns
        -------
        result : dict with keys:
            'h_final' : ndarray - final elevation grid
            'thickness_change' : ndarray - h_final - h_initial
            'thickness_evolution' : ndarray (ny, nx, nt) - full history
            'area_series' : list of float - moving area per time step
        """
        ny, nx = h_initial.shape

        if dx == 1.0 or dy == 1.0:
            QgsMessageLog.logMessage(
                "SurfaceEvolutionSolver.run(): dx and/or dy are 1.0 (pixel units). "
                "thickness_change magnitudes and area_series values will be in pixel "
                "units, not metres/m².  Pass real-world grid spacing for physically "
                "meaningful results, or use run_with_grid_spacing() instead.",
                "TestPlugin", Qgis.Warning)

        n_steps = len(vx_series)
        h = h_initial.copy()
        h_saved = h_initial.copy()
        thickness_evolution = np.zeros((ny, nx, n_steps))
        area_series = []

        for step in range(n_steps):
            vx = vx_series[step]
            vy = vy_series[step]
            dt = dt_series[step]

            # Handle NaN in velocities
            vx = np.where(np.isnan(vx), 0.0, vx)
            vy = np.where(np.isnan(vy), 0.0, vy)

            # Compute interior mask if not provided
            if interior_masks is not None:
                interior_mask = interior_masks[step]
            else:
                interior_mask, _ = detect_moving_area(vx, vy)

            n_interior = interior_mask.sum()
            # area = n_interior pixels × pixel area (dx=dy=1 in this method)
            area_series.append(float(n_interior * dx * dy))

            h_new = h.copy()

            # Surface evolution PDE: dh/dt = -div(h * f * v)
            # h and vx/vy share the same grid orientation (h_initial was
            # flipped upstream to match the decomposer Y_flipped output).
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    if not interior_mask[i, j]:
                        continue  # h_new[i,j] already equals h[i,j]

                    v_eff_x = self.f * vx[i, j]
                    v_eff_y = self.f * vy[i, j]

                    # X-direction: d(h*f*vx)/dx
                    term1_x = v_eff_x * (h[i, j+1] - h[i, j-1]) / (2 * dx)
                    term2_x = h[i, j] * (self.f * vx[i, j+1] - self.f * vx[i, j-1]) / (2 * dx)

                    # Y-direction: d(h*f*vy)/dy
                    term1_y = v_eff_y * (h[i+1, j] - h[i-1, j]) / (2 * dy)
                    term2_y = h[i, j] * (self.f * vy[i+1, j] - self.f * vy[i-1, j]) / (2 * dy)

                    flux_divergence = (term1_x + term2_x) + (term1_y + term2_y)
                    dh_dt = -flux_divergence

                    if np.isnan(dh_dt):
                        dh_dt = 0.0

                    # Unit derivation:
                    #   vx      [mm/day]  = displacement[mm] / dt_days[day]
                    #   term1_x [mm/day]  = vx[mm/day] * (dh/dx)[m/m]        (dimensionless ratio)
                    #   term2_x [mm/day]  = h[m] * (d(f·vx)/dx)[mm/day / m]  → mm/day
                    #   flux_divergence [mm/day], dh_dt [mm/day]
                    #   delta_h = dt[day] * dh_dt[mm/day] / 1000             → [m]
                    # No days→years conversion needed: vx is already in mm/day.
                    delta_h = dt * dh_dt / 1000.0
                    # Clamp: thickness cannot go negative (physical constraint).
                    h_new[i, j] = max(h[i, j] + delta_h, 0.0)

            # Boundary conditions: fixed to initial values
            h_new[:, 0] = h_saved[:, 0]
            h_new[:, -1] = h_saved[:, -1]
            h_new[0, :] = h_saved[0, :]
            h_new[-1, :] = h_saved[-1, :]

            h = h_new.copy()
            thickness_evolution[:, :, step] = h_new

            if progress_callback and step % max(1, n_steps // 20) == 0:
                pct = int(100 * (step + 1) / n_steps)
                progress_callback(pct, f"Surface evolution step {step + 1}/{n_steps}")

        thickness_change = h - h_saved
        moving_area_km2 = np.mean(area_series) / 1e6 if area_series else 0.0

        QgsMessageLog.logMessage(
            f"Surface evolution complete: {n_steps} steps, "
            f"mean moving area = {moving_area_km2:.3f} km²",
            "TestPlugin", Qgis.Info)

        return {
            'h_final': h,
            'h_initial': h_saved,
            'thickness_change': thickness_change,
            'thickness_evolution': thickness_evolution,
            'area_series': area_series,
            'moving_area_km2': moving_area_km2,
        }

    def run_with_grid_spacing(self, h_initial, vx_series, vy_series,
                               dt_series, dx, dy, interior_masks=None,
                               progress_callback=None):
        """
        Same as run() but with explicit grid spacing.
        This is the preferred method when coordinate arrays are available.
        """
        ny, nx = h_initial.shape
        n_steps = len(vx_series)
        h = h_initial.copy()
        h_saved = h_initial.copy()
        thickness_evolution = np.zeros((ny, nx, n_steps))
        area_series = []

        for step in range(n_steps):
            vx = np.where(np.isnan(vx_series[step]), 0.0, vx_series[step])
            vy = np.where(np.isnan(vy_series[step]), 0.0, vy_series[step])
            dt = dt_series[step]

            if interior_masks is not None:
                interior_mask = interior_masks[step]
            else:
                interior_mask, _ = detect_moving_area(vx, vy)

            area_series.append(float(interior_mask.sum() * dx * dy))
            h_new = h.copy()

            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    if not interior_mask[i, j]:
                        continue  # h_new[i,j] already equals h[i,j]

                    v_eff_x = self.f * vx[i, j]
                    v_eff_y = self.f * vy[i, j]

                    term1_x = v_eff_x * (h[i, j+1] - h[i, j-1]) / (2 * dx)
                    term2_x = h[i, j] * (self.f * vx[i, j+1] - self.f * vx[i, j-1]) / (2 * dx)
                    term1_y = v_eff_y * (h[i+1, j] - h[i-1, j]) / (2 * dy)
                    term2_y = h[i, j] * (self.f * vy[i+1, j] - self.f * vy[i-1, j]) / (2 * dy)

                    flux_divergence = (term1_x + term2_x) + (term1_y + term2_y)
                    dh_dt = -flux_divergence

                    if np.isnan(dh_dt):
                        dh_dt = 0.0

                    delta_h = dt * dh_dt / 1000.0
                    # Clamp: thickness cannot go negative (physical constraint).
                    h_new[i, j] = max(h[i, j] + delta_h, 0.0)

            h_new[:, 0] = h_saved[:, 0]
            h_new[:, -1] = h_saved[:, -1]
            h_new[0, :] = h_saved[0, :]
            h_new[-1, :] = h_saved[-1, :]

            h = h_new.copy()
            thickness_evolution[:, :, step] = h_new

            if progress_callback and step % max(1, n_steps // 20) == 0:
                pct = int(100 * (step + 1) / n_steps)
                progress_callback(pct, f"Surface evolution step {step + 1}/{n_steps}")

        return {
            'h_final': h,
            'h_initial': h_saved,
            'thickness_change': h - h_saved,
            'thickness_evolution': thickness_evolution,
            'area_series': area_series,
            'moving_area_km2': np.mean(area_series) / 1e6 if area_series else 0.0,
        }


# ---------------------------------------------------------------------------
# Thickness estimation from empirical relationships
# ---------------------------------------------------------------------------

def estimate_thickness_from_area(area_m2):
    """
    Legacy wrapper — prefer estimate_depth_from_area().

    WARNING: The regression source (Simoni et al., 2013) and the logarithm
    base (ln vs log₁₀) for both formulas are UNVERIFIED.  If the original
    regression used log₁₀ (common in geomorphology), replacing np.log/np.exp
    with np.log10/10** would reduce the prefactor by ~4.7×.  Do NOT use these
    values in published work without tracing the original source.

    Assumed form — natural logarithm (np.log = ln):

    Circular geometry (attributed to Simoni et al., 2013 — UNVERIFIED):
        ln(D) = 0.44·ln(A) - 1.18  →  D = e^(-1.18)·A^0.44 ≈ 0.307·A^0.44  [m]

    Unknown-geometry width estimate (source UNVERIFIED):
        ln(W) = 0.42·ln(A) + 1.13  →  W = e^(1.13)·A^0.42  ≈ 3.09·A^0.42   [m]
        NOTE: W appears to represent along-slope *width*, not depth/thickness.
        Values of W are typically ~10× larger than D for the same area, which
        is physically unreasonable for a thickness but plausible for a width.

    Parameters
    ----------
    area_m2 : float
        Moving area in square meters (m²).

    Returns
    -------
    D : float
        Estimated depth/thickness for circular geometry [metres].
    W : float
        Estimated along-slope width for unknown geometry [metres].
        (Labelled 'thickness' historically but likely represents width.)
    """
    if area_m2 <= 0:
        return 0.0, 0.0

    log_A = np.log(area_m2)
    D = np.exp(0.44 * log_A - 1.18)
    W = np.exp(0.42 * log_A + 1.13)

    return float(D), float(W)


# ---------------------------------------------------------------------------
# Empirical depth-area scaling formulas (D = alpha * A^beta)
# ---------------------------------------------------------------------------

#: Registry of empirical area-to-depth formulas.
#: Each entry: alpha, beta coefficients for D = alpha * A^beta (D in metres,
#: A in square metres), plus metadata for display.
DEPTH_FORMULAS = {
    'guzzetti_2009': {
        'label': 'Guzzetti et al. (2009) — Mixed types, Central Italy',
        'alpha': 0.1078,
        'beta': 0.4130,
        'ref': (
            'Guzzetti, F., Ardizzone, F., Cardinali, M., Rossi, M., Valigi, D. '
            '(2009): Landslide volumes and landslide mobilization rates in Umbria, '
            'central Italy. Earth Planet. Sci. Lett., 279, 222–229. '
            'doi:10.1016/j.epsl.2009.01.005. '
            'NOTE: the paper publishes V = 0.074·A^1.45 (volume–area); the D–A '
            'coefficients here (α=0.1078, β=0.413) are from a direct depth–area '
            'regression and differ from D = V/A = 0.074·A^0.45. Verify against '
            'Table 1 of the original paper before use.'
        ),
        'desc': (
            'Derived from 677 mixed landslides (slides, flows, complex) in '
            'Umbria, central Italy (flysch, marly limestones). Most widely '
            'cited general-purpose formula. Valid for A ≈ 10¹–10⁷ m².'
        ),
    },
    'larsen_2010_soil': {
        'label': 'Larsen et al. (2010) — Soil / regolith slides (global)',
        'alpha': 0.4840,
        'beta': 0.3290,
        'ref': (
            'Larsen, I.J., Montgomery, D.R., Korup, O. (2010): Landslide erosion '
            'controlled by hillslope material. Nature Geoscience, 3, 247–251. '
            'doi:10.1038/ngeo776. '
            'Coefficients verified: log10(D) = 0.329·log10(A) − 0.315 '
            '→ α = 10^−0.315 = 0.484.'
        ),
        'desc': (
            'Global compilation of ~2,500 soil/regolith landslides across '
            'diverse climatic and tectonic settings. Best for shallow '
            'failures in unconsolidated or weathered material (A ≈ 1–10⁹ m²).'
        ),
    },
    'larsen_2010_bedrock': {
        'label': 'Larsen et al. (2010) — Bedrock slides (global)',
        'alpha': 0.1570,
        'beta': 0.4120,
        'ref': (
            'Larsen, I.J., Montgomery, D.R., Korup, O. (2010): Landslide erosion '
            'controlled by hillslope material. Nature Geoscience, 3, 247–251. '
            'doi:10.1038/ngeo776. '
            'Coefficients verified: log10(D) = 0.412·log10(A) − 0.804 '
            '→ α = 10^−0.804 = 0.157.'
        ),
        'desc': (
            'Global compilation of ~1,700 bedrock landslides. Best for rock '
            'slides and deep-seated failures in hard rock. Bedrock slides '
            'are deeper than soil slides for a given area (A ≈ 1–10⁹ m²).'
        ),
    },
    'larsen_2010_all': {
        'label': 'Larsen et al. (2010) — All types combined (global)',
        'alpha': 0.2840,
        'beta': 0.3510,
        'ref': (
            'Larsen, I.J., Montgomery, D.R., Korup, O. (2010): Landslide erosion '
            'controlled by hillslope material. Nature Geoscience, 3, 247–251. '
            'doi:10.1038/ngeo776. '
            'Coefficients verified: log10(D) = 0.351·log10(A) − 0.547 '
            '→ α = 10^−0.547 = 0.284.'
        ),
        'desc': (
            'Combined global dataset (~4,231 landslides), all material types. '
            'Use when landslide type (soil vs. bedrock) is unknown '
            '(A ≈ 1–10⁹ m²).'
        ),
    },
    'marc_2018': {
        'label': 'Marc et al. (2018) — Rainfall-triggered (global)',
        'alpha': 0.1330,
        'beta': 0.3990,
        'ref': (
            'UNVERIFIED — precise source paper for α=0.133, β=0.399 not confirmed. '
            'Possible source: Marc, O., et al. (2018): Initial insights from a global '
            'database of rainfall-induced landslide inventories. '
            'Earth Surf. Dynam., 6, 903–922. doi:10.5194/esurf-6-903-2018. '
            'Cross-check coefficients against the original publication before use.'
        ),
        'desc': (
            'Rainfall-triggered landslides globally across diverse climatic '
            'and lithological settings. Consistent with Guzzetti (2009) '
            'exponent; broader geographic coverage (A ≈ 10¹–10⁷ m²).'
        ),
    },
    'tanyas_2017': {
        'label': 'Tanyaş et al. (2017) — Coseismic landslides (global)',
        'alpha': 0.1370,
        'beta': 0.4000,
        'ref': (
            'Tanyaş, H., van Westen, C.J., Allstadt, K.E., et al. (2017): '
            'Presentation and Analysis of a Worldwide Database of '
            'Earthquake-Induced Landslide Inventories. '
            'J. Geophys. Res. Earth Surf., 122(10), 1991–2015. '
            'doi:10.1002/2017JF004236. '
            'NOTE: previous entry had wrong issue (8) and wrong pages (1595–1615); '
            'correct is issue 10, pp. 1991–2015. '
            'Depth–area coefficients (α=0.137, β=0.400) should be verified '
            'against the paper tables before use.'
        ),
        'desc': (
            'Earthquake-triggered landslides globally, compiled from major '
            'seismic events (Wenchuan 2008, Haiti 2010, etc.). Use for '
            'seismically induced slope failures (A ≈ 10¹–10⁸ m²).'
        ),
    },
    'guthrie_2004': {
        'label': 'Guthrie & Evans (2004) — Shallow debris (BC, Canada)',
        'alpha': 0.0500,
        'beta': 0.4500,
        'ref': (
            'Guthrie, R.H. & Evans, S.G. (2004): Analysis of landslide frequencies '
            'and characteristics in a natural system, coastal British Columbia. '
            'Earth Surf. Process. Landforms, 29(11), 1321–1339. '
            'doi:10.1002/esp.1095. '
            'UNVERIFIED: depth–area coefficients (α=0.050, β=0.450) not confirmed '
            'in the cited paper. The paper may report volume–area or frequency '
            'relationships rather than a direct D–A fit. Verify before use.'
        ),
        'desc': (
            'Shallow debris avalanches and flows in steep, forested coastal '
            'terrain (British Columbia, Canada). Not suitable for deep-seated '
            'or rotational slides. Valid for A ≈ 10²–10⁵ m².'
        ),
    },
}


def estimate_depth_from_area(area_m2, formula_key='guzzetti_2009'):
    """
    Estimate landslide sliding surface depth from planimetric area
    using empirical power-law relationships:  D = alpha * A^beta.

    Parameters
    ----------
    area_m2 : float
        Landslide planimetric area in square metres.
    formula_key : str
        Key into DEPTH_FORMULAS. Default: 'guzzetti_2009'.

    Returns
    -------
    depth : float
        Estimated sliding surface depth in metres.
    info : dict
        Metadata for the chosen formula (label, alpha, beta, ref, desc).
    """
    info = DEPTH_FORMULAS.get(formula_key, DEPTH_FORMULAS['guzzetti_2009'])
    if area_m2 <= 0:
        return 0.0, info
    depth = info['alpha'] * (area_m2 ** info['beta'])
    return float(depth), info


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------

class InSARDecompositionProcessor:
    """
    Full InSAR decomposition and surface evolution pipeline.

    Reads multi-orbit PS-InSAR CSV data, decomposes LOS velocities,
    runs surface evolution, and exports results.
    """

    def __init__(self, csv_paths, dtm_path,
                 slope_azimuth_deg=270.0, slope_angle_deg=50.0,
                 friction_coeff=0.01, grid_nx=100, grid_ny=100,
                 max_distance=100.0, date_col_offset=28,
                 smoothing_sigma=1.5, depth_formula='guzzetti_2009',
                 include_vertical_velocity=False,
                 movement_threshold_mmday=0.1):
        self.csv_paths = csv_paths
        self.dtm_path = dtm_path
        self.slope_azimuth_deg = slope_azimuth_deg
        self.slope_angle_deg = slope_angle_deg
        self.friction_coeff = friction_coeff
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.max_distance = max_distance
        self.date_col_offset = date_col_offset
        self.smoothing_sigma = smoothing_sigma
        self.depth_formula = depth_formula
        # If True, vertical slope-component (DT) is included in vy;
        # if False, only slope-parallel motion (DK) drives the PDE.
        self.include_vertical_velocity = include_vertical_velocity
        # Minimum velocity magnitude [mm/day] to classify a cell as moving.
        self.movement_threshold_mmday = movement_threshold_mmday

    def load_csv_datasets(self):
        """
        Load CSV files into dataset dicts.
        Expected columns: easting, northing, incidence_angle, track_angle,
        mean_velocity, plus time-series displacement columns.
        """
        import pandas as pd

        datasets = []
        for path in self.csv_paths:
            df = pd.read_csv(path, sep=None, engine='python')

            # Normalize column names: handle X/Y -> easting/northing
            col_map = {}
            cols_lower = {c.lower(): c for c in df.columns}
            if 'easting' not in cols_lower and 'x' in cols_lower:
                col_map[cols_lower['x']] = 'easting'
            if 'northing' not in cols_lower and 'y' in cols_lower:
                col_map[cols_lower['y']] = 'northing'
            if col_map:
                df = df.rename(columns=col_map)

            ds = {
                'easting': df['easting'].values,
                'northing': df['northing'].values,
                'incidence_angle': df['incidence_angle'].values,
                'track_angle': df['track_angle'].values,
                'mean_velocity': df['mean_velocity'].values,
                'dataframe': df,
                'columns': list(df.columns),
            }
            datasets.append(ds)

        return datasets

    def run(self, progress_callback=None):
        """
        Execute the full pipeline:
        1. Load CSV data
        2. Initialize height from DTM
        3. For each time step: decompose LOS -> get velocity -> evolve surface
        4. Return results

        Returns
        -------
        results : dict
        """
        import pandas as pd

        if progress_callback:
            progress_callback(2, "Loading CSV datasets...")

        datasets = self.load_csv_datasets()
        n_datasets = len(datasets)

        if progress_callback:
            progress_callback(5, "Parsing dates...")

        # Parse dates from the smallest dataset
        col_counts = [len(ds['columns']) for ds in datasets]
        smallest_idx = int(np.argmin(col_counts))
        smallest_df = datasets[smallest_idx]['dataframe']
        all_columns = list(smallest_df.columns)

        dates_all = []
        for i, ds in enumerate(datasets):
            dates_all.append(
                parse_egms_dates(ds['columns'], self.date_col_offset))

        dates_smallest = dates_all[smallest_idx]
        n_timesteps = len(dates_smallest)

        if n_timesteps < 2:
            raise ValueError(
                f"Not enough time steps found ({n_timesteps}). "
                f"Check date_col_offset ({self.date_col_offset}).")

        # Compute grid coordinates
        all_x = np.concatenate([ds['easting'] for ds in datasets])
        all_y = np.concatenate([ds['northing'] for ds in datasets])
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        x = np.linspace(x_min, x_max, self.grid_nx)
        y = np.linspace(y_min, y_max, self.grid_ny)
        dx = (x_max - x_min) / max(self.grid_nx - 1, 1)
        dy = (y_max - y_min) / max(self.grid_ny - 1, 1)

        if progress_callback:
            progress_callback(8, "Initializing height from DTM...")

        # Initialize height on the ascending-y grid, then flip to match the
        # decomposer's Y_flipped orientation (row 0 = y_max) so that h and
        # vx/vy share the same row ordering throughout the PDE solver.
        h_initial = np.flipud(initialize_height_from_dtm(self.dtm_path, x, y))

        if progress_callback:
            progress_callback(10, "Running time-series decomposition...")

        # Decomposer
        decomposer = LOSDecomposer(
            slope_azimuth_deg=self.slope_azimuth_deg,
            slope_angle_deg=self.slope_angle_deg,
            grid_nx=self.grid_nx,
            grid_ny=self.grid_ny,
            max_distance=self.max_distance,
            smoothing_sigma=self.smoothing_sigma,
        )

        # Build per-timestep velocity fields via decomposition
        vx_series = []
        vy_series = []
        dt_series = []
        interior_masks = []

        move = self.date_col_offset

        for t_idx in range(n_timesteps):
            if t_idx == 0:
                continue  # need previous step for dt

            # Calculate dt
            dt_days = (dates_smallest[t_idx] - dates_smallest[t_idx - 1]).days
            if dt_days <= 0:
                dt_days = 1
            dt_series.append(float(dt_days))

            # Column index in smallest dataset
            col_idx_smallest = move + t_idx
            date_target = dates_smallest[t_idx]

            # For each dataset, find the closest date column
            velocity_data = []
            for i, ds in enumerate(datasets):
                df = ds['dataframe']
                if i == smallest_idx:
                    col_name = all_columns[col_idx_smallest]
                else:
                    closest = find_closest_date_index(date_target, dates_all[i])
                    col_name = ds['columns'][move + closest]

                ds_copy = dict(ds)
                ds_copy['mean_velocity'] = df[col_name].values
                velocity_data.append(ds_copy)

            # Decompose for this time step
            result = decomposer.decompose(velocity_data)

            d_x = result['DK']   # slope-parallel displacement [mm], epoch
            d_y = result['DT']   # transverse (across-slope) displacement [mm], epoch

            # Sign convention:
            #   Positive LOS = motion TOWARD satellite (range decrease).
            #   After slope-frame decomposition, positive DK therefore means
            #   motion toward the satellite projected onto the slope — which
            #   for ascending right-looking geometry (heading ~350°, look right)
            #   is predominantly UP-SLOPE.
            #   The PDE dh/dt = -div(h·f·v) removes mass from where div(v) > 0
            #   (diverging / accelerating flow). To model DOWN-SLOPE transport
            #   (mass leaving the source zone) we need vx to point DOWN-SLOPE,
            #   so we negate DK before dividing by dt.
            #   If your data uses the opposite LOS sign (away = positive), remove
            #   the negation or the model will thicken the source and thin the toe.
            vx = -d_x / dt_days   # [mm/day], positive = down-slope in slope frame
            if self.include_vertical_velocity:
                vy = -d_y / dt_days   # [mm/day], positive = across-slope
            else:
                vy = np.zeros_like(vx)

            vx_series.append(vx)
            vy_series.append(vy)

            # Moving area detection
            interior_mask, _ = detect_moving_area(
                vx, vy,
                threshold_mmday=self.movement_threshold_mmday,
            )
            interior_masks.append(interior_mask)

            if progress_callback:
                pct = 10 + int(60 * t_idx / n_timesteps)
                progress_callback(
                    pct,
                    f"Decomposing time step {t_idx}/{n_timesteps}...")

        if progress_callback:
            progress_callback(75, "Running surface evolution...")

        # Surface evolution
        solver = SurfaceEvolutionSolver(
            f=self.friction_coeff,
            grid_nx=self.grid_nx,
            grid_ny=self.grid_ny,
        )

        def evo_progress(pct, msg):
            if progress_callback:
                progress_callback(75 + int(pct * 0.2), msg)

        evo_result = solver.run_with_grid_spacing(
            h_initial=h_initial,
            vx_series=vx_series,
            vy_series=vy_series,
            dt_series=dt_series,
            dx=dx, dy=dy,
            interior_masks=interior_masks,
            progress_callback=evo_progress,
        )

        # Depth estimation from moving area using selected empirical formula
        mean_area_m2 = np.mean(evo_result['area_series']) if evo_result['area_series'] else 0
        depth_est, formula_info = estimate_depth_from_area(mean_area_m2, self.depth_formula)

        # Legacy thickness estimates kept for backward compatibility
        D_est, W_est = estimate_thickness_from_area(mean_area_m2)

        if progress_callback:
            progress_callback(98, "Preparing results...")

        results = {
            'h_initial': h_initial,
            'h_final': evo_result['h_final'],
            'thickness_change': evo_result['thickness_change'],
            'thickness_evolution': evo_result['thickness_evolution'],
            'area_series': evo_result['area_series'],
            'moving_area_km2': evo_result['moving_area_km2'],
            'depth_estimate': depth_est,
            'depth_formula_info': formula_info,
            'thickness_circular': D_est,
            'thickness_unknown_geom': W_est,
            'x': x, 'y': y, 'dx': dx, 'dy': dy,
            'last_vx': vx_series[-1] if vx_series else None,
            'last_vy': vy_series[-1] if vy_series else None,
            'n_timesteps': len(dt_series),
        }

        if progress_callback:
            progress_callback(100, "Done!")

        QgsMessageLog.logMessage(
            f"InSAR decomposition complete: {len(dt_series)} time steps, "
            f"depth estimate ({formula_info['label']}) = {depth_est:.2f} m, "
            f"moving area = {evo_result['moving_area_km2']:.3f} km²",
            "TestPlugin", Qgis.Info)

        return results
