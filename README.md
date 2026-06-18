# 3D GemPy Model Builder — QGIS Plugin

A QGIS plugin for building 3D geological models with [GemPy](https://www.gempy.org/) and exporting results as 3D meshes or GeoTIFFs.

---

## Features

- **3D Geological Modelling** — Import surface points, orientations and a DTM to compute an implicit 3D model via GemPy
- **Structural Groups** — Define formation stacking relations (ERODE, ONLAP, FAULT, BASEMENT)
- **Interactive 3D Viewer** — Visualize the model in PyVista with opacity control and surface legend
- **Mesh Export** — Export geological surfaces as STL, VTK, OBJ or PLY
- **GeoTIFF Export** — Export a 2D geological map slice from the 3D model

---

## Requirements

### QGIS

QGIS 3.0 or later — <https://qgis.org/download/>

### Python

> [!WARNING]
> The plugin runs inside QGIS's bundled Python environment. All packages must be installed into **that** environment, not a separate system Python. Installing into your OS Python will have no effect.

#### Find the QGIS Python executable

Open the QGIS Python Console (**Plugins → Python Console**) and run:

```python
import sys
print(sys.executable)
```

This prints a path such as `C:\Program Files\QGIS 3.x\apps\Python312\python.exe`. Use that exact path in the commands below (replace `<qgis-python>`).

---

## Installation

### 1. Install the plugin

Copy (or clone) this repository into your QGIS plugins folder:

```
# Windows
%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\3D_gempy_model_builder

# Linux / macOS
~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/3D_gempy_model_builder
```

Then enable it in QGIS via **Plugins → Manage and Install Plugins → Installed → GemPy Model Builder**.

### 2. Install Python dependencies

> [!IMPORTANT]
> On Windows, open the terminal **as Administrator**. On Linux/macOS, you may need `sudo` or write access to the QGIS Python directory.

#### Required packages

```bash
<qgis-python> -m pip install numpy scipy rasterio pandas gempy gempy-viewer pyvista pyvistaqt numpy-stl pymeshfix pymeshlab
```

#### Example (Windows, QGIS 3.38)

```bat
"C:\Program Files\QGIS 3.38\apps\Python312\python.exe" -m pip install numpy scipy rasterio pandas gempy gempy-viewer pyvista pyvistaqt numpy-stl pymeshfix pymeshlab
```

### 3. Restart QGIS

> [!IMPORTANT]
> After installing packages, restart QGIS so the new libraries are visible to the plugin.

---

## Dependency Reference

### Required packages

> [!WARNING]
> The six packages below are **hard required** — they are imported at module level with no fallback. The plugin will crash immediately on startup or at model-compute time if any of them are missing.

| Package | pip install name | Why it is needed | If missing |
|---------|-----------------|------------------|------------|
| `numpy` | `numpy` | Foundation for all numerical operations: array math, coordinate transforms, mesh geometry, velocity field computations | **Plugin crashes** |
| `scipy` | `scipy` | Spatial indexing (`cKDTree`) for nearest-neighbour queries; `griddata` interpolation for surface reconstruction; `ndimage` morphological operations for raster clean-up | **Plugin crashes** |
| `pyvista` | `pyvista` | Builds and renders 3D meshes (geological surfaces, topography grid, exported STL/VTK/OBJ/PLY files) | **Plugin crashes** |
| `pyvistaqt` | `pyvistaqt` | Embeds the PyVista 3D viewport as a Qt widget inside the QGIS dialog | **Plugin crashes** |
| `gempy` | `gempy` | Core implicit geological modelling engine — interpolates formation surfaces from scattered input points using a co-kriging method | **Plugin crashes** at compute time |
| `gempy-viewer` | `gempy-viewer` | Fallback visualisation used when the primary PyVista view is unavailable; also used for GemPy result inspection | **Plugin crashes** at compute time |

### Packages with graceful fallback

> [!WARNING]
> These packages do **not** crash the plugin on startup, but missing them silently disables major functionality. Read the "If missing" column carefully before deciding to skip any of them.

| Package | pip install name | Why it is needed | If missing |
|---------|-----------------|------------------|------------|
| `rasterio` | `rasterio` | Reads DTM/DEM raster files (GeoTIFF) to clip the 3D geological model to the topographic surface; also writes the GeoTIFF geological map export | A warning is logged; DTM loading is skipped and the model has **no topography clipping**; GeoTIFF export is silently disabled |
| `pandas` | `pandas` | Reads and rewrites the geology / orientation CSV files during the GemPy import pipeline (AOI clipping, axis-alignment rotation, Arrow-string compatibility) | The **GemPy compute step raises an `ImportError`** at runtime; the plugin loads but cannot build a model |
| `numpy-stl` | `numpy-stl` | Reads and writes STL mesh files for the topography surface export | STL topography export is silently skipped; other mesh formats (VTK, OBJ, PLY) are unaffected |
| `pymeshfix` | `pymeshfix` | Watertight mesh repair (Marco Attene's MeshFix) applied to every surface before STL export — fixes self-intersections, non-manifold edges and holes that PyVista's `clean` alone cannot | A warning is logged; export falls back to PyVista clean/normals only, so meshes may still contain non-manifold edges |
| `pymeshlab` | `pymeshlab` | MeshLab's geometry-processing engine; removes folded / sliver faces along the topography boundary (`meshing_remove_folded_faces` + degenerate/non-manifold cleanup) before the watertight pymeshfix pass | A warning is logged; folded-face removal is skipped, so the topography box may keep folded boundary triangles (the pymeshfix watertight pass still runs) |

> [!WARNING]
> `rasterio` is particularly important. Without it the geological model is computed without a topographic boundary, which produces unrealistic results for any near-surface geology. Install it unless your model extent is entirely subsurface.

> [!NOTE]
> `pandas` missing does not crash the plugin on startup, but the GemPy compute step will raise an `ImportError` the first time it is used. Install it before attempting to build a model.

---

## Input Data Format

### Geology — Surface Points CSV

| X | Y | Z | formation |
|---|---|---|-----------|
| 500.0 | 1200.0 | 350.0 | sandstone |

### Geology — Orientations CSV

| X | Y | Z | azimuth | dip | polarity | formation |
|---|---|---|---------|-----|----------|-----------|
| 500.0 | 1200.0 | 350.0 | 45.0 | 30.0 | 1 | sandstone |

---

## Usage Overview

1. Open the plugin from the QGIS toolbar or **Plugins** menu.
2. **Phase 1 — Load Data**: select geology CSV files, an orientation CSV, and a DTM raster layer.
3. **Phase 2 — Structural Groups**: define formation stacking order and fault relationships.
4. **Phase 3 — Compute Model**: set grid resolution and extent, then click **Compute GemPy Model**.
5. **Phase 4 — Visualise & Export**: explore the 3D model in the PyVista viewer; export meshes or a GeoTIFF.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## References

- de la Varga, M., Schaaf, A., & Wellmann, F. (2019). *GemPy 1.0: open-source stochastic geological modeling and inversion*. Geoscientific Model Development, 12, 1–32.
