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
<qgis-python> -m pip install numpy scipy pandas shapely rasterio pyvista pyvistaqt gempy gempy-viewer numpy-stl
```

#### Example (Windows, QGIS 3.38)

```bat
"C:\Program Files\QGIS 3.38\apps\Python312\python.exe" -m pip install numpy scipy pandas shapely rasterio pyvista pyvistaqt gempy gempy-viewer numpy-stl
```

### 3. Restart QGIS

> [!IMPORTANT]
> After installing packages, restart QGIS so the new libraries are visible to the plugin.

---

## Dependency Reference

The plugin checks these packages when you open it. It lists all the missing or broken ones in one window, with the commands that fix them.

### Required packages

| Package | pip install name | Why it is needed |
|---------|-----------------|------------------|
| `numpy` | `numpy` | Numerical operations |
| `scipy` | `scipy` | Interpolation and spatial indexing |
| `pandas` | `pandas` | Reading and rewriting the input CSV files |
| `shapely` | `shapely` | AOI clipping and rotation |
| `rasterio` | `rasterio` | Reading the DTM, GeoTIFF export |
| `pyvista` | `pyvista` | 3D meshes and rendering |
| `pyvistaqt` | `pyvistaqt` | 3D viewer inside the dialog |
| `gempy` | `gempy` | Geological modelling engine |
| `gempy_viewer` | `gempy-viewer` | Model visualisation |

### Optional package

| Package | pip install name | Why it is needed | If missing |
|---------|-----------------|------------------|------------|
| `stl` | `numpy-stl` | Writing `Topography.stl` | The topography STL is skipped; everything else still works |

> [!WARNING]
> Install `numpy-stl`, **not** `stl`. The PyPI package `stl` is unrelated and overwrites files of `numpy-stl`, which causes `AttributeError: module 'stl' has no attribute 'BaseStl'`. To fix it:
>
> ```bash
> <qgis-python> -m pip uninstall -y stl numpy-stl
> <qgis-python> -m pip install numpy-stl
> ```

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
