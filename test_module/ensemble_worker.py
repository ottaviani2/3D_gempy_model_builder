# -*- coding: utf-8 -*-
"""
Self-contained worker for the probabilistic ensemble.

This module is imported both by the parent QGIS process and by the child
processes of the ``ProcessPoolExecutor`` used in
``test_module_dialog.EnsembleTask``.  It MUST NOT import qgis / Qt / pyvista:
the child processes start from a bare QGIS-Python interpreter and only have
numpy / pandas / gempy available.  Keeping the heavy GemPy import *inside* the
functions lets the module itself import cheaply (only numpy at top level), so a
spawned child pays the GemPy import cost once, not at module load.

A single realization is fully independent: it perturbs the orientation CSV with
its own seed, rebuilds and recomputes a GemPy model, and returns the integer
lithology block.  That independence is what makes the ensemble embarrassingly
parallel.
"""
import os
import tempfile

import numpy as np


# ---------------------------------------------------------------------------
# Orientation perturbation
# ---------------------------------------------------------------------------
def perturb_orientation_csv(or_path, dip_sigma, az_sigma, seed):
    """Write a temp copy of an orientation CSV with the dip and azimuth
    perturbed by Gaussian noise and return its path.

    ``seed`` may be anything ``numpy.random.default_rng`` accepts (an int or a
    ``SeedSequence``); passing a distinct per-realization seed keeps the whole
    ensemble reproducible regardless of the order in which workers finish.

    Handles both GemPy orientation conventions:
      * polar    — columns 'dip' and 'azimuth' (degrees)
      * gradient — columns 'G_x','G_y','G_z'; converted to dip/azimuth,
                   perturbed, then converted back (norm and polarity preserved).
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    df = pd.read_csv(or_path)
    lower = {c.lower(): c for c in df.columns}

    dip_col = lower.get('dip')
    az_col = lower.get('azimuth')
    gx_col = lower.get('g_x') or lower.get('gx')
    gy_col = lower.get('g_y') or lower.get('gy')
    gz_col = lower.get('g_z') or lower.get('gz')

    if dip_col is not None and az_col is not None:
        dip = df[dip_col].to_numpy(dtype=float)
        az = df[az_col].to_numpy(dtype=float)
        dip = np.clip(dip + rng.normal(0.0, dip_sigma, dip.shape), 0.0, 90.0)
        az = (az + rng.normal(0.0, az_sigma, az.shape)) % 360.0
        df[dip_col] = dip
        df[az_col] = az
    elif None not in (gx_col, gy_col, gz_col):
        gx = df[gx_col].to_numpy(dtype=float)
        gy = df[gy_col].to_numpy(dtype=float)
        gz = df[gz_col].to_numpy(dtype=float)
        norm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        norm[norm == 0.0] = 1.0
        # GemPy gradient convention:
        #   G = (sin(dip)sin(az), sin(dip)cos(az), cos(dip)) * polarity
        pol = np.sign(gz)
        pol[pol == 0.0] = 1.0
        ux, uy, uz = gx / norm, gy / norm, gz / norm
        dip = np.degrees(np.arccos(np.clip(np.abs(uz), -1.0, 1.0)))
        az = np.degrees(np.arctan2(ux, uy)) % 360.0
        dip = np.clip(dip + rng.normal(0.0, dip_sigma, dip.shape), 0.0, 90.0)
        az = (az + rng.normal(0.0, az_sigma, az.shape)) % 360.0
        dr = np.radians(dip)
        ar = np.radians(az)
        df[gx_col] = np.sin(dr) * np.sin(ar) * norm * pol
        df[gy_col] = np.sin(dr) * np.cos(ar) * norm * pol
        df[gz_col] = np.cos(dr) * norm * pol
    else:
        raise RuntimeError(
            "Orientation CSV has neither 'dip'/'azimuth' nor "
            "'G_x'/'G_y'/'G_z' columns — cannot perturb orientations.")

    fd, tmp = tempfile.mkstemp(suffix='_or_perturbed.csv')
    os.close(fd)
    df.to_csv(tmp, index=False)
    return tmp


# ---------------------------------------------------------------------------
# Lithology-block extraction (defensive across GemPy versions)
# ---------------------------------------------------------------------------
def extract_lith_block(geo_model):
    """Return the 1-D lithology-id block of a computed GemPy model, or None
    when it cannot be located."""
    sol = getattr(geo_model, 'solutions', None)
    if sol is None:
        return None
    for path in (('raw_arrays', 'lith_block'), ('lith_block',)):
        obj = sol
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok:
            try:
                arr = np.asarray(obj).ravel()
                if arr.size > 0:
                    return arr
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Build + compute one model
# ---------------------------------------------------------------------------
def build_and_compute_block(sp_path, or_path, extent, resolution,
                            groups_cfg, topo_xyz):
    """Build and compute a GemPy model with the given extent, resolution and
    structural-group layout but with the supplied (perturbed) input CSVs, and
    return its integer lithology block.

    This mirrors the deterministic build path one-to-one so a realization is
    identical to the main model except for the perturbed orientations.
    """
    import pandas as pd
    import gempy as gp
    from gempy_engine.core.data.stack_relation_type import StackRelationType
    from gempy.core.data.structural_group import FaultsRelationSpecialCase

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

    # Arrow-string compatibility shim — same as the deterministic path.
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
    try:
        geo_model = gp.create_geomodel(
            project_name='Ensemble',
            extent=extent,
            resolution=resolution,
            importer_helper=gp.data.ImporterHelper(
                path_to_orientations=or_path,
                path_to_surface_points=sp_path,
            )
        )
    finally:
        for key, val in pd_opts:
            try:
                pd.set_option(key, val)
            except (KeyError, pd.errors.OptionError):
                pass

    # Structural groups — identical layout to the deterministic run.
    default_group = geo_model.structural_frame.get_group_by_name(
        "default_formation")
    all_elements = {
        el.name: el
        for grp in geo_model.structural_frame.structural_groups
        for el in grp.elements
    }
    for el in list(default_group.elements):
        default_group.elements.remove(el)

    groups_created = 0
    for i, cfg in enumerate(groups_cfg):
        elements = [all_elements[fn] for fn in cfg['formations']
                    if fn in all_elements]
        if not elements:
            continue
        gp.add_structural_group(
            model=geo_model,
            group_index=i,
            structural_group_name=cfg['name'],
            elements=elements,
            structural_relation=relation_map[cfg['relation']],
            fault_relations=fault_rel_map[cfg['fault_relation']],
        )
        groups_created += 1

    if not default_group.elements and groups_created > 0:
        try:
            geo_model.structural_frame.structural_groups.remove(default_group)
        except ValueError:
            pass
    elif groups_created == 0:
        for el in all_elements.values():
            default_group.elements.append(el)

    if topo_xyz is not None:
        gp.set_topography_from_arrays(
            grid=geo_model.grid, xyz_vertices=topo_xyz)

    geo_model.input_transform.apply_anisotropy(gp.data.GlobalAnisotropy.NONE)
    gp.compute_model(geo_model)
    return extract_lith_block(geo_model)


# ---------------------------------------------------------------------------
# Top-level pool entry point
# ---------------------------------------------------------------------------
def run_realization(args):
    """Compute one ensemble realization.  This is the callable submitted to the
    process pool, so it must be importable by reference and fully self-contained.

    ``args`` is a tuple::

        (index, sp_path, or_path, extent, resolution, groups_cfg,
         topo_xyz, dip_sigma, az_sigma, seed)

    Returns ``(index, block_int32_or_None, error_string_or_None)``.  Errors are
    returned rather than raised so one bad realization never kills the pool.
    """
    (idx, sp_path, or_path, extent, resolution, groups_cfg, topo_xyz,
     dip_sigma, az_sigma, seed) = args
    tmp = None
    try:
        tmp = perturb_orientation_csv(or_path, dip_sigma, az_sigma, seed)
        block = build_and_compute_block(
            sp_path, tmp, extent, resolution, groups_cfg, topo_xyz)
        if block is None:
            return (idx, None, "lithology block not found")
        return (idx, np.round(block).astype(np.int32), None)
    except Exception as e:
        return (idx, None, str(e))
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass
