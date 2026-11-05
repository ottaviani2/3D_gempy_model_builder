# Test data

A small example dataset for trying the plugin. All layers use WGS 84 / UTM zone 30N (EPSG:32630).

| File | QGIS layer | Content |
|------|------------|---------|
| `layer_i.shp` | Geology | 74 surface points: formations `14`, `18`, `26`, `Fault_1`, `Fault_2` |
| `layer_o.shp` | Orientation | 20 orientation measurements with azimuth, dip and polarity |
| `DTM_30_Clip_Filled_Model.tif` | DTM | Elevation raster, 78 x 70 cells of about 27.5 m |

## How to try it

1. Download this folder and keep the files with the same name together.
2. Add `layer_i.shp`, `layer_o.shp` and the DTM to a QGIS project.
3. Open the plugin and select the three layers.
4. Keep the automatic structural groups: `Fault_1` and `Fault_2` become faults, and the other formations form one group.
5. Set a low resolution such as 20 x 20 x 20 and a model depth such as 500 m, then compute the model.

These values are only a starting point for a quick test.

## Things to know

- 24 of the 74 surface points and 5 of the 20 orientations fall outside the DTM extent.
- One point has `X` and `Y` attributes that differ from its geometry by about 44 m and 34 m, in both shapefiles. The plugin uses the attributes.
