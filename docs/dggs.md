# DGGS (Discrete Global Grid System) Support

This document describes how DGGS cubes are represented and which processes
support them in `openeo-processes-dask-slim`.

The implementation follows **option 3** from the
[DGGS discussion](https://github.com/Open-EO/openeo-processes/pull/562):
a dedicated DGGS dimension type that preserves cell topology, hierarchy,
and resolution semantics.

**HEALPix** is the concrete grid system used for validation, but the
interface is DGGS-generic wherever possible.

## Representation

A DGGS data cube is an `xr.Dataset` with a cell ID dimension instead of
planar `x`/`y` dimensions.

### Minimal Example

```python
Dimensions:
  t: 3
  healpix_index: 48

Coordinates:
  t                (t) datetime64[ns]
  healpix_index    (healpix_index) int64
  lat              (healpix_index) float64
  lon              (healpix_index) float64

Data variables:
  band_0           (t, healpix_index) float64

Attributes:
  dggs_grid_system:    "healpix"
  dggs_resolution:     2
  dggs_cell_id_dim:    "healpix_index"
  dggs_order:          "ring"
  crs:                 "healpix:2"
```

### Metadata Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `dggs_grid_system` | Grid system identifier | `"healpix"`, `"h3"`, `"s2"` |
| `dggs_resolution` | Resolution level | `1024` (HEALPix nside) |
| `dggs_cell_id_dim` | Name of the cell ID dimension | `"healpix_index"` |
| `dggs_order` | Cell ordering scheme | `"ring"`, `"nested"` |
| `crs` | CRS identifying grid + resolution | `"healpix:1024"` |

### Legacy HEALPix Detection

Cubes with only `healpix_index` dimension, `crs="healpix:*"`, and
`healpix_nside` attribute are also detected as DGGS cubes automatically.

## Detection Helpers

All in `openeo_processes_dask_slim.process_implementations.cubes.dggs`:

| Function | Returns |
|----------|---------|
| `is_dggs_cube(data)` | `True` if the cube is a DGGS cube |
| `get_dggs_dim(data)` | Name of the cell ID dimension, or `None` |
| `get_dggs_grid_system(data)` | Grid system name, or `None` |
| `get_dggs_resolution(data)` | Resolution level, or `None` |
| `get_dggs_order(data)` | Ordering scheme, or `None` |
| `get_dggs_crs(data)` | CRS string, or `None` |
| `require_dggs_cube(data)` | Cell dim name, raises `DimensionNotAvailable` if not DGGS |

## Accessor Behavior

For DGGS cubes, the `data.openeo` accessor:

- `data.openeo.spatial_dims` — returns `("healpix_index",)` (or the configured cell dim)
- `data.openeo.x_dim` — returns `None`
- `data.openeo.y_dim` — returns `None`

Planar `x`/`y` cubes are unaffected.

## Supported Processes

### Standard Processes (DGGS-Extended)

These standard openEO processes automatically detect DGGS cubes and use
lat/lon cell-center semantics instead of planar slicing.

| Process | DGGS Behavior |
|---------|---------------|
| `filter_bbox` | Selects cells whose center falls within the bounding box. Handles EPSG:4326 reprojection and antimeridian wrapping. |
| `filter_spatial` | Selects cells whose center falls within GeoJSON polygon(s). Supports FeatureCollection, Feature, Polygon. |
| `reduce_spatial` | Reduces over the cell ID dimension (e.g. `healpix_index`). Works with any reducer (mean, sum, min, max, etc). |
| `aggregate_spatial` | Aggregates cell values per geometry, returning a Dataset with a `geometry` dimension. Supports FeatureCollection (one entry per feature). |
| `mask_polygon` | Replaces cells outside a polygon with nodata. Supports `inside` flag and custom `replacement`. |

### Extension Processes

These DGGS-specific extension processes are available as proposals.

#### `filter_dggs`

Select DGGS cells by cell ID or parent cell ID.

```
filter_dggs(data, cells=None, parent_cells=None, nside=None, order=None)
```

- `cells` — explicit cell IDs at the cube resolution
- `parent_cells` — select all children of coarse parent cells
- `nside` — parent cell resolution (defaults to cube resolution)
- `order` — ordering scheme (defaults to cube order; nested recommended for parent queries)

#### `resample_dggs`

Change DGGS resolution (downsampling only).

```
resample_dggs(data, resolution, order=None, reducer="mean")
```

- `resolution` — target coarser nside
- `order` — output ordering (defaults to cube order)
- `reducer` — aggregation: `mean`, `sum`, `min`, `max`, `count`, `mode`

#### `dggs_to_raster`

Convert DGGS cube to a regular `x`/`y` projected grid.

```
dggs_to_raster(data, projection=4326, resolution=None, bbox=None, method="nearest")
```

- `projection` — target EPSG code (default: 4326)
- `resolution` — grid resolution in projection units
- `bbox` — bounding box dict with `west`, `east`, `south`, `north`
- `method` — resampling method (only `"nearest"` in v1)

#### `apply_neighborhood_dggs`

Apply a process over HEALPix k-ring neighborhoods.

```
apply_neighborhood_dggs(data, process, k=None, radius=None, include_center=True, context=None)
```

- `k` — graph-distance neighbor rings (1–4)
- `radius` — angular radius in degrees (not yet implemented)
- `include_center` — include the central pixel (default: `True`)
- `context` — additional data passed to the process

#### `mask_polygon_dggs`

Apply a polygon mask to a DGGS cube.

```
mask_polygon_dggs(data, mask, replacement=None, inside=False)
```

- `mask` — GeoJSON polygon/multipolygon
- `replacement` — nodata replacement value (default: `NaN`)
- `inside` — if `True`, mask the *inside* instead of the outside

## Cell Inclusion Rule

All spatial selection processes use **center-point containment** (v1 default):
a cell is included if its center (`lat`, `lon`) intersects the geometry.
Exact HEALPix cell polygon intersection is not yet implemented.

## Known v1 Limitations

- Geometry selection uses **cell centers** only; exact cell-polygon intersection is deferred.
- `resample_dggs` supports **downsampling only**; no upsampling in v1.
- `apply_neighborhood_dggs` supports **bounded small neighborhoods** (k ≤ 4).
- `apply_neighborhood_dggs` with **angular radius** is not yet implemented.
- `aggregate_spatial` for planar `x`/`y` cubes is not yet implemented.
- `save_result` is out of scope.
- `resample_dggs` parent-child resolution change gives correct results for
  **nested** ordering. Ring ordering is supported via `ring_to_nested`
  conversion but may be less intuitive.

## Testing

88 DGGS-specific tests are in `tests/test_dggs*.py`:

| File | Tests |
|------|-------|
| `test_dggs.py` | 24 — fixtures, detection, helpers, accessor |
| `test_filter_dggs.py` | 17 — filter_bbox, filter_spatial, filter_dggs |
| `test_reduce_dggs.py` | 12 — reduce_spatial, aggregate_spatial |
| `test_resample_dggs.py` | 18 — resample_dggs, dggs_to_raster |
| `test_neighborhood_dggs.py` | 16 — apply_neighborhood_dggs, mask_polygon_dggs |
| `test_dggs_laziness.py` | 10 — systematic dask laziness checks |

An optional credential-gated integration test is in
`tests/test_dggs_integration.py` (skipped by default; requires Icechunk
STAC credentials).

## Dependencies

Add `astropy-healpix` for DGGS support:

```
pip install openeo-processes-dask-slim[dggs]
```

The `dggs` extra installs `astropy-healpix`, which covers all needed
HEALPix operations: pixel↔lat/lon, ring↔nested, neighbours, boundaries,
and pixel area/resolution. No other DGGS libraries are required.

## Upstream Alignment

This implementation provides concrete service-provider feedback for the
[openEO DGGS discussion](https://github.com/Open-EO/openeo-processes/pull/562).
When contributing process specifications upstream, names should be
generalized to DGGS-level concepts:

| Here | Upstream suggestion |
|------|-------------------|
| `filter_dggs` | DGGS cell filtering |
| `resample_dggs` | DGGS resolution change |
| `dggs_to_raster` | DGGS-to-raster conversion |
| `apply_neighborhood_dggs` | DGGS topology-aware neighborhood processing |
