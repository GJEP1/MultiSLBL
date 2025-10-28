# MultiSLBL Script: Formulas and Parameters

## Core Mathematical Formulas

### 1. **Standard SLBL (Failure Mode)**
```
z_temp = mean(neighbours) - C
```
- Updates the elevation if `z_temp < z_prev` (lowering terrain)
- Simulates mass failure/erosion

### 2. **Inverse SLBL (Filling Mode)**
```
z_temp = mean(neighbours) + C
```
- Updates the elevation if `z_temp > z_prev` (raising terrain)
- Simulates deposition/filling

### 3. **Tolerance C from e/Lrh Method**
```
C = (4 × e / Lrh) × (Δx²)
```
Where:
- `e` = E_RATIO (erosion/deposition rate parameter)
- `Lrh` = Runout horizontal length (calculated from polygon geometry)
- `Δx` = Pixel size (cell resolution in meters)

### 4. **Tolerance C from Area-Shape Method**
```
C = 4 × k × e × sqrt(A) × (Δx²)
```
Where:
- `k` = SHAPE_K (shape factor)
- `e` = E_RATIO (erosion/deposition rate parameter)
- `A` = Polygon area (in square meters)
- `Δx` = Pixel size (cell resolution in meters)

---

## Parameter Reference

### **File Paths & Directories**

| Parameter | Type | Description |
|-----------|------|-------------|
| `DEM_PATH` | string | Path to input Digital Elevation Model (DEM) raster file |
| `SCENARIO_DIR` | string | Directory containing scenario polygon shapefiles |
| `OUT_DIR` | string | Directory for output rasters and results |
| `CSV_SUMMARY` | string | Path for summary CSV output file |
| `TARGET_EPSG` | integer | Target coordinate reference system (e.g., 25832 for UTM Zone 32N) |

---

### **SLBL Mode Selection**

| Parameter | Values | Description |
|-----------|--------|-------------|
| `SLBL_MODE` | `"failure"` or `"inverse"` | **"failure"**: Lowers terrain (erosion/mass failure)<br>**"inverse"**: Raises terrain (deposition/filling) |

---

### **Tolerance Calculation Method**

| Parameter | Values | Description |
|-----------|--------|-------------|
| `C_FROM` | `"manual"`, `"e_Lrh"`, `"area_shape"` or list | **Manual**: Specify tolerance directly in TOLERANCES<br>**e_Lrh**: Calculate C using formula #3<br>**area_shape**: Calculate C using formula #4<br>Can be a list like `["manual", "e_Lrh"]` to run multiple methods |

---

### **Tolerance Sweep Parameters**

#### For `C_FROM = "manual"`:
| Parameter | Type | Effect |
|-----------|------|--------|
| `TOLERANCES` | list of floats | Directly specifies C value in meters. **Larger values** = more aggressive terrain modification per iteration. Example: `[0.001, 0.002, 0.005, 0.01]` |

#### For `C_FROM = "e_Lrh"`:
| Parameter | Type | Effect |
|-----------|------|--------|
| `E_RATIOS` | list of floats | Controls erosion/deposition rate relative to runout length. **Larger values** = more aggressive modification. Typical range: 0.0001 to 0.10. Example: `[0.0001, 0.0002]` |

#### For `C_FROM = "area_shape"`:
| Parameter | Type | Effect |
|-----------|------|--------|
| `E_RATIOS` | list of floats | Erosion/deposition rate parameter. **Larger values** = more modification |
| `SHAPE_KS` | list of floats | Shape factor controlling how polygon geometry affects tolerance. **Larger values** = stronger area influence. Example: `[0.05, 0.1]` |

**Note**: For `area_shape`, all combinations of E_RATIOS × SHAPE_KS are tested (Cartesian product).

---

### **Iteration Control**

| Parameter | Type | Effect |
|-----------|------|--------|
| `MAX_ITERS` | integer | Maximum number of SLBL iterations. **Higher values** allow more terrain modification but increase computation time. Default: 5000 |
| `STOP_EPS` | float | Convergence threshold for elevation change (meters). Process stops when max change per iteration falls below this. **Smaller values** = more precise results but slower. Default: 1e-3 (0.001 m) |
| `STOP_VOL_EPS` | float or None | Convergence threshold for volume change (m³). Process stops when volume change falls below this. `None` disables. |

---

### **Depth Limiting**

| Parameter | Type | Effect |
|-----------|------|--------|
| `MAX_DEPTHS` | list | Maximum allowed deposition/erosion depth in meters. `[None]` = no limit. Example: `[None, 5.0, 10.0]` tests unlimited, 5m max, and 10m max depths |

---

### **Neighborhood Configuration**

| Parameter | Type | Effect |
|-----------|------|--------|
| `NEIGHBOURS` | integer | Number of neighbor cells to consider. **4** = orthogonal only (N,S,E,W). **8** = includes diagonals (recommended). |

---

### **Feathering Parameters**

| Parameter | Type | Effect |
|-----------|------|--------|
| `FEATHER_TOL_K` | list of integers | Distance in pixels for feathering tolerance near polygon edges. **Larger values** = smoother transitions, more gradual edge effects. `[0]` disables. Example: `[50, 100]` |

---

### **Gradation Parameters**

| Parameter | Values | Effect |
|-----------|--------|--------|
| `GRADATION_ENABLE` | boolean | Enable spatially-varying tolerance within polygon |
| `GRADATION_SPLIT_METHOD` | `"elev"` or `"y"` | **"elev"**: Split by elevation quantile<br>**"y"**: Split by northing coordinate |
| `GRADATION_ELEV_Q` | float (0-1) | Quantile for elevation-based split. Default: 0.50 (median) |
| `GRADATION_K_PAIRS` | list of tuples or None | Explicit (upper, lower) K pairs for gradation. Example: `[(50,100), (50,150)]`. If `None`, auto-generates from `FEATHER_TOL_K` |

**Effect**: Upper zone uses smaller K (stronger modification), lower zone uses larger K (gentler modification).

---

### **Constraints**

| Parameter | Type | Effect |
|-----------|------|--------|
| `FIXED_PATH` | string or None | Path to shapefile of fixed (unchangeable) areas. `None` = no constraints |
| `ALL_TOUCHED` | boolean | If `True`, rasterizes polygons using all touched pixels (more inclusive). If `False`, only fully covered pixels |
| `BUFFER_PIXELS` | integer | Number of pixels to buffer polygon outward. **Larger values** expand the affected area |

---

### **Floor Constraints**

| Parameter | Type | Effect |
|-----------|------|--------|
| `USE_Z_FLOOR` | float or False | Minimum elevation limit (meters). Prevents erosion below this absolute elevation. `False` disables |
| `USE_MINMAX` | boolean | Keep `False` (legacy parameter) |

---

### **Performance & Hardware**

| Parameter | Type | Effect |
|-----------|------|--------|
| `USE_GPU` | boolean | Enable GPU acceleration with CuPy. **True** = much faster for large rasters if CUDA GPU available |
| `N_PROCESSES` | integer | Number of CPU processes for parallel scenario processing. GPU mode forces this to 1. Higher values process multiple scenarios simultaneously |
| `MP_CHUNKSIZE` | integer | Number of jobs per process chunk. Affects load balancing. Typically 1-5 |

---

### **Output Control**

| Parameter | Type | Effect |
|-----------|------|--------|
| `WRITE_RASTERS` | boolean | Write output GeoTIFF rasters (base elevation, thickness) |
| `WRITE_ASCII` | boolean | Write ASCII grid format (.asc) for thickness rasters (for AvaFrame compatibility) |
| `WRITE_XSECTIONS` | boolean | Generate cross-section CSV profiles |
| `XSECT_LINES_SOURCE` | list of paths | Paths to shapefile(s) or directories containing cross-section line geometries |
| `XSECT_STEP_M` | float or None | Sample interval along cross-section lines in meters. `None` uses pixel resolution |
| `XSECT_CLIP_TO_POLY` | boolean | If `True`, only sample cross-sections within scenario polygon |
| `XSECT_DIRNAME` | string | Subdirectory name for cross-section outputs |

---

### **Debug Options**

| Parameter | Type | Effect |
|-----------|------|--------|
| `DEBUG_WRITE_TOL_RASTER` | boolean | Write tolerance (C) field as separate raster for inspection |
| `DEBUG_REPORT_CLAMP` | boolean | Print statistics about depth clamping if `MAX_DEPTHS` is active |

---

## How Parameters Affect Results

### **Increasing Tolerance (C)**
- **Direct increase** (manual TOLERANCES): More aggressive per-iteration change
- **E_RATIOS increase**: Stronger erosion/deposition relative to geometry
- **SHAPE_K increase**: Amplifies area-based tolerance
- **Result**: Faster convergence, deeper/higher final terrain changes

### **Feathering (FEATHER_TOL_K)**
- **Larger K**: Smoother edge transitions, tolerance gradually decreases toward polygon boundary
- **Smaller K**: Sharper transitions, uniform tolerance until near edge
- **Result**: Affects realism of avalanche/landslide deposition edges

### **Gradation**
- Creates spatially-varying tolerance (upper zone vs. lower zone)
- **Upper zone (smaller K)**: More deposition/erosion
- **Lower zone (larger K)**: Less modification
- **Result**: Mimics natural processes where upper slopes contribute more material

### **Iteration Limits**
- **MAX_ITERS**: Safety valve; increase if process terminates before convergence
- **STOP_EPS**: Quality vs. speed tradeoff; smaller = more precise
- **Result**: Balance between accuracy and computation time

### **Depth Limits (MAX_DEPTHS)**
- Caps maximum local change from original DEM
- **Result**: Prevents unrealistic deep pits or tall mounds; ensures physical plausibility

---

## Typical Parameter Combinations

### **Aggressive Run (Fast, Exploratory)**
```python
TOLERANCES = [0.001,0.02]
E_RATIOS = [1,5]
MAX_ITERS = 2000
STOP_EPS = 1e-4
FEATHER_TOL_K = [0,150]
```

### **Parameter Sweep (Comprehensive)**
```python
C_FROM = ["manual", "e_Lrh", "area_shape"]
TOLERANCES = [0.001, 0.002, 0.005, 0.01]
E_RATIOS = [1,2,3,4,5]
SHAPE_KS = [0.05, 0.1, 0.5, 1.0]
FEATHER_TOL_K = [0, 100, 200]
MAX_DEPTHS = [None, 5.0, 10.0, 20.0]
MAX_ITERS = 2000
```

### **Conservative Run (Slow, Precise)**
```python
TOLERANCES = [0.001,0.002]
E_RATIOS = [2,3,4]
MAX_ITERS = 5000
STOP_EPS = 1e-7
FEATHER_TOL_K = [100,150]
```

*Warning: Cartesian product creates many scenarios!*

---

## References

Formulas from: [Surface Level Backfill (SLBL) Paper](https://esurf.copernicus.org/articles/7/439/2019/)
