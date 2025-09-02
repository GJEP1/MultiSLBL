MultiSLBL

Batch processing of SLBL thickness for one or more scenarios with optional GPU acceleration.
The engine auto-detects an NVIDIA GPU via CuPy and falls back to fast NumPy on CPU if a GPU isn’t available.

The SLBL Scenario Viewer (Streamlit) lets you explore results: map + cross-lines, oriented cross-sections, and per-scenario P10/P50/P90 tables.

Quick start
Create a clean conda env (Windows/Linux)
conda create -n slbl_env python=3.11 -y
conda activate slbl_env

Core geo stack (use conda-forge on Windows)
conda install -c conda-forge gdal rasterio geopandas shapely pyproj rtree -y

Python libs
pip install numpy pandas scipy matplotlib streamlit

GPU (optional but recommended): CUDA 12.x + CuPy
pip install cupy-cuda12x nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12

(Windows) Make sure CUDA is on PATH (adjust version if needed)
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
setx PATH "%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

Verify GPU from the shell (should end with "GPU OK. Sum: 96.0")
python -c "import cupy as cp, cupyx.scipy.ndimage as nd; cp.show_config(); \
x=cp.arange(9,dtype=cp.float32).reshape(3,3); k=cp.array([[0,1,0],[1,0,1],[0,1,0]],dtype=cp.float32); \
y=nd.convolve(x,k,mode='constant',cval=0); print('GPU OK. Sum:', float(y.sum().get()))"


If the check fails, see Troubleshooting (GPU) below.

Repository layout (suggested)
MultiSLBL/
├─ engine/                 # batch solver (GPU/CPU)
│  └─ SLBL_batch_cuda.py   # <ENGINE_PY> — main script (file name may differ in your repo)
├─ viewer/
│  └─ SLBL_viewer.py       # Streamlit viewer (v4.4+)
├─ scenarios/              # one .shp per scenario (polygons, EPSG as your DEM)
├─ xsections/              # optional cross-section polylines (.shp)
├─ outputs/
│  ├─ slbl_summary.csv     # appended/overwritten by the engine
│  ├─ slbl_scenario_stats.csv
│  └─ xsections/<Scenario>/<Scenario>_<Label>_xsect_<LineID>_pN.csv
└─ README.md

What the engine does
Computes SLBL base and thickness inside each scenario polygon using an iterative neighbour scheme.
Feathering K (in pixels) grows the tolerance from the polygon edge toward the interior.
Gradated K: optionally use a smaller K in the “upper” part and a larger K in the “lower” part (split by DEM elevation quantile or by rows).

Exports:
slbl_summary.csv (one row per run/label)
slbl_scenario_stats.csv (per-scenario quantiles)
optional GeoTIFFs (base/thickness)
optional cross-section CSVs sampled along polylines

The engine auto-uses GPU when CuPy is usable; otherwise it runs on CPU.

What the viewer does
Small map tile of the scenario with subtle line IDs and W/E or S/N arrows.
Cross-sections oriented to match the map (W→E or S→N) with shaded thickness.
Inclusion manager in the sidebar: pick which parameter sets count toward P10/P50/P90.
Smoothing slider (rolling mean).
Downloadable percentile table (volumes in million m³; depths in meters).

Running the engine
Open engine/<ENGINE_PY> (file name in this repo may be SLBL_batch_cuda.py or similar) and set the user paths near the top:

DEM_PATH       = r"D:\Python\SLBL\path\dtm.tif"
SCENARIO_DIR   = r"D:\Python\SLBL\path\scenarios"
OUT_DIR        = r"D:\Python\SLBL\path\outputs"
CSV_SUMMARY    = r"D:\Python\SLBL\path\outputs\slbl_summary.csv"
TARGET_EPSG    = must match DEM

Tune run parameters (examples shown in the script):
TOLERANCES: e.g. [0.01, 0.02, 0.03, 0.04, 0.05] (meters)
MAX_DEPTHS: [None] or [40, 60]
NEIGHBOURS: 4 or 8
Feathering: FEATHER_TOL_K = [50, 100, 150, 200]

Gradation:
GRADATION_ENABLE       = True
GRADATION_SPLIT_METHOD = "elev"   # or "y"
GRADATION_ELEV_Q       = 0.50     # median split
GRADATION_K_PAIRS      = None     # auto-generate (upper < lower)

Cross-sections:
WRITE_XSECTIONS    = True
XSECT_LINES_SOURCE = [r"D:\Python\SLBL\Skarfjell\xsections"]
XSECT_STEP_M       = None  # defaults to DEM pixel

Run it:
conda activate slbl_cf
python engine/SLBL_batch_cuda.py   # or the actual filename in your repo

Outputs land in outputs/.
The label naming is: tolXXcm|tolYY.YYm + nolimit|mdNNm + fkKpx or fkK1toK2pxg (g = gradated).

Running the viewer
Make sure the summary and x-sections from the engine exist.

Start the Streamlit app:
conda activate slbl_cf
streamlit run viewer/SLBL_viewer.py

In the sidebar:
Summary CSV → point to outputs/slbl_summary.csv
Scenario shapefile folder → your scenarios/
X-sections root → outputs/xsections/
Optional: turn smoothing on, adjust window (odd number), and show map tile.
Use the Inclusion manager to include/exclude parameter sets that should count toward the percentile table.
Download the percentile CSV from the bottom table.

GPU notes
Driver: NVIDIA R552+ (supports CUDA 12.6 at the time of writing).
Toolkit (optional): CUDA 12.6 from NVIDIA.

Python packages:
cupy-cuda12x (>= 13)
nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12 (ensure CUDA DLLs are present)

The engine automatically tries GPU first and logs e.g.
[GPU] CuPy available — using GPU kernels or a clear CPU fallback message.

Performance tips
GPU: biggest win for large rasters; keep everything float32/float64 (as in the code).
STOP_EPS: looser threshold converges quicker; default 1e-4 is a good balance.
NEIGHBOURS: 4 is faster; 8 is slightly smoother but more work per iteration.
WRITE_RASTERS: set False during tuning; writing tiled GeoTIFFs can dominate wall time.
N_PROCESSES: the script uses multiprocessing over scenarios × labels; if you’re already on GPU, you often don’t need very many processes. Start with the default and reduce if you see pressure on VRAM or CPU.

Troubleshooting (GPU)
CuPy failed to load cudart64_12.dll
Install CUDA runtime wheels and ensure CUDA is on PATH:
pip install -U nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
setx PATH "%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

cupyx.scipy.ndimage missing
Upgrade CuPy: pip install -U cupy-cuda12x (you want v13+).

Viewer can’t find shapefile CRS / DEM EPSG
Reproject to the engine’s TARGET_EPSG (e.g., 25833). DEM and shapefiles must match.

Very slow / stuck
Ensure GPU verification passes; otherwise you’re on CPU. Also try turning off raster writes and reduce the number of concurrent processes.

Inputs & assumptions
DEM: single-band float raster with valid EPSG set; projected CRS in meters (e.g., EPSG:25833).
Scenarios: one polygon .shp per scenario; CRS matches DEM. Multi-polygons are handled (dissolve).
Cross-sections (optional): polyline .shp (LineString/MultiLineString); CRS matches DEM.

Outputs

slbl_summary.csv (per run/label):
Scenario, Label, Tol_m, Feather_Gradated, Feather_K_px, Feather_K_upper_px, Feather_K_lower_px,
MaxDepth_m, MeanDepth_m, Depth_P10_m, Depth_P50_m, Depth_P90_m,
Volume_m3, MaskArea_m2, XSection_CSVs, Error (if any)

slbl_scenario_stats.csv (per scenario):
N_runs, volume and depth **min/mean/p10/p50/p90/max`
outputs/xsections/<Scenario>/*xsect*.csv:
Distance_m, X, Y, DEM_z, Base_z, Thickness_m, LineID, LinePart

Viewer niceties
Map tile shows line numbers at the arrow head and subtle W/E or S/N markers.
Cross-sections are auto-oriented to match the map (W→E for E–W lines, S→N for N–S).
Percentile table shows volumes in million m³ (raw m³ removed for clarity).
