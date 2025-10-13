#!/usr/bin/env python
# SLBL batch processing — CPU/GPU
#
# Notes (paper formulas) https://esurf.copernicus.org/articles/7/439/2019/:
#   1) z_temp = mean(neighbours) - C, update if z_temp < z_prev (failure mode)
#   2) Inverse SLBL: z_temp = mean(neighbours) + C, update if z_temp > z_prev
#   3) C = (4 * e / Lrh) * (Δx^2)
#   4) C = 4 * k * e * sqrt(A) * (Δx^2)   (inventory formulation)

import os, sys, re, traceback
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import rowcol
from shapely.geometry import box, LineString, MultiLineString
from shapely.ops import unary_union
from itertools import product
from multiprocessing import Pool, cpu_count
from pyproj import datadir as _proj_datadir

# ---------------------------
# Thread caps to avoid oversubscription (MP TUNING)
# ---------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# ---------------------------
# User parameters
# ---------------------------
DEM_PATH       = r"D:\Python\SLBL\yourpath\DEM.tif"
SCENARIO_DIR   = r"D:\Python\SLBL\yourpath\scenarios"
OUT_DIR        = r"D:\Python\SLBL\yourpath\outputs"
CSV_SUMMARY    = r"D:\Python\SLBL\yourpath\outputs\slbl_summary.csv"
TARGET_EPSG    = 25832

# --- SLBL mode & tolerance source ---
SLBL_MODE      = "failure"    # "failure" (lowering) or "inverse" (filling)

# C_FROM can be a string or a list of methods to sweep in one run.
# Allowed values: "manual", "e_Lrh", "area_shape"
# Examples:
#   C_FROM = "manual"
#   C_FROM = ["manual","e_Lrh","area_shape"]
C_FROM         = ["manual", "e_Lrh","area_shape"]

# --- Parameter sweeps ---
# For "e_Lrh" → sweep E (zmax/Lrh)
E_RATIOS       = [0.0001, 0.0002, 0.0003]
# For "area_shape" → sweep E and K together (Cartesian product)
SHAPE_KS       = [0.1, 0.2, 0.4]

# Back-compat (singletons still supported if you prefer):
E_RATIO        = 0.10     # used only if E_RATIOS is not a list
SHAPE_K        = 1.0      # used only if SHAPE_KS is not a list

# For "manual" → sweep tolerances (meters)
TOLERANCES     = [0.001, 0.002, 0.005, 0.01]

# Hard stops / neighbour rule
MAX_DEPTHS     = [None]
NEIGHBOURS     = 8
STOP_EPS       = 1e-3
STOP_VOL_EPS   = None               # e.g. 1.0 (m^3)
MAX_ITERS      = 5000

# Optional constraints
FIXED_PATH     = None
ALL_TOUCHED    = True
BUFFER_PIXELS  = 4

# Multiprocessing policy
N_PROCESSES    = 1
MP_CHUNKSIZE   = 2

WRITE_RASTERS   = True
WRITE_XSECTIONS = True
XSECT_LINES_SOURCE = [ r"D:\Python\SLBL\yourpath\xsections" ]
XSECT_STEP_M      = None
XSECT_CLIP_TO_POLY= False
XSECT_DIRNAME     = "xsections"

USE_Z_FLOOR    = False
USE_MINMAX     = False        # keep False

FEATHER_TOL_K  = [0, 50, 100]  # 0 disables feathering

# Gradation controls
GRADATION_ENABLE       = False
GRADATION_SPLIT_METHOD = "elev"       # "elev" or "y"
GRADATION_ELEV_Q       = 0.50
GRADATION_K_PAIRS      = None         # e.g. [(50,100),(50,150),(100,150)]

# GPU
USE_GPU = True

DEBUG_WRITE_TOL_RASTER = False
DEBUG_REPORT_CLAMP     = True

CSV_SCENARIO_STATS = os.path.join(os.path.dirname(CSV_SUMMARY), "slbl_scenario_stats.csv")

print("[ENV] Python:", sys.executable)
print("[ENV] pyproj data dir:", _proj_datadir.get_data_dir())

# ---------------------------
# Optional GPU backend (CuPy) — CUPY FIX
# ---------------------------
HAVE_CUPY = False
cp = None
try:
    if USE_GPU:
        import cupy as cp
        import cupyx.scipy.ndimage as cpx_nd
        HAVE_CUPY = (cp.cuda.runtime.getDeviceCount() > 0)
        if HAVE_CUPY:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            if isinstance(name, bytes): name = name.decode(errors="ignore")
            print(f"[GPU] Detected device: {name}")
        else:
            print("[GPU] No CUDA device detected")
except Exception as e:
    print(f"[GPU] CuPy not usable ({type(e).__name__}: {e})")
    HAVE_CUPY = False

# ---------------------------
# IO / CRS helpers
# ---------------------------
def read_dem(dem_path):
    src = rasterio.open(dem_path)
    if src.crs is None:
        raise RuntimeError("DEM has no CRS.")
    dem_epsg = src.crs.to_epsg()
    if dem_epsg != TARGET_EPSG:
        raise RuntimeError(f"DEM EPSG is {dem_epsg}, expected {TARGET_EPSG}.")
    dem = src.read(1, masked=True).astype("float64")
    prof = src.profile.copy()
    pixel_area = abs(src.transform.a) * abs(src.transform.e)
    dem_bounds = box(*src.bounds)
    print(f"[INFO] DEM pixel={abs(src.transform.a)} m, shape={dem.shape}")
    return src, dem, prof, pixel_area, dem_bounds, dem_epsg

def load_and_prepare_gdf(vec_path, target_epsg, pixel_size=None, buffer_pixels=0):
    gdf = gpd.read_file(vec_path)
    if gdf.empty: raise ValueError("Empty shapefile")
    if gdf.crs is None: raise ValueError("Shapefile CRS is undefined")
    if gdf.crs.to_epsg() != target_epsg:
        gdf = gdf.to_crs(epsg=target_epsg)
    if "geometry" in gdf and gdf.geom_type.isin(["Polygon","MultiPolygon"]).any():
        gdf["__diss__"] = 1
        gdf = gdf.dissolve(by="__diss__", as_index=False).drop(columns="__diss__", errors="ignore")
    if buffer_pixels and pixel_size and gdf.geom_type.isin(["Polygon","MultiPolygon"]).any():
        gdf["geometry"] = gdf.buffer(buffer_pixels * pixel_size)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)]
    if gdf.empty: raise ValueError("Geometry empty after cleaning/buffering.")
    return gdf

def load_lines_gdf(vec_path, target_epsg):
    if not vec_path: return None
    gdf = gpd.read_file(vec_path)
    if gdf.empty: raise ValueError("Cross-section file is empty.")
    if gdf.crs is None: raise ValueError("Cross-section file CRS is undefined.")
    if gdf.crs.to_epsg() != target_epsg:
        gdf = gdf.to_crs(epsg=target_epsg)
    gdf = gdf[gdf.geom_type.isin(["LineString","MultiLineString"])]
    if gdf.empty: raise ValueError("No LineString/MultiLineString geometries.")
    return gdf

def _collect_line_paths(source):
    if not source: return []
    items = list(source) if isinstance(source,(list,tuple)) else [source]
    out = []
    for item in items:
        if os.path.isdir(item):
            out += [os.path.join(item,f) for f in os.listdir(item) if f.lower().endswith(".shp")]
        elif str(item).lower().endswith(".shp"):
            out.append(item)
    return sorted(set(out))

def load_lines_many(source, target_epsg):
    paths = _collect_line_paths(source)
    if not paths: return None, 0
    frames = []
    for p in paths:
        try:
            gdf = load_lines_gdf(p, target_epsg)
            if not gdf.empty: frames.append(gdf)
        except Exception as e:
            print(f"[WARN] load {p}: {e}")
    if not frames: return None, 0
    merged = pd.concat(frames, ignore_index=True)
    return merged, len(paths)

# ---------------------------
# Rasterization / neighbours
# ---------------------------
def vector_to_mask(gdf, like_src, all_touched=True, burn_value=1):
    shapes = ((geom, burn_value) for geom in gdf.geometry if geom and not geom.is_empty)
    mask = features.rasterize(
        shapes=shapes,
        out_shape=(like_src.height, like_src.width),
        transform=like_src.transform,
        fill=0, all_touched=all_touched, dtype="uint8"
    ).astype(bool)
    return mask

# FAST CPU mean
try:
    from scipy.ndimage import convolve as _sp_convolve
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def _neigh_mean_cpu(arr, valid, mode8=False):
    if _HAVE_SCIPY:
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float32) if mode8 else \
                 np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.float32)
        a = np.where(valid & np.isfinite(arr), arr, 0.0).astype(np.float32)
        m = (valid & np.isfinite(arr)).astype(np.float32)
        s = _sp_convolve(a, kernel, mode="constant", cval=0.0)
        c = _sp_convolve(m, kernel, mode="constant", cval=0.0)
        out = np.divide(s, c, out=np.full_like(s, np.nan, dtype=np.float32), where=(c>0))
        return out.astype(np.float64)
    # fallback
    H, W = arr.shape
    s = np.zeros((H,W), dtype="float64")
    c = np.zeros((H,W), dtype="float64")
    shifts4 = [(-1,0),(1,0),(0,-1),(0,1)]
    shifts8 = shifts4 + [(-1,-1),(-1,1),(1,-1),(1,1)]
    shifts = shifts8 if mode8 else shifts4
    for dy, dx in shifts:
        y0s, y1s = max(0, dy), H + min(0, dy)
        x0s, x1s = max(0, dx), W + min(0, dx)
        y0d, y1d = max(0,-dy), H + min(0,-dy)
        x0d, x1d = max(0,-dx), W + min(0,-dx)
        if y0s>=y1s or x0s>=x1s: continue
        neigh = arr[y0s:y1s, x0s:x1s]
        vmask = valid[y0s:y1s, x0s:x1s] & np.isfinite(neigh)
        s[y0d:y1d, x0d:x1d] += np.where(vmask, neigh, 0.0)
        c[y0d:y1d, x0d:x1d] += vmask.astype("float64")
    out = np.divide(s, c, out=np.full_like(s, np.nan), where=(c>0))
    return out

def _neigh_mean_gpu(arr_g, valid_g, mode8=False):
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float32) if mode8 else \
             np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.float32)
    k = cp.asarray(kernel)
    arrv = cp.where(valid_g & cp.isfinite(arr_g), arr_g, 0.0)
    cntm = valid_g.astype(cp.float32)
    s = cpx_nd.convolve(arrv, k, mode="constant", cval=0.0)
    c = cpx_nd.convolve(cntm, k, mode="constant", cval=0.0)
    out = cp.where(c>0, s/c, cp.nan)
    return out

def neighbour_mean(arr, valid, mode8=False, use_gpu=False):
    if use_gpu and HAVE_CUPY:
        arr_g   = arr if isinstance(arr, cp.ndarray) else cp.asarray(arr)
        valid_g = valid if isinstance(valid, cp.ndarray) else cp.asarray(valid)
        return _neigh_mean_gpu(arr_g, valid_g, mode8)
    return _neigh_mean_cpu(arr, valid, mode8)

# ---------------------------
# SLBL utils (z-floor, gradation)
# ---------------------------
def compute_z_floor(inmask, dem, use_gpu=False):
    try:
        if use_gpu and HAVE_CUPY:
            ring = cpx_nd.binary_dilation(cp.asarray(inmask), iterations=1) & (~cp.asarray(inmask))
            arr = cp.asarray(dem.filled(np.nan))
            if cp.any(ring):
                return float(cp.nanmin(arr[ring]).get())
            return np.nan
        else:
            from scipy.ndimage import binary_dilation
            ring = binary_dilation(inmask, iterations=1) & (~inmask)
            arr = dem.filled(np.nan)
            if np.any(ring):
                return float(np.nanmin(arr[ring]))
            return np.nan
    except Exception as e:
        print("[WARN] compute_z_floor:", e)
        return np.nan

def build_upper_lower_masks(dem, inmask, method="elev", elev_q=0.5):
    arr = dem.filled(np.nan)
    if method == "elev":
        vals = arr[inmask]
        if vals.size and np.isfinite(vals).any():
            z_split = float(np.nanquantile(vals, float(elev_q)))
            upper = inmask & (arr >= z_split)
            lower = inmask & (arr <  z_split)
            return upper, lower, {"mode":"elev","z_split":z_split}
    H, W = inmask.shape
    r = np.arange(H).reshape(H,1)
    split = int(H*float(elev_q))
    upper = inmask & (r < split)
    lower = inmask & (r >= split)
    return upper, lower, {"mode":"y","row_split":split}

def _edt_cpu(mask):
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(mask.astype(np.uint8))

def _edt_gpu(mask_g):
    return cpx_nd.distance_transform_edt(mask_g)

def compute_feathered_tol(inmask, tol_scalar, K, use_gpu=False):
    H, W = inmask.shape
    tol_arr = np.full((H,W), float(tol_scalar), dtype="float64")
    if not K or K<=0: return tol_arr
    try:
        if use_gpu and HAVE_CUPY:
            dist = _edt_gpu(cp.asarray(inmask))
            w = cp.clip(dist/float(K), 0.0, 1.0)
            t = float(tol_scalar)*w
            t[~cp.asarray(inmask)] = float(tol_scalar)
            return cp.asnumpy(t)
        else:
            dist = _edt_cpu(inmask)
            w = np.clip(dist/float(K), 0.0, 1.0)
            tol_arr = tol_scalar * w
            tol_arr[~inmask] = tol_scalar
            return tol_arr
    except Exception as e:
        print("[WARN] feather:", e); return tol_arr

def compute_feathered_tol_gradated(inmask, tol_scalar, K_upper, K_lower, upper_mask, lower_mask, use_gpu=False):
    H, W = inmask.shape
    tol_arr = np.full((H,W), float(tol_scalar), dtype="float64")
    try:
        if use_gpu and HAVE_CUPY:
            du = _edt_gpu(cp.asarray(upper_mask))
            dl = _edt_gpu(cp.asarray(lower_mask))
            w = cp.zeros((H,W), dtype=cp.float32)
            if K_upper and K_upper>0: w[cp.asarray(upper_mask)] = du[cp.asarray(upper_mask)]/float(K_upper)
            if K_lower and K_lower>0: w[cp.asarray(lower_mask)] = dl[cp.asarray(lower_mask)]/float(K_lower)
            w = cp.clip(w, 0.0, 1.0)
            t = float(tol_scalar)*w
            t[~cp.asarray(inmask)] = float(tol_scalar)
            return cp.asnumpy(t)
        else:
            du = _edt_cpu(upper_mask)
            dl = _edt_cpu(lower_mask)
            w = np.zeros((H,W), dtype="float64")
            if K_upper and K_upper>0: w[upper_mask] = du[upper_mask]/float(K_upper)
            if K_lower and K_lower>0: w[lower_mask] = dl[lower_mask]/float(K_lower)
            w = np.clip(w, 0.0, 1.0)
            tol_arr = tol_scalar * w
            tol_arr[~inmask] = tol_scalar
            return tol_arr
    except Exception as e:
        print("[WARN] feather-grad:", e); return tol_arr

def save_gtiff(path, arr, profile, nodata=np.nan):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prof = profile.copy()
    prof.update(dtype="float32", count=1, compress="deflate", predictor=3,
                tiled=True, blockxsize=256, blockysize=256, nodata=nodata)
    try:
        with rasterio.open(path,"w",**prof) as dst:
            dst.write(arr.astype("float32"),1)
    except Exception:
        prof2 = profile.copy()
        prof2.update(dtype="float32", count=1, compress="deflate", predictor=3, tiled=False, nodata=nodata)
        with rasterio.open(path,"w",**prof2) as dst:
            dst.write(arr.astype("float32"),1)

def format_tol_label(tol_m):
    t = float(abs(tol_m)) if np.isscalar(tol_m) else float(np.nanmax(tol_m))
    return f"tol{t:.4f}m"

def format_md_label(max_depth):
    return f"md{int(max_depth)}m" if max_depth is not None else "nolimit"

def format_fk_label(K):
    if isinstance(K,(list,tuple)) and len(K)==2:
        return f"fk{int(K[0])}to{int(K[1])}pxg"
    return f"fk{int(K)}px" if (K is not None and K>0) else "fk0"

def _sanitize(s):
    if s is None: return "line"
    s = re.sub(r"\s+","_",str(s).strip())
    s = re.sub(r"[^A-Za-z0-9_\-]+","",s)
    return s or "line"

def _line_parts(geom):
    if isinstance(geom, LineString): yield geom
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            if isinstance(g, LineString): yield g

def _densify_line(line: LineString, step_m: float):
    L = line.length
    if L<=0 or (step_m or 0)<=0:
        return [(line.coords[0][0], line.coords[0][1], 0.0),
                (line.coords[-1][0],line.coords[-1][1], L)]
    n = max(1, int(np.floor(L/step_m)))
    dists = [i*step_m for i in range(n+1)]
    if dists[-1] < L: dists.append(L)
    pts = []
    for d in dists:
        p = line.interpolate(d); pts.append((p.x,p.y,d))
    return pts

def _sample_arrays(xy_dist_pts, transform, dem_arr, base_arr, thick_arr, inmask=None):
    xs = [x for x,_,_ in xy_dist_pts]; ys = [y for _,y,_ in xy_dist_pts]
    rows, cols = rowcol(transform, xs, ys, op=float)
    rows = np.rint(rows).astype(int); cols = np.rint(cols).astype(int)
    H, W = dem_arr.shape; out=[]
    for (x,y,d), r, c in zip(xy_dist_pts, rows, cols):
        if r<0 or c<0 or r>=H or c>=W: continue
        if inmask is not None and not inmask[r,c]: continue
        dz = float(dem_arr[r,c]) if np.isfinite(dem_arr[r,c]) else np.nan
        bz = float(base_arr[r,c]) if np.isfinite(base_arr[r,c]) else np.nan
        th = float(thick_arr[r,c]) if np.isfinite(thick_arr[r,c]) else np.nan
        out.append((d,x,y,dz,bz,th))
    return out

def _infer_line_id(row):
    for k in ["name","Name","NAME","id","ID","Id","label","Label"]:
        if k in row.index and pd.notna(row[k]): return _sanitize(row[k])
    return f"line{int(row.name)}"

# ---------------------------
# Geometry helpers for C estimation
# ---------------------------
def _poly_dims(g):
    """Return (Lrh, w) from oriented minimum bounding rectangle (meters)."""
    try:
        mbr = g.minimum_rotated_rectangle
        if mbr.is_empty: return np.nan, np.nan
        coords = np.asarray(mbr.exterior.coords)
        if coords.shape[0] < 5: return np.nan, np.nan
        edges = np.sqrt(np.sum(np.diff(coords[:4], axis=0)**2, axis=1))
        l1, l2 = np.sort(edges)[::-1][:2]
        Lrh = float(l1)
        w   = float(l2)
        return Lrh, w
    except Exception:
        return np.nan, np.nan

def _auto_C_from_e_Lrh(e, Lrh, dx):
    # C = (4 * e / Lrh) * dx^2
    if not np.isfinite(e) or not np.isfinite(Lrh) or Lrh <= 0 or dx <= 0: return np.nan
    return float((4.0 * e / float(Lrh)) * (dx*dx))

def _auto_C_from_area_shape(e, A, k, dx):
    # C = 4 * k * e * sqrt(A) * dx^2
    if not np.isfinite(e) or not np.isfinite(A) or A <= 0 or dx <= 0: return np.nan
    return float((4.0 * k * e * np.sqrt(A)) * (dx*dx))

# ---------------------------
# Iterative solver (CPU/GPU) — exact SLBL logic
# ---------------------------
def slbl_iterative(dem, inmask, tol_m=0.1, stop_eps=1e-4, max_iters=2000,
                   neighbours=4, max_depth=None, fixed_mask=None, z_floor=None,
                   use_gpu=False, mode="failure", stop_vol_eps=None, pixel_area=1.0):
    """
    mode='failure'  → lower:  target = mean - C, update if target < base
    mode='inverse'  → fill:   target = mean + C, update if target > base
    """
    H, W = dem.shape
    dem_np = dem.filled(np.nan)
    valid_np = np.isfinite(dem_np)
    work_mask = inmask & valid_np
    if not np.any(work_mask):
        base0 = dem_np.copy()
        thick0 = np.zeros_like(base0)
        return base0, thick0

    if np.isscalar(tol_m):
        tol_np = np.full((H,W), float(abs(tol_m)), dtype="float64")
    else:
        tol_np = np.abs(np.asarray(tol_m, dtype="float64"))
        if tol_np.shape != (H,W): raise ValueError("tol array shape mismatch")

    mode8 = (neighbours == 8)
    is_failure = (str(mode).lower() == "failure")

    # GPU path
    if use_gpu and HAVE_CUPY:
        dem_g  = cp.asarray(dem_np)
        base_g = dem_g.copy()
        tol_g  = cp.asarray(tol_np)
        valid_g= cp.asarray(valid_np)
        work_g = cp.asarray(work_mask)
        fix_g  = cp.asarray(fixed_mask) if fixed_mask is not None else cp.zeros_like(work_g, dtype=cp.bool_)
        floor_md_g = (dem_g - float(max_depth)) if (max_depth is not None) else None
        zfloor = float(z_floor) if (z_floor is not None and np.isfinite(z_floor)) else None

        prev_vol = None
        for itr in range(int(max_iters)):
            mean_nb = neighbour_mean(base_g, valid_g, mode8=mode8, use_gpu=True)
            target  = mean_nb - tol_g if is_failure else mean_nb + tol_g
            if is_failure:
                candidate = cp.where(work_g & (~fix_g) & (~cp.isnan(target)), target, base_g)
                if floor_md_g is not None:
                    candidate = cp.maximum(candidate, floor_md_g)
                if zfloor is not None:
                    candidate = cp.maximum(candidate, zfloor)
                new_base = cp.where(candidate < base_g, candidate, base_g)
            else:
                candidate = cp.where(work_g & (~fix_g) & (~cp.isnan(target)), target, base_g)
                new_base = cp.where(candidate > base_g, candidate, base_g)

            diff = cp.abs(new_base[work_g] - base_g[work_g])
            delta = float(cp.nanmax(diff).get()) if diff.size else 0.0
            base_g = new_base

            if stop_vol_eps is not None:
                if is_failure:
                    thickness_g = cp.where(work_g, cp.maximum(dem_g - base_g, 0.0), 0.0)
                else:
                    thickness_g = cp.where(work_g, cp.maximum(base_g - dem_g, 0.0), 0.0)
                vol = float(cp.nansum(thickness_g).get()) * float(pixel_area)
                if prev_vol is not None and abs(vol - prev_vol) < float(stop_vol_eps):
                    print(f"[ITR-GPU] converged (volΔ<{stop_vol_eps}) at iter={itr}")
                    break
                prev_vol = vol

            if itr < 5 or itr % 25 == 0:
                print(f"[ITR-GPU] iter={itr:4d} delta={delta:.6f}")
            if delta < stop_eps:
                print(f"[ITR-GPU] converged at iter={itr} (delta<{stop_eps})")
                break

        if is_failure:
            thickness_g = dem_g - base_g
        else:
            thickness_g = base_g - dem_g
        thickness_g = cp.where(work_g, cp.maximum(thickness_g, 0.0), 0.0)

        base = cp.asnumpy(base_g)
        thick = cp.asnumpy(thickness_g)
        return base, thick

    # CPU path
    base = dem_np.copy()
    fix = fixed_mask if fixed_mask is not None else np.zeros_like(work_mask, dtype=bool)
    prev_vol = None
    for itr in range(int(max_iters)):
        mean_nb = _neigh_mean_cpu(base, valid_np, mode8=mode8)
        target    = mean_nb - tol_np if is_failure else mean_nb + tol_np

        if is_failure:
            candidate = np.where(work_mask & (~fix) & (~np.isnan(target)), target, base)
            if max_depth is not None:
                candidate = np.maximum(candidate, dem_np - float(max_depth))
            if z_floor is not None and np.isfinite(z_floor):
                candidate = np.maximum(candidate, float(z_floor))
            new_base = np.where(candidate < base, candidate, base)
        else:
            candidate = np.where(work_mask & (~fix) & (~np.isnan(target)), target, base)
            new_base = np.where(candidate > base, candidate, base)

        diff  = np.abs(new_base[work_mask] - base[work_mask])
        delta = 0.0 if diff.size == 0 or np.all(np.isnan(diff)) else float(np.nanmax(diff))
        base[:] = new_base

        if stop_vol_eps is not None:
            if is_failure:
                thickness = np.where(work_mask, np.maximum(dem_np - base, 0.0), 0.0)
            else:
                thickness = np.where(work_mask, np.maximum(base - dem_np, 0.0), 0.0)
            vol = float(np.nansum(thickness) * float(pixel_area))
            if prev_vol is not None and abs(vol - prev_vol) < float(stop_vol_eps):
                print(f"[ITR] converged (volΔ<{stop_vol_eps}) at iter={itr}")
                break
            prev_vol = vol

        if itr < 5 or itr % 25 == 0:
            print(f"[ITR] iter={itr:4d} delta={delta:.6f}")
        if delta < stop_eps:
            print(f"[ITR] converged at iter={itr} (delta<{stop_eps})")
            break

    if is_failure:
        thickness = dem_np - base
    else:
        thickness = base - dem_np
    thickness = np.where(work_mask, np.maximum(thickness, 0.0), 0.0)
    return base, thickness

# ---------------------------
# Per-job execution
# ---------------------------
def format_tol_label(tol_m):
    t = float(abs(tol_m)) if np.isscalar(tol_m) else float(np.nanmax(tol_m))
    return f"tol{t:.4f}m"

def format_md_label(max_depth):
    return f"md{int(max_depth)}m" if max_depth is not None else "nolimit"

def format_fk_label(K):
    if isinstance(K,(list,tuple)) and len(K)==2:
        return f"fk{int(K[0])}to{int(K[1])}pxg"
    return f"fk{int(K)}px" if (K is not None and K>0) else "fk0"

def format_e_label(e):
    if e == "" or e is None:
        return "eNA"
    s = f"{float(e):.3e}"        # e.g., '5.000e-05'
    return "e" + s.replace("+", "p").replace("-", "m")  # 'e5.000e m05'

def _label_cfrom(method, tol_used, e_ratio, shape_k):
    if method == "manual":
        return f"{format_tol_label(tol_used)}"
    elif method == "e_Lrh":
        return f"{format_e_label(e_ratio)}"
    elif method == "area_shape":
        return f"{format_e_label(e_ratio)}_k{float(shape_k):.3f}"
    return method

def run_one_job(job):
    # job: (shp_path, method, tol_m_in, e_ratio, shape_k, max_depth, K)
    shp_path, method, tol_m_in, e_ratio, shape_k, max_depth, K = job
    name = os.path.splitext(os.path.basename(shp_path))[0]
    try:
        src, dem, prof, pixel_area, dem_bounds, dem_epsg = read_dem(DEM_PATH)
        dx = abs(src.transform.a)
        pixel_size = dx

        gdf = load_and_prepare_gdf(shp_path, target_epsg=dem_epsg, pixel_size=pixel_size, buffer_pixels=BUFFER_PIXELS)
        if not gdf.intersects(gpd.GeoSeries([dem_bounds], crs=f"EPSG:{dem_epsg}").iloc[0]).any():
            raise ValueError("Polygon does not overlap DEM extent")
        inmask = vector_to_mask(gdf, src, all_touched=ALL_TOUCHED, burn_value=1)
        if inmask.sum() == 0: raise ValueError("Rasterized mask has 0 cells")

        dem_valid = ~np.ma.getmaskarray(dem)
        work_mask = inmask & dem_valid
        if work_mask.sum() == 0: raise ValueError("Mask overlaps 0 valid DEM cells")

        # Geom measures
        poly_union = unary_union(gdf.geometry)
        area_A = float(poly_union.area)
        Lrh, width_w = _poly_dims(poly_union)

        # Tolerance selection (tol_used must be scalar here; feathering may spatialize it later)
        if method == "manual":
            tol_used = float(tol_m_in)
        elif method == "e_Lrh":
            tol_used = _auto_C_from_e_Lrh(float(e_ratio), Lrh, dx)
            if not np.isfinite(tol_used):
                raise ValueError("Auto C (e_Lrh) failed; check E_RATIO/Lrh/DEM pixel size.")
        elif method == "area_shape":
            tol_used = _auto_C_from_area_shape(float(e_ratio), area_A, float(shape_k), dx)
            if not np.isfinite(tol_used):
                raise ValueError("Auto C (area_shape) failed; check E_RATIO/area/DEM pixel size.")
        else:
            raise ValueError(f"Unknown C_FROM method '{method}'")

        # Feathering / gradation
        on_gpu = USE_GPU and HAVE_CUPY
        is_grad = isinstance(K,(list,tuple)) and len(K)==2 and (K[0] is not None) and (K[1] is not None)
        if is_grad and (K[0] < K[1]):
            upper_mask, lower_mask, gd = build_upper_lower_masks(dem, inmask, method=GRADATION_SPLIT_METHOD, elev_q=GRADATION_ELEV_Q)
            tol_eff = compute_feathered_tol_gradated(inmask, tol_used, float(K[0]), float(K[1]),
                                                     upper_mask, lower_mask, use_gpu=on_gpu)
            grad_meta = {"mode":gd.get("mode","elev"), **gd, "K_upper":float(K[0]), "K_lower":float(K[1])}
        else:
            if is_grad and K[0] >= K[1]:
                print(f"[WARN] Gradation expects K_upper < K_lower; got {K}. Using constant K={K[1]}")
                K = K[1]
            tol_eff = compute_feathered_tol(inmask, tol_used, (K if not isinstance(K,(list,tuple)) else K), use_gpu=on_gpu)
            grad_meta = {"mode":"constant"}

        label_core = _label_cfrom(method, tol_used, e_ratio, shape_k)
        label = f"{label_core}_{format_md_label(max_depth)}_{format_fk_label(K)}_{SLBL_MODE}"
        print(f"\n[SCN] >>> {name} | {label} | C_FROM={method} | tol_used={tol_used:.6f} m | Lrh={Lrh:.2f} m, w={width_w:.2f} m, A={area_A:.2f} m²")

        fixed_mask = None
        if FIXED_PATH:
            f_gdf = load_and_prepare_gdf(FIXED_PATH, target_epsg=dem_epsg)
            fixed_mask = vector_to_mask(f_gdf, src, all_touched=ALL_TOUCHED, burn_value=1)

        z_floor = compute_z_floor(inmask, dem, use_gpu=(USE_GPU and HAVE_CUPY)) if USE_Z_FLOOR else None

        base, thick = slbl_iterative(
            dem=dem, inmask=inmask, tol_m=tol_eff, stop_eps=STOP_EPS, max_iters=MAX_ITERS,
            neighbours=NEIGHBOURS, max_depth=max_depth, fixed_mask=fixed_mask, z_floor=z_floor,
            use_gpu=on_gpu, mode=SLBL_MODE, stop_vol_eps=STOP_VOL_EPS, pixel_area=pixel_area
        )

        if DEBUG_WRITE_TOL_RASTER:
            try:
                tol_arr_out = tol_eff if not np.isscalar(tol_eff) else np.full(dem.shape, float(tol_eff), dtype='float32')
                save_gtiff(os.path.join(OUT_DIR, f"{name}_{label}_tol_target.tif"), tol_arr_out, prof, nodata=np.nan)
            except Exception as _e:
                print("[DBG] tol raster failed:", _e)

        if DEBUG_REPORT_CLAMP and SLBL_MODE=="failure":
            denom = float(work_mask.sum()) if work_mask.sum() else 1.0
            zclamp = 0.0; mdclamp = 0.0
            if USE_Z_FLOOR and z_floor is not None and np.isfinite(z_floor):
                zclamp = 100.0 * float(np.count_nonzero(work_mask & (base <= (z_floor + 1e-6)))) / denom
            if max_depth is not None:
                floor_md = dem.filled(np.nan) - float(max_depth)
                mdclamp = 100.0 * float(np.count_nonzero(work_mask & (base <= (floor_md + 1e-6)))) / denom
            print(f"[CLAMP] z_floor%={zclamp:.1f} | max_depth%={mdclamp:.1f}")

        pos = (thick > 0)
        max_depth_m  = float(np.nanmax(thick)) if np.isfinite(thick).any() else 0.0
        mean_depth_m = float(np.nanmean(thick[pos])) if np.any(pos) else 0.0
        volume_m3    = float(np.nansum(thick) * pixel_area) if np.isfinite(thick).any() else 0.0

        if np.any(pos):
            dvals = thick[pos].astype("float64")
            p10_d, p50_d, p90_d = np.nanpercentile(dvals, [10, 50, 90])
        else:
            p10_d = p50_d = p90_d = 0.0

        footprint_px   = int(work_mask.sum())
        mask_area_m2   = float(footprint_px * pixel_area)
        poly_area_m2   = float(gdf.area.sum())

        dem_vals = dem.filled(np.nan)[work_mask]
        dem_min  = float(np.nanmin(dem_vals)) if dem_vals.size else np.nan
        dem_max  = float(np.nanmax(dem_vals)) if dem_vals.size else np.nan
        dem_mean = float(np.nanmean(dem_vals)) if dem_vals.size else np.nan
        dem_rel  = float(dem_max - dem_min) if np.isfinite(dem_max) and np.isfinite(dem_min) else np.nan

        base_path = thick_path = ""
        if WRITE_RASTERS:
            base_suffix  = "slbl_base.tif"  if SLBL_MODE=="failure" else "slbl_fill_base.tif"
            thick_suffix = "slbl_thickness.tif" if SLBL_MODE=="failure" else "slbl_fill_height.tif"
            base_path  = os.path.join(OUT_DIR, f"{name}_{label}_{base_suffix}")
            thick_path = os.path.join(OUT_DIR, f"{name}_{label}_{thick_suffix}")
            save_gtiff(base_path,  base,  prof, nodata=np.nan)
            save_gtiff(thick_path, thick, prof, nodata=0.0)

        # Cross-sections
        wrote_x_count = 0
        if WRITE_XSECTIONS and XSECT_LINES_SOURCE:
            try:
                lines_gdf, _nfiles = load_lines_many(XSECT_LINES_SOURCE, dem_epsg)
                if lines_gdf is not None and not lines_gdf.empty:
                    poly_union_cs = unary_union(gdf.geometry) if XSECT_CLIP_TO_POLY else None
                    xs_dir = os.path.join(OUT_DIR, XSECT_DIRNAME, name); os.makedirs(xs_dir, exist_ok=True)
                    step_eff = float(XSECT_STEP_M) if (XSECT_STEP_M and XSECT_STEP_M>0) else float(pixel_size)
                    for _, row in lines_gdf.iterrows():
                        lid = _infer_line_id(row); geom=row.geometry
                        if XSECT_CLIP_TO_POLY and poly_union_cs is not None:
                            geom = geom.intersection(poly_union_cs)
                            if geom.is_empty: continue
                        for li, part in enumerate(_line_parts(geom)):
                            pts = _densify_line(part, step_eff)
                            samples = _sample_arrays(pts, src.transform, dem.filled(np.nan), base, thick,
                                                     inmask=(work_mask if XSECT_CLIP_TO_POLY else None))
                            if not samples: continue
                            dfp = pd.DataFrame(samples, columns=["Distance_m","X","Y","DEM_z","Base_z","Thickness_m"])
                            dfp.insert(0,"LinePart",li); dfp.insert(0,"LineID",lid); dfp.insert(0,"Scenario",name); dfp.insert(0,"Label",label)
                            fn = f"{name}_{label}_xsect_{_sanitize(lid)}_p{li}.csv"
                            dfp.to_csv(os.path.join(xs_dir, fn), index=False)
                            wrote_x_count += 1
                    print(f"[XSECT] wrote {wrote_x_count} CSV(s)")
            except Exception as xe:
                print(f"[WARN] x-sections: {xe}")

        print(f"[DONE] {name} | {label}: max={max_depth_m:.3f} m, mean={mean_depth_m:.3f} m, vol={volume_m3:.1f} m³ | A={mask_area_m2:.1f} m² | xsects={wrote_x_count}")

        return {
            "Scenario": name, "DEM": os.path.basename(DEM_PATH), "Label": label,
            "Mode": SLBL_MODE,
            "C_From": method, "Tol_m": round(float(tol_used), 6),
            "E_Ratio": ("" if method=="manual" else float(e_ratio)),
            "Shape_k": (float(shape_k) if method=="area_shape" else ""),
            "Lrh_m": round(Lrh,3) if np.isfinite(Lrh) else "",
            "Width_w_m": round(width_w,3) if np.isfinite(width_w) else "",
            "PolyArea_m2": round(area_A,3),
            "Neighbours": NEIGHBOURS,
            "Feather_K_px": (int(K) if (isinstance(K,(int,float)) and K is not None) else np.nan),
            "Feather_Gradated": isinstance(K,(list,tuple)),
            "Feather_K_upper_px": (int(K[0]) if isinstance(K,(list,tuple)) else ""),
            "Feather_K_lower_px": (int(K[1]) if isinstance(K,(list,tuple)) else ""),
            "Gradation_Mode": grad_meta.get("mode","constant"),
            "Gradation_Zsplit": grad_meta.get("z_split",""),
            "Gradation_RowSplit": grad_meta.get("row_split",""),
            "MaxDepthLimit_m": ("" if max_depth is None else float(max_depth)),
            "MaxDepth_m": round(max_depth_m,3), "MeanDepth_m": round(mean_depth_m,3),
            "Depth_P10_m": round(float(p10_d),3), "Depth_P50_m": round(float(p50_d),3), "Depth_P90_m": round(float(p90_d),3),
            "Volume_m3": round(volume_m3,3),
            "Footprint_px": int(footprint_px), "MaskArea_m2": round(mask_area_m2,3),
            "DEM_Min_m": round(dem_min,3) if np.isfinite(dem_min) else "",
            "DEM_Max_m": round(dem_max,3) if np.isfinite(dem_max) else "",
            "DEM_Mean_m": round(dem_mean,3) if np.isfinite(dem_mean) else "",
            "DEM_Relief_m": round(dem_rel,3) if np.isfinite(dem_rel) else "",
            "WroteRasters": bool(WRITE_RASTERS), "XSection_CSVs": int(wrote_x_count),
            "BaseRaster": base_path, "ThicknessRaster": thick_path
        }

    except Exception as e:
        label = "error"
        tb = traceback.format_exc()
        print(f"\n[ERROR] {name} | {label}\n{tb}\n", file=sys.stderr, flush=True)
        return {
            "Scenario": name, "Label": label, "Mode": SLBL_MODE,
            "C_From": method, "Tol_m": float(tol_m_in) if np.isscalar(tol_m_in) else float('nan'),
            "E_Ratio": ("" if method=="manual" else float(e_ratio)),
            "Shape_k": (float(shape_k) if method=="area_shape" else ""),
            "Feather_K_px": int(K) if isinstance(K,(int,float)) else np.nan,
            "Feather_Gradated": isinstance(K,(list,tuple)),
            "Feather_K_upper_px": (int(K[0]) if isinstance(K,(list,tuple)) else ""),
            "Feather_K_lower_px": (int(K[1]) if isinstance(K,(list,tuple)) else ""),
            "MaxDepthLimit_m": ("" if max_depth is None else float(max_depth)),
            "Error": f"{type(e).__name__}: {e}", "Traceback": tb,
            "WroteRasters": False, "XSection_CSVs": 0,
            "BaseRaster": "", "ThicknessRaster": ""
        }

# ---------------------------
# Scenario-level stats writer
# ---------------------------
def _write_scenario_stats(df_all: pd.DataFrame, out_path: str):
    if df_all.empty or "Volume_m3" not in df_all.columns: return
    def q10(s): return s.quantile(0.10, interpolation="linear")
    def q50(s): return s.quantile(0.50, interpolation="linear")
    def q90(s): return s.quantile(0.90, interpolation="linear")
    by_scn = df_all.groupby(["Scenario","Mode","C_From","E_Ratio","Shape_k"]).agg(
        N_runs=("Volume_m3","size"),
        Volume_m3_min=("Volume_m3","min"), Volume_m3_mean=("Volume_m3","mean"),
        Volume_m3_p10=("Volume_m3",q10),   Volume_m3_p50=("Volume_m3",q50),
        Volume_m3_p90=("Volume_m3",q90),   Volume_m3_max=("Volume_m3","max"),
        MaxDepth_m_min=("MaxDepth_m","min"), MaxDepth_m_mean=("MaxDepth_m","mean"),
        MaxDepth_m_p10=("MaxDepth_m",q10),   MaxDepth_m_p50=("MaxDepth_m",q50),
        MaxDepth_m_p90=("MaxDepth_m",q90),   MaxDepth_m_max=("MaxDepth_m","max"),
    ).reset_index()
    by_scn.to_csv(out_path, index=False)
    print(f"✅ Scenario stats written: {out_path}")

# ---------------------------
# Job builder
# ---------------------------
def _as_list(x, fallback=None):
    if isinstance(x, (list, tuple)): 
        return [v for v in x]
    return [x if x is not None else fallback]

def _get_cfrom_list():
    cf = C_FROM
    if isinstance(cf, (list, tuple)): 
        return [str(v) for v in cf]
    return [str(cf)]

def _get_feather_Ks_list():
    Ks = FEATHER_TOL_K
    return list(Ks) if isinstance(Ks,(list,tuple)) else [Ks]

def _get_gradation_pairs(base_Ks):
    if not GRADATION_ENABLE: return []
    if GRADATION_K_PAIRS:
        return [(int(a),int(b)) for (a,b) in GRADATION_K_PAIRS if a is not None and b is not None and a<b]
    base_Ks = sorted(int(k) for k in base_Ks if k is not None)
    out=[]
    for i, ku in enumerate(base_Ks):
        for kl in base_Ks[i+1:]: out.append((ku, kl))  # ku < kl
    return out

def _build_jobs(shp_list):
    """Return list of jobs: (shp, method, tol_in, e_ratio, shape_k, max_depth, K)"""
    cf_list = _get_cfrom_list()
    Ks      = _get_feather_Ks_list()
    K_pairs = _get_gradation_pairs(Ks)

    # Build lists (ensure lists exist)
    e_list = _as_list(E_RATIOS, fallback=E_RATIO)
    k_list = _as_list(SHAPE_KS, fallback=SHAPE_K)
    tol_list = _as_list(TOLERANCES, fallback=None)

    jobs = []

    for shp in shp_list:
        for md in MAX_DEPTHS:
            # constant feather K
            for K in Ks:
                for method in cf_list:
                    method = method.strip()
                    if method == "manual":
                        for tol in tol_list:
                            jobs.append((shp, "manual", float(tol), "", "", md, K))
                    elif method == "e_Lrh":
                        for e in e_list:
                            jobs.append((shp, "e_Lrh", 0.0, float(e), "", md, K))
                    elif method == "area_shape":
                        for e, k in product(e_list, k_list):
                            jobs.append((shp, "area_shape", 0.0, float(e), float(k), md, K))
                    else:
                        raise ValueError(f"Unknown C_FROM method '{method}'")

            # gradated K pairs
            for pair in K_pairs:
                for method in cf_list:
                    method = method.strip()
                    if method == "manual":
                        for tol in tol_list:
                            jobs.append((shp, "manual", float(tol), "", "", md, pair))
                    elif method == "e_Lrh":
                        for e in e_list:
                            jobs.append((shp, "e_Lrh", 0.0, float(e), "", md, pair))
                    elif method == "area_shape":
                        for e, k in product(e_list, k_list):
                            jobs.append((shp, "area_shape", 0.0, float(e), float(k), md, pair))
                    else:
                        raise ValueError(f"Unknown C_FROM method '{method}'")
    return jobs

# ---------------------------
# Driver
# ---------------------------
def main():
    shp_list = [os.path.join(SCENARIO_DIR, f) for f in os.listdir(SCENARIO_DIR) if f.lower().endswith(".shp")]
    if not shp_list:
        print(f"No shapefiles found in {SCENARIO_DIR}")
        return

    jobs = _build_jobs(shp_list)

    # GPU ⇒ single process; CPU ⇒ cap
    if USE_GPU and HAVE_CUPY:
        procs = 1
    else:
        procs = max(1, int(N_PROCESSES)) if N_PROCESSES else min(8, cpu_count())

    print(f"Running {len(jobs)} jobs on {procs} process(es)...")
    print(f"  Mode        : {SLBL_MODE}")
    print(f"  C_FROM      : {_get_cfrom_list()}")
    print(f"  E_RATIOS    : {_as_list(E_RATIOS, fallback=E_RATIO)}")
    print(f"  SHAPE_KS    : {_as_list(SHAPE_KS, fallback=SHAPE_K)}")
    print(f"  Tolerances  : {_as_list(TOLERANCES)}")
    print(f"  MaxDepths   : {MAX_DEPTHS}")
    print(f"  Feather K   : {_get_feather_Ks_list()}")
    if GRADATION_ENABLE:
        print(f"  Gradation   : pairs={_get_gradation_pairs(_get_feather_Ks_list())} (upper→lower), split={GRADATION_SPLIT_METHOD}@{GRADATION_ELEV_Q}")

    if procs <= 1:
        rows = [run_one_job(j) for j in jobs]
    else:
        with Pool(processes=procs, maxtasksperchild=1) as pool:
            rows = list(pool.imap_unordered(run_one_job, jobs, chunksize=MP_CHUNKSIZE))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CSV_SUMMARY), exist_ok=True)
    df.to_csv(CSV_SUMMARY, index=False)
    print(f"\n✅ Summary written: {CSV_SUMMARY}")

    cols = [c for c in [
        "Scenario","Mode","C_From","E_Ratio","Shape_k","Tol_m",
        "Lrh_m","Width_w_m","PolyArea_m2",
        "Label","MaxDepthLimit_m","MaxDepth_m","MeanDepth_m","Depth_P50_m","Volume_m3",
        "MaskArea_m2","XSection_CSVs","Error"
    ] if c in df.columns]
    if cols: print(df[cols])

    try:
        ok = df
        if "Error" in ok.columns:
            ok = ok[ok["Error"].astype(str).str.len().fillna(0) == 0]
        ok = ok.dropna(subset=["Scenario","Volume_m3"]).copy()
        if not ok.empty:
            _write_scenario_stats(ok, CSV_SCENARIO_STATS)
    except Exception as e:
        print("[WARN] scenario stats failed:", e)

if __name__ == "__main__":
    main()
