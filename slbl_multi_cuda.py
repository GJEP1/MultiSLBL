#!/usr/bin/env python
# SLBL batch processing — exact pySLBL-compatible core, CPU/GPU, non-ArcGIS
# Gustav edition (v4)

import os, sys, re, math, traceback
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import rowcol
from shapely.geometry import box, LineString, MultiLineString
from shapely import ops as sops

# ---------------------------
# Environment thread caps
# ---------------------------
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS","1")
os.environ.setdefault("NUMEXPR_MAX_THREADS","1")

# ---------------------------
# User parameters
# ---------------------------
DEM_PATH       = r"D:\Python\SLBL\Skarfjell\dtm1_25833_clean.tif"
SCENARIO_DIR   = r"D:\Python\SLBL\Skarfjell\scenarios"
OUT_DIR        = r"D:\Python\SLBL\Skarfjell\outputs"
CSV_SUMMARY    = r"D:\Python\SLBL\Skarfjell\outputs\slbl_summary.csv"
TARGET_EPSG    = 25833

# --- pySLBL-compatible iteration parameters ---
NEIGHBOURS     = 4                   # 4 or 8
CRITERIA       = "average"           # "average" or "minmax"
MODE           = "failure"           # "failure" (lowering) or "inverse" (filling)
STOP_EPS       = 1e-4                # stop when max|Δthickness| <= STOP_EPS
MAX_VOLUME     = math.inf            # m^3; use math.inf for no limit
NOT_DEEPEN     = True                # z_min clamp around polygon
MAX_DEPTHS     = [None]              # meters; None for unlimited

# --- Tolerance sources (C) and sweeps ---
C_FROM         = ["manual"]          # "manual", "e_Lrh", "area_shape" (or a list of them)
TOLERANCES     = [0.01, 0.02, 0.03, 0.04]  # meters; used when "manual" in C_FROM
E_RATIOS       = [0.0010]            # used with "e_Lrh"
SHAPE_KS       = [1.0]               # used with "area_shape"
E_RATIO        = 0.0010              # singleton fallback
SHAPE_K        = 1.0                 # singleton fallback

# --- Mask rasterization & buffer ---
ALL_TOUCHED    = True
BUFFER_PIXELS  = 2                   # outward buffer around polygons (pySLBL pads extent by 2 px)

# --- GPU and MP ---
USE_GPU        = True                # auto-disables if no CUDA
N_PROCESSES    = max(1, cpu_count()-1)
MP_CHUNKSIZE   = 1

# --- Outputs ---
WRITE_RASTERS      = False
WRITE_XSECTIONS    = True
XSECT_LINES_SOURCE = r"D:\Python\SLBL\Skarfjell\xsections"  # folder or single .shp
XSECT_STEP_M       = 5.0
XSECT_DIRNAME      = "xsections"

# ---------------------------
# GPU backend (CuPy)
# ---------------------------
HAVE_CUPY = False
cp = None
try:
    if USE_GPU:
        import cupy as cp
        HAVE_CUPY = cp.cuda.runtime.getDeviceCount() > 0
        if HAVE_CUPY:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            if isinstance(name, bytes): name = name.decode(errors="ignore")
            print(f"[GPU] Using: {name}")
        else:
            print("[GPU] No CUDA device detected; falling back to CPU.")
except Exception as e:
    print(f"[GPU] CuPy unavailable: {e}")
    HAVE_CUPY = False

# Enforce single-process mode if GPU is used
if HAVE_CUPY:
    print("[GPU] Enabling single-process mode to give the GPU full control.")
    globals()["N_PROCESSES"] = 1

# ---------------------------
# Helpers: IO / CRS / mask
# ---------------------------
def read_dem(path):
    src = rasterio.open(path)
    if src.crs is None: raise RuntimeError("DEM has no CRS.")
    if src.crs.to_epsg() != TARGET_EPSG:
        raise RuntimeError(f"DEM EPSG {src.crs.to_epsg()} != TARGET_EPSG {TARGET_EPSG}")
    arr = src.read(1, masked=True).astype("float64")
    pixel = abs(src.transform.a)
    pix_area = abs(src.transform.a) * abs(src.transform.e)
    return src, arr, src.profile.copy(), pixel, pix_area, box(*src.bounds)

def load_gdf(path, epsg, pixel_size=None, buffer_pixels=0):
    gdf = gpd.read_file(path)
    if gdf.empty: raise ValueError("Empty shapefile.")
    if gdf.crs is None: raise ValueError("Shapefile has no CRS.")
    if gdf.crs.to_epsg() != epsg: gdf = gdf.to_crs(epsg=epsg)
    # dissolve multipolygons for mask (like pySLBL "processing extent")
    if gdf.geom_type.isin(["Polygon","MultiPolygon"]).any():
        gdf["__diss__"]=1
        gdf = gdf.dissolve(by="__diss__", as_index=False).drop(columns="__diss__", errors="ignore")
    if buffer_pixels and pixel_size and gdf.geom_type.isin(["Polygon","MultiPolygon"]).any():
        gdf["geometry"] = gdf.buffer(buffer_pixels*pixel_size)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)]
    return gdf

def rasterize_mask(gdf, like_src, all_touched=True):
    shapes = ((geom, 1) for geom in gdf.geometry if geom and not geom.is_empty)
    mask = features.rasterize(
        shapes=shapes,
        out_shape=(like_src.height, like_src.width),
        transform=like_src.transform,
        fill=0, dtype="uint8", all_touched=all_touched
    ).astype(bool)
    return mask

def fill_nodata_nearest(dem_ma):
    """Fill DEM NoData with nearest (pySLBL behaviour)."""
    arr = dem_ma.filled(np.nan)
    if np.isfinite(arr).all(): return np.ma.array(arr, mask=False)
    try:
        from scipy.interpolate import griddata
        H, W = arr.shape
        yy, xx = np.mgrid[0:H, 0:W]
        pts = np.column_stack((xx[np.isfinite(arr)], yy[np.isfinite(arr)]))
        vals= arr[np.isfinite(arr)]
        filled = griddata(pts, vals, (xx,yy), method="nearest")
        return np.ma.array(filled, mask=np.isnan(filled))
    except Exception:
        a = arr.copy()
        for _ in range(8):  # crude grow
            n = a.copy()
            for dy,dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                n = np.where(np.isfinite(n), n,
                             np.roll(np.where(np.isfinite(a),a,np.nan), (dy,dx), (0,1)))
            a = np.where(np.isfinite(a), a, n)
            if np.isfinite(a).all(): break
        return np.ma.array(a, mask=~np.isfinite(a))

def compute_z_min_border(mask, dem_ma):
    """z_min from 1-pixel border ring (pySLBL 'Expand 1')."""
    try:
        from scipy.ndimage import binary_dilation
        ring = binary_dilation(mask, iterations=1) & (~mask)
    except Exception:
        H,W = mask.shape
        ring = np.zeros_like(mask, dtype=bool)
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            y0s, y1s = max(0,dy), H+min(0,dy)
            x0s, x1s = max(0,dx), W+min(0,dx)
            y0d, y1d = max(0,-dy), H+min(0,-dy)
            x0d, x1d = max(0,-dx), W+min(0,-dx)
            if y0s<y1s and x0s<x1s:
                ring[y0d:y1d, x0d:x1d] |= mask[y0s:y1s, x0s:x1s]
        ring &= ~mask
    arr = dem_ma.filled(np.nan)
    if not np.any(ring): return np.nan
    return float(np.nanmin(arr[ring]))

# ---------------------------
# Neighbour stats — CPU & GPU
# ---------------------------
NEIGH_SHIFTS4 = [(-1,0),(1,0),(0,-1),(0,1)]
NEIGH_SHIFTS8 = NEIGH_SHIFTS4 + [(-1,-1),(-1,1),(1,-1),(1,1)]

def _stack_neighbours_cpu(arr, valid, mode8=False):
    H,W = arr.shape
    shifts = NEIGH_SHIFTS8 if mode8 else NEIGH_SHIFTS4
    st = np.full((H,W,len(shifts)), np.nan, dtype="float64")
    for k,(dy,dx) in enumerate(shifts):
        y0s, y1s = max(0,dy), H+min(0,dy)
        x0s, x1s = max(0,dx), W+min(0,dx)
        y0d, y1d = max(0,-dy), H+min(0,-dy)
        x0d, x1d = max(0,-dx), W+min(0,-dx)
        if y0s>=y1s or x0s>=x1s: continue
        nb = arr[y0s:y1s, x0s:x1s]
        vm = valid[y0s:y1s, x0s:x1s]
        blk = np.where(vm, nb, np.nan)
        st[y0d:y1d, x0d:x1d, k] = blk
    return st

def _stack_neighbours_gpu(arr_g, valid_g, mode8=False):
    shifts = NEIGH_SHIFTS8 if mode8 else NEIGH_SHIFTS4
    H,W = arr_g.shape
    st = cp.full((H,W,len(shifts)), cp.nan, dtype=cp.float64)
    for k,(dy,dx) in enumerate(shifts):
        nb = cp.full_like(arr_g, cp.nan)
        vm = cp.full_like(valid_g, False)
        y0s, y1s = max(0,dy), H+min(0,dy)
        x0s, x1s = max(0,dx), W+min(0,dx)
        y0d, y1d = max(0,-dy), H+min(0,-dy)
        x0d, x1d = max(0,-dx), W+min(0,-dx)
        if y0s>=y1s or x0s>=x1s: continue
        nb[y0d:y1d, x0d:x1d] = arr_g[y0s:y1s, x0s:x1s]
        vm[y0d:y1d, x0d:x1d] = valid_g[y0s:y1s, x0s:x1s]
        st[:,:,k] = cp.where(vm, nb, cp.nan)
    return st

def neighbour_stat(arr, valid, criteria="average", mode8=False, use_gpu=False):
    if use_gpu and HAVE_CUPY:
        arr_g   = arr if isinstance(arr, cp.ndarray) else cp.asarray(arr)
        valid_g = valid if isinstance(valid, cp.ndarray) else cp.asarray(valid)
        st = _stack_neighbours_gpu(arr_g, valid_g, mode8)
        if criteria == "minmax":
            mx = cp.nanmax(st, axis=2)
            mn = cp.nanmin(st, axis=2)
            return (mx + mn) / 2.0
        s = cp.nansum(st, axis=2)
        c = cp.sum(cp.isfinite(st), axis=2)
        return cp.where(c>0, s/c, cp.nan)
    # CPU
    st = _stack_neighbours_cpu(arr, valid, mode8)
    if criteria == "minmax":
        mx = np.nanmax(st, axis=2)
        mn = np.nanmin(st, axis=2)
        return (mx + mn) / 2.0
    s = np.nansum(st, axis=2)
    c = np.sum(np.isfinite(st), axis=2)
    out = np.full_like(s, np.nan, dtype="float64")
    np.divide(s, c, out=out, where=(c>0))
    return out

# ---------------------------
# Geometry helpers for auto C
# ---------------------------
def _poly_dims(g):
    mbr = g.minimum_rotated_rectangle
    if mbr.is_empty: return np.nan, np.nan
    coords = np.asarray(mbr.exterior.coords)
    if coords.shape[0] < 5: return np.nan, np.nan
    edges = np.sqrt(np.sum(np.diff(coords[:4], axis=0)**2, axis=1))
    Lrh, w = np.sort(edges)[::-1][:2]
    return float(Lrh), float(w)

def _auto_C_from_e_Lrh(e, Lrh, dx):
    if not np.isfinite(e) or not np.isfinite(Lrh) or Lrh <= 0: return np.nan
    return float((4.0 * e / Lrh) * (dx*dx))

def _auto_C_from_area_shape(e, A, k, dx):
    if not np.isfinite(e) or not np.isfinite(A) or A<=0: return np.nan
    return float((4.0 * k * e * math.sqrt(A)) * (dx*dx))

# ---------------------------
# Exact pySLBL-like iterative solver (CPU/GPU)
# ---------------------------
def slbl_iterative_exact(
    dem_ma, inmask, tol_scalar, *,
    criteria="average", neighbours=4, mode="failure",
    stop_eps=1e-4, max_volume=math.inf, pixel_area=1.0,
    max_depth=None, not_deepen=True, use_gpu=False
):
    """GPU/CPU implementation with identical pySLBL semantics."""
    dem_np = fill_nodata_nearest(dem_ma).filled(np.nan).astype("float64")
    valid_np = np.isfinite(dem_np)
    work_mask_np = inmask & valid_np
    if not np.any(work_mask_np):
        return dem_np.copy(), np.zeros_like(dem_np), 0

    z_floor = compute_z_min_border(inmask, np.ma.array(dem_np, mask=~valid_np)) if not_deepen else np.nan
    floor_md_np = (dem_np - float(max_depth)) if (max_depth is not None) else None

    mode8    = (neighbours == 8)
    failure  = (str(mode).lower() == "failure")
    tol      = float(abs(tol_scalar))
    max_volf = float(max_volume)

    on_gpu = bool(use_gpu and HAVE_CUPY)
    xp = cp if on_gpu else np

    if on_gpu:
        print("[GPU] SLBL iteration running on CUDA.")
        dem     = cp.asarray(dem_np)
        valid   = cp.asarray(valid_np)
        work_m  = cp.asarray(work_mask_np)
        in_m    = cp.asarray(inmask)
        floor_md = cp.asarray(floor_md_np) if floor_md_np is not None else None
    else:
        dem     = dem_np
        valid   = valid_np
        work_m  = work_mask_np
        in_m    = inmask
        floor_md = floor_md_np

    base       = dem.copy()
    thick_prev = xp.zeros_like(base)
    nb_iter    = 0
    z_floor_backend = float(z_floor)

    while True:
        nb_iter += 1
        nb = neighbour_stat(base, valid, criteria=criteria, mode8=mode8, use_gpu=on_gpu)
        target = (nb - tol) if failure else (nb + tol)
        cand = xp.where(work_m & xp.isfinite(target), target, base)
        if floor_md is not None:
            cand = xp.maximum(cand, floor_md)
        if math.isfinite(z_floor_backend):
            cand = xp.maximum(cand, z_floor_backend)
        new_base = xp.minimum(cand, base) if failure else xp.maximum(cand, base)
        new_base = xp.where(in_m, new_base, dem)

        thick  = xp.abs(dem - new_base) if failure else xp.abs(new_base - dem)
        grid_d = xp.abs(thick - thick_prev)
        delta  = float((xp.nanmax(grid_d)).get() if on_gpu else xp.nanmax(grid_d))
        volume = float(((xp.nansum(thick))*pixel_area).get() if on_gpu else (xp.nansum(thick))*pixel_area)

        base[:]       = new_base
        thick_prev[:] = thick

        if nb_iter % 100 == 0 or nb_iter <= 5:
            max_t = float((xp.nanmax(thick)).get() if on_gpu else xp.nanmax(thick))
            print(f"[ITR] iter={nb_iter} delta={delta:.6e} max_thick={max_t:.3f} vol={volume:.3f} m^3")

        if not (delta > stop_eps and volume < max_volf):
            break
        if nb_iter >= 5000:
            print("[WARN] Reached 5000 iterations; stopping.")
            break

    if on_gpu:
        base  = cp.asnumpy(base)
        thick = cp.asnumpy(thick)

    return base, (dem_np - base if failure else base - dem_np), nb_iter

# ---------------------------
# Outputs & labels
# ---------------------------
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

# ---- v3-compatible label + xsection helpers ----
def format_tol_label(tol_m):
    t = abs(float(tol_m)) if np.isscalar(tol_m) else float(np.nanmax(tol_m))
    return f"tol{int(round(t*100)):02d}cm" if t < 1.0 else f"tol{t:.2f}m"

def format_md_label(max_depth):
    return f"md{int(max_depth)}m" if max_depth is not None else "nolimit"

def format_fk_label(K):
    if isinstance(K,(list,tuple)) and len(K)==2:
        return f"fk{int(K[0])}to{int(K[1])}pxg"
    return f"fk{int(K)}px" if (K is not None and K>0) else "fk0"

# --- x-section helpers (safe) ---
def _sanitize(s):
    if s is None: return "line"
    s = re.sub(r"\s+","_",str(s).strip())
    s = re.sub(r"[^A-Za-z0-9_\-]+","",s)
    return s or "line"

def _line_parts(geom):
    from shapely.geometry import LineString, MultiLineString, GeometryCollection
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
    if isinstance(geom, GeometryCollection):
        parts = []
        for g in geom.geoms:
            parts.extend(_line_parts(g))
        return parts
    try:
        return [g for g in geom.geoms if hasattr(g, "coords")]
    except Exception:
        return []

def _densify_line(line, step_m):
    L = float(line.length)
    if L <= 0:
        p0, p1 = line.coords[0], line.coords[-1]
        return [(p0[0], p0[1], 0.0), (p1[0], p1[1], L)]
    step = max(float(step_m) if step_m else 0.0, 0.001)
    n = max(1, int(L // step))
    dists = [i*step for i in range(n+1)]
    if dists[-1] < L: dists.append(L)
    pts = []
    for d in dists:
        p = line.interpolate(d)
        pts.append((float(p.x), float(p.y), float(d)))
    return pts

def _sample_arrays(xy_dist_pts, transform, dem_arr, base_arr, thick_arr, inmask=None):
    xs = [x for x,_,_ in xy_dist_pts]; ys = [y for _,y,_ in xy_dist_pts]
    rows, cols = rowcol(transform, xs, ys, op=float)
    rows = np.rint(rows).astype(int); cols = np.rint(cols).astype(int)
    H, W = dem_arr.shape
    out=[]
    for (x,y,d), r, c in zip(xy_dist_pts, rows, cols):
        if r<0 or c<0 or r>=H or c>=W: continue
        if inmask is not None and not inmask[r,c]: continue
        dz = float(dem_arr[r,c]) if np.isfinite(dem_arr[r,c]) else np.nan
        bz = float(base_arr[r,c]) if np.isfinite(base_arr[r,c]) else np.nan
        th = float(thick_arr[r,c]) if np.isfinite(thick_arr[r,c]) else np.nan
        out.append((d,x,y,dz,bz,th))
    return out

def _infer_line_id(row_or_geom, idx=0):
    try:
        row = row_or_geom
        for k in ("name","Name","NAME","id","ID","Id","label","Label"):
            if (hasattr(row,"__contains__") and k in row) and pd.notna(row[k]):
                return _sanitize(row[k])
    except Exception:
        pass
    return f"line{int(idx)}"

# ---------------------------
# Single job runner
# ---------------------------
@dataclass
class Job:
    shp_path: str
    method: str
    tol_in: float
    e_ratio: float
    shape_k: float
    max_depth: float|None

def run_one(job: Job):
    name = os.path.splitext(os.path.basename(job.shp_path))[0]
    try:
        src, dem_ma, prof, dx, pix_area, dem_bounds = read_dem(DEM_PATH)
        dem_epsg = src.crs.to_epsg()
        gdf = load_gdf(job.shp_path, dem_epsg, pixel_size=dx, buffer_pixels=BUFFER_PIXELS)
        if not gdf.intersects(gpd.GeoSeries([dem_bounds], crs=f"EPSG:{dem_epsg}").iloc[0]).any():
            raise ValueError("Polygon does not overlap DEM.")
        inmask = rasterize_mask(gdf, src, ALL_TOUCHED)
        if inmask.sum()==0: raise ValueError("Mask rasterized to 0 pixels.")

        # Geometry for auto C
        poly_union = sops.unary_union(gdf.geometry)
        A = float(poly_union.area)
        Lrh, _ = _poly_dims(poly_union)

        # Choose C (tol)
        if job.method == "manual":
            tol = float(job.tol_in)
        elif job.method == "e_Lrh":
            tol = _auto_C_from_e_Lrh(float(job.e_ratio), Lrh, dx)
        elif job.method == "area_shape":
            tol = _auto_C_from_area_shape(float(job.e_ratio), A, float(job.shape_k), dx)
        else:
            raise ValueError(f"Unknown C_FROM method {job.method}")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("Computed tolerance (C) is invalid. Check geometry and parameters.")

        base, thick, niter = slbl_iterative_exact(
            dem_ma, inmask, tol_scalar=tol,
            criteria=CRITERIA, neighbours=NEIGHBOURS, mode=MODE,
            stop_eps=STOP_EPS, max_volume=MAX_VOLUME, pixel_area=pix_area,
            max_depth=job.max_depth, not_deepen=NOT_DEEPEN,
            use_gpu=HAVE_CUPY
        )

        # v3-style label
        K = 0  # Feathering/gradation K; keep 0 if not used
        label_core = format_tol_label(tol)
        label = f"{label_core}_{format_md_label(job.max_depth)}_{format_fk_label(K)}_{MODE}"

        # Outputs
        out_dir = os.path.join(OUT_DIR, name)
        os.makedirs(out_dir, exist_ok=True)
        if WRITE_RASTERS:
            save_gtiff(os.path.join(out_dir, f"{name}_{label}_slbl.tif"), base, prof, nodata=np.nan)
            save_gtiff(os.path.join(out_dir, f"{name}_{label}_thick.tif"), thick, prof, nodata=np.nan)

        # ---- Cross-sections (v3-compatible) ----
        wrote_x_count = 0
        if WRITE_XSECTIONS and XSECT_LINES_SOURCE:
            try:
                def _load_lines(src_path, epsg):
                    if os.path.isdir(src_path):
                        from glob import glob
                        files = [p for p in glob(os.path.join(src_path, "*.shp"))]
                        gdfs = [gpd.read_file(p) for p in files if os.path.isfile(p)]
                        if not gdfs:
                            return gpd.GeoDataFrame(geometry=[]), 0
                        gdfL = pd.concat(gdfs, ignore_index=True)
                    else:
                        gdfL = gpd.read_file(src_path)
                    if gdfL.crs is None or gdfL.crs.to_epsg() != epsg:
                        gdfL = gdfL.to_crs(epsg=epsg)
                    if "geometry" not in gdfL.columns:
                        gdfL = gpd.GeoDataFrame(gdfL, geometry=gdfL.geometry)
                    return gdfL, len(gdfL)

                lines_gdf, _ = _load_lines(XSECT_LINES_SOURCE, src.crs.to_epsg())
                if lines_gdf is None or lines_gdf.empty:
                    print("[XSECT] No lines found.")
                else:
                    XSECT_CLIP_TO_POLY = True
                    poly_union_cs = poly_union if XSECT_CLIP_TO_POLY else None

                    xs_dir = os.path.join(OUT_DIR, XSECT_DIRNAME, name)
                    os.makedirs(xs_dir, exist_ok=True)

                    dem_arr  = dem_ma.filled(np.nan)
                    base_arr = base
                    thick_arr= thick
                    work_mask= inmask

                    step_eff = float(XSECT_STEP_M) if (XSECT_STEP_M and XSECT_STEP_M>0) else float(abs(src.transform.a))

                    for i in range(len(lines_gdf)):
                        row = lines_gdf.iloc[i]
                        geom = row.geometry
                        if geom is None or geom.is_empty:
                            continue
                        if XSECT_CLIP_TO_POLY and poly_union_cs is not None:
                            try:
                                geom = geom.intersection(poly_union_cs)
                                if geom.is_empty:
                                    continue
                            except Exception:
                                pass

                        parts = _line_parts(geom)
                        if not parts:
                            continue

                        lid = _infer_line_id(row, idx=i)
                        for li, part in enumerate(parts):
                            pts = _densify_line(part, step_eff)
                            samples = _sample_arrays(pts, src.transform, dem_arr, base_arr, thick_arr,
                                                     inmask=(work_mask if XSECT_CLIP_TO_POLY else None))
                            if not samples:
                                continue
                            dfp = pd.DataFrame(samples, columns=["Distance_m","X","Y","DEM_z","Base_z","Thickness_m"])
                            dfp.insert(0,"LinePart",li)
                            dfp.insert(0,"LineID",_sanitize(lid))
                            dfp.insert(0,"Scenario",name)
                            dfp.insert(0,"Label",label)

                            fn = f"{name}_{label}_xsect_{_sanitize(lid)}_p{li}.csv"
                            dfp.to_csv(os.path.join(xs_dir, fn), index=False)
                            wrote_x_count += 1

                    print(f"[XSECT] wrote {wrote_x_count} CSV(s) → {xs_dir}")
            except Exception as xe:
                print(f"[WARN] x-sections: {xe}")

        # ---- Metrics & summary row (v3-ish schema) ----
        abs_thick = np.abs(thick)
        max_depth_m  = float(np.nanmax(abs_thick)) if np.isfinite(abs_thick).any() else 0.0
        mask_pos     = (inmask & np.isfinite(abs_thick))
        mean_depth_m = float(np.nanmean(abs_thick[mask_pos])) if np.any(mask_pos) else 0.0
        volume_m3    = float(np.nansum(abs_thick) * pix_area)

        row_out = {
            "Scenario": name, "DEM": os.path.basename(DEM_PATH), "Label": label,
            "Mode": MODE,
            "C_From": job.method, "Tol_m": round(float(tol), 6),
            "E_Ratio": ("" if job.method=="manual" else float(job.e_ratio)),
            "Shape_k": (float(job.shape_k) if job.method=="area_shape" else ""),
            "Lrh_m": float(Lrh) if np.isfinite(Lrh) else "",
            "Width_w_m": "",  # optional; not used by viewer logic
            "PolyArea_m2": float(A),
            "Neighbours": NEIGHBOURS,
            "Feather_K_px": np.nan, "Feather_Gradated": False,
            "Feather_K_upper_px": "", "Feather_K_lower_px": "",
            "Gradation_Mode": "constant", "Gradation_Zsplit": "", "Gradation_RowSplit": "",
            "MaxDepthLimit_m": ("" if job.max_depth is None else float(job.max_depth)),
            "MaxDepth_m": round(max_depth_m,3), "MeanDepth_m": round(mean_depth_m,3),
            "Depth_P10_m": "", "Depth_P50_m": "", "Depth_P90_m": "",
            "Volume_m3": round(volume_m3,3),
            "Footprint_px": int(inmask.sum()), "MaskArea_m2": float(inmask.sum())*pix_area,
            "WroteRasters": bool(WRITE_RASTERS), "XSection_CSVs": int(wrote_x_count),
        }
        return row_out

    except Exception as e:
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")
        return dict(Scenario=name, error=f"{type(e).__name__}: {e}")

# ---------------------------
# Batch orchestration
# ---------------------------
def build_jobs():
    shp_paths = [os.path.join(SCENARIO_DIR,f) for f in os.listdir(SCENARIO_DIR)
                 if f.lower().endswith(".shp")]
    if not shp_paths:
        raise RuntimeError(f"No shapefiles in {SCENARIO_DIR}")

    methods = C_FROM if isinstance(C_FROM,(list,tuple)) else [C_FROM]
    e_vals  = E_RATIOS if isinstance(E_RATIOS,(list,tuple)) else [E_RATIO]
    k_vals  = SHAPE_KS if isinstance(SHAPE_KS,(list,tuple)) else [SHAPE_K]
    tol_vals= TOLERANCES if isinstance(TOLERANCES,(list,tuple)) else [TOLERANCES]
    md_vals = MAX_DEPTHS if isinstance(MAX_DEPTHS,(list,tuple)) else [MAX_DEPTHS]

    jobs=[]
    for shp in shp_paths:
        for md in md_vals:
            for m in methods:
                if m == "manual":
                    for t in tol_vals:
                        jobs.append(Job(shp, m, float(t), float(e_vals[0]), float(k_vals[0]), md))
                elif m == "e_Lrh":
                    for e in e_vals:
                        jobs.append(Job(shp, m, float("nan"), float(e), float(k_vals[0]), md))
                elif m == "area_shape":
                    for e,k in product(e_vals, k_vals):
                        jobs.append(Job(shp, m, float("nan"), float(e), float(k), md))
                else:
                    raise ValueError(f"Unknown method {m}")
    return jobs

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = build_jobs()
    print(f"[INFO] Jobs: {len(jobs)}   (MP={N_PROCESSES}, GPU={'on' if HAVE_CUPY else 'off'})")

    if N_PROCESSES > 1:
        with Pool(processes=N_PROCESSES) as pool:
            rows = pool.map(run_one, jobs, chunksize=MP_CHUNKSIZE)
    else:
        rows = [run_one(j) for j in jobs]

    df = pd.DataFrame(rows)
    df.to_csv(CSV_SUMMARY, index=False)
    print(f"[DONE] Wrote {CSV_SUMMARY}")

if __name__ == "__main__":
    main()
