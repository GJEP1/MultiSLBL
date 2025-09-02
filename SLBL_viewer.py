#!/usr/bin/env python
"""
SLBL Scenario Viewer

"""

from __future__ import annotations
import os, glob, io, re, math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional: map polygon support
try:
    import geopandas as gpd
except Exception:
    gpd = None

# ---------------------------
# Defaults
# ---------------------------
DEFAULT_SUMMARY    = r"D:\Python\SLBL\path\outputs\slbl_summary.csv"
DEFAULT_SCEN_DIR   = r"D:\Python\SLBL\path\scenarios"
DEFAULT_XSECT_ROOT = r"D:\Python\SLBL\path\outputs\xsections"

st.set_page_config(page_title="SLBL Scenario Viewer", layout="wide")
st.title("SLBL Scenario Viewer")

# ----------------------------------------
# Sidebar containers (control visual order)
# ----------------------------------------
inc_box    = st.sidebar.container()   # Inclusion manager (first)
inputs_box = st.sidebar.container()   # Inputs
plot_box   = st.sidebar.container()   # Plot options
layout_box = st.sidebar.container()   # Layout

# ---------------------------
# Inputs
# ---------------------------
with inputs_box:
    st.header("Inputs")
    summary_path = st.text_input("Summary CSV", DEFAULT_SUMMARY)
    scen_dir     = st.text_input("Scenario shapefile folder", DEFAULT_SCEN_DIR)
    xsect_root   = st.text_input("X-sections root", DEFAULT_XSECT_ROOT)

with plot_box:
    st.header("Plot options")
    smooth_on  = st.checkbox("Smooth cross-sections", value=True)
    smooth_win = st.slider("Smoothing window (samples)", 3, 301, 31, step=2)
    st.caption("Centered rolling mean. Window must be odd.")

with layout_box:
    st.header("Layout")
    show_map_tile = st.checkbox("Show map + crosslines tile", value=True)

# ---------------------------
# Helpers
# ---------------------------
def _safe_read_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)

def find_xsect_files(xroot: str, scenario: str) -> list[str]:
    folder = os.path.join(xroot, scenario)
    if not os.path.isdir(folder):
        return []
    return sorted(glob.glob(os.path.join(folder, f"{scenario}_*_xsect_*.csv")))

def parse_label_from_filename(filepath: str, scenario: str) -> str:
    base = os.path.basename(filepath)
    if not base.startswith(scenario + "_"):
        return ""
    return base[len(scenario) + 1:].split("_xsect_")[0]

def parse_lineid_from_filename(filepath: str) -> str:
    base = os.path.basename(filepath)
    try:
        return base.split("_xsect_")[1].split("_p")[0]
    except Exception:
        return "line"

# --- Robust label parser (tol, md, K or Ku/Kl) ---
def parse_params_from_label(lab: str) -> dict:
    # tolXXcm or tol0.03m
    tol = None
    m_tol = re.search(r"tol(\d+(?:\.\d+)?)(cm|m)", lab)
    if m_tol:
        val = float(m_tol.group(1))
        tol = val / 100.0 if m_tol.group(2) == "cm" else val

    # mdXm or nolimit
    md = None
    m_md = re.search(r"_(md(\d+(?:\.\d+)?)m|nolimit)_", lab)
    if m_md:
        if m_md.group(1).startswith("md"):
            md = float(m_md.group(2))
        else:
            md = None  # nolimit

    # gradated fk{Ku}to{Kl}pxg
    m_g = re.search(r"fk(\d+)to(\d+)pxg", lab)
    if m_g:
        return {"tol": tol, "md": md, "grad": True,
                "Ku": int(m_g.group(1)), "Kl": int(m_g.group(2)), "K": None}

    # constant fk{K}px (ensure not the gradated one)
    m_c = re.search(r"fk(\d+)px(?!g)", lab)
    if m_c:
        return {"tol": tol, "md": md, "grad": False,
                "Ku": None, "Kl": None, "K": int(m_c.group(1))}

    return {"tol": tol, "md": md, "grad": None, "Ku": None, "Kl": None, "K": None}

def png_from_matplotlib(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

# ---- Build crossline geometries with IDs + orientation metadata ----
def _build_xlines_indexed(csv_paths: list[str]) -> dict:
    lines: dict[str, dict] = {}
    tmp: dict[str, list[pd.DataFrame]] = {}
    for p in csv_paths:
        lid = parse_lineid_from_filename(p)
        try:
            dfp = _safe_read_csv(p)[["X","Y","LinePart"]]
        except Exception:
            continue
        for c in ["X","Y","LinePart"]:
            if c in dfp.columns:
                dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
        dfp = dfp.dropna(subset=["X","Y"])
        tmp.setdefault(lid, []).append(dfp)

    for lid, parts in tmp.items():
        try:
            parts_sorted = sorted(parts, key=lambda d: d["LinePart"].iloc[0] if "LinePart" in d.columns else 0)
        except Exception:
            parts_sorted = parts
        xy_list = []
        for dfp in parts_sorted:
            if {"X","Y"}.issubset(dfp.columns) and len(dfp) >= 2:
                xy_list.append(dfp[["X","Y"]].to_numpy())
        if not xy_list:
            continue
        xy = np.vstack(xy_list)

        # Endpoints + dominant orientation
        i_w = int(np.nanargmin(xy[:,0])); i_e = int(np.nanargmax(xy[:,0]))
        west_pt = (float(xy[i_w,0]), float(xy[i_w,1]))
        east_pt = (float(xy[i_e,0]), float(xy[i_e,1]))
        dx, dy = east_pt[0]-west_pt[0], east_pt[1]-west_pt[1]
        ang = abs(math.degrees(math.atan2(dy, dx)))
        if ang > 90: ang = 180 - ang
        orient = "WE" if ang <= 45 else "NS"

        lines[lid] = {"xy": xy, "west_pt": west_pt, "east_pt": east_pt, "orient": orient}
    return lines

# ---- Map plotting (smaller) with subtle IDs + W/E/N/S + thin arrows ----
def plot_polygon_outline(shp_path: str, xlines_idx: dict | None = None) -> bytes | None:
    if gpd is None or not os.path.exists(shp_path):
        return None
    try:
        gdf = gpd.read_file(shp_path)
        if gdf.empty:
            return None
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        gdf.plot(ax=ax, facecolor=(0,0,0,0), edgecolor="black", linewidth=1.0)

        if xlines_idx:
            for lid, info in xlines_idx.items():
                xy = info["xy"]
                if isinstance(xy, np.ndarray) and xy.ndim == 2 and xy.shape[0] >= 2:
                    ax.plot(xy[:,0], xy[:,1], linewidth=1.0, color="black")
                    wx, wy = info["west_pt"]; ex, ey = info["east_pt"]
                    if info.get("orient") == "WE":
                        labels = ("W","E"); start = (wx, wy); end = (ex, ey)
                    else:
                        # N–S by Y
                        if wy <= ey:
                            start = (wx, wy); end = (ex, ey); labels = ("S","N")
                        else:
                            start = (ex, ey); end = (wx, wy); labels = ("S","N")
                    ax.text(start[0], start[1], labels[0], fontsize=7, color="dimgray", ha="right", va="bottom")
                    ax.text(end[0], end[1],   labels[1], fontsize=7, color="dimgray", ha="left",  va="bottom")
                    ax.annotate("", xy=end, xytext=start,
                                arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.45, color="dimgray"))
                    ax.text(end[0], end[1], f" {lid}", fontsize=7, color="gray", ha="left", va="center")

        ax.set_aspect('equal','box')
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title(os.path.basename(shp_path))
        ax.tick_params(axis='both', which='both', labelsize=7)
        return png_from_matplotlib(fig)
    except Exception:
        return None

# ---- Cross-section plotting with orientation (WE => W→E; NS => S→N) ----
def plot_xsection(df: pd.DataFrame, title: str, smooth: bool, window: int, orient: str) -> bytes:
    df = df.copy()
    for c in ["Distance_m","DEM_z","Base_z","X","Y"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Distance_m" in df.columns:
        df = df.sort_values("Distance_m")

    # Enforce display direction
    if orient == "WE":
        if "X" in df.columns and df["X"].notna().sum() >= 2:
            if df["X"].iloc[-1] < df["X"].iloc[0]:
                df = df.iloc[::-1].copy()
                if "Distance_m" in df.columns and df["Distance_m"].notna().any():
                    maxd = float(df["Distance_m"].max()); df["Distance_m"] = maxd - df["Distance_m"]
                if "Distance_m" in df.columns:
                    df = df.sort_values("Distance_m")
        xlab = "Distance (m)  (W → E)"; tdir = "(W → E)"
    else:
        if "Y" in df.columns and df["Y"].notna().sum() >= 2:
            if df["Y"].iloc[-1] < df["Y"].iloc[0]:
                df = df.iloc[::-1].copy()
                if "Distance_m" in df.columns and df["Distance_m"].notna().any():
                    maxd = float(df["Distance_m"].max()); df["Distance_m"] = maxd - df["Distance_m"]
                if "Distance_m" in df.columns:
                    df = df.sort_values("Distance_m")
        xlab = "Distance (m)  (S → N)"; tdir = "(S → N)"

    if smooth and window >= 3 and window % 2 == 1:
        if "DEM_z" in df.columns:  df["DEM_z"]  = df["DEM_z"].rolling(window, center=True, min_periods=1).mean()
        if "Base_z" in df.columns: df["Base_z"] = df["Base_z"].rolling(window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(df["Distance_m"], df["DEM_z"], label="DEM_z")
    ax.plot(df["Distance_m"], df["Base_z"], label="Base_z")
    if {"Distance_m","DEM_z","Base_z"}.issubset(df.columns):
        dist = df["Distance_m"].to_numpy(); demz = df["DEM_z"].to_numpy(); basez = df["Base_z"].to_numpy()
        mask = np.isfinite(dist) & np.isfinite(demz) & np.isfinite(basez)
        try:
            ax.fill_between(dist[mask], basez[mask], demz[mask], where=(demz[mask] >= basez[mask]), alpha=0.3)
        except Exception:
            pass
    ax.set_xlabel(xlab); ax.set_ylabel("Elevation (m)"); ax.set_title(f"{title}  {tdir}")
    ax.legend()
    return png_from_matplotlib(fig)

# ---------------------------
# Load summary & normalize types
# ---------------------------
if not os.path.isfile(summary_path):
    st.warning("Summary CSV not found. Set a valid path in the sidebar.")
    st.stop()
summary = _safe_read_csv(summary_path)
if summary.empty:
    st.error("Summary CSV appears empty.")
    st.stop()

# Type normalization for matching
def _coerce_bool(s):
    return s.astype(str).str.lower().isin(["true","1","yes","y","t"])

for col in ["Tol_m","Feather_K_px","Neighbours","MaxDepthLimit_m","MaxDepth_m",
            "MeanDepth_m","Volume_m3","XSection_CSVs",
            "Feather_K_upper_px","Feather_K_lower_px"]:
    if col in summary.columns:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
if "Feather_Gradated" in summary.columns:
    summary["Feather_Gradated"] = _coerce_bool(summary["Feather_Gradated"]).fillna(False)

# Scenario list
scenarios = sorted(summary["Scenario"].dropna().unique().tolist())

# Build Scenario -> available Labels (from x-sections; fallback to summary)
label_index: dict[str, list[str]] = {}
for scn in scenarios:
    files = find_xsect_files(xsect_root, scn)
    labs = sorted({parse_label_from_filename(f, scn) for f in files if parse_label_from_filename(f, scn)})
    if not labs:
        # fallback from summary if needed
        sub = summary[summary["Scenario"] == scn]
        labs = sorted([str(l) for l in sub.get("Label", pd.Series(dtype=str)).dropna().unique().tolist()])
    label_index[scn] = labs

# ---------------------------
# Inclusion manager (TOP of sidebar, expanded)
# ---------------------------
with inc_box:
    st.header("Inclusion manager")
    if "include_map" not in st.session_state:
        st.session_state.include_map = {scn: {lab: True for lab in label_index.get(scn, [])} for scn in scenarios}
    for scn in scenarios:
        labs = label_index.get(scn, [])
        if not labs:
            continue
        included_n = sum(st.session_state.include_map.get(scn, {}).values())
        with st.expander(f"{scn} ({included_n}/{len(labs)} included)", expanded=True):
            cols = st.columns([1,1,1])
            if cols[0].button("All", key=f"{scn}_all"):
                st.session_state.include_map[scn] = {lab: True for lab in labs}
            if cols[1].button("None", key=f"{scn}_none"):
                st.session_state.include_map[scn] = {lab: False for lab in labs}
            if cols[2].button("Invert", key=f"{scn}_inv"):
                st.session_state.include_map[scn] = {lab: not st.session_state.include_map[scn].get(lab, False) for lab in labs}
            for lab in labs:
                cur = st.session_state.include_map[scn].get(lab, True)
                st.session_state.include_map[scn][lab] = st.checkbox(lab, value=cur, key=f"{scn}__{lab}")

# ---------------------------
# Scenario picker + label navigator
# ---------------------------
left, center = st.columns([1.1, 3.2])
with left:
    st.subheader("Scenarios")
    scenario = st.selectbox("Pick scenario", scenarios, key="scenario_select")

# Scenario-specific summary & sorted order
sub = summary[summary["Scenario"] == scenario].copy()
if "Tol_m" in sub.columns: sub["Tol_m"] = pd.to_numeric(sub["Tol_m"], errors="coerce")
if "Feather_K_px" in sub.columns: sub["Feather_K_px"] = pd.to_numeric(sub["Feather_K_px"], errors="coerce")
sub = sub.sort_values(["Tol_m","Feather_K_px"], ascending=[True, True])

# Label -> x-section files
xs_files = find_xsect_files(xsect_root, scenario)
label_map: dict[str, list[str]] = {}
for f in xs_files:
    lab = parse_label_from_filename(f, scenario)
    if lab:
        label_map.setdefault(lab, []).append(f)
labels = sorted([l for l in label_map.keys() if l])

# --- Matching logic: prefer exact Label, else parse + match ---
def pick_stats_row_for_label(sub_df: pd.DataFrame, label: str) -> pd.Series | None:
    # 1) Exact label match if column exists
    if "Label" in sub_df.columns:
        cand = sub_df[sub_df["Label"].astype(str) == label]
        if not cand.empty:
            return cand.iloc[0]

    # 2) Parse & match
    p = parse_params_from_label(label)
    cand = sub_df.copy()

    # tol
    if p["tol"] is not None and "Tol_m" in cand.columns:
        cand = cand[np.isclose(cand["Tol_m"].astype(float), float(p["tol"]), rtol=0, atol=1e-12)]

    # max depth
    if "MaxDepthLimit_m" in cand.columns:
        if p["md"] is None:
            cand = cand[(cand["MaxDepthLimit_m"].isna()) | (cand["MaxDepthLimit_m"].astype(str).eq(""))]
        else:
            cand = cand[np.isclose(cand["MaxDepthLimit_m"].astype(float), float(p["md"]), rtol=0, atol=1e-9)]

    # K
    if p["grad"] is True and all(col in cand.columns for col in ["Feather_Gradated","Feather_K_upper_px","Feather_K_lower_px"]):
        cand = cand[(cand["Feather_Gradated"] == True) &
                    (cand["Feather_K_upper_px"].astype(float) == float(p["Ku"])) &
                    (cand["Feather_K_lower_px"].astype(float) == float(p["Kl"]))]

    elif p["grad"] is False and "Feather_K_px" in cand.columns:
        # accept gradated==False or missing/NaN
        if "Feather_Gradated" in cand.columns:
            ok_grad = (cand["Feather_Gradated"] == False) | cand["Feather_Gradated"].isna()
            cand = cand[ok_grad]
        cand = cand[cand["Feather_K_px"].astype(float) == float(p["K"])]

    if not cand.empty:
        return cand.iloc[0]
    return sub_df.iloc[0] if not sub_df.empty else None

# Session state for label index
if "_last_scn" not in st.session_state or st.session_state._last_scn != scenario:
    st.session_state.label_idx = 0
    st.session_state._last_scn = scenario
if labels:
    st.session_state.label_idx = max(0, min(st.session_state.get("label_idx", 0), len(labels)-1))

with center:
    st.subheader("Parameters (Label)")
    if not labels:
        st.info("No cross-section CSVs found for this scenario.")
    else:
        c1, c2, c3 = st.columns([1,3,1])
        if c1.button("← Prev", use_container_width=True):
            st.session_state.label_idx = (st.session_state.label_idx - 1) % len(labels)
        chosen = c2.selectbox(" ", labels, index=st.session_state.label_idx, label_visibility="collapsed")
        st.session_state.label_idx = labels.index(chosen)
        if c3.button("Next →", use_container_width=True):
            st.session_state.label_idx = (st.session_state.label_idx + 1) % len(labels)
        label = labels[st.session_state.label_idx]
        st.markdown(f"**Selected:** `{label}`")
        stats_row = pick_stats_row_for_label(sub, label)

# Optional map tile
map_png = None
xlines_idx = {}
shp_path = os.path.join(scen_dir, f"{scenario}.shp")
if labels and show_map_tile:
    parts_for_label = label_map.get(label, [])
    xlines_idx = _build_xlines_indexed(parts_for_label)
    map_png = plot_polygon_outline(shp_path, xlines_idx=xlines_idx if xlines_idx else None)

# ---------------------------
# Cross-sections grid + stats panel
# ---------------------------
st.markdown("---")
st.subheader("All cross-sections for this parameter set")
if labels:
    # Build LineID -> list of files for the chosen label
    line_map: dict[str, list[str]] = {}
    for f in label_map[label]:
        lid = parse_lineid_from_filename(f)
        line_map.setdefault(lid, []).append(f)
    try:
        line_ids = sorted(line_map.keys(), key=lambda x: int(x))
    except Exception:
        line_ids = sorted(line_map.keys())

    tiles: list[tuple[str, str|None, bytes|None]] = [("stats", None, None)]
    for lid in line_ids:
        parts = sorted(line_map[lid])
        dfs = [_safe_read_csv(p) for p in parts]
        xs_df = pd.concat(dfs, ignore_index=True)
        orient = xlines_idx.get(lid, {}).get("orient", "WE")
        xs_png = plot_xsection(xs_df, title=f"{scenario} — {label} — line {lid}",
                               smooth=smooth_on, window=smooth_win, orient=orient)
        tiles.append(("xs", str(lid), xs_png))
    if show_map_tile:
        tiles.append(("map", None, None))

    # Render 2-column grid
    for i in range(0, len(tiles), 2):
        colA, colB = st.columns(2)
        for col, (kind, lid, payload) in zip((colA, colB), tiles[i:i+2]):
            if kind == "stats":
                if stats_row is None:
                    col.info("No stats found for this label.")
                    continue
                present = [c for c in [
                    "Tol_m","Feather_K_px","Feather_Gradated","Feather_K_upper_px","Feather_K_lower_px",
                    "Neighbours","MaxDepthLimit_m","MaxDepth_m","MeanDepth_m","Volume_m3","XSection_CSVs",
                    "Gradation_Mode","Gradation_Zsplit","Gradation_RowSplit","Label"
                ] if c in getattr(stats_row, "index", [])]

                col.markdown(f"### {scenario}")
                col.caption(f"Parameter set: `{label}`")

                mcols = col.columns(3)
                vol_val = float(stats_row["Volume_m3"]) if "Volume_m3" in present and pd.notna(stats_row["Volume_m3"]) else None
                mcols[0].metric("Volume", f"{vol_val/1e6:.3f} million m³" if vol_val is not None else "—")
                mcols[1].metric("Max depth (m)", f"{float(stats_row['MaxDepth_m']):.3f}" if "MaxDepth_m" in present and pd.notna(stats_row["MaxDepth_m"]) else "—")
                mcols[2].metric("Mean depth (m)", f"{float(stats_row['MeanDepth_m']):.3f}" if "MeanDepth_m" in present and pd.notna(stats_row["MeanDepth_m"]) else "—")

                rest_cols = [c for c in present if c not in {"Volume_m3","MaxDepth_m","MeanDepth_m"}]
                if rest_cols:
                    col.dataframe(pd.DataFrame([stats_row[rest_cols]]), use_container_width=True)
                else:
                    col.info("No additional stats columns found.")

            elif kind == "xs":
                orient = xlines_idx.get(lid, {}).get("orient", "WE")
                cap = "W → E" if orient == "WE" else "S → N"
                col.image(payload, caption=f"Line {lid}  ({cap})", use_container_width=True)

            elif kind == "map":
                col.markdown("**Map + crosslines**  (subtle IDs at line end; W/E or S/N arrows)")
                if map_png is not None:
                    col.image(map_png, caption=os.path.basename(shp_path), use_container_width=True)
                else:
                    col.info("Polygon outline unavailable (missing shapefile or GeoPandas not installed).")

# ---------------------------
# Percentile calculations (per Scenario) — volumes only in million m³
# ---------------------------
st.markdown("---")
st.subheader("P10 / P50 / P90 by Scenario (based on included parameter sets)")

if not {"Volume_m3","MaxDepth_m","MeanDepth_m"}.intersection(summary.columns):
    st.info("No percentile-eligible columns found.")
else:
    rows = []
    for scn in scenarios:
        labs = label_index.get(scn, [])
        if not labs:
            continue
        include_flags = st.session_state.include_map.get(scn, {})
        sub_scn = summary[summary["Scenario"] == scn]

        stats_rows = []
        for lab in labs:
            if not include_flags.get(lab, True):
                continue
            r = pick_stats_row_for_label(sub_scn, lab)
            if r is not None and "Volume_m3" in r.index and pd.notna(r["Volume_m3"]):
                stats_rows.append(r)

        if not stats_rows:
            continue
        dfS = pd.DataFrame(stats_rows)

        def _pct(series: pd.Series, q: float):
            s = pd.to_numeric(series, errors="coerce")
            s = s[np.isfinite(s)]
            return float(np.nanpercentile(s, q)) if len(s) else np.nan

        out = {"Scenario": scn, "Included_count": len(dfS)}
        if "Volume_m3" in dfS.columns:
            out.update({
                "P10_Volume_million_m3": _pct(dfS["Volume_m3"], 10) / 1e6,
                "P50_Volume_million_m3": _pct(dfS["Volume_m3"], 50) / 1e6,
                "P90_Volume_million_m3": _pct(dfS["Volume_m3"], 90) / 1e6,
            })
        if "MaxDepth_m" in dfS.columns:
            out.update({
                "P10_MaxDepth_m": _pct(dfS["MaxDepth_m"], 10),
                "P50_MaxDepth_m": _pct(dfS["MaxDepth_m"], 50),
                "P90_MaxDepth_m": _pct(dfS["MaxDepth_m"], 90),
            })
        if "MeanDepth_m" in dfS.columns:
            out.update({
                "P10_MeanDepth_m": _pct(dfS["MeanDepth_m"], 10),
                "P50_MeanDepth_m": _pct(dfS["MeanDepth_m"], 50),
                "P90_MeanDepth_m": _pct(dfS["MeanDepth_m"], 90),
            })
        rows.append(out)

    if not rows:
        st.info("No scenarios have any included parameter sets selected.")
    else:
        pct_df = pd.DataFrame(rows).sort_values("Scenario").reset_index(drop=True)
        cols_order = ["Scenario", "Included_count"]
        vol_cols = [c for c in ["P10_Volume_million_m3","P50_Volume_million_m3","P90_Volume_million_m3"] if c in pct_df.columns]
        depth_cols = [c for c in [
            "P10_MaxDepth_m","P50_MaxDepth_m","P90_MaxDepth_m",
            "P10_MeanDepth_m","P50_MeanDepth_m","P90_MeanDepth_m"
        ] if c in pct_df.columns]
        pct_df = pct_df[cols_order + vol_cols + depth_cols]

        st.dataframe(pct_df, use_container_width=True)
        st.download_button("Download CSV",
                           data=pct_df.to_csv(index=False).encode("utf-8"),
                           file_name="scenario_percentiles.csv",
                           mime="text/csv")
