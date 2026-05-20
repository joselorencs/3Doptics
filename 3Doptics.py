# app.py - 3D OPTICS with Translation Scan surface visualization
import os
import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
import json
import streamlit.components.v1 as components
import uuid
import urllib.parse
import plotly.io as pio
import io

st.set_page_config(page_title="3D OPTICS", layout="wide", initial_sidebar_state="expanded")
capture_basename = "plot_view"

def extract_temp_from_filename(filename: str):
    """Extract temperature value from a filename."""
    name = filename
    name = name.replace("_", " ").replace("-", " ").replace(".", " ")
    num_re = r'([+-]?\d{1,4}(?:[.,]\d+)?)'
    patterns = [
        rf'{num_re}\s*(?:°\s*[Cc]|deg\s*[Cc]|C\b)',
        rf'{num_re}\s*(?:K\b)',
        rf'(?:T|temp|temperature)\s*[:=_]?\s*{num_re}',
        num_re
    ]
    for pat in patterns:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            raw = m.group(1)
            raw = raw.replace(",", ".")
            try:
                val = float(raw)
            except Exception:
                continue
            if 1900 <= int(math.floor(val)) <= 2100:
                continue
            if val < -500 or val > 5000:
                continue
            return val
    return None

def detect_scan_column_indices(df, expected_col_count=3):
    """Detects column indices for Translation Scan data (X, Y, Thickness)."""
    if df.shape[1] < expected_col_count:
        raise ValueError(f"Expected at least {expected_col_count} columns, found {df.shape[1]}")
    
    cols = [str(c).lower().strip() for c in df.columns]
    
    x_patterns = [r'\bx\b', r'x[\s_-]*(position|pos|axis)', r'eje\s*x']
    y_patterns = [r'\by\b', r'y[\s_-]*(position|pos|axis)', r'eje\s*y']
    thickness_patterns = [r'\bthickness\b', r'espesor', r'thicknes', r'\bt\b(?!ime)', r'thick']
    
    x_idx, y_idx, thickness_idx = None, None, None
    
    for i, col in enumerate(cols):
        if x_idx is None and any(re.search(pat, col) for pat in x_patterns):
            x_idx = i
        elif y_idx is None and any(re.search(pat, col) for pat in y_patterns):
            y_idx = i
        elif thickness_idx is None and any(re.search(pat, col) for pat in thickness_patterns):
            thickness_idx = i
    
    if x_idx is not None and y_idx is not None and thickness_idx is not None:
        return (x_idx, y_idx, thickness_idx)
    
    if x_idx is None:
        x_idx = 0
    if y_idx is None:
        y_idx = 1
    if thickness_idx is None:
        thickness_idx = 2
    
    return (x_idx, y_idx, thickness_idx)

def read_spectrum(file_like, scale_to_percent=True):
    """Reads a two-column file (wavelength, property)."""
    file_like.seek(0)
    df = None
    for sep in [";", ",", "\t"]:
        try:
            file_like.seek(0)
            df = pd.read_csv(file_like, sep=sep, header=0)
            if df.shape[1] >= 2:
                break
        except Exception:
            file_like.seek(0)
            continue

    if df is None or df.shape[1] < 2:
        raise ValueError("File must have two columns (wavelength and property).")

    df = df.iloc[:, :2].copy()
    df.columns = ["wavelength", "prop"]
    df = df.dropna().sort_values("wavelength")

    df = df.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)

    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
    df["prop"] = pd.to_numeric(df["prop"], errors="coerce")
    df = df.dropna()

    if df.empty:
        raise ValueError("No valid numeric data found in file.")

    if scale_to_percent:
        if df["prop"].max() <= 1.01:
            df["prop"] *= 100.0

    return df

def read_translation_scan(file_like):
    """Reads a three-column file (X position, Y position, Thickness)."""
    file_like.seek(0)
    df = None
    for sep in [";", ",", "\t"]:
        try:
            file_like.seek(0)
            df = pd.read_csv(file_like, sep=sep, header=0)
            if df.shape[1] >= 3:
                break
        except Exception:
            file_like.seek(0)
            continue

    if df is None or df.shape[1] < 3:
        raise ValueError("File must have at least three columns (X, Y, Thickness).")

    x_idx, y_idx, thickness_idx = detect_scan_column_indices(df)
    
    df = df.iloc[:, [x_idx, y_idx, thickness_idx]].copy()
    df.columns = ["X", "Y", "thickness"]
    
    df = df.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df["thickness"] = pd.to_numeric(df["thickness"], errors="coerce")
    df = df.dropna()

    if df.empty:
        raise ValueError("No valid numeric data found in file.")

    return df

def expand_uploaded_to_spectra(uploaded_files, scale_to_percent=True, mode_key="spectroscopy"):
    """Accepts multiple uploaded files and expands to spectra."""
    class InMemoryFile(io.StringIO):
        def __init__(self, s, name):
            super().__init__(s)
            self.name = name

    entries = []
    idx_global = 0

    for up in uploaded_files:
        try:
            text = up.getvalue().decode("utf-8", errors="ignore")
        except Exception:
            text = up.read().decode("utf-8", errors="ignore")

        df_raw = None
        for sep in [";", ",", "\t"]:
            try:
                df_try = pd.read_csv(io.StringIO(text), sep=sep, header=0)
                if df_try.shape[1] >= 2:
                    df_raw = df_try
                    break
            except Exception:
                continue

        if df_raw is None or df_raw.shape[1] < 2:
            st.warning(f"File **{up.name}** must have at least two columns. Skipping.")
            continue

        df_raw = df_raw.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)

        wl = pd.to_numeric(df_raw.iloc[:, 0], errors="coerce")
        if wl.dropna().empty:
            st.warning(f"Could not parse wavelength column in **{up.name}**. Skipping.")
            continue

        for j in range(1, df_raw.shape[1]):
            col_series = pd.to_numeric(df_raw.iloc[:, j], errors="coerce")
            if col_series.dropna().empty:
                continue

            df_two = pd.DataFrame({"wavelength": wl, "prop": col_series}).dropna().sort_values("wavelength")
            if df_two.empty:
                continue

            if scale_to_percent and df_two["prop"].max() <= 1.01:
                df_two["prop"] *= 100.0

            col_label = str(df_raw.columns[j])
            temp_guess = None
            parsed_from_col = extract_temp_from_filename(col_label)
            if parsed_from_col is not None:
                temp_guess = float(parsed_from_col)
            else:
                parsed_from_name = extract_temp_from_filename(up.name)
                if parsed_from_name is not None:
                    temp_guess = float(parsed_from_name)

            csv_text = df_two.to_csv(index=False)
            synthetic_name = f"{up.name}__col{j}__{col_label}"
            memfile = InMemoryFile(csv_text, synthetic_name)

            temp_key = f"{mode_key}_temp_{idx_global}"
            idx_global += 1

            entries.append({
                "fileobj": memfile,
                "label": f"{up.name} — column: {col_label}",
                "temp_key": temp_key,
                "temp_default": float(temp_guess) if temp_guess is not None else None
            })

    return entries

# Sidebar
st.sidebar.title("3D OPTICS: Controls")
st.sidebar.markdown("Global settings for processing and display")

resolution = st.sidebar.slider("Wavelength grid resolution", 200, 4000, 1000, step=100)
use_overlap = st.sidebar.checkbox("Use overlapped wavelength range (recommended)", value=True)
apply_smooth = st.sidebar.checkbox("Apply Savitzky–Golay smoothing (optional)", value=False)
if apply_smooth:
    window = st.sidebar.slider("S-G window length (odd)", 5, 101, 11, step=2)
    poly = st.sidebar.slider("S-G polynomial order", 1, 5, 3)
else:
    window = None
    poly = None

st.sidebar.markdown("---")
st.sidebar.header("3D colorscale")
colorscales = ["Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo",
               "Jet", "Hot", "Blues", "Reds", "YlGnBu", "RdBu", "Viridis_r"]
colorscale_choice = st.sidebar.selectbox("Choose colorscale", options=colorscales, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **José Lorenzo Calderón Solís**")
st.sidebar.markdown('[📧 Contact: joselorencs@gmail.com](mailto:joselorencs@gmail.com)')

# Main
st.title("3D OPTICS")
st.markdown("Interactive 3D viewer for **Spectroscopy** and **Ellipsometry** datasets.")

if "main_mode" not in st.session_state:
    st.session_state.main_mode = "Home"

main_mode = st.radio("Select workspace", options=["Home", "Spectroscopy", "Ellipsometry"], 
                     index=["Home","Spectroscopy","Ellipsometry"].index(st.session_state.main_mode))
st.session_state.main_mode = main_mode

def nearest_index(val, lst):
    arr = np.array(lst, dtype=float)
    return int(np.abs(arr - float(val)).argmin())

def process_and_display_translation_scan(file_objs, options, mode_key="translation_scan"):
    """Processes Translation Scan data and displays 3D surface with interpolation."""
    signature = tuple((f.name,) for f in file_objs)

    force_key = f"{mode_key}_force_recompute"
    sig_key = f"{mode_key}_scan_signature"
    data_key = f"{mode_key}_scan_data"

    gen_btn_key = f"{mode_key}_generate_btn"
    if st.button("Generate 3D map", key=gen_btn_key):
        st.session_state[force_key] = True

    need_process = False
    if sig_key not in st.session_state:
        need_process = True
    elif st.session_state.get(sig_key) != signature:
        need_process = True
    elif st.session_state.get(force_key, False):
        need_process = True

    if need_process:
        st.session_state[force_key] = False
        try:
            all_data = []
            for f in file_objs:
                f.seek(0)
                df = read_translation_scan(f)
                all_data.append(df)

            scan_data = pd.concat(all_data, ignore_index=True)
            
            if scan_data.empty:
                st.error("No valid data found in uploaded files.")
                return

            st.session_state[sig_key] = signature
            st.session_state[data_key] = scan_data

        except Exception as e:
            st.error(f"Error processing translation scan data: {e}")
            return

    if data_key in st.session_state:
        scan_data = st.session_state[data_key]

        # Create interpolated surface
        x_pts = scan_data["X"].values
        y_pts = scan_data["Y"].values
        z_pts = scan_data["thickness"].values
        
        x_min, x_max = x_pts.min(), x_pts.max()
        y_min, y_max = y_pts.min(), y_pts.max()
        
        xi = np.linspace(x_min, x_max, 50)
        yi = np.linspace(y_min, y_max, 50)
        XI, YI = np.meshgrid(xi, yi)
        
        points = np.column_stack([x_pts, y_pts])
        ZI = griddata(points, z_pts, (XI, YI), method='cubic', fill_value=np.mean(z_pts))
        
        fig = go.Figure(
            data=[
                go.Surface(
                    x=XI,
                    y=YI,
                    z=ZI,
                    surfacecolor=ZI,
                    colorscale=options.get("colorscale", "Viridis"),
                    colorbar=dict(title="Thickness (nm)"),
                    hovertemplate="X: %{x:.3f} cm<br>Y: %{y:.3f} cm<br>Thickness: %{z:.2f} nm<extra></extra>"
                )
            ]
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="X-position (cm)",
                yaxis_title="Y-position (cm)",
                zaxis_title="Thickness (nm)"
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=700,
        )

        uid = uuid.uuid4().hex[:8]
        div_container = f"plotly-container-{mode_key}-{uid}"
        div_query_selector = f"#{div_container} .plotly-graph-div"

        fig_html_fragment = pio.to_html(fig, include_plotlyjs='cdn', full_html=False, default_height=700)

        html = f'''
        <div id="{div_container}" style="width:100%;">{fig_html_fragment}</div>
        <div style="margin-top:8px;">
        <button id="capture-btn-{uid}">Capture image</button>
        </div>
        '''

        components.html(html, height=820, scrolling=True)

        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("X range (cm)", f"{scan_data['X'].min():.3f} to {scan_data['X'].max():.3f}")
        with col2:
            st.metric("Y range (cm)", f"{scan_data['Y'].min():.3f} to {scan_data['Y'].max():.3f}")
        with col3:
            st.metric("Thickness range (nm)", f"{scan_data['thickness'].min():.2f} to {scan_data['thickness'].max():.2f}")

        csv_bytes = scan_data.to_csv(index=False).encode("utf-8")
        st.download_button("Download processed data (CSV)", data=csv_bytes, file_name="translation_scan_data.csv", mime="text/csv")

    else:
        st.info("No processed data available. Click 'Generate 3D map' to process the uploaded files.")

# HOME
if main_mode == "Home":
    st.header("Welcome to 3D OPTICS")
    banner_path = "3Doptics_banner.png"
    if os.path.exists(banner_path):
        st.image(banner_path, width="stretch")
    else:
        st.info("Insert your banner image to show it here.")

# ELLIPSOMETRY
elif main_mode == "Ellipsometry":
    st.header("Ellipsometry - Translation Scan Mode")
    st.markdown("Upload translation scan data with three columns: X position, Y position, and Thickness.")
    
    uploaded = st.file_uploader(
        "Upload translation scan files (CSV/TXT/DAT). Each file should contain three columns: X, Y, Thickness.",
        accept_multiple_files=True,
        type=["csv", "txt", "dat"],
        key="translation_uploader"
    )

    if uploaded:
        mode_key = "translation_scan"
        file_objs = list(uploaded)
        options = {"colorscale": colorscale_choice}
        
        process_and_display_translation_scan(
            file_objs,
            options,
            mode_key=mode_key
        )
    else:
        st.info("Upload files to enable processing.")

# SPECTROSCOPY
elif main_mode == "Spectroscopy":
    st.header("Spectroscopy")
    st.info("Spectroscopy mode coming soon...")
