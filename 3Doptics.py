# app.py - 3D OPTICS (Integrated Transmittance & Ellipsometry) with filename temp extraction
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

# --- Page config
st.set_page_config(page_title="3D OPTICS", layout="wide", initial_sidebar_state="expanded")

# default base name for capture files (used as modal default)
capture_basename = "plot_view"

# ---------------------------
# Helper: extract temperature from filename
# ---------------------------
def extract_temp_from_filename(filename: str):
    """
    Try to extract a temperature value from a filename.
    Heuristics:
     - Prefer numbers followed by °C, C, degC or K.
     - Otherwise take the first numeric token plausible as temperature.
     - Ignore 4-digit years (1900-2100).
     - Accept comma or dot decimals.
    Returns float or None if not found.
    """
    name = filename
    # Normalize separators to spaces
    name = name.replace("_", " ").replace("-", " ").replace(".", " ")
    # Patterns: prefer with unit markers
    num_re = r'([+-]?\d{1,4}(?:[.,]\d+)?)'
    patterns = [
        rf'{num_re}\s*(?:°\s*[Cc]|deg\s*[Cc]|C\b)',    # e.g. 100C, 100 °C, 100degC
        rf'{num_re}\s*(?:K\b)',                       # Kelvin
        rf'(?:T|temp|temperature)\s*[:=_]?\s*{num_re}',# T=100 or temp_100
        num_re                                        # fallback: any number
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
            # ignore years like 1900-2100
            if 1900 <= int(math.floor(val)) <= 2100:
                continue
            # sanity bounds
            if val < -500 or val > 5000:
                continue
            return val
    return None

# ---------------------------
# Helper: detect column headers (supports X, Y, thickness/Psi/n variants)
# ---------------------------
def detect_scan_column_indices(df, expected_col_count=3):
    """
    Detects column indices for Translation Scan data (X, Y, and a 3rd property).
    The 3rd property can be: Thickness, Psi, or n.
    Supports variations like "X", "X position", "eje X", "X_position", etc.
    If headers can't be matched, assumes order: [X, Y, Property]
    Returns tuple: (x_idx, y_idx, property_idx, property_name) where property_name is 'thickness', 'psi', or 'n'.
    """
    if df.shape[1] < expected_col_count:
        raise ValueError(f"Expected at least {expected_col_count} columns, found {df.shape[1]}")
    
    cols = [str(c).lower().strip() for c in df.columns]
    
    # Define patterns for each column type
    x_patterns = [r'\bx\b', r'x[\s_-]*(position|pos|axis)', r'eje\s*x']
    y_patterns = [r'\by\b', r'y[\s_-]*(position|pos|axis)', r'eje\s*y']
    thickness_patterns = [r'\bthickness\b', r'espesor', r'thicknes', r'\bt\b(?!ime)', r'thick']
    psi_patterns = [r'\bpsi\b', r'ψ']
    n_patterns = [r'\bn\b', r'n[\s_-]*(refractive|index|ri)', r'refractive[\s_-]*index']
    
    x_idx, y_idx, property_idx = None, None, None
    property_name = 'thickness'  # default
    
    # Try to match column headers
    for i, col in enumerate(cols):
        if x_idx is None and any(re.search(pat, col) for pat in x_patterns):
            x_idx = i
        elif y_idx is None and any(re.search(pat, col) for pat in y_patterns):
            y_idx = i
        elif property_idx is None:
            # Check for specific property types
            if any(re.search(pat, col) for pat in psi_patterns):
                property_idx = i
                property_name = 'psi'
            elif any(re.search(pat, col) for pat in n_patterns):
                property_idx = i
                property_name = 'n'
            elif any(re.search(pat, col) for pat in thickness_patterns):
                property_idx = i
                property_name = 'thickness'
    
    # If all found, return
    if x_idx is not None and y_idx is not None and property_idx is not None:
        return (x_idx, y_idx, property_idx, property_name)
    
    # Fallback: assume order X, Y, Property (first 3 columns)
    if x_idx is None:
        x_idx = 0
    if y_idx is None:
        y_idx = 1
    if property_idx is None:
        property_idx = 2
    
    return (x_idx, y_idx, property_idx, property_name)

# ---------------------------
# Helper: read file as dataframe (supports CSV, TXT, DAT, XLSX)
# ---------------------------
def read_file_to_dataframe(file_like, filename):
    """
    Reads various file formats (CSV, TXT, DAT, XLSX) and returns a dataframe.
    Supports multiple separators for text files.
    """
    # Check file extension
    file_ext = os.path.splitext(filename)[1].lower()
    
    # Try Excel format first if .xlsx
    if file_ext == '.xlsx':
        try:
            file_like.seek(0)
            df = pd.read_excel(file_like, header=0)
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            st.warning(f"Could not read **{filename}** as Excel file: {e}")
            return None
    
    # Try CSV/TXT format with various separators
    df = None
    for sep in [";", ",", "\t"]:
        try:
            file_like.seek(0)
            df = pd.read_csv(file_like, sep=sep, header=0)
            if df.shape[1] >= 2:
                return df
        except Exception:
            file_like.seek(0)
            continue
    
    return None

# ---------------------------
# Helper: spectrum reader (robust to separators; conditional scaling to percent)
# ---------------------------
def read_spectrum(file_like, filename, scale_to_percent=True):
    """
    Reads a two-column file (wavelength, property) from CSV/TXT/XLSX formats.
    If scale_to_percent==True and values are in [0,1], multiply by 100.
    Returns dataframe with columns: wavelength, prop
    """
    df = read_file_to_dataframe(file_like, filename)
    
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

    # Scale fractions to percentage only if requested
    if scale_to_percent:
        if df["prop"].max() <= 1.01:
            df["prop"] *= 100.0

    return df

# ---------------------------
# Helper: read translation scan data (X, Y, and Property: Thickness/Psi/n)
# ---------------------------
def read_translation_scan(file_like, filename):
    """
    Reads a three-column file (X position, Y position, and a 3rd property).
    The 3rd property can be: Thickness (nm), Psi (°), or n (refractive index).
    Supports CSV/TXT/XLSX formats with various separators and comma/dot decimals.
    Allows negative values for X and Y.
    Returns dataframe with columns: X, Y, and property_value (plus property_name attribute).
    """
    df = read_file_to_dataframe(file_like, filename)
    
    if df is None or df.shape[1] < 3:
        raise ValueError("File must have at least three columns (X, Y, Property).")

    # Detect column indices and property type
    x_idx, y_idx, property_idx, property_name = detect_scan_column_indices(df)
    
    df = df.iloc[:, [x_idx, y_idx, property_idx]].copy()
    df.columns = ["X", "Y", "property"]
    
    # Replace decimal commas with dots
    df = df.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df["property"] = pd.to_numeric(df["property"], errors="coerce")
    df = df.dropna()

    if df.empty:
        raise ValueError("No valid numeric data found in file.")

    # Store property_name as an attribute
    df.property_name = property_name

    return df

# ---------------------------
# Helper: one file reader
# ---------------------------
def expand_uploaded_to_spectra(uploaded_files, scale_to_percent=True, mode_key="spectroscopy"):
    """
    Accepts multiple uploaded files (CSV, TXT, DAT, XLSX).
    Each file can contain:
      - Two columns (wavelength, property)  → one spectrum
      - One wavelength + multiple property columns → multiple spectra from same file

    Returns a list of entries ready for processing.
    """
    class InMemoryFile(io.StringIO):
        def __init__(self, s, name):
            super().__init__(s)
            self.name = name

    entries = []
    idx_global = 0

    for up in uploaded_files:
        try:
            # Read file to dataframe
            df_raw = read_file_to_dataframe(up, up.name)
            
            if df_raw is None or df_raw.shape[1] < 2:
                st.warning(f"File **{up.name}** must have at least two columns (wavelength + property). Skipping.")
                continue

            # Replace decimal commas with dots
            df_raw = df_raw.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)

            wl = pd.to_numeric(df_raw.iloc[:, 0], errors="coerce")
            if wl.dropna().empty:
                st.warning(f"Could not parse wavelength column in file **{up.name}**. Skipping file.")
                continue

            # Process each property column
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
        
        except Exception as e:
            st.warning(f"Error processing file **{up.name}**: {e}")
            continue

    return entries

# ---------------------------
# Sidebar (global)
# ---------------------------
st.sidebar.title("3D OPTICS: Controls")
st.sidebar.markdown("Global settings for processing and display")

resolution = st.sidebar.slider("Wavelength grid resolution", 200, 4000, 1000, step=100)
use_overlap = st.sidebar.checkbox("Use overlapped wavelength range (recommended)", value=True)
apply_smooth = st.sidebar.checkbox("Apply Savitzky–Goyal smoothing (optional)", value=False)
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
st.sidebar.markdown('[💼 LinkedIn](https://www.linkedin.com/in/jos%C3%A9-lorenzo-calder%C3%B3n-sol%C3%ADs-44b086341/)')

# ---------------------------
# Main header / navigation
# ---------------------------
st.title("3D OPTICS")
st.markdown("Interactive 3D viewer for **Spectroscopy** and **Ellipsometry** datasets as a function of temperature.")
st.markdown("You can upload **multiple two-column files** (wavelength (nm) and optical property) registered at different temperatures or **one file with more than two columns** (the first one for wavelength).")
st.markdown("""
**CHOOSE A WORKING MODE** BELOW TO START:
""")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Spectroscopy")
    st.write("Upload spectra measured at different temperatures (°C). Modes: Transmittance, Absorbance, Reflectance.")
with c2:
    st.subheader("Ellipsometry")
    st.write("Upload ellipsometric data (Ψ, Δ, n, k) measured at different temperatures (°C), or Translation Scan data.")

if "main_mode" not in st.session_state:
    st.session_state.main_mode = "Home"

main_mode = st.radio("Select workspace", options=["Home", "Spectroscopy", "Ellipsometry"], index=["Home","Spectroscopy","Ellipsometry"].index(st.session_state.main_mode))
st.session_state.main_mode = main_mode

# ---------------------------
# Utility: nearest index for floats
# ---------------------------
def nearest_index(val, lst):
    arr = np.array(lst, dtype=float)
    return int(np.abs(arr - float(val)).argmin())

# ---------------------------
# Core processing/display function (updated)
# ---------------------------
def process_and_display_spectra(file_objs, temps, options, property_name="Transmittance", mode_key="spectroscopy", is_percent=True):
    """
    Processes uploaded files and displays 3D surface, 2D heatmap and 2D cut.
    - is_percent: True  -> label units " (%)" and scale 0..1 -> 0..100 in reader
                  False -> no unit suffix and no automatic scaling; accepts any numeric values
    Caches results per mode_key in st.session_state to avoid recompute.
    The colorscale represents TEMPERATURE (°C), not the property values.
    """
    # Build signature for current inputs
    signature = tuple((f.name, float(t)) for f, t in zip(file_objs, temps))

    # Mode-scoped keys
    force_key = f"{mode_key}_force_recompute"
    sig_key = f"{mode_key}_spectra_signature"
    wl_key = f"{mode_key}_wl_grid"
    Z_key = f"{mode_key}_Z_sorted"
    temps_key = f"{mode_key}_temps_list"
    selected_temp_key = f"{mode_key}_selected_temp"

    # Unit label logic: percent, degrees for Psi/Delta, or none
    if is_percent:
        unit_label = " (%)"
    else:
        if property_name in ["Psi", "Delta", "Ψ", "Δ"]:
            unit_label = " (°)"
        else:
            unit_label = ""

    # Button to force recompute (unique key per mode)
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
            # Read and validate
            spectra = []
            wl_mins = []
            wl_maxs = []
            for f in file_objs:
                f.seek(0)
                df = read_spectrum(f, f.name, scale_to_percent=is_percent)
                # For spectroscopic percent properties, warn if outside 0-100
                if is_percent:
                    if (df["prop"].min() < -1e-6) or (df["prop"].max() > 100.0001):
                        st.warning(f"File **{f.name}** has values outside 0–100. They will still be processed.")
                spectra.append(df)
                wl_mins.append(df["wavelength"].min())
                wl_maxs.append(df["wavelength"].max())

            # Determine wavelength grid
            if options.get("use_overlap", True):
                wl_min = max(wl_mins)
                wl_max = min(wl_maxs)
                if wl_min >= wl_max:
                    st.error("No overlapped wavelength range between files. Disable 'Use overlapped wavelength range' or check files.")
                    st.stop()
            else:
                wl_min = min(wl_mins)
                wl_max = max(wl_maxs)

            wl_grid = np.linspace(wl_min, wl_max, int(options.get("resolution", 1000)))

            # Interpolate and optional smoothing
            interpolated = []
            for df in spectra:
                x = df["wavelength"].values
                y = df["prop"].values
                interp = np.interp(wl_grid, x, y)
                if not options.get("use_overlap", True):
                    mask = (wl_grid < x.min()) | (wl_grid > x.max())
                    interp[mask] = np.nan
                if options.get("apply_smooth", False):
                    w = min(options.get("window", 11), len(interp) - (1 - (len(interp) % 2)))
                    if w % 2 == 0:
                        w -= 1
                    if w >= 5:
                        try:
                            interp = savgol_filter(interp, window_length=w, polyorder=options.get("poly", 3), mode="interp")
                        except Exception:
                            pass
                interpolated.append(interp)

            Z = np.vstack(interpolated)
            temps_arr = np.array(temps, dtype=float)

            # Sort by temperature
            sort_idx = np.argsort(temps_arr)
            temps_sorted = temps_arr[sort_idx]
            Z_sorted = Z[sort_idx, :]

            # Fill NaNs along wavelength where necessary
            for irow in range(Z_sorted.shape[0]):
                row = Z_sorted[irow, :]
                if np.isnan(row).any():
                    nans = np.isnan(row)
                    if (~nans).sum() >= 2:
                        row[nans] = np.interp(wl_grid[nans], wl_grid[~nans], row[~nans])
                    else:
                        row[nans] = 0.0
                    Z_sorted[irow, :] = row

            # Store results in session_state (mode-scoped keys)
            st.session_state[sig_key] = signature
            st.session_state[wl_key] = wl_grid
            st.session_state[Z_key] = Z_sorted
            st.session_state[temps_key] = [float(t) for t in temps_sorted]

        except Exception as e:
            st.error(f"Error processing spectra: {e}")
            return  # abort display

    # If processed data exists, display
    if wl_key in st.session_state and Z_key in st.session_state and temps_key in st.session_state:
        wl_grid_local = st.session_state[wl_key]
        Z_sorted_local = st.session_state[Z_key]
        temps_list_local = st.session_state[temps_key]

        # 3D surface with temperature as colorscale
        WL_mesh, T_mesh = np.meshgrid(wl_grid_local, np.array(temps_list_local, dtype=float))
        
        # Create a colorscale array based on temperature values
        colorscale_array = np.array(temps_list_local, dtype=float)
        
        fig = go.Figure(
            data=[
                go.Surface(
                    x=WL_mesh,
                    y=T_mesh,
                    z=Z_sorted_local,
                    surfacecolor=colorscale_array[:, None] * np.ones_like(wl_grid_local),
                    colorscale=options.get("colorscale", colorscale_choice),
                    colorbar=dict(title="Temperature (°C)"),
                    hovertemplate=f"λ=%{{x:.1f}} nm<br>T=%{{y:.3f}} °C<br>{property_name}=%{{z:.2f}}{unit_label}"
                )
            ]
        )
        fig.update_layout(
            scene=dict(xaxis_title="Wavelength (nm)", yaxis_title="Temperature (°C)", zaxis_title=f"{property_name}{unit_label}"),
            margin=dict(l=0, r=0, t=30, b=0),
            height=700,
        )
        
        # Render plot via plotly.io.to_html + interactive capture modal
        uid = uuid.uuid4().hex[:8]
        div_container = f"plotly-container-{mode_key}-{uid}"
        div_query_selector = f"#{div_container} .plotly-graph-div"
        modal_id = f"modal-{mode_key}-{uid}"
        btn_id = f"capture-btn-{mode_key}-{uid}"
        do_id = f"do-capture-{uid}"
        cancel_id = f"cancel-{uid}"
        fmt_sel = f"fmt-{uid}"
        dpi_sel = f"dpi-{uid}"
        name_input = f"name-{uid}"

        fig_html_fragment = pio.to_html(fig, include_plotlyjs='cdn', full_html=False, default_height=700)

        html = f'''
        <div id="{div_container}" style="width:100%;">{fig_html_fragment}</div>

        <div style="margin-top:8px;">
        <button id="{btn_id}">Capture image</button>
        <span style="margin-left:12px;color:#666;font-size:0.95em;">Filename base: <strong>{capture_basename}_{mode_key}</strong></span>
        </div>

        <!-- Modal -->
        <div id="{modal_id}" style="display:none; position:fixed; left:50%; top:50%; transform:translate(-50%,-50%); z-index:10000;
            background:#fff; border:1px solid #ccc; padding:14px; box-shadow:0 8px 24px rgba(0,0,0,0.2); min-width:320px;">
        <div style="font-weight:600; margin-bottom:8px;">Capture options</div>
        <div style="margin-bottom:6px;">
            <label>Format:
            <select id="{fmt_sel}">
                <option>PNG</option>
                <option>JPG</option>
            </select>
            </label>
        </div>
        <div style="margin-bottom:6px;">
            <label>DPI:
            <select id="{dpi_sel}">
                <option>96</option>
                <option>150</option>
                <option>300</option>
                <option>600</option>
                <option>1200</option>
            </select>
            </label>
        </div>
        <div style="margin-bottom:8px;">
            <label>Filename: <input id="{name_input}" style="width:60%" value="{capture_basename}_{mode_key}"/></label>
        </div>
        <div style="text-align:right;">
            <button id="{do_id}">Capture</button>
            <button id="{cancel_id}" style="margin-left:8px;">Cancel</button>
        </div>
        </div>

        <script>
        setTimeout(function(){{
        try {{
            var inner = document.querySelector("{div_query_selector}");
            if (!inner) {{
            var cont = document.getElementById("{div_container}");
            if (cont) cont.insertAdjacentHTML('afterbegin', '<div style="color:#b00;padding:10px;">Plot element not found (Plotly may not have initialised). Open console for details.</div>');
            console.error("Plotly inner div not found: selector {div_query_selector}");
            return;
            }}

            var btn = document.getElementById("{btn_id}");
            var modal = document.getElementById("{modal_id}");
            var doBtn = document.getElementById("{do_id}");
            var cancelBtn = document.getElementById("{cancel_id}");

            if (btn) btn.onclick = function(){{ modal.style.display = 'block'; }};
            if (cancelBtn) cancelBtn.onclick = function(){{ modal.style.display = 'none'; }};

            if (doBtn) doBtn.onclick = function(){{
            try {{
                var fmt = document.getElementById("{fmt_sel}").value.toLowerCase();
                if (fmt === 'jpg') fmt = 'jpeg';
                var dpi = parseInt(document.getElementById("{dpi_sel}").value, 10) || 96;
                var baseDpi = 96;
                var scale = dpi / baseDpi;
                var w = Math.round(inner.clientWidth * scale);
                var h = Math.round(inner.clientHeight * scale);

                Plotly.toImage(inner, {{format: fmt, width: w, height: h}})
                .then(function(dataUrl) {{
                    var a = document.createElement('a');
                    a.href = dataUrl;
                    var ext = fmt === 'jpeg' ? 'jpg' : fmt;
                    var filename = document.getElementById("{name_input}").value || "{capture_basename}_{mode_key}";
                    a.download = filename + "." + ext;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    modal.style.display = 'none';
                }})
                .catch(function(err) {{
                    alert('Could not generate image: ' + err);
                    console.error("Plotly.toImage error:", err);
                    modal.style.display = 'none';
                }});
            }} catch (err) {{
                alert('Capture error: ' + err);
                console.error("Capture exception:", err);
                modal.style.display = 'none';
            }}
            }};
        }} catch (e) {{
            console.error("Initialization error for capture UI:", e);
        }}
        }}, 80);
        </script>
        '''

        components.html(html, height=820, scrolling=True)

        # 2D heatmap
        st.subheader("2D map (Wavelength vs Temperature)")
        st.markdown(
            f"This heatmap is a 2D projection of the 3D surface. "
            f"The X axis is the wavelength (nm), the Y axis is the temperature (°C), and the color represents the {property_name}{unit_label}. "
        )
        fig2 = px.imshow(Z_sorted_local, x=np.round(wl_grid_local, 6), y=np.round(temps_list_local, 6), origin="lower", aspect="auto",
                         labels={"x": "Wavelength (nm)", "y": "Temperature", "color": f"{property_name}{unit_label}"})
        fig2.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig2, config={"responsive": True, "autosizable": True})

        # 2D cut: spectrum at chosen temperature (persistent selection)
        st.subheader(f"2D cut: {property_name} at a chosen temperature")

        if selected_temp_key not in st.session_state:
            st.session_state[selected_temp_key] = float(temps_list_local[len(temps_list_local)//2])

        if float(st.session_state[selected_temp_key]) not in [float(t) for t in temps_list_local]:
            idx_near = nearest_index(st.session_state[selected_temp_key], temps_list_local)
            st.session_state[selected_temp_key] = float(temps_list_local[idx_near])

        chosen_temp = st.selectbox(
            "Select temperature:",
            options=temps_list_local,
            format_func=lambda x: f"{float(x):.3f} °C",
            key=selected_temp_key
        )

        idx = nearest_index(chosen_temp, temps_list_local)
        trans_cut = Z_sorted_local[idx, :]

        fig_cut = px.line(x=wl_grid_local, y=trans_cut,
                          labels={"x": "Wavelength (nm)", "y": f"{property_name}{unit_label}"},
                          title=f"{property_name} at {float(chosen_temp):.3f} °C")
        st.plotly_chart(fig_cut, config={"responsive": True, "autosizable": True})

        col_name = property_name.lower().replace(" ", "_")
        df_cut = pd.DataFrame({"wavelength": wl_grid_local, col_name: trans_cut})
        csv_bytes = df_cut.to_csv(index=False).encode("utf-8")
        st.download_button(f"Download selected {property_name} spectrum (CSV)", data=csv_bytes, file_name=f"{col_name}_spectrum_{float(chosen_temp):.3f}C.csv", mime="text/csv")

    else:
        st.info("No processed data available. Click 'Generate 3D map' to process the uploaded files.")

# ---------------------------
# Function for Translation Scan 3D display
# ---------------------------
def process_and_display_translation_scan(file_objs, options, mode_key="translation_scan"):
    """
    Processes Translation Scan data (X, Y, and Property: Thickness/Psi/n) and displays 3D surface.
    Colorscale is based on the Property values (Thickness, Psi, or n).
    Interpolates data to create a smooth surface using cubic interpolation.
    """
    # Build signature for current inputs
    signature = tuple((f.name,) for f in file_objs)

    # Mode-scoped keys
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
            # Read and validate all files
            all_data = []
            property_types = []
            for f in file_objs:
                f.seek(0)
                df = read_translation_scan(f, f.name)
                all_data.append(df)
                property_types.append(df.property_name)

            # Concatenate all data
            scan_data = pd.concat(all_data, ignore_index=True)
            
            if scan_data.empty:
                st.error("No valid data found in uploaded files.")
                return

            # Determine the property type (use the most common one if mixed)
            property_name = max(set(property_types), key=property_types.count)
            
            # Store results in session_state
            st.session_state[sig_key] = signature
            st.session_state[data_key] = scan_data
            st.session_state[f"{mode_key}_property_name"] = property_name

        except Exception as e:
            st.error(f"Error processing translation scan data: {e}")
            return

    # If processed data exists, display
    if data_key in st.session_state:
        scan_data = st.session_state[data_key]
        property_name = st.session_state.get(f"{mode_key}_property_name", "thickness")

        # Determine units and labels based on property type
        if property_name == "psi":
            property_label = "Psi"
            unit_label = " (°)"
            color_label = "Psi (°)"
        elif property_name == "n":
            property_label = "n"
            unit_label = ""
            color_label = "Refractive Index"
        else:  # thickness
            property_label = "Thickness"
            unit_label = " (nm)"
            color_label = "Thickness (nm)"

        # Create interpolated surface using griddata
        x_pts = scan_data["X"].values
        y_pts = scan_data["Y"].values
        z_pts = scan_data["property"].values
        
        # Create a regular grid for interpolation
        x_min, x_max = x_pts.min(), x_pts.max()
        y_min, y_max = y_pts.min(), y_pts.max()
        
        # Create grid with 50x50 resolution
        xi = np.linspace(x_min, x_max, 50)
        yi = np.linspace(y_min, y_max, 50)
        XI, YI = np.meshgrid(xi, yi)
        
        # Interpolate property values on the grid using cubic interpolation
        points = np.column_stack([x_pts, y_pts])
        ZI = griddata(points, z_pts, (XI, YI), method='cubic', fill_value=np.mean(z_pts))
        
        # Create 3D surface plot
        fig = go.Figure(
            data=[
                go.Surface(
                    x=XI,
                    y=YI,
                    z=ZI,
                    surfacecolor=ZI,
                    colorscale=options.get("colorscale", "Viridis"),
                    colorbar=dict(title=color_label),
                    hovertemplate="X: %{x:.3f} cm<br>Y: %{y:.3f} cm<br>" + property_label + ": %{z:.2f}" + unit_label + "<extra></extra>"
                )
            ]
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="X-position (cm)",
                yaxis_title="Y-position (cm)",
                zaxis_title=f"{property_label}{unit_label}"
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=700,
        )

        # Render plot via plotly.io.to_html + interactive capture modal
        uid = uuid.uuid4().hex[:8]
        div_container = f"plotly-container-{mode_key}-{uid}"
        div_query_selector = f"#{div_container} .plotly-graph-div"
        modal_id = f"modal-{mode_key}-{uid}"
        btn_id = f"capture-btn-{mode_key}-{uid}"
        do_id = f"do-capture-{uid}"
        cancel_id = f"cancel-{uid}"
        fmt_sel = f"fmt-{uid}"
        dpi_sel = f"dpi-{uid}"
        name_input = f"name-{uid}"

        fig_html_fragment = pio.to_html(fig, include_plotlyjs='cdn', full_html=False, default_height=700)

        html = f'''
        <div id="{div_container}" style="width:100%;">{fig_html_fragment}</div>

        <div style="margin-top:8px;">
        <button id="{btn_id}">Capture image</button>
        <span style="margin-left:12px;color:#666;font-size:0.95em;">Filename base: <strong>{capture_basename}_{mode_key}</strong></span>
        </div>

        <!-- Modal -->
        <div id="{modal_id}" style="display:none; position:fixed; left:50%; top:50%; transform:translate(-50%,-50%); z-index:10000;
            background:#fff; border:1px solid #ccc; padding:14px; box-shadow:0 8px 24px rgba(0,0,0,0.2); min-width:320px;">
        <div style="font-weight:600; margin-bottom:8px;">Capture options</div>
        <div style="margin-bottom:6px;">
            <label>Format:
            <select id="{fmt_sel}">
                <option>PNG</option>
                <option>JPG</option>
            </select>
            </label>
        </div>
        <div style="margin-bottom:6px;">
            <label>DPI:
            <select id="{dpi_sel}">
                <option>96</option>
                <option>150</option>
                <option>300</option>
                <option>600</option>
                <option>1200</option>
            </select>
            </label>
        </div>
        <div style="margin-bottom:8px;">
            <label>Filename: <input id="{name_input}" style="width:60%" value="{capture_basename}_{mode_key}"/></label>
        </div>
        <div style="text-align:right;">
            <button id="{do_id}">Capture</button>
            <button id="{cancel_id}" style="margin-left:8px;">Cancel</button>
        </div>
        </div>

        <script>
        setTimeout(function(){{
        try {{
            var inner = document.querySelector("{div_query_selector}");
            if (!inner) {{
            var cont = document.getElementById("{div_container}");
            if (cont) cont.insertAdjacentHTML('afterbegin', '<div style="color:#b00;padding:10px;">Plot element not found (Plotly may not have initialised). Open console for details.</div>');
            console.error("Plotly inner div not found: selector {div_query_selector}");
            return;
            }}

            var btn = document.getElementById("{btn_id}");
            var modal = document.getElementById("{modal_id}");
            var doBtn = document.getElementById("{do_id}");
            var cancelBtn = document.getElementById("{cancel_id}");

            if (btn) btn.onclick = function(){{ modal.style.display = 'block'; }};
            if (cancelBtn) cancelBtn.onclick = function(){{ modal.style.display = 'none'; }};

            if (doBtn) doBtn.onclick = function(){{
            try {{
                var fmt = document.getElementById("{fmt_sel}").value.toLowerCase();
                if (fmt === 'jpg') fmt = 'jpeg';
                var dpi = parseInt(document.getElementById("{dpi_sel}").value, 10) || 96;
                var baseDpi = 96;
                var scale = dpi / baseDpi;
                var w = Math.round(inner.clientWidth * scale);
                var h = Math.round(inner.clientHeight * scale);

                Plotly.toImage(inner, {{format: fmt, width: w, height: h}})
                .then(function(dataUrl) {{
                    var a = document.createElement('a');
                    a.href = dataUrl;
                    var ext = fmt === 'jpeg' ? 'jpg' : fmt;
                    var filename = document.getElementById("{name_input}").value || "{capture_basename}_{mode_key}";
                    a.download = filename + "." + ext;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    modal.style.display = 'none';
                }})
                .catch(function(err) {{
                    alert('Could not generate image: ' + err);
                    console.error("Plotly.toImage error:", err);
                    modal.style.display = 'none';
                }});
            }} catch (err) {{
                alert('Capture error: ' + err);
                console.error("Capture exception:", err);
                modal.style.display = 'none';
            }}
            }};
        }} catch (e) {{
            console.error("Initialization error for capture UI:", e);
        }}
        }}, 80);
        </script>
        '''

        components.html(html, height=820, scrolling=True)

        # Summary statistics
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("X range (cm)", f"{scan_data['X'].min():.3f} to {scan_data['X'].max():.3f}")
        with col2:
            st.metric("Y range (cm)", f"{scan_data['Y'].min():.3f} to {scan_data['Y'].max():.3f}")
        with col3:
            st.metric(f"{property_label} range{unit_label}", f"{scan_data['property'].min():.2f} to {scan_data['property'].max():.2f}")

        # Download data as CSV
        csv_bytes = scan_data.to_csv(index=False).encode("utf-8")
        st.download_button("Download processed data (CSV)", data=csv_bytes, file_name="translation_scan_data.csv", mime="text/csv")

    else:
        st.info("No processed data available. Click 'Generate 3D map' to process the uploaded files.")

# ---------------------------

# HOME
if main_mode == "Home":
    st.header("Welcome to 3D OPTICS")
    banner_path = "3Doptics_banner.png"
    if os.path.exists(banner_path):
        st.image(banner_path, width="stretch")
    else:
        st.info("Insert your banner image at /assets/banner.png to show it here.")
    
# SPECTROSCOPY
elif main_mode == "Spectroscopy":
    st.header("Spectroscopy")
    st.markdown("Modes: Transmittance, Absorbance, Reflectance")

    if "spectro_submode" not in st.session_state:
        st.session_state["spectro_submode"] = "Transmittance"

    spectro_submode = st.selectbox(
        "Select spectroscopy mode",
        options=["Transmittance", "Absorbance", "Reflectance"],
        key="spectro_submode"
    )

    uploaded = st.file_uploader(
        "Upload spectrum files (CSV/TXT/DAT/XLSX). One file = one temperature, or a single file with multiple property columns (first column = wavelength).",
        accept_multiple_files=True,
        type=["csv", "txt", "dat", "xlsx"],
        key="spectro_uploader"
    )

    if uploaded:
        mode_key = "spectroscopy"
        entries = expand_uploaded_to_spectra(uploaded, scale_to_percent=True, mode_key=mode_key)

        if not entries:
            st.info("No valid spectra found in uploaded files.")
        else:
            st.markdown("**Detected spectra (from uploaded files). Edit temperatures if needed:**")
            cols = st.columns(2)
            file_objs = []
            temps = []
            for i, e in enumerate(entries):
                col = cols[i % 2]
                col.write(f"**{e['label']}**")
                if e["temp_key"] not in st.session_state:
                    if e["temp_default"] is not None:
                        st.session_state[e["temp_key"]] = float(e["temp_default"])
                    else:
                        st.session_state[e["temp_key"]] = 25.0
                temp_val = col.number_input(
                    f"Temperature for {e['label']}",
                    step=0.1,
                    format="%.3f",
                    key=e["temp_key"]
                )

                file_objs.append(e["fileobj"])
                temps.append(float(temp_val))

            options = {"resolution": resolution, "use_overlap": use_overlap, "apply_smooth": apply_smooth,
                    "window": window, "poly": poly, "colorscale": colorscale_choice}

            process_and_display_spectra(
                file_objs,
                temps,
                options,
                property_name=spectro_submode,
                mode_key=mode_key,
                is_percent=True
            )
    else:
        st.info("Upload files to enable processing.")


# ELLIPSOMETRY
elif main_mode == "Ellipsometry":
    st.header("Ellipsometry")
    st.markdown("Modes: Ψ, Δ, n (refractive index), k (extinction coefficient), Translation Scan")

    # Display → internal mapping
    ellip_map = {
        "Ψ": "Psi",
        "Δ": "Delta",
        "n": "n",
        "k": "k",
        "Translation Scan": "translation_scan"
    }

    if "ellip_submode" not in st.session_state:
        st.session_state["ellip_submode"] = "Ψ"

    ellip_submode_display = st.selectbox(
        "Select ellipsometry mode",
        options=list(ellip_map.keys()),
        key="ellip_submode"
    )
    ellip_submode_internal = ellip_map[ellip_submode_display]

    # Translation Scan mode
    if ellip_submode_display == "Translation Scan":
        st.markdown("Upload translation scan data with three columns: X position, Y position, and Property (Thickness, Psi, or n). Supports various column header formats and file types (CSV/TXT/DAT/XLSX).")
        
        uploaded = st.file_uploader(
            "Upload translation scan files (CSV/TXT/DAT/XLSX). Each file should contain three columns: X, Y, and Property (Thickness/Psi/n).",
            accept_multiple_files=True,
            type=["csv", "txt", "dat", "xlsx"],
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

    # Standard ellipsometry modes (Ψ, Δ, n, k)
    else:
        uploaded = st.file_uploader(
            "Upload ellipsometry files (CSV/TXT/DAT/XLSX). One file = one temperature, or a single file with multiple property columns (first column = wavelength).",
            accept_multiple_files=True,
            type=["csv", "txt", "dat", "xlsx"],
            key="ellip_uploader"
        )

        if uploaded:
            mode_key = "ellipsometry"
            entries = expand_uploaded_to_spectra(uploaded, scale_to_percent=False, mode_key=mode_key)

            if not entries:
                st.info("No valid spectra found in uploaded files.")
            else:
                st.markdown("**Detected ellipsometry spectra. Edit temperatures if needed:**")
                cols = st.columns(2)
                file_objs = []
                temps = []
                for i, e in enumerate(entries):
                    col = cols[i % 2]
                    col.write(f"**{e['label']}**")
                    if e["temp_key"] not in st.session_state:
                        if e["temp_default"] is not None:
                            st.session_state[e["temp_key"]] = float(e["temp_default"])
                        else:
                            st.session_state[e["temp_key"]] = 25.0
                    temp_val = col.number_input(
                        f"Temperature for {e['label']}",
                        step=0.1,
                        format="%.3f",
                        key=e["temp_key"]
                    )

                    file_objs.append(e["fileobj"])
                    temps.append(float(temp_val))

                options = {"resolution": resolution, "use_overlap": use_overlap, "apply_smooth": apply_smooth,
                        "window": window, "poly": poly, "colorscale": colorscale_choice}

                process_and_display_spectra(
                    file_objs,
                    temps,
                    options,
                    property_name=ellip_submode_display,
                    mode_key=mode_key,
                    is_percent=False
                )
        else:
            st.info("Upload files to enable processing.")
