==========================================
3D OPTICS — Interactive Spectroscopy & Ellipsometry Viewer
==========================================

Author: José Lorenzo Calderón Solís  
Email: joselorencs@gmail.com  
Version: 1.0  
Framework: Streamlit + Plotly + Pandas + NumPy  
------------------------------------------

DESCRIPTION
-----------
3D OPTICS is an interactive web application for visualizing spectroscopy and ellipsometry data 
as a function of wavelength and temperature. It generates 3D surfaces, 2D heatmaps, and 
temperature-specific spectra from multiple measurement files.

Supported modes:
- Spectroscopy: Transmittance, Absorbance, Reflectance
- Ellipsometry: Ψ, Δ, n, k

Each mode accepts multiple files, either:
  • Multiple 2-column files (wavelength, property)
  • A single file containing multiple property columns (first = wavelength)

The program automatically detects temperature values from filenames or column headers.

------------------------------------------
HOW TO RUN
------------------------------------------
1. Make sure you have Python 3.10 or later installed.

2. Install the required dependencies:
   > pip install -r requirements.txt

3. Launch the Streamlit web app:
   > streamlit run 3Doptics.py

4. The app will open automatically in your web browser (default: http://localhost:8501)

------------------------------------------
USAGE
------------------------------------------
- Select your working mode: Spectroscopy or Ellipsometry.
- Upload one or more CSV, TXT, or DAT files.
- Adjust settings (resolution, color scale, smoothing, etc.) from the sidebar.
- Click "Generate 3D map" to visualize your data.
- You can interactively rotate, zoom, and export images (PNG/JPG) at various DPI.
- The software automatically detects temperatures from filenames (e.g., “sample_40C.txt”).

------------------------------------------
OUTPUTS
------------------------------------------
- 3D interactive surface (Wavelength, Temperature, Optical Property)
- 2D heatmap projection
- 2D line cut at selected temperature
- CSV export of the selected spectrum
- Image export of the 3D visualization (PNG/JPG)

------------------------------------------
DEVELOPER NOTES
------------------------------------------
To customize the interface:
- Edit the file `3Doptics.py`
- Change the banner image: `3Doptics_banner.png`
- You can compile it into a standalone executable using PyInstaller:
   > pyinstaller --onefile --windowed --icon="3Doptics.ico" 3Doptics.py

------------------------------------------
LICENSE
------------------------------------------
This project is distributed for academic and research use.
(c) 2025 José Lorenzo Calderón Solís. All rights reserved.
