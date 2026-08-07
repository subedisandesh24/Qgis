import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from geopy.geocoders import Nominatim
import io
import zipfile
import tempfile
import os

st.set_page_config(page_title="Nepal Scientific Map & Location Search", layout="wide")

st.title("🗺️ Nepal Scientific Map Generator with Location Search")

# --- SHAPEFILE & GEOCODING HELPERS ---

def extract_all_shapefiles_from_zip(uploaded_file):
    """Extracts zip file (handling nested folders) and returns dict of layers."""
    tmpdir = tempfile.mkdtemp()
    zip_path = os.path.join(tmpdir, "uploaded.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmpdir)
        
    shp_dict = {}
    for root, dirs, files in os.walk(tmpdir):
        for file in files:
            if file.endswith('.shp'):
                layer_name = os.path.splitext(file)[0]
                full_path = os.path.join(root, file)
                shp_dict[layer_name] = full_path
                
    return shp_dict

@st.cache_data(ttl=3600)
def geocode_place_name(search_query):
    """Geocodes location name to Lat/Lon coordinates using OpenStreetMap Nominatim."""
    try:
        geolocator = Nominatim(user_agent="nepal_gis_app")
        # Append Nepal if not specified to narrow search
        query = search_query if "nepal" in search_query.lower() else f"{search_query}, Nepal"
        location = geolocator.geocode(query)
        if location:
            return {
                "address": location.address,
                "lat": location.latitude,
                "lon": location.longitude,
                "google_maps_url": f"https://www.google.com/maps?q={location.latitude},{location.longitude}"
            }
        else:
            return None
    except Exception as e:
        return None

def run_idw_interpolation(x, y, z, grid_x, grid_y, power=2):
    dist = np.hypot(grid_x[..., np.newaxis] - x, grid_y[..., np.newaxis] - y)
    dist = np.where(dist == 0, 1e-12, dist)
    weights = 1.0 / (dist ** power)
    weights /= weights.sum(axis=-1, keepdims=True)
    zi = np.sum(weights * z, axis=-1)
    return zi

def mask_grid_to_shapefile(gdf, grid_x, grid_y):
    unified_geom = gdf.geometry.unary_union
    points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    mask = np.zeros(len(points), dtype=bool)
    if isinstance(unified_geom, Polygon):
        polys = [unified_geom]
    elif isinstance(unified_geom, MultiPolygon):
        polys = list(unified_geom.geoms)
    else:
        polys = []
        
    for poly in polys:
        path = Path(np.array(poly.exterior.coords))
        mask |= path.contains_points(points)
        
    return mask.reshape(grid_x.shape)

def add_north_arrow(ax, position=(0.93, 0.88)):
    ax.annotate('N', xy=position, xytext=(position[0], position[1] - 0.08),
                xycoords='axes fraction', ha='center', va='bottom',
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.8, color='black'))

# --- SIDEBAR CONTROLS ---

st.sidebar.header("1. Upload Layers")
shp_zip = st.sidebar.file_uploader("Upload Boundary Shapefile (.zip)", type=["zip"])

gdf_boundary = None
if shp_zip:
    shp_dict = extract_all_shapefiles_from_zip(shp_zip)
    if shp_dict:
        selected_layer_name = st.sidebar.selectbox("Select Layer / Admin Level:", list(shp_dict.keys()))
        gdf_boundary = gpd.read_file(shp_dict[selected_layer_name])
        if gdf_boundary.crs is None:
            gdf_boundary.set_crs(epsg=4326, inplace=True)

# 📍 LOCATION SEARCH ENGINE
st.sidebar.header("2. 🔍 Search Location / Station")
search_location = st.sidebar.text_input(
    "Search Research Site / Place Name", 
    value="Horticulture Research Station Malepatan Pokhara"
)

found_site = None
if search_location:
    found_site = geocode_place_name(search_location)
    if found_site:
        st.sidebar.success(f"📍 **Found:** {found_site['address'][:60]}...")
        st.sidebar.caption(f"**Coordinates:** {found_site['lat']:.4f}°N, {found_site['lon']:.4f}°E")
        st.sidebar.markdown(f"[📍 Open in Google Maps]({found_site['google_maps_url']})")
    else:
        st.sidebar.warning("Location not found automatically. You can enter manual Lat/Lon below.")
        manual_lat = st.sidebar.number_input("Manual Latitude", value=28.2111, format="%.5f")
        manual_lon = st.sidebar.number_input("Manual Longitude", value=83.9785, format="%.5f")
        found_site = {
            "address": search_location,
            "lat": manual_lat,
            "lon": manual_lon,
            "google_maps_url": f"https://www.google.com/maps?q={manual_lat},{manual_lon}"
        }

# Upload Sample Data
data_file = st.sidebar.file_uploader("Upload Sample Data (Excel / CSV)", type=["csv", "xlsx", "xls"])
df_data = None
if data_file:
    file_ext = os.path.splitext(data_file.name)[1].lower()
    df_data = pd.read_csv(data_file) if file_ext == '.csv' else pd.read_excel(data_file)

map_mode = st.sidebar.radio("3. Select Map Type", ["Hierarchical Locator Map (Study Area)", "IDW Spatial Interpolation Map"])

# --- RENDER MAPS ---

if map_mode == "Hierarchical Locator Map (Study Area)":
    if gdf_boundary is None:
        st.info("👈 Upload your `.zip` shapefile in the sidebar to begin.")
    else:
        st.sidebar.header("4. Locator Settings")
        region_col = st.sidebar.selectbox("Select Name Column", gdf_boundary.columns)
        
        unique_regions = sorted(gdf_boundary[region_col].dropna().astype(str).unique().tolist())
        selected_study_area = st.sidebar.selectbox("Select Study Area Region:", unique_regions)
        
        map_title = st.sidebar.text_input("Map Title", value=f"Location Map of {selected_study_area}")
        base_color = st.sidebar.color_picker("Country Base Color", value="#F0F0F0")
        highlight_color = st.sidebar.color_picker("Study Area Color", value="#E6E65C")
        
        study_area_gdf = gdf_boundary[gdf_boundary[region_col].astype(str) == selected_study_area]
        
        # PLOT
        fig = plt.figure(figsize=(10, 8), dpi=300)
        
        # Main Detail Map
        ax_main = fig.add_subplot(1, 1, 1)
        study_area_gdf.plot(ax=ax_main, color=highlight_color, edgecolor="black", linewidth=1.2)
        
        # Plot Searched Location Marker
        if found_site:
            ax_main.scatter(found_site["lon"], found_site["lat"], marker="*", color="red", s=180, zorder=10, edgecolor="black", label=search_location)
            ax_main.annotate(search_location, xy=(found_site["lon"], found_site["lat"]),
                            xytext=(found_site["lon"] + 0.02, found_site["lat"] + 0.02),
                            fontsize=9, fontweight='bold',
                            arrowprops=dict(arrowstyle="->", color="black", lw=1),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8))
            ax_main.legend(loc="lower left", fontsize=8)

        ax_main.set_title(map_title, fontsize=13, fontweight='bold', pad=15)
        ax_main.grid(True, linestyle="--", alpha=0.5)
        ax_main.set_xlabel("Longitude (°E)", fontsize=9)
        ax_main.set_ylabel("Latitude (°N)", fontsize=9)
        add_north_arrow(ax_main, (0.95, 0.88))
        
        # Inset Locator Map
        ax_inset = fig.add_axes([0.65, 0.65, 0.28, 0.28])
        gdf_boundary.plot(ax=ax_inset, color=base_color, edgecolor="#666666", linewidth=0.4)
        study_area_gdf.plot(ax=ax_inset, color=highlight_color, edgecolor="red", linewidth=1.0)
        
        bounds = study_area_gdf.total_bounds
        rect = patches.Rectangle((bounds[0], bounds[1]), bounds[2]-bounds[0], bounds[3]-bounds[1],
                                 linewidth=1.2, edgecolor='red', facecolor='none')
        ax_inset.add_patch(rect)
        ax_inset.set_title("Country Locator", fontsize=7, fontweight='bold')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

        st.pyplot(fig)
        
        # Download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button("📥 Download Publication Map (300 DPI PNG)", buf.getvalue(), "locator_map.png", "image/png")

elif map_mode == "IDW Spatial Interpolation Map":
    if gdf_boundary is None or df_data is None:
        st.info("👈 Upload both **Boundary Shapefile (.zip)** and **Sample Points (Excel/CSV)** in the sidebar.")
    else:
        st.sidebar.header("4. Interpolation Settings")
        col_names = df_data.columns.tolist()
        
        lat_col = st.sidebar.selectbox("Latitude Column", col_names)
        lon_col = st.sidebar.selectbox("Longitude Column", col_names)
        
        numeric_cols = df_data.select_dtypes(include=[np.number]).columns.tolist()
        target_var = st.sidebar.selectbox("Target Variable to Interpolate", numeric_cols)
        
        colormap = st.sidebar.selectbox("Color Ramp", ["YlGn", "viridis", "plasma", "Spectral_r", "RdYlBu_r", "coolwarm"])
        grid_res = st.sidebar.slider("Interpolation Resolution", 100, 400, 200)
        idw_power = st.sidebar.slider("IDW Power Parameter", 1.0, 4.0, 2.0)
        
        # INTERPOLATION
        clean_df = df_data.dropna(subset=[lat_col, lon_col, target_var])
        x = clean_df[lon_col].values
        y = clean_df[lat_col].values
        z = clean_df[target_var].values
        
        bounds = gdf_boundary.total_bounds
        gx = np.linspace(bounds[0], bounds[2], grid_res)
        gy = np.linspace(bounds[1], bounds[3], grid_res)
        grid_x, grid_y = np.meshgrid(gx, gy)
        
        zi = run_idw_interpolation(x, y, z, grid_x, grid_y, power=idw_power)
        mask = mask_grid_to_shapefile(gdf_boundary, grid_x, grid_y)
        zi_masked = np.where(mask, zi, np.nan)
        
        # PLOT
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        contour = ax.contourf(grid_x, grid_y, zi_masked, levels=15, cmap=colormap)
        cbar = fig.colorbar(contour, ax=ax, shrink=0.7, pad=0.03)
        cbar.set_label(target_var, fontsize=10, fontweight='bold')
        
        gdf_boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8)
        
        # Plot Research Station Pin if searched
        if found_site:
            ax.scatter(found_site["lon"], found_site["lat"], marker="*", color="red", s=180, zorder=10, edgecolor="black")
            ax.annotate(search_location, xy=(found_site["lon"], found_site["lat"]),
                        xytext=(found_site["lon"] + 0.01, found_site["lat"] + 0.01),
                        fontsize=9, fontweight='bold', bbox=dict(boxstyle="round", fc="white", ec="black"))

        ax.set_title(f"Spatial Interpolation of {target_var}", fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        add_north_arrow(ax, (0.93, 0.88))
        
        st.pyplot(fig)
        
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
        st.download_button("📥 Download Interpolation Map (300 DPI)", buf_png.getvalue(), "interpolation_map.png", "image/png")
