import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import io
import os
import tempfile
import zipfile
import plotly.express as px
import numpy as np

# App configuration
st.set_page_config(page_title="BPD Patrol Optimization Dashboard", layout="wide")

# Title and description
st.title("Step 1: Load Patrol Zones and Incident Data")

st.markdown("""
This tool supports Berkeley Police Department's efforts to evaluate and rebalance patrol workloads using GIS and service call data.  

**Step 1** allows you to:
- Upload the current patrol zone shapefile
- Upload annual service call records
- Visualize patrol zones and incident distribution
- Identify workload imbalance across sectors
""")
st.markdown("---")
st.header("Step 1: Upload Patrol Zones and Incident Data")

col1, col2 = st.columns(2)

# ========= Helper Function =========
def load_default_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return f.read()
    return None

# ========= Patrol Zones (Shapefile) =========
with col1:
    st.subheader("Upload Current Patrol Zones")
    st.write("Upload a ZIP file containing your shapefile components (.shp, .shx, .dbf, etc.). It must include a `Sector` column and geometry.")

    shapefile = st.file_uploader("Upload Shapefile ZIP", type=['zip'], key="shapefile")

    default_shapefile_path = "data/current_beats.zip"
    default_shapefile = load_default_file(default_shapefile_path)

    if shapefile is None and default_shapefile is not None:
        st.markdown("*Using default patrol zone shapefile:* `current_beats.zip`")
        st.download_button("Download Default Patrol Zones", data=default_shapefile, file_name="current_beats.zip")

    # Load shapefile
    file_to_use = shapefile if shapefile else io.BytesIO(default_shapefile) if default_shapefile else None

    if file_to_use:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(file_to_use.getvalue())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            shp_file = next((os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")), None)
            if shp_file:
                beats = gpd.read_file(shp_file)
                st.session_state.beats = beats
            else:
                st.error("No .shp file found in the ZIP.")

# ========= Incident CSV =========
with col2:
    st.subheader("Upload Annual Service Call Data")
    st.write("Upload a CSV file that includes at least `lat` and `lon` columns to geolocate incidents.")

    incident_file = st.file_uploader("Upload Incident CSV", type=['csv'], key="incident")

    default_incident_path = "data/2024.csv"
    default_csv = load_default_file(default_incident_path)

    if incident_file is None and default_csv is not None:
        st.markdown("*Using default incident dataset:* `2024.csv`")
        st.download_button("Download Default Incident CSV", data=default_csv, file_name="2024.csv")
        st.markdown(
        "_Note: `Time Spent Responding` is a derived column calculated as the sum of travel time "
        "(`Response Time Mins` − `Dispatch Time Mins`) and `On Scene Time Mins`._"
    )

    # Load CSV
    if incident_file:
        df = pd.read_csv(incident_file)
        st.session_state.incidents_df = df
    elif default_csv:
        df = pd.read_csv(io.BytesIO(default_csv))
        st.session_state.incidents_df = df
    else:
        df = None

st.markdown("---")
# Visualization and analysis
if 'beats' in locals() and 'df' in locals():
    st.header("Visualize Current Patrol Zones")
    st.markdown("The map below displays the uploaded patrol sectors, labeled by sector ID.")

    # Create the map using GeoPandas with different colors for each sector
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate a colormap with 14 distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, 14))
    
    # Plot each sector with a different color
    for idx, row in beats.iterrows():
        sector_num = int(row['Sector']) - 1  # Convert to 0-based index
        beats[beats['Sector'] == row['Sector']].plot(
            ax=ax,
            color=colors[sector_num],
            alpha=0.7,
            edgecolor='black'
        )
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, f"Sector {row['Sector']}", 
                fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.title('Current Patrol Sector Boundaries')
    plt.axis('off')
    st.pyplot(fig)
    st.markdown("---")
    st.header("Service Call Distribution by Sector")
    st.markdown("Each service call is assigned to a sector using spatial joins. The bar chart below summarizes total incidents by sector.")

    # Convert incident coordinates to Point geometry
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    incident_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=beats.crs)

    # Spatial join
    incidents_by_sector = gpd.sjoin(incident_gdf, beats[['Sector', 'geometry']], how='left', predicate='within')
    sector_counts = incidents_by_sector['Sector'].value_counts().reset_index()
    sector_counts.columns = ['Sector', 'Count']
    
    # Sort by sector number
    sector_counts['Sector'] = sector_counts['Sector'].astype(int)
    sector_counts = sector_counts.sort_values('Sector')

    # Create interactive Plotly bar chart
    fig = px.bar(sector_counts, 
                 x='Sector', 
                 y='Count',
                 title='Number of Service Calls per Sector',
                 labels={'Sector': 'Sector Number', 'Count': 'Number of Calls'},
                 hover_data={'Count': True})
    
    fig.update_traces(marker_color='rgb(55, 83, 109)')
    fig.update_layout(
        xaxis_title="Sector Number",
        yaxis_title="Number of Calls",
        showlegend=False,
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
     
    st.header("Workload Imbalance Assessment")
    max_calls = sector_counts['Count'].max()
    min_calls = sector_counts['Count'].min()
    imbalance = max_calls - min_calls
    imbalance_percentage = (imbalance / min_calls) * 100

    st.markdown(f"""
    - **Highest sector workload:** {max_calls} calls  
    - **Lowest sector workload:** {min_calls} calls  
    - **Call volume gap:** {imbalance} calls  
    - **Relative imbalance:** {imbalance_percentage:.2f}%  
    """)
    st.markdown("---")
    st.header("Sector-Level Workload Summary")
    st.markdown("Sorted by sector ID, this table presents the number of service calls assigned to each patrol zone.")
    
    # Format the dataframe for display
    display_df = sector_counts.copy()
    display_df['Sector'] = display_df['Sector'].astype(int)
    display_df = display_df.sort_values('Sector')
    display_df.columns = ['Sector Number', 'Number of Calls']
    
    st.dataframe(display_df, use_container_width=True)
