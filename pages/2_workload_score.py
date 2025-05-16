import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from sklearn.preprocessing import MinMaxScaler
import os, tempfile, zipfile, re
import matplotlib.colors as mcolors

# Page setup
st.set_page_config(page_title="Step 2 – Define Custom Workload Score", layout="wide")
st.title("Step 2:Define and Visualize Patrol Workload Score (WLS)")

st.markdown("""
In this step, you'll define how **patrol workload** should be measured by customizing the **Workload Score (WLS)**.  
This score helps us evaluate how much operational effort each service call requires, so we can compare and balance workload across each patrol sector.

            
### Why this matters:
Berkeley's patrol zones are unevenly loaded. Some sectors respond to more urgent or complex calls than others.  
Just counting total calls isn't enough. We also need to account for:
- **How urgent the call was**
- **How long it took officers to respond**
- **What the outcome was (e.g., arrest, case filed)**

By combining these elements into a single score (WLS), we can:
- **Better understand real workload of each service call**
- **Identify patrol sectors that are overburdened or underutilized**
- **Use that insight to redraw patrol boundaries in the next step**
""")
st.markdown("---")

# === Validate session state ===
if "beats" not in st.session_state or "incidents_df" not in st.session_state:
    st.error("❗Please go to **Step 1** and upload patrol zones and incident data first.")
    st.stop()

beats = st.session_state.beats
df = st.session_state.incidents_df.copy()


# === Helper functions ===

def extract_priority(level):
    if pd.isna(level) or level.strip() == "":
        return 0
    match = re.search(r'(\d+)', level)
    if match:
        return int(match.group(1)) if match.group(1) != "1F" else 1
    return 0

def replace_outliers_with_zero(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df[column] = df[column].apply(lambda x: 0 if x < lower or x > upper else x)
    return df

def get_disposition_weight(dispo):
    if pd.isna(dispo) or dispo.strip() == "":
        return 0.3
    dispo = dispo.lower()
    if "arrest" in dispo:
        return 1.0
    elif "case" in dispo:
        return 0.7
    else:
        return 0.3

# === Preprocess incident data  

df = df[~((df['lat'] == 37.87053067971685) & (df['lon'] == -122.273288))]
df = df[['Priority', 'lat', 'lon', 'Time Spent Responding', 'Dispositions']]
df = replace_outliers_with_zero(df, 'Time Spent Responding')
df['Priority Numeric'] = df['Priority'].apply(extract_priority)
df['Priority Weight'] = df['Priority Numeric'].map({1:1.0, 2:0.7, 3:0.4, 4:0.2, 5:0.1}).fillna(0.0)
df['Scaled Response Time'] = MinMaxScaler().fit_transform(df[['Time Spent Responding']])
df['Disposition Weight'] = df['Dispositions'].apply(get_disposition_weight)
 

# === Let users customize weights ===

st.header("Step 2.1: Choose How You Define Workload Score")

st.markdown("""
You decide what matters most when measuring patrol workload.  
Adjust the inputs below to assign weights to each factor:

- **Urgency of call (priority)**
- **Response time**
- **Disposition (outcome)**

The total must sum to exactly 1.0 — you are directly defining how workload is calculated.
""")


col1, col2, col3 = st.columns(3)
w1 = col1.number_input("Weight for Priority", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
w2 = col2.number_input("Weight for Response Time", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
w3 = col3.number_input("Weight for Disposition", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

total_weight = w1 + w2 + w3
if abs(total_weight - 1.0) > 0.001:
    st.error(f"❌ The weights must sum to exactly 1.0. Current total: {total_weight:.2f}")
    st.stop()
else:
    st.success(f"✅ Weights are valid and sum to 1.0. (Priority: {w1:.2f}, Response Time: {w2:.2f}, Disposition: {w3:.2f})")

    # Store weights in session state
    st.session_state.wls_weights = {
        "priority": w1,
        "response_time": w2,
        "disposition": w3
    }

    df["WLS"] = (
        w1 * df["Priority Weight"] +
        w2 * df["Scaled Response Time"] +
        w3 * df["Disposition Weight"]
    )

# === Grid-Based Visualization ===
st.markdown("---")
st.header("Step 2.2: Visualize Workload Score (WLS) by Grid")

st.markdown("""
We divide Berkeley into a uniform grid to **visualize where workload is concentrated** across the city. Each service call is assigned to its nearest grid cell, and the total workload (WLS) is summed per grid. 
This method helps us detect **hotspots of operational demand**, regardless of current patrol zones.
""")

minx, miny, maxx, maxy = beats.total_bounds
cell_size = 0.001
grid_cells = [box(x, y, x + cell_size, y + cell_size)
              for x in np.arange(minx, maxx, cell_size)
              for y in np.arange(miny, maxy, cell_size)]
grid = gpd.GeoDataFrame(geometry=grid_cells, crs=beats.crs)
grid["Grid_ID"] = grid.index
grid["centroid"] = grid.geometry.centroid
centroids = grid.copy()
centroids["geometry"] = centroids["centroid"]
grid["Sector"] = gpd.sjoin(centroids, beats, how="left", predicate="intersects")["Sector"].values
grid.dropna(subset=["Sector"], inplace=True)
grid["Sector"] = grid["Sector"].astype(int)

points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs="EPSG:4326")
joined = gpd.sjoin_nearest(points, grid, how="left")
grid_npps = joined.groupby("Grid_ID")["WLS"].sum().reset_index()
grid = grid.merge(grid_npps, on="Grid_ID", how="left").fillna({"WLS": 0})

fig, ax = plt.subplots(figsize=(12, 8))
norm = plt.Normalize(grid["WLS"].min(), grid["WLS"].max())
grid.plot(column="WLS", cmap="Reds", edgecolor="black", alpha=0.8, norm=norm, ax=ax)
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="Reds"), ax=ax, label="Total WLS per Grid")
plt.title("Grid-Level Workload Score (WLS)", fontsize=14)
plt.axis("off")
st.pyplot(fig)

# === Sector-Level Aggregation ===
st.markdown("---")
st.header("Step 2.3: Aggregate WLS to Sector Level")

st.markdown("""
Finally, we sum up the WLS within each patrol sector to see which areas are **most overloaded**.  
This tells us where to consider rebalancing in Step 3.
""")

# Calculate sector-level WLS
sector_npps = grid.groupby("Sector")["WLS"].sum().reset_index(name="Sector_WLS")


# Prepare DataFrames for merge
beats = beats.copy()  # Create a copy to avoid modifying the original
sector_npps = sector_npps.copy()

# Drop existing Sector_WLS and Load Category if they exist
beats = beats.drop(columns=["Sector_WLS", "Load Category"], errors='ignore')

beats["Sector"] = beats["Sector"].astype(str)
sector_npps["Sector"] = sector_npps["Sector"].astype(str)

# Perform the merge
beats = beats.merge(sector_npps[["Sector", "Sector_WLS"]], on="Sector", how="left")


# Display the results
display_df = (beats[["Sector", "Sector_WLS"]]
             .sort_values("Sector")
             .rename(columns={"Sector": "Sector ID", "Sector_WLS": "Total WLS"}))
st.dataframe(display_df)

# Create the visualization
fig, ax = plt.subplots(figsize=(12, 8))
norm = mcolors.Normalize(vmin=beats["Sector_WLS"].min(), vmax=beats["Sector_WLS"].max())
beats.plot(column="Sector_WLS", cmap="Reds", norm=norm, edgecolor="black", linewidth=1, alpha=0.8, ax=ax)

for idx, row in beats.iterrows():
    centroid = row.geometry.centroid
    ax.text(centroid.x, centroid.y, f"{row['Sector']}", ha="center", va="center", fontsize=10, fontweight="bold")

plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="Reds"), ax=ax, label="Total WLS per Sector")
plt.title("Sector-Level Workload Score Heatmap", fontsize=14)
plt.axis("off")
st.pyplot(fig)
st.markdown("---")

# === Sector Classification ===
st.markdown("### WLS-Based Sector Classification for Rebalancing")

# Calculate thresholds and classify sectors
mean_wls = beats["Sector_WLS"].mean()
threshold_high = mean_wls * 1.2
threshold_low = mean_wls * 0.8

def classify_sector(wls):
    if pd.isna(wls):
        return "No Data"
    elif wls > threshold_high:
        return "Overburdened"
    elif wls < threshold_low:
        return "Underutilized"
    else:
        return "Balanced"

# Create Load Category and prepare final display
beats["Load Category"] = beats["Sector_WLS"].apply(classify_sector)

result_df = (beats[["Sector", "Sector_WLS", "Load Category"]]
             .sort_values("Sector")
             .rename(columns={
                 "Sector": "Sector ID",
                 "Sector_WLS": "Total WLS",
                 "Load Category": "Classification"
             })
             .round(2))

st.dataframe(result_df)

# Classification explanation
overburdened_sectors = beats[beats["Load Category"] == "Overburdened"]["Sector"].tolist()
underutilized_sectors = beats[beats["Load Category"] == "Underutilized"]["Sector"].tolist()

st.info(f"""
To guide data-driven patrol boundary adjustments, we classify each sector based on its total workload (WLS).

**Classification criteria:**
- **Overburdened**: WLS > {threshold_high:.2f} (120% of mean)
  → Sector {', '.join(map(str, sorted(overburdened_sectors)))}
- **Balanced**: {threshold_low:.2f} ≤ WLS ≤ {threshold_high:.2f}
- **Underutilized**: WLS < {threshold_low:.2f} (80% of mean)
  → Sector {', '.join(map(str, sorted(underutilized_sectors)))}

Sectors flagged as **overburdened** or **underutilized** are key candidates for boundary adjustment in Step 3.
""")

# Store updated beats in session state
st.session_state.beats = beats

 

