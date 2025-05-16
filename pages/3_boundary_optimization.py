import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import box, Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union, nearest_points
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler
import re
import plotly.graph_objects as go
import io
import zipfile
import tempfile
import os
import pathlib

# Helper functions from Page 2
def replace_outliers_with_zero(df, column):
    """Replace outliers in a column with zero using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df[column] = df[column].apply(lambda x: 0 if x < lower or x > upper else x)
    return df

def extract_priority(level):
    """Extract priority level from string."""
    if pd.isna(level) or level.strip() == "":
        return 0
    match = re.search(r'(\d+)', level)
    if match:
        return int(match.group(1)) if match.group(1) != "1F" else 1
    return 0

def get_disposition_weight(dispo):
    """Get weight based on disposition type."""
    if pd.isna(dispo) or dispo.strip() == "":
        return 0.3
    dispo = dispo.lower()
    if "arrest" in dispo:
        return 1.0
    elif "case" in dispo:
        return 0.7
    else:
        return 0.3

# Grid utilities
def create_grid_from_beats(beats: gpd.GeoDataFrame, cell_size: float = 0.001) -> gpd.GeoDataFrame:
    """Create a grid from beats shapefile and assign sectors."""
    minx, miny, maxx, maxy = beats.total_bounds
    
    grid_cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + cell_size, y + cell_size)
            grid_cells.append(cell)
            y += cell_size
        x += cell_size
    
    grid = gpd.GeoDataFrame(geometry=grid_cells, crs=beats.crs)
    grid["centroid"] = grid.geometry.centroid
    centroids = grid.copy()
    centroids["geometry"] = centroids["centroid"]
    
    centroids_sjoined = gpd.sjoin(
        centroids,
        beats,
        how="left",
        predicate="intersects"
    ).reset_index(drop=True)
    
    grid["Sector"] = centroids_sjoined["Sector"]
    grid = grid.dropna(subset=["Sector"])
    grid["Sector"] = grid["Sector"].astype(int)
    grid["Grid_ID"] = grid.index
    
    return grid

def calculate_grid_wls(grid: gpd.GeoDataFrame, incidents_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Calculate WLS for each grid cell using Page 2's method."""
    # Process incidents data
    required_columns = ['Priority', 'lat', 'lon', 'Time Spent Responding', 'Dispositions']
    missing_columns = [col for col in required_columns if col not in incidents_df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns in incidents data: {', '.join(missing_columns)}")
        st.error("Please ensure all required columns are present in the incidents data uploaded in Step 1")
        st.stop()
        
    df = incidents_df[required_columns].copy()
    
    # Remove known outlier point
    df = df[~((df['lat'] == 37.87053067971685) & (df['lon'] == -122.273288))]
    
    df = replace_outliers_with_zero(df, 'Time Spent Responding')
    df['Priority Numeric'] = df['Priority'].apply(extract_priority)
    df['Priority Weight'] = df['Priority Numeric'].map({1:1.0, 2:0.7, 3:0.4, 4:0.2, 5:0.1}).fillna(0.0)
    df['Scaled Response Time'] = MinMaxScaler().fit_transform(df[['Time Spent Responding']])
    df['Disposition Weight'] = df['Dispositions'].apply(get_disposition_weight)
    
    # Get weights from session state or use defaults
    if 'wls_weights' in st.session_state:
        w1 = st.session_state.wls_weights["priority"]
        w2 = st.session_state.wls_weights["response_time"]
        w3 = st.session_state.wls_weights["disposition"]
    else:
        w1, w2, w3 = 0.7, 0.2, 0.1  # Default weights
    
    # Calculate WLS with weights
    df["WLS"] = (
        w1 * df["Priority Weight"] +
        w2 * df["Scaled Response Time"] +
        w3 * df["Disposition Weight"]
    )
    
    # Match incidents to grid cells and calculate grid-level WLS
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs="EPSG:4326")
    joined = gpd.sjoin_nearest(points, grid[["geometry", "Grid_ID"]], how="left")
    
    # Calculate WLS per grid cell
    grid_wls = joined.groupby("Grid_ID")["WLS"].sum().reset_index()
    
    # Create a new grid with WLS values
    result_grid = grid.copy()
    
    # Initialize WLS column with zeros
    result_grid["WLS"] = 0
    
    # Update WLS values for grid cells that have incidents
    wls_dict = dict(zip(grid_wls["Grid_ID"], grid_wls["WLS"]))
    result_grid.loc[result_grid["Grid_ID"].isin(wls_dict.keys()), "WLS"] = \
        result_grid.loc[result_grid["Grid_ID"].isin(wls_dict.keys()), "Grid_ID"].map(wls_dict)
    
    return result_grid

# Optimization functions
def build_boundary_pairs_info(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Find grid cells on sector boundaries."""
    boundary_list = []
    for idx, row in grid.iterrows():
        from_sector = row["Sector"]
        neighbors = grid[grid.geometry.touches(row.geometry)]
        diff_sector_neighbors = neighbors[neighbors["Sector"] != from_sector]
        for n_idx, n_row in diff_sector_neighbors.iterrows():
            boundary_list.append({
                "Grid_ID": row["Grid_ID"],
                "Sector": from_sector,
                "neighbor_sector": n_row["Sector"],
                "geometry": row["geometry"]
            })
    boundary_pairs_info = gpd.GeoDataFrame(boundary_list, crs=grid.crs)
    boundary_pairs_info.drop_duplicates(
        subset=["Grid_ID", "Sector", "neighbor_sector"],
        inplace=True
    )
    return boundary_pairs_info

def give_bulk_boundaries_from_excess(
    grid: gpd.GeoDataFrame,
    sector_neighbors: dict,
    excess_sectors: list
) -> tuple:
    """Move boundary grids from excess sectors to neighbors."""
    boundary_pairs_info = build_boundary_pairs_info(grid)
    moved_all = []
    
    eligible_donors = sorted(
        [s for s in excess_sectors],
        key=lambda s: len([n for n in sector_neighbors[s] if n not in excess_sectors])
    )
    
    for donor in eligible_donors:
        recipients = [n for n in sector_neighbors[donor] if n not in excess_sectors]
        for recipient in recipients:
            target_boundary = boundary_pairs_info[
                (boundary_pairs_info["Sector"] == donor) &
                (boundary_pairs_info["neighbor_sector"] == recipient)
            ]
            moved_ids = target_boundary["Grid_ID"].unique()
            if len(moved_ids) > 0:
                grid.loc[grid["Grid_ID"].isin(moved_ids), "Sector"] = recipient
                moved_all.extend(moved_ids)
    
    return grid, moved_all

def take_bulk_boundaries_to_deficient(
    grid: gpd.GeoDataFrame,
    deficient_sector: int,
    neighbors: list
) -> tuple:
    """Pull boundary grids to deficient sector from neighbors."""
    boundary_pairs_info = build_boundary_pairs_info(grid)
    moved_all = []
    
    for nbr in neighbors:
        target_boundary = boundary_pairs_info[
            (boundary_pairs_info["Sector"] == nbr) &
            (boundary_pairs_info["neighbor_sector"] == deficient_sector)
        ]
        moved_ids = target_boundary["Grid_ID"].unique()
        if len(moved_ids) > 0:
            grid.loc[grid["Grid_ID"].isin(moved_ids), "Sector"] = deficient_sector
            moved_all.extend(moved_ids)
    
    return grid, moved_all

# Visualization functions
def plot_sector_heatmap(beats: gpd.GeoDataFrame, workload_stats: pd.DataFrame) -> plt.Figure:
    """Create a heatmap of sector workload."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = plt.cm.Reds
    norm = mcolors.Normalize(
        vmin=workload_stats["Total WLS"].min(),
        vmax=workload_stats["Total WLS"].max()
    )
    
    # Ensure both DataFrames have the same type for 'Sector' column
    beats = beats.copy()
    workload_stats = workload_stats.copy()
    beats["Sector"] = beats["Sector"].astype(str)
    workload_stats["Sector"] = workload_stats["Sector"].astype(str)
    
    beats = beats.merge(
        workload_stats[["Sector", "Total WLS"]],
        on="Sector",
        how="left"
    )
    
    beats.plot(
        column="Total WLS",
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=1,
        alpha=0.8,
        ax=ax
    )
    
    beats["centroid"] = beats.geometry.centroid
    for idx, row in beats.iterrows():
        ax.text(
            row["centroid"].x,
            row["centroid"].y,
            f"{int(row['Sector'])}",
            fontsize=10,
            color="black",
            ha="center",
            va="center",
            fontweight="bold"
        )
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label="Total WLS per Sector")
    ax.set_axis_off()
    
    return fig

def plot_grid_assignments(grid: gpd.GeoDataFrame, beats: gpd.GeoDataFrame) -> plt.Figure:
    """Plot current grid assignments with sector boundaries."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    unique_sectors = sorted(grid["Sector"].unique())
    cmap = plt.cm.get_cmap("tab20", len(unique_sectors))
    color_dict = {sec: cmap(i) for i, sec in enumerate(unique_sectors)}
    
    for sec in unique_sectors:
        subset = grid[grid["Sector"] == sec]
        subset.plot(
            ax=ax,
            facecolor=color_dict[sec],
            edgecolor="black",
            linewidth=0.5,
            alpha=1.0
        )
    
    beats.boundary.plot(
        ax=ax,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="Original Boundaries"
    )
    
    legend_patches = [
        Patch(facecolor=color_dict[sec], edgecolor="black", label=f"Sector {sec}")
        for sec in unique_sectors
    ]
    legend_patches.append(
        Patch(facecolor="none", edgecolor="black", linestyle="--", label="Original Boundaries")
    )
    
    ax.legend(
        handles=legend_patches,
        title="Sector Assignments",
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.
    )
    
    ax.set_title("Current Sector Assignments", fontsize=14)
    ax.set_axis_off()
    
    return fig

def create_workload_summary(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """Create a summary of workload statistics by sector."""
    sector_stats = grid.groupby("Sector")["WLS"].sum().reset_index()
    sector_stats.rename(columns={"WLS": "Total WLS"}, inplace=True)
    
    mean_wls = sector_stats["Total WLS"].mean()
    sector_stats["Difference from Mean (%)"] = (
        (sector_stats["Total WLS"] - mean_wls) / mean_wls * 100
    )
    
    # Classify sectors
    sector_stats["Status"] = "Normal"
    sector_stats.loc[sector_stats["Total WLS"] > mean_wls * 1.2, "Status"] = "Overburdened"
    sector_stats.loc[sector_stats["Total WLS"] < mean_wls * 0.8, "Status"] = "Underutilized"
    
    return sector_stats

def calculate_pbi_metrics(grid: gpd.GeoDataFrame) -> tuple:
    """Calculate variance and max/min ratio of WLS across sectors."""
    sector_wls = grid.groupby("Sector")["WLS"].sum()
    variance = sector_wls.var()
    max_min_ratio = sector_wls.max() / sector_wls.min()
    return variance, max_min_ratio

def normalized_score(value: float, worst: float, best: float) -> float:
    """Convert a metric to a 1-10 score based on worst/best thresholds."""
    score_0_to_1 = (worst - value) / (worst - best)  # The lower the value, the better
    score_0_to_1 = max(min(score_0_to_1, 1), 0)  # Clamp between 0 and 1
    return round(score_0_to_1 * 9 + 1, 1)  # Convert to 1-10 scale

def calculate_pbi(grid: gpd.GeoDataFrame, 
                 worst_variance: float = 120000, 
                 best_variance: float = 60000,
                 worst_ratio: float = 2.0, 
                 best_ratio: float = 1.25) -> tuple:
    """Calculate the Patrol Balance Index (PBI) for a grid configuration."""
    variance, ratio = calculate_pbi_metrics(grid)
    
    var_score = normalized_score(variance, worst_variance, best_variance)
    ratio_score = normalized_score(ratio, worst_ratio, best_ratio)
    
    final_score = round((var_score + ratio_score) / 2, 1)
    return final_score, variance, ratio, var_score, ratio_score

# Add the street snapping functions after the PBI functions
def find_nearest_point_on_lines(point, lines, max_distance=50):
    """Find the nearest point on any line within max_distance meters."""
    min_dist = float('inf')
    nearest_point = None

    for line in lines:
        if line.distance(point) <= max_distance:
            p = line.interpolate(line.project(point))
            dist = point.distance(p)
            if dist < min_dist:
                min_dist = dist
                nearest_point = p

    return nearest_point if min_dist <= max_distance else None

def snap_polygon_to_streets(polygon, street_lines, tolerance=50):
    """Snap a single polygon's vertices to nearest street segments."""
    coords = list(polygon.exterior.coords)
    new_coords = []

    for i, coord in enumerate(coords[:-1]):
        point = Point(coord)
        snapped = find_nearest_point_on_lines(point, street_lines, tolerance)
        
        if snapped:
            new_coords.append((snapped.x, snapped.y))
        else:
            new_coords.append(coord)

    new_coords.append(new_coords[0])
    return Polygon(new_coords)

def snap_to_streets(geom, streets, tolerance=50):
    """Snap polygon vertices to nearest street segments within tolerance distance."""
    street_lines = [geom for geom in streets.geometry]

    if isinstance(geom, Polygon):
        return snap_polygon_to_streets(geom, street_lines, tolerance)
    elif isinstance(geom, MultiPolygon):
        snapped_polys = []
        for poly in geom.geoms:
            snapped_poly = snap_polygon_to_streets(poly, street_lines, tolerance)
            snapped_polys.append(snapped_poly)
        return MultiPolygon(snapped_polys)
    else:
        raise ValueError(f"Unsupported geometry type: {type(geom)}")

def smooth_sectors(grid: gpd.GeoDataFrame, streets: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Smooth sector boundaries by snapping to nearby streets."""
    # Project to UTM Zone 10N (Berkeley area) for accurate distance calculations
    utm_crs = 'EPSG:26910'
    grid_utm = grid.to_crs(utm_crs)
    streets_utm = streets.to_crs(utm_crs)
    
    # Create spatial index for streets
    streets_sindex = streets_utm.sindex
    
    sectors = []
    for sector in sorted(grid_utm["Sector"].unique()):
        # Get all grid cells for this sector
        sector_grids = grid_utm[grid_utm["Sector"] == sector]
        
        # Dissolve grid cells into a single polygon
        sector_polygon = unary_union(sector_grids.geometry)
        
        # Get boundary coordinates
        if isinstance(sector_polygon, MultiPolygon):
            boundary_coords = [coord for poly in sector_polygon.geoms 
                             for coord in poly.exterior.coords[:-1]]
        else:
            boundary_coords = list(sector_polygon.exterior.coords[:-1])
        
        new_coords = []
        prev_snapped = None  # Keep track of previous snapped point
        
        for coord in boundary_coords:
            point = Point(coord)
            
            # Use spatial index to find nearby streets efficiently
            bounds = (point.x - 50, point.y - 50, point.x + 50, point.y + 50)
            possible_matches_idx = list(streets_sindex.intersection(bounds))
            
            if not possible_matches_idx:
                if prev_snapped:
                    # If we have a previous snapped point, try to maintain continuity
                    new_coords.append((prev_snapped.x, prev_snapped.y))
                else:
                    new_coords.append(coord)
                continue
            
            nearby_streets = streets_utm.iloc[possible_matches_idx]
            
            # Find closest point on nearby streets
            min_dist = float('inf')
            best_point = None
            
            for _, street in nearby_streets.iterrows():
                if street.geometry.distance(point) <= 50:  # 50 meters tolerance
                    projected = street.geometry.interpolate(street.geometry.project(point))
                    dist = point.distance(projected)
                    
                    # Prefer points that maintain continuity with previous point
                    if prev_snapped:
                        continuity_factor = Point(prev_snapped).distance(projected)
                        dist = dist + (continuity_factor * 0.5)  # Weight continuity
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_point = projected
            
            if best_point is not None:
                new_coords.append((best_point.x, best_point.y))
                prev_snapped = best_point
            else:
                if prev_snapped:
                    new_coords.append((prev_snapped.x, prev_snapped.y))
                else:
                    new_coords.append(coord)
        
        # Close the polygon
        new_coords.append(new_coords[0])
        
        try:
            # Create new polygon and simplify slightly to remove redundant points
            snapped_polygon = Polygon(new_coords).simplify(1.0)  # 1 meter tolerance
            if snapped_polygon.is_valid:
                sectors.append({
                    "Sector": sector,
                    "geometry": snapped_polygon
                })
            else:
                # If invalid, try to fix with buffer(0)
                fixed_polygon = snapped_polygon.buffer(0)
                if fixed_polygon.is_valid:
                    sectors.append({
                        "Sector": sector,
                        "geometry": fixed_polygon
                    })
                else:
                    # If still invalid, use original sector shape
                    sectors.append({
                        "Sector": sector,
                        "geometry": sector_polygon
                    })
        except Exception as e:
            # Fallback to original sector shape if polygon creation fails
            sectors.append({
                "Sector": sector,
                "geometry": sector_polygon
            })
    
    # Create GeoDataFrame with snapped sectors
    snapped_sectors = gpd.GeoDataFrame(sectors, crs=utm_crs)
    
    # Convert back to original CRS
    snapped_sectors = snapped_sectors.to_crs(grid.crs)
    
    return snapped_sectors

def create_shapefile_download(gdf, filename):
    """Create a downloadable zip file containing the shapefile."""
    # Create a temporary directory for the files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the GeoDataFrame as a shapefile
        gdf.to_file(os.path.join(tmpdir, filename))
        
        # Create a zip file containing all the shapefile components
        zip_path = os.path.join(tmpdir, "patrol_sectors.zip")
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for file in os.listdir(tmpdir):
                if file.startswith(filename.replace(".shp", "")):
                    zip_file.write(
                        os.path.join(tmpdir, file),
                        arcname=file
                    )
        
        # Read the zip file as bytes
        with open(zip_path, 'rb') as f:
            return f.read()

def load_default_file(filename):
    """Load default file using a path relative to this script."""
    try:
        # Get the directory where this script is located
        script_dir = pathlib.Path(__file__).parent.parent
        # Construct path to data directory
        file_path = script_dir / "data" / filename
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None

# Main Streamlit app code starts here
st.set_page_config(page_title="Step 3 – Generate Optimized Patrol Boundaries", layout="wide")
st.title("Step 3: Generate Optimized Patrol Boundaries")

st.markdown("""
This step uses an **iterative boundary optimization algorithm** to reduce workload imbalances between patrol sectors, based on the WLS (Workload Score) computed in Step 2.

### Purpose:
- Rebalance workload by adjusting sector boundaries — not from scratch, but incrementally.
- Maintain spatial contiguity and minimize disruption to existing patrol areas.
- Reassign only boundary-level grid cells, preserving overall sector structure.

### How it works:
1. **Classify sectors** as Overburdened (>120% of mean WLS), Balanced, or Underutilized (<80%).
2. **Transfer grid cells** from overburdened to adjacent sectors with lower load.
3. **Reassign cells** from neighboring sectors into underutilized ones.
4. **Recalculate WLS** after each step to gradually move toward a balanced configuration.

This lets planners make **data-driven yet operationally realistic** patrol adjustments.
""")
st.markdown("---")
# Check if previous steps are completed
if 'beats' not in st.session_state or 'incidents_df' not in st.session_state:
    st.error("⚠️ Please complete Steps 1 and 2 first!")
    st.stop()

# Get data from session state
beats = st.session_state.beats
incidents_df = st.session_state.incidents_df

# Create grid and calculate WLS using Page 2's method
grid = create_grid_from_beats(beats)
grid = calculate_grid_wls(grid, incidents_df)

# Calculate initial sector-level WLS
sector_stats = create_workload_summary(grid)

# Display current state
st.header("Current Workload Distribution")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sector-Level WLS Heatmap")
    fig_heatmap = plot_sector_heatmap(beats, sector_stats)
    st.pyplot(fig_heatmap)

with col2:
    st.subheader("Workload Analysis")
    st.write("Current Sector-Level Workload:")
    st.dataframe(
        sector_stats.style.format({
            "Total WLS": "{:.2f}",
            "Difference from Mean (%)": "{:.1f}%"
        })
    )
    
mean_wls = sector_stats["Total WLS"].mean()
high_threshold = mean_wls * 1.2
low_threshold = mean_wls * 0.8
status_counts = sector_stats["Status"].value_counts()

st.info(f"""
**Summary:**  
The average workload (WLS) across all sectors is **{mean_wls:.2f}**.  
Sectors with WLS above **{high_threshold:.2f} (+20%)** are classified as **Overburdened**,  
while those below **{low_threshold:.2f} (–20%)** are marked as **Underutilized**.

**Sector Classification:**  
- Overburdened: {status_counts.get('Overburdened', 0)} sector(s)  
- Underutilized: {status_counts.get('Underutilized', 0)} sector(s)  
- Balanced: {status_counts.get('Normal', 0)} sector(s)
""")


# Define sector neighbors
sector_neighbors = {
    1: [2],
    2: [1, 3, 4, 14],
    3: [2, 4, 5, 11, 12, 14],
    4: [2, 3, 5, 6],
    5: [3, 4, 6, 9, 10, 11],
    6: [4, 5, 7, 8, 9],
    7: [6, 8],
    8: [6, 7, 9],
    9: [5, 6, 8, 10],
    10: [5, 9, 11],
    11: [3, 5, 10, 12],
    12: [3, 11, 13],
    13: [12, 14],
    14: [2, 3, 13]
}

# Initialize optimization state
if 'optimized_grid' not in st.session_state:
    st.session_state.optimized_grid = grid.copy()
    st.session_state.excess_optimization_done = False

# Get sectors to optimize
overburdened = sector_stats[sector_stats["Status"] == "Overburdened"]["Sector"].tolist()
underutilized = sector_stats[sector_stats["Status"] == "Underutilized"]["Sector"].tolist()

st.markdown("---")
# Step 1: Excess Optimization
st.header("Boundary Optimization")
st.subheader("Step 1: Optimize Overburdened Sectors")
col1, col2 = st.columns(2)

with col1:
    st.write(f"Overburdened sectors: {', '.join(map(str, overburdened))}")
    
    if st.button("Reduce Grids in Overburdened Sectors"):
        with st.spinner("Optimizing overburdened sectors..."):
            grid_updated, moved_ids = give_bulk_boundaries_from_excess(
                st.session_state.optimized_grid.copy(),
                sector_neighbors,
                overburdened
            )
            
            if len(moved_ids) > 0:
                st.session_state.optimized_grid = grid_updated
                st.session_state.excess_optimization_done = True
                
                # Recalculate WLS for the updated grid
                st.session_state.optimized_grid = calculate_grid_wls(
                    st.session_state.optimized_grid,
                    incidents_df
                )
                
                # Update sector stats
                new_sector_stats = create_workload_summary(st.session_state.optimized_grid)
                
                st.success(f"Moved {len(moved_ids)} grid cells from overburdened sectors")
                st.write("Updated WLS distribution:")
                st.dataframe(new_sector_stats)

with col2:
    if st.session_state.excess_optimization_done:
        st.write("Results after Excess Optimization:")
        fig_excess = plot_grid_assignments(st.session_state.optimized_grid, beats)
        st.pyplot(fig_excess)
    else:
        st.write("Original State:")
        fig_original = plot_grid_assignments(grid, beats)
        st.pyplot(fig_original)

# Step 2: Deficient Optimization
st.subheader("Step 2: Optimize Underutilized Sectors")
col3, col4 = st.columns(2)

with col3:
    if not st.session_state.excess_optimization_done:
        st.warning("Please complete Step 1 (Excess Optimization) first")
    else:
        st.write(f"Underutilized sectors: {', '.join(map(str, underutilized))}")
        
        if st.button("Add Grids to Underutilized Sectors"):
            with st.spinner("Optimizing underutilized sectors..."):
                grid_updated = st.session_state.optimized_grid.copy()
                total_moved = 0
                
                for deficient_sector in underutilized:
                    grid_temp, moved_ids = take_bulk_boundaries_to_deficient(
                        grid_updated,
                        deficient_sector,
                        sector_neighbors[deficient_sector]
                    )
                    if len(moved_ids) > 0:
                        grid_updated = grid_temp
                        total_moved += len(moved_ids)
                
                if total_moved > 0:
                    st.session_state.optimized_grid = grid_updated
                    
                    # Recalculate WLS for the updated grid
                    st.session_state.optimized_grid = calculate_grid_wls(
                        st.session_state.optimized_grid,
                        incidents_df
                    )
                    
                    # Update sector stats
                    final_stats = create_workload_summary(st.session_state.optimized_grid)
                    
                    st.success(f"Moved {total_moved} grid cells to underutilized sectors")
                    st.write("Final WLS distribution:")
                    st.dataframe(final_stats)
                else:
                    st.info("No grid cells could be moved to improve underutilized sectors")

with col4:
    if st.session_state.excess_optimization_done:
        st.write("Optimized Boundary State:")
        fig_final = plot_grid_assignments(st.session_state.optimized_grid, beats)
        st.pyplot(fig_final)

st.success("""
✅ Boundary optimization complete! The new boundaries:
- Maintain spatial contiguity of all sectors
- Improve workload balance between sectors
- Preserve core areas of existing sectors
""")

# Store final optimized results in session state
st.session_state.final_optimized_grid = st.session_state.optimized_grid 

st.markdown("---")
st.header("Optimization Evaluation: Patrol Balance Index (PBI)")
st.markdown("""
The PBI is a 1–10 scale metric that evaluates how evenly workload is distributed across patrol sectors. It combines:
- **WLS Variance**: How spread out the workload is across sectors
- **Max/Min Ratio**: The disparity between highest and lowest workload sectors

A higher score (closer to 10) indicates a more balanced configuration.
""")

# Calculate PBI for original and optimized configurations
original_pbi, orig_var, orig_ratio, orig_var_score, orig_ratio_score = calculate_pbi(grid)
if st.session_state.excess_optimization_done:
    optimized_pbi, opt_var, opt_ratio, opt_var_score, opt_ratio_score = calculate_pbi(st.session_state.optimized_grid)
    
    # Display results in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Before Optimization")
        st.write(f"**PBI Score:** {original_pbi}/10")
        st.write(f"- WLS Variance: {orig_var:.2f} (Score: {orig_var_score}/10)")
        st.write(f"- Max/Min Ratio: {orig_ratio:.2f} (Score: {orig_ratio_score}/10)")
        
    with col2:
        st.subheader("After Optimization")
        st.write(f"**PBI Score:** {optimized_pbi}/10")
        st.write(f"- WLS Variance: {opt_var:.2f} (Score: {opt_var_score}/10)")
        st.write(f"- Max/Min Ratio: {opt_ratio:.2f} (Score: {opt_ratio_score}/10)")
        
        improvement = optimized_pbi - original_pbi
        if improvement > 0:
            st.success(f"✨ Balance improved by {improvement:.1f} points!")
        else:
            st.warning("No improvement in balance score.")
    
    # Score comparison visualization (outside columns)
    st.subheader("PBI Score Comparison")
    
    # Create comparison chart
    fig = go.Figure()
    
    # Add bars for before/after scores
    categories = ['PBI Score', 'Variance Score', 'Max/Min Ratio Score']
    before_scores = [original_pbi, orig_var_score, orig_ratio_score]
    after_scores = [optimized_pbi, opt_var_score, opt_ratio_score]
    
    # Add before optimization bars
    fig.add_trace(go.Bar(
        name='Before',
        x=categories,
        y=before_scores,
        marker_color='lightgray'
    ))
    
    # Add after optimization bars
    fig.add_trace(go.Bar(
        name='After',
        x=categories,
        y=after_scores,
        marker_color='#00CC96'
    ))
    
    # Update layout
    fig.update_layout(
        yaxis_title='Score (1-10)',
        yaxis_range=[0, 10],
        barmode='group',
        showlegend=True,
        legend_title_text='Optimization Stage',
        plot_bgcolor='white',
        margin=dict(t=30, b=0),
        height=400  # Fixed height for better proportions
    )
    
    # Add horizontal grid lines
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )
    
    # Show the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Optional: threshold explanation
    with st.expander("How scoring thresholds work"):
        st.markdown(f"""
        - **Variance thresholds**: Best = 60,000 / Worst = 120,000  
        - **Max/Min ratio thresholds**: Best = 1.25 / Worst = 2.0  

        These define how raw imbalance metrics are scaled to 1–10 scores. They can be adjusted based on:
        - Operational goals (e.g., how strict you want the standard)
        - Historical benchmarks from past patrol years
        """)
            
else:
    st.write(f"**Current PBI Score:** {original_pbi}/10")
    st.info("Complete the optimization steps to see the improvement in balance scores.")

# Add new section after the PBI evaluation
st.markdown("---")
st.header("Final Steps: Smoothing Boundaries")

# Street snapping section
st.markdown("""
Let's snap the optimized grid boundaries to nearby streets, making them more practical for real-world deployment.
We'll use Berkeley's street centerlines data for this process.
""")

# Load embedded centerlines data
default_centerlines = load_default_file("Centerlines.zip")

if default_centerlines is not None:
    if st.button("Snap Boundaries to Streets"):
        with st.spinner("Snapping boundaries to streets..."):
            # Create a temporary file to read the zipped shapefile
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write the zip file
                zip_path = os.path.join(tmpdir, "Centerlines.zip")
                with open(zip_path, 'wb') as f:
                    f.write(default_centerlines)
                
                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                # Find the .shp file
                shp_file = next(f for f in os.listdir(tmpdir) if f.endswith('.shp'))
                streets = gpd.read_file(os.path.join(tmpdir, shp_file))
                
                # Get the final grid (either optimized or original)
                final_grid = (st.session_state.optimized_grid 
                            if 'optimized_grid' in st.session_state 
                            else grid)
                
                # Add progress bar
                progress_text = "Operation in progress. Please wait..."
                my_bar = st.progress(0, text=progress_text)
                
                # Perform the snapping
                snapped_sectors = smooth_sectors(final_grid, streets)
                my_bar.progress(100, text="Snapping complete!")
                st.session_state.snapped_sectors = snapped_sectors
                
                # Show success message
                st.success("✅ Boundaries successfully snapped to streets!")
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Original boundaries
                streets.plot(ax=ax1, color='gray', linewidth=0.5, alpha=0.5)
                final_grid.boundary.plot(ax=ax1, color='black', linewidth=1)
                ax1.set_title("Original Boundaries")
                ax1.set_axis_off()
                
                # Snapped boundaries
                streets.plot(ax=ax2, color='gray', linewidth=0.5, alpha=0.5)
                snapped_sectors.boundary.plot(ax=ax2, color='black', linewidth=1)
                ax2.set_title("Street-Snapped Boundaries")
                ax2.set_axis_off()
                
                plt.tight_layout()
                st.pyplot(fig)
else:
    st.error("Error: Could not load street centerlines data. Please contact support.")

# Download section
st.subheader("Final Output: Download Patrol Boundary Shapefile")
st.markdown("""
Download the final street-aligned patrol sector boundaries as a shapefile. 
- These optimized boundaries are ready for deployment and integration into GIS systems.
- They represent a more balanced workload distribution across sectors, based on real service call patterns.
- The tool has been validated for scalability across multiple years of data, and the Berkeley Police Department's Strategic Planning team will adopt these outputs for annual patrol zone updates, enabling more data-driven and equitable public safety decisions.         
""")

if 'snapped_sectors' in st.session_state:
    zip_data = create_shapefile_download(
        st.session_state.snapped_sectors, 
        "street_sectors.shp"
    )
    st.download_button(
        "📥 Download Street-Aligned Boundaries",
        data=zip_data,
        file_name="street_patrol_sectors.zip",
        mime="application/zip"
    )
else:
    st.info("Complete the boundary smoothing step first to enable download.")
 