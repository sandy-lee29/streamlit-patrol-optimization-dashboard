# [Berkeley Police Patrol Optimization Dashboard](https://app-patrol-optimization-dashboard.streamlit.app/) 

A Streamlit-based web application for optimizing police patrol sector boundaries based on service call data and workload analysis.

## Overview

This tool helps law enforcement agencies optimize their patrol sector boundaries by:
- Analyzing service call distribution patterns
- Calculating workload scores for each sector
- Optimizing boundaries to balance workload across sectors
- Snapping final boundaries to street networks for practical deployment

## Features

### 1. Service Call Analysis
- Upload and visualize service call data
- Interactive maps showing incident distribution
- Automatic workload calculation per sector
- Real-time statistics and visualizations

### 2. Workload Score Calculation
- Considers multiple factors:
  - Call priority levels
  - Response times
  - Time spent on scene
  - Call dispositions
- Weighted scoring system for balanced evaluation

### 3. Boundary Optimization
- Data-driven boundary adjustment
- Maintains spatial contiguity
- Balances workload across sectors
- Patrol Balance Index (PBI) scoring
- Street network alignment

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/streamlit-patrol-optimization-dashboard.git

# Navigate to the project directory
cd streamlit-patrol-optimization-dashboard

# Install required packages
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run service_call_overview.py
```

2. Upload Required Data:
   - Current patrol zone shapefile (ZIP containing .shp, .shx, .dbf)
   - Annual service call records (CSV)

3. Follow the three-step process:
   - Step 1: Data Upload & Initial Analysis
   - Step 2: Workload Score Calculation
   - Step 3: Boundary Optimization

## Data Requirements

### Patrol Zone Shapefile
- Must include a 'Sector' column
- Valid polygon geometries
- Projected coordinate system

### Service Call Data (CSV)
Required columns:
- `lat`: Latitude
- `lon`: Longitude
- `Priority`: Call priority level
- `Time Spent Responding`: Response duration
- `Dispositions`: Call outcomes

## Output

The tool generates:
- Optimized patrol sector boundaries
- Workload distribution analysis
- Street-aligned boundary shapefiles
- Performance metrics (PBI scores)

## Dependencies

- streamlit>=1.31.0
- geopandas>=0.14.0
- pandas>=2.0.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- plotly>=5.18.0
- numpy>=1.24.0
- shapely>=2.0.0

## Acknowledgments

This tool was developed to support the Berkeley Police Department's Office of Strategic Planning and Accountability (OSPA) in their efforts to create more equitable and efficient patrol distributions.
