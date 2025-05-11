#!/usr/bin/env python
import logging
import sys
from src.services.data_loader import DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Create data loader
data_loader = DataLoader()

# Load geographical data
try:
    gdf = data_loader.load_geographical_data()
    print(f"\nSuccessfully loaded {len(gdf)} municipalities from geoBoundaries")
    print("\nFirst 5 municipalities:")
    print(gdf[["shapeName", "name_lower"]].head())

    # Print column names
    print("\nAvailable columns:")
    for col in gdf.columns:
        print(f"- {col}")

    # Print coordinate reference system
    print(f"\nCoordinate reference system: {gdf.crs}")

except Exception as e:
    print(f"Error loading data: {e}")
