#!/usr/bin/env python3
# Script to examine the structure of the wrangled data

import geopandas as gpd

# Load municipal data from wrangled shapefile
print("Loading wrangled data...")
municipalities = gpd.read_file("data/wrangled_data/wrangled_data.shp")

# Display basic information
print(f"Dataset shape: {municipalities.shape}")
print("\nColumns:")
print(municipalities.columns.tolist())

# Display data types
print("\nData types:")
print(municipalities.dtypes)

# Display a sample of the data
print("\nSample data:")
print(municipalities.head())

# Display summary statistics for numeric columns
print("\nSummary statistics:")
print(municipalities.describe())

# Check for missing values
print("\nMissing values:")
print(municipalities.isnull().sum())

# If the dataset contains municipality codes/names, list them
if "name" in municipalities.columns:
    print("\nMunicipality names:")
    print(municipalities["name"].tolist())
elif "kommune" in municipalities.columns:
    print("\nMunicipality names:")
    print(municipalities["kommune"].tolist())
elif "KOMNAVN" in municipalities.columns:
    print("\nMunicipality names:")
    print(municipalities["KOMNAVN"].tolist())

print("\nExamination complete.")
