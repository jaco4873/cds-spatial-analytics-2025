#!/usr/bin/env python3
# Examine the wrangled shapefile data to see what attributes are already included

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load municipal data from wrangled shapefile
print("Loading wrangled data...")
municipalities = gpd.read_file("data/wrangled_data/wrangled_data.shp")

# Display basic information
print(f"Dataset shape: {municipalities.shape}")
print("\nColumns available in the shapefile:")
for col in municipalities.columns:
    print(f" - {col}")

# Check for election-related columns
election_cols = [
    col
    for col in municipalities.columns
    if "turnout" in col.lower()
    or "vote" in col.lower()
    or "elect" in col.lower()
    or "KV" in col
    or "VtrTrnt" in col
]
if election_cols:
    print("\nElection-related columns found:")
    for col in election_cols:
        print(f" - {col}")

# Look for columns related to municipality IDs or years
id_cols = [
    col
    for col in municipalities.columns
    if "id" in col.lower()
    or "kommun" in col.lower()
    or "nr" in col.lower()
    or "2005" in col
    or "2009" in col
]
if id_cols:
    print("\nMunicipality ID or year-related columns found:")
    for col in id_cols:
        print(f" - {col}")

# Print sample data rows
print("\nSample data (first 3 rows):")
print(municipalities.head(3).T)  # Transpose for easier reading

# Summary stats for numeric columns
print("\nSummary statistics for numeric columns:")
print(municipalities.describe().T)  # Transpose for easier reading

# Check if the dist column exists for identifying mergers
if "dist" in municipalities.columns:
    print("\nDistribution of 'dist' values for identifying mergers:")
    print(municipalities["dist"].describe())
    print(
        f"Number of municipalities with dist > 0: {(municipalities['dist'] > 0).sum()}"
    )
    print(
        f"Number of municipalities with dist = 0: {(municipalities['dist'] == 0).sum()}"
    )

# If VoterTurnout columns exist for both years, check for the change
turnout_2005_col = [
    col
    for col in municipalities.columns
    if "VtrTrnt" in col or "turnout_2005" in col.lower()
]
turnout_2009_col = [
    col for col in municipalities.columns if "turnout_2009" in col.lower()
]

if turnout_2005_col and turnout_2009_col:
    print("\nTurnout data for both 2005 and 2009 found. Can calculate change directly.")
elif "VtrTrnt" in municipalities.columns:
    print("\nFound 2005 turnout data (VtrTrnt column).")
    # Basic stats on turnout
    print("\nVoter turnout statistics (2005):")
    print(municipalities["VtrTrnt"].describe())

# Check for pre-calculated turnout change
change_cols = [col for col in municipalities.columns if "change" in col.lower()]
if change_cols:
    print("\nTurnout change columns found:")
    for col in change_cols:
        print(f" - {col}")

print("\nExamination complete.")
