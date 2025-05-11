#!/usr/bin/env python3
# municipal_reform/data_loader.py

import geopandas as gpd
import pandas as pd
import logging
import os
from pathlib import Path


class DataLoader:
    """Handles loading, cleaning, and preparing data for municipal reform analysis."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.municipalities = None
        self.ensure_output_dir()

    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs("output", exist_ok=True)

    def load_municipal_data(
        self, shapefile_path="data/wrangled_data/wrangled_data.shp"
    ):
        """Load municipal data from wrangled shapefile."""
        self.logger.info("Loading wrangled data...")
        root_dir = Path(__file__).parent.parent.parent
        self.municipalities = gpd.read_file(root_dir / shapefile_path)
        self.logger.info("Logging head of wrangled data:")
        self.logger.info(self.municipalities.head())
        self.logger.info("Logging shape of wrangled data:")
        self.logger.info(self.municipalities.shape)
        return self.municipalities

    def clean_data(self):
        """Clean the dataset by removing rows with missing essential data."""
        if self.municipalities is None:
            self.logger.error("No data loaded. Call load_municipal_data first.")
            return None

        # Remove rows with missing essential data
        self.municipalities = self.municipalities.dropna(
            subset=["KommnNr", "VtrTrnt"], how="any"
        )
        self.logger.info(
            f"Shape after removing NaN values: {self.municipalities.shape}"
        )

        # Fix voter turnout ratio where it exceeds 1.0 (likely data error)
        self.municipalities.loc[self.municipalities["VtrTrnt"] > 1.0, "VtrTrnt"] = (
            self.municipalities.loc[self.municipalities["VtrTrnt"] > 1.0, "VtrTrnt"]
            / 10
        )
        self.logger.info(
            f"Max voter turnout after correction: {self.municipalities['VtrTrnt'].max()}"
        )

        # Display basic information
        self.logger.info(f"Dataset shape: {self.municipalities.shape}")
        self.logger.info("\nColumns:")
        self.logger.info(self.municipalities.columns.tolist())
        self.logger.info("\nSample data:")
        self.logger.info(self.municipalities.head())
        self.logger.info("\nMissing values:")
        self.logger.info(self.municipalities.isnull().sum())

        return self.municipalities

    def load_voter_data(self, voter_data_path="data/ValgData2009.csv"):
        """Load voter turnout data from CSV and merge with municipal data."""
        if self.municipalities is None:
            self.logger.error("No data loaded. Call load_municipal_data first.")
            return None

        self.logger.info("\nLoading 2009 voter data...")
        root_dir = Path(__file__).parent.parent.parent
        valgdata2009 = pd.read_csv(root_dir / voter_data_path, sep=";")
        self.logger.info(f"2009 data shape: {valgdata2009.shape}")

        # Rename columns for clarity
        valgdata2009.columns = [
            "Gruppe",
            "KommuneNr",
            "KV2009_Stemmeberettigede",
            "KV2009_Afgivne",
        ]

        # Calculate 2009 turnout
        valgdata2009["VoterTurnout2009"] = (
            valgdata2009["KV2009_Afgivne"] / valgdata2009["KV2009_Stemmeberettigede"]
        )

        # Convert kommune number to numeric to match with shapefile
        valgdata2009["KommuneNr"] = pd.to_numeric(valgdata2009["KommuneNr"])

        # Merge with municipal data - using a left join to keep all municipalities
        self.municipalities = self.municipalities.merge(
            valgdata2009[
                [
                    "KommuneNr",
                    "KV2009_Stemmeberettigede",
                    "KV2009_Afgivne",
                    "VoterTurnout2009",
                ]
            ],
            left_on="KommnNr",
            right_on="KommuneNr",
            how="left",
        )

        self.logger.info(f"Shape after merging 2009 data: {self.municipalities.shape}")
        return self.municipalities

    def prepare_data(self):
        """Prepare data for analysis by renaming columns and calculating metrics."""
        if self.municipalities is None:
            self.logger.error("No data loaded. Call load_municipal_data first.")
            return None

        # Rename our 2005 columns for clarity
        self.municipalities = self.municipalities.rename(
            columns={
                "KV2005___S": "KV2005_Stemmeberettigede",  # Eligible voters
                "KV2005___A": "KV2005_Afgivne",  # Actual votes
                "VtrTrnt": "VoterTurnout2005",  # Turnout as ratio
                "navn_x": "MunicipalityName",
                "navn_y": "MunicipalityName2",
            }
        )

        # Convert turnout to percentage for analysis
        self.municipalities["VoterTurnout2005_pct"] = (
            self.municipalities["VoterTurnout2005"] * 100
        )
        self.municipalities["VoterTurnout2009_pct"] = (
            self.municipalities["VoterTurnout2009"] * 100
        )

        # Calculate turnout change (in percentage points)
        self.municipalities["turnout_change"] = (
            self.municipalities["VoterTurnout2009_pct"]
            - self.municipalities["VoterTurnout2005_pct"]
        )

        self.logger.info("\nTurnout change statistics:")
        self.logger.info(self.municipalities["turnout_change"].describe())

        # Identify merged municipalities based on boundary changes
        self.municipalities["boundary_changed"] = self.municipalities["dist"] > 0
        self.municipalities["merged"] = self.municipalities["boundary_changed"]

        # Count boundary-changed vs unchanged municipalities
        boundary_changed_count = self.municipalities["boundary_changed"].sum()
        unchanged_count = len(self.municipalities) - boundary_changed_count
        self.logger.info(
            f"\nBoundary changes: {boundary_changed_count} changed, {unchanged_count} unchanged"
        )

        # Count merged vs unchanged using our definition
        merged_count = self.municipalities["merged"].sum()
        unchanged_count = len(self.municipalities) - merged_count
        self.logger.info(
            f"\nMunicipalities: {merged_count} merged, {unchanged_count} unchanged"
        )

        return self.municipalities
