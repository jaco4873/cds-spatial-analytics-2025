import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
from functools import lru_cache
from utils.data_mappings import MUNICIPALITY_ENCODING_FIXES


class DataLoader:
    def __init__(self, data_dir="data"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.election_data_dir = self.data_dir / "kmdvalg"
        self.geodata_dir = self.data_dir / "geodata"

    @lru_cache(maxsize=1)
    def load_2001_data(self):
        """Load 2001 election data with post-merger municipality mapping."""
        path = self.election_data_dir / "aggregated_kommune_data_2001.csv"
        self._check_file_exists(path)
        self.logger.info("Loading 2001 election data (cached)")
        return pd.read_csv(path, encoding="utf-8-sig")

    @lru_cache(maxsize=1)
    def load_2005_data(self):
        """Load 2005 election data."""
        path = self.election_data_dir / "kommune_data_2005.csv"
        self._check_file_exists(path)
        self.logger.info("Loading 2005 election data (cached)")
        return pd.read_csv(path, encoding="utf-8-sig")

    @lru_cache(maxsize=1)
    def load_2009_data(self):
        """Load 2009 election data."""
        path = self.election_data_dir / "kommune_data_2009.csv"
        self._check_file_exists(path)
        self.logger.info("Loading 2009 election data (cached)")
        return pd.read_csv(path, encoding="utf-8-sig")

    def load_all_election_data(self):
        """Load all election years."""
        return (self.load_2001_data(), self.load_2005_data(), self.load_2009_data())

    @lru_cache(maxsize=1)
    def load_geographical_data(self):
        """Load Danish municipality boundaries from shapefile."""
        shapefile_path = (
            self.geodata_dir
            / "geoBoundaries-DNK-ADM2-all"
            / "geoBoundaries-DNK-ADM2.shp"
        )
        self._check_file_exists(shapefile_path)

        try:
            gdf = gpd.read_file(str(shapefile_path), engine="pyogrio")
            self.logger.info(f"Loaded {len(gdf)} municipalities from shapefile")

            # Fix encoding issues with Danish characters
            if "shapeName" in gdf.columns:
                gdf["shapeName"] = gdf["shapeName"].apply(
                    lambda name: MUNICIPALITY_ENCODING_FIXES.get(name, name)
                )
                gdf["name_lower"] = gdf["shapeName"].str.lower()

            return gdf

        except Exception as e:
            self.logger.error(f"Failed to load shapefile data: {e}")
            raise

    def _check_file_exists(self, file_path):
        """Verify file exists with helpful error message."""
        if not file_path.exists():
            raise FileNotFoundError(f"Required data file not found: {file_path}")
