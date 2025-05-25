# src/services/data_merger.py
import logging
import pandas as pd
import geopandas as gpd
from functools import lru_cache
from skrub import fuzzy_join
from utils.data_mappings import DISPLAY_NAME_MAPPING
from utils.copenhagen_municipalities import COPENHAGEN_MUNICIPALITIES


class DataMerger:
    """Handles merging of election data with geographical data and extraction of specific regions."""

    def __init__(self):
        """Initialize the data merger with a logger."""
        self.logger = logging.getLogger(__name__)
        self._cached_spatial_data: (
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame] | None
        ) = None

    def get_spatial_data(
        self,
        geo_df: gpd.GeoDataFrame,
        election_2001: pd.DataFrame,
        election_2005: pd.DataFrame,
        election_2009: pd.DataFrame,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Merge election data with geographical boundaries to create spatial datasets.

        Args:
            geo_df: GeoDataFrame with municipality boundaries
            election_2001: DataFrame with 2001 election data
            election_2005: DataFrame with 2005 election data
            election_2009: DataFrame with 2009 election data

        Returns:
            Tuple of GeoDataFrames for 2001, 2005, and 2009 with merged spatial and election data
        """
        self.logger.info("Merging election data with spatial boundaries")
        return self.merge_election_with_spatial(
            geo_df, election_2001, election_2005, election_2009
        )

    def merge_election_with_spatial(
        self,
        geo_df: gpd.GeoDataFrame,
        election_2001: pd.DataFrame,
        election_2005: pd.DataFrame,
        election_2009: pd.DataFrame,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Join election data with spatial boundaries.

        Args:
            geo_df: GeoDataFrame with municipality boundaries
            election_2001: DataFrame with 2001 election data
            election_2005: DataFrame with 2005 election data
            election_2009: DataFrame with 2009 election data

        Returns:
            Tuple of GeoDataFrames for 2001, 2005, and 2009 with merged spatial and election data
        """
        # Add canonical names for consistent joining
        election_2001["canonical_name"] = (
            election_2001["post_merger_kommune"]
            .map(DISPLAY_NAME_MAPPING)
            .fillna(election_2001["post_merger_kommune"])
        )

        election_2005["canonical_name"] = (
            election_2005["name"]
            .map(DISPLAY_NAME_MAPPING)
            .fillna(election_2005["name"])
        )

        election_2009["canonical_name"] = (
            election_2009["name"]
            .map(DISPLAY_NAME_MAPPING)
            .fillna(election_2009["name"])
        )

        # Perform fuzzy joins
        merged_2001 = self._perform_fuzzy_join(geo_df, election_2001, "canonical_name")
        merged_2005 = self._perform_fuzzy_join(geo_df, election_2005, "canonical_name")
        merged_2009 = self._perform_fuzzy_join(geo_df, election_2009, "canonical_name")

        # Log match rates
        self._log_match_rates(merged_2001, merged_2005, merged_2009)

        # Apply display name standardization
        for df in [merged_2001, merged_2005]:
            if "name" in df.columns:
                for old_name, new_name in DISPLAY_NAME_MAPPING.items():
                    df.loc[df["name"] == old_name, "name"] = new_name

        return merged_2001, merged_2005, merged_2009

    def _perform_fuzzy_join(
        self, geo_df: gpd.GeoDataFrame, election_df: pd.DataFrame, match_column: str
    ) -> gpd.GeoDataFrame:
        """Perform fuzzy join between geographical and election data.

        Args:
            geo_df: GeoDataFrame with municipality boundaries
            election_df: DataFrame with election data
            match_column: Column in election_df to match with shapeName in geo_df

        Returns:
            GeoDataFrame with merged data
        """
        self.logger.info(f"Performing fuzzy join on {match_column}")

        joined = fuzzy_join(
            left=geo_df,
            right=election_df,
            left_on="shapeName",
            right_on=match_column,
            suffix="",
            max_dist=0.3,
            ref_dist="random_pairs",
            add_match_info=True,
        )

        return gpd.GeoDataFrame(joined, geometry="geometry")

    def _log_match_rates(
        self,
        merged_2001: gpd.GeoDataFrame,
        merged_2005: gpd.GeoDataFrame,
        merged_2009: gpd.GeoDataFrame,
    ) -> None:
        """Log match rates and unmatched municipalities.

        Args:
            merged_2001: Merged GeoDataFrame for 2001
            merged_2005: Merged GeoDataFrame for 2005
            merged_2009: Merged GeoDataFrame for 2009
        """
        # Calculate and log match rates
        match_rate_2001 = (
            merged_2001["canonical_name"].notna().sum() / len(merged_2001)
        ) * 100
        match_rate_2005 = (
            merged_2005["canonical_name"].notna().sum() / len(merged_2005)
        ) * 100
        match_rate_2009 = (merged_2009["name"].notna().sum() / len(merged_2009)) * 100

        self.logger.info(f"Match rate for 2001 data: {match_rate_2001:.2f}%")
        self.logger.info(f"Match rate for 2005 data: {match_rate_2005:.2f}%")
        self.logger.info(f"Match rate for 2009 data: {match_rate_2009:.2f}%")

        # Log unmatched municipalities if any
        if match_rate_2001 < 100:
            self._log_unmatched(merged_2001, "canonical_name", "2001")
        if match_rate_2005 < 100:
            self._log_unmatched(merged_2005, "canonical_name", "2005")
        if match_rate_2009 < 100:
            self._log_unmatched(merged_2009, "name", "2009")

    def _log_unmatched(
        self, merged_df: gpd.GeoDataFrame, column: str, year: str
    ) -> None:
        """Log unmatched municipalities for a specific year.

        Args:
            merged_df: Merged GeoDataFrame
            column: Column to check for NaN values
            year: Year for logging
        """
        unmatched = merged_df[merged_df[column].isna()]["shapeName"].tolist()
        self.logger.warning(f"Unmatched municipalities in {year} data: {unmatched}")

    def extract_copenhagen_area(
        self,
        merged_2001: gpd.GeoDataFrame,
        merged_2005: gpd.GeoDataFrame,
        merged_2009: gpd.GeoDataFrame,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Extract Copenhagen metropolitan area data for all years.

        While geographical boundaries are identical, each dataset contains
        different election statistics for the respective year.

        Args:
            merged_2001: Merged GeoDataFrame for 2001
            merged_2005: Merged GeoDataFrame for 2005
            merged_2009: Merged GeoDataFrame for 2009

        Returns:
            Tuple of GeoDataFrames for Copenhagen area in 2001, 2005, and 2009
        """
        self.logger.info(
            f"Extracting Copenhagen area ({len(COPENHAGEN_MUNICIPALITIES)} municipalities) for all years"
        )

        # Filter using canonical_name since it's consistently present in all datasets
        metro_2001 = merged_2001[
            merged_2001["canonical_name"].isin(COPENHAGEN_MUNICIPALITIES)
        ].copy()
        metro_2005 = merged_2005[
            merged_2005["canonical_name"].isin(COPENHAGEN_MUNICIPALITIES)
        ].copy()
        metro_2009 = merged_2009[
            merged_2009["canonical_name"].isin(COPENHAGEN_MUNICIPALITIES)
        ].copy()

        # Log results
        self.logger.info(
            f"Found {len(metro_2001)} Copenhagen municipalities in 2001 aggregated data"
        )
        self.logger.info(
            f"Found {len(metro_2005)} Copenhagen municipalities in 2005 data"
        )
        self.logger.info(
            f"Found {len(metro_2009)} Copenhagen municipalities in 2009 data"
        )

        # Log missing municipalities if any
        for year, metro_data in [
            ("2001", metro_2001),
            ("2005", metro_2005),
            ("2009", metro_2009),
        ]:
            if len(metro_data) < len(COPENHAGEN_MUNICIPALITIES):
                found = set(metro_data["canonical_name"].dropna().unique())
                missing = set(COPENHAGEN_MUNICIPALITIES) - found
                self.logger.warning(f"Missing municipalities in {year} data: {missing}")

        return metro_2001, metro_2005, metro_2009
