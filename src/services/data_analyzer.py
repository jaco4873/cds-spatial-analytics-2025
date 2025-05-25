import logging
import pandas as pd
import geopandas as gpd
from typing import Any
from functools import lru_cache

from services.data_loader import DataLoader
from services.data_merger import DataMerger


class DataAnalyzer:
    """Analyzes municipal election data to assess the impact of the 2007 Danish municipal reform."""

    def __init__(self, data_loader: DataLoader, data_merger: DataMerger):
        """Initialize the data analyzer with a data loader and merger.

        Args:
            data_loader: An instance of DataLoader for retrieving the necessary data
            data_merger: An instance of DataMerger for merging spatial and election data
        """
        self.logger = logging.getLogger(__name__)
        self.data_loader = data_loader
        self.data_merger = data_merger
        self.municipality_gdf: gpd.GeoDataFrame | None = None
        self._cached_spatial_data: (
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame] | None
        ) = None

    def _load_geographical_data(self) -> None:
        """Load geographical data if not already loaded."""
        if self.municipality_gdf is None:
            self.municipality_gdf = self.data_loader.load_geographical_data()

    def _calculate_total_councilors(self, election_df: pd.DataFrame) -> int:
        """Calculate total councilors from election data.

        Args:
            election_df: DataFrame containing election data with counselor_count column

        Returns:
            Total number of councilors
        """
        return election_df["counselor_count"].sum()

    def analyze_voter_turnout(self) -> dict[str, Any]:
        """Analyze voter turnout before and after the reform.

        Returns:
            Dict with analysis results including average turnout and changes
        """
        # Load specific years needed for this analysis
        election_2005_df = self.data_loader.load_2005_data()
        election_2009_df = self.data_loader.load_2009_data()

        # Calculate overall turnout
        overall_turnout_2005 = (
            election_2005_df["optalte_stemmer"].sum()
            / election_2005_df["stemmeberettigede"].sum()
            * 100
        )
        overall_turnout_2009 = (
            election_2009_df["optalte_stemmer"].sum()
            / election_2009_df["stemmeberettigede"].sum()
            * 100
        )

        # Calculate average municipal turnout
        avg_turnout_2005 = election_2005_df["stemmeprocent"].mean()
        avg_turnout_2009 = election_2009_df["stemmeprocent"].mean()

        # Calculate median municipal turnout
        median_turnout_2005 = election_2005_df["stemmeprocent"].median()
        median_turnout_2009 = election_2009_df["stemmeprocent"].median()

        # Calculate variance and standard deviation
        variance_2005 = election_2005_df["stemmeprocent"].var()
        variance_2009 = election_2009_df["stemmeprocent"].var()

        # Identify top 5 and bottom 5 municipalities by turnout
        top5_2005 = election_2005_df.nlargest(5, "stemmeprocent")[
            ["name", "stemmeprocent"]
        ]
        bottom5_2005 = election_2005_df.nsmallest(5, "stemmeprocent")[
            ["name", "stemmeprocent"]
        ]

        top5_2009 = election_2009_df.nlargest(5, "stemmeprocent")[
            ["name", "stemmeprocent"]
        ]
        bottom5_2009 = election_2009_df.nsmallest(5, "stemmeprocent")[
            ["name", "stemmeprocent"]
        ]

        # Identify municipalities present in both datasets (unchanged)
        common_municipalities = set(election_2005_df["name"]).intersection(
            set(election_2009_df["name"])
        )

        # Create a subset for unchanged municipalities
        unchanged_2005 = election_2005_df[
            election_2005_df["name"].isin(common_municipalities)
        ]
        unchanged_2009 = election_2009_df[
            election_2009_df["name"].isin(common_municipalities)
        ]

        # Calculate average turnout change for unchanged municipalities
        unchanged_avg_2005 = unchanged_2005["stemmeprocent"].mean()
        unchanged_avg_2009 = unchanged_2009["stemmeprocent"].mean()
        unchanged_turnout_change = unchanged_avg_2009 - unchanged_avg_2005

        # Merge the datasets on municipality name for direct comparison
        merged_df = pd.merge(
            unchanged_2005[["name", "stemmeprocent"]],
            unchanged_2009[["name", "stemmeprocent"]],
            on="name",
            suffixes=("_2005", "_2009"),
        )

        # Calculate turnout change for each unchanged municipality
        merged_df["turnout_change"] = (
            merged_df["stemmeprocent_2009"] - merged_df["stemmeprocent_2005"]
        )

        # Count municipalities with increased/decreased turnout
        increased_turnout = len(merged_df[merged_df["turnout_change"] > 0])
        decreased_turnout = len(merged_df[merged_df["turnout_change"] < 0])
        unchanged_turnout = len(merged_df[merged_df["turnout_change"] == 0])

        # Return results as a dictionary
        return {
            "overall_turnout": {
                "2005": overall_turnout_2005,
                "2009": overall_turnout_2009,
                "change": overall_turnout_2009 - overall_turnout_2005,
            },
            "average_turnout": {
                "2005": avg_turnout_2005,
                "2009": avg_turnout_2009,
                "change": avg_turnout_2009 - avg_turnout_2005,
            },
            "median_turnout": {
                "2005": median_turnout_2005,
                "2009": median_turnout_2009,
                "change": median_turnout_2009 - median_turnout_2005,
            },
            "turnout_variance": {
                "2005": variance_2005,
                "2009": variance_2009,
                "change": variance_2009 - variance_2005,
            },
            "top5_municipalities": {
                "2005": top5_2005.to_dict("records"),
                "2009": top5_2009.to_dict("records"),
            },
            "bottom5_municipalities": {
                "2005": bottom5_2005.to_dict("records"),
                "2009": bottom5_2009.to_dict("records"),
            },
            "unchanged_municipalities": {
                "count": len(common_municipalities),
                "avg_turnout_2005": unchanged_avg_2005,
                "avg_turnout_2009": unchanged_avg_2009,
                "avg_change": unchanged_turnout_change,
                "increased_turnout": increased_turnout,
                "decreased_turnout": decreased_turnout,
                "unchanged_turnout": unchanged_turnout,
                "detail": merged_df.to_dict("records"),
            },
        }

    def analyze_representation_density(self) -> dict[str, Any]:
        """Analyze representation density (citizens per councilor) before and after the reform.
        Uses 2001 data for pre-reform and 2009 for post-reform.

        Returns:
            Dict with analysis results including average representation density and changes
        """
        # Load the years needed for this analysis
        election_2001_df = self.data_loader.load_2001_data()
        election_2009_df = self.data_loader.load_2009_data()

        # Calculate total councilors
        total_councilors_2001 = self._calculate_total_councilors(election_2001_df)
        total_councilors_2009 = self._calculate_total_councilors(election_2009_df)

        # Get total population (using eligible voters as a proxy)
        total_population_2001 = election_2001_df["stemmeberettigede"].sum()
        total_population_2009 = election_2009_df["stemmeberettigede"].sum()

        # Calculate overall representation density (citizens per councilor)
        density_2001 = total_population_2001 / total_councilors_2001
        density_2009 = total_population_2009 / total_councilors_2009

        # Calculate density for each municipality
        spatial_2001 = self.calculate_representation_density(election_2001_df, 2001)
        spatial_2009 = self.calculate_representation_density(election_2009_df, 2009)

        # Calculate average municipal representation density
        avg_density_2001 = spatial_2001["density"].mean()
        avg_density_2009 = spatial_2009["density"].mean()

        # Return results with 2001 data instead of 2005
        return {
            "total_councilors": {
                "2001": total_councilors_2001,
                "2009": total_councilors_2009,
                "change": total_councilors_2009 - total_councilors_2001,
                "percent_change": (total_councilors_2009 - total_councilors_2001)
                / total_councilors_2001
                * 100,
            },
            "overall_density": {
                "2001": density_2001,
                "2009": density_2009,
                "change": density_2009 - density_2001,
                "percent_change": (density_2009 - density_2001) / density_2001 * 100,
            },
            "avg_municipal_density": {
                "2001": avg_density_2001,
                "2009": avg_density_2009,
                "change": avg_density_2009 - avg_density_2001,
                "percent_change": (avg_density_2009 - avg_density_2001)
                / avg_density_2001
                * 100,
            },
            "municipalities_count": {
                "2001": len(election_2001_df),
                "2009": len(election_2009_df),
                "change": len(election_2009_df) - len(election_2001_df),
                "percent_change": (len(election_2009_df) - len(election_2001_df))
                / len(election_2001_df)
                * 100,
            },
        }

    def prepare_turnout_change_data(self) -> gpd.GeoDataFrame:
        """Prepare data for turnout change visualization between 2005 and 2009.

        Returns:
            GeoDataFrame with turnout change data for visualization
        """
        # Get spatial data using data_merger
        _, spatial_2005, spatial_2009 = self.get_spatial_data()

        # For municipalities that exist in both datasets, calculate turnout change
        # Use shapeName for joining as it's more reliable than the fuzzy matched names
        common_names = set(spatial_2005["shapeName"]).intersection(
            set(spatial_2009["shapeName"])
        )

        # Filter to common municipalities
        spatial_2005_common = spatial_2005[spatial_2005["shapeName"].isin(common_names)]
        spatial_2009_common = spatial_2009[spatial_2009["shapeName"].isin(common_names)]

        # Check that we have matched data for these municipalities
        spatial_2005_common = spatial_2005_common[
            spatial_2005_common["stemmeprocent"].notna()
        ]
        spatial_2009_common = spatial_2009_common[
            spatial_2009_common["stemmeprocent"].notna()
        ]

        # Merge data for comparison
        comparison_df = pd.merge(
            spatial_2005_common[["shapeName", "stemmeprocent", "geometry"]],
            spatial_2009_common[["shapeName", "stemmeprocent"]],
            on="shapeName",
            suffixes=("_2005", "_2009"),
        )

        # Calculate turnout change
        comparison_df["turnout_change"] = (
            comparison_df["stemmeprocent_2009"] - comparison_df["stemmeprocent_2005"]
        )

        # Convert to GeoDataFrame
        comparison_gdf = gpd.GeoDataFrame(
            comparison_df, geometry="geometry", crs=spatial_2005.crs
        )

        return comparison_gdf

    def get_spatial_data(
        self,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Get merged spatial data for all years.

        Returns:
            Tuple of GeoDataFrames for 2001, 2005, and 2009 with merged spatial and election data
        """
        # Load geographic data if not already loaded
        self._load_geographical_data()

        # Load election data for all years
        election_2001_df = self.data_loader.load_2001_data()
        election_2005_df = self.data_loader.load_2005_data()
        election_2009_df = self.data_loader.load_2009_data()

        # Check if we've already computed this result
        if self._cached_spatial_data is None:
            self._cached_spatial_data = self.data_merger.get_spatial_data(
                self.municipality_gdf,
                election_2001_df,
                election_2005_df,
                election_2009_df,
            )

        return self._cached_spatial_data

    @lru_cache(maxsize=1)
    def get_copenhagen_metropolitan_area(
        self,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Get spatial data for Copenhagen metropolitan area.

        Returns:
            Tuple of GeoDataFrames for 2001, 2005, and 2009 Copenhagen area
        """
        # Get all spatial data
        spatial_2001, spatial_2005, spatial_2009 = self.get_spatial_data()

        # Use data_merger to extract Copenhagen area
        metro_2001, metro_2005, metro_2009 = self.data_merger.extract_copenhagen_area(
            spatial_2001, spatial_2005, spatial_2009
        )

        # Return only 2005 and 2009 as that's what's used in the visualizer
        return metro_2001, metro_2005, metro_2009

    def get_quasi_experimental_comparison(self) -> dict[str, Any]:
        """Perform quasi-experimental comparison between merged and unchanged municipalities.

        Returns:
            Dict with comparison results showing differential effects by municipality type
        """
        # Get the spatial data that contains the merged flag
        _, spatial_2005, spatial_2009 = self.get_spatial_data()

        # Identify municipalities that were merged vs unchanged based on the merged flag
        merged_2005 = spatial_2005[spatial_2005["merged"] == True]  # noqa: E712
        unchanged_2005 = spatial_2005[spatial_2005["merged"] != True]  # noqa: E712

        # For unchanged municipalities, find their counterparts in 2009
        unchanged_names = set(unchanged_2005["name"]).intersection(
            set(spatial_2009["name"])
        )

        unchanged_2005 = unchanged_2005[unchanged_2005["name"].isin(unchanged_names)]
        unchanged_2009 = spatial_2009[spatial_2009["name"].isin(unchanged_names)]

        # Calculate average metrics
        turnout_unchanged_2005 = unchanged_2005["stemmeprocent"].mean()
        turnout_unchanged_2009 = unchanged_2009["stemmeprocent"].mean()
        turnout_merged_2005 = merged_2005["stemmeprocent"].mean()

        # Calculate average population sizes
        pop_unchanged_2005 = unchanged_2005["stemmeberettigede"].mean()
        pop_unchanged_2009 = unchanged_2009["stemmeberettigede"].mean()

        return {
            "unchanged_municipalities": {
                "count": len(unchanged_names),
                "turnout_2005": turnout_unchanged_2005,
                "turnout_2009": turnout_unchanged_2009,
                "turnout_change": turnout_unchanged_2009 - turnout_unchanged_2005,
                "population_2005": pop_unchanged_2005,
                "population_2009": pop_unchanged_2009,
                "population_change": pop_unchanged_2009 - pop_unchanged_2005,
            },
            "merged_municipalities": {
                "count": len(merged_2005),
                "turnout_2005": turnout_merged_2005,
                # We can't directly calculate 2009 values for merged municipalities
                # since they don't exist in the same form
            },
        }

    def generate_analysis_report(self) -> dict[str, Any]:
        """Generate a comprehensive analysis report combining all analyses.

        Returns:
            Dict with complete analysis results for all metrics
        """
        # Load the data for years needed for this analysis
        election_2001_df = self.data_loader.load_2001_data()
        election_2009_df = self.data_loader.load_2009_data()

        # Calculate total councilors
        total_councilors_2001 = self._calculate_total_councilors(election_2001_df)
        total_councilors_2009 = self._calculate_total_councilors(election_2009_df)

        turnout_analysis = self.analyze_voter_turnout()
        representation_analysis = self.analyze_representation_density()
        experimental_comparison = self.get_quasi_experimental_comparison()

        # Combine all analyses
        report = {
            "summary": {
                "total_councilors_before": total_councilors_2001,
                "total_councilors_after": total_councilors_2009,
                "councilors_reduction_percentage": (
                    total_councilors_2001 - total_councilors_2009
                )
                / total_councilors_2001
                * 100,
                "overall_turnout_before": turnout_analysis["overall_turnout"]["2005"],
                "overall_turnout_after": turnout_analysis["overall_turnout"]["2009"],
                "overall_turnout_change": turnout_analysis["overall_turnout"]["change"],
                "representation_density_before": representation_analysis[
                    "overall_density"
                ]["2001"],
                "representation_density_after": representation_analysis[
                    "overall_density"
                ]["2009"],
                "representation_density_increase_percentage": representation_analysis[
                    "overall_density"
                ]["percent_change"],
            },
            "voter_turnout": turnout_analysis,
            "representation_density": representation_analysis,
            "quasi_experimental_comparison": experimental_comparison,
        }

        return report

    def calculate_representation_density(
        self, spatial_data: gpd.GeoDataFrame, year: int
    ) -> gpd.GeoDataFrame:
        """Calculate representation density (citizens per councilor) for spatial dataset."""
        # Create a deep copy to avoid modifying the original
        result_data = spatial_data.copy(deep=True)

        if year == 2001:
            self.logger.info("Calculating 2001 density on post-merger level")

            # Create temporary columns only for this calculation
            temp_data = result_data.copy()

            # Check if we have post_merger_kommune to aggregate
            if "post_merger_kommune" in temp_data.columns:
                # Perform the aggregation on temporary data
                voter_counts = temp_data.groupby("post_merger_kommune")[
                    "stemmeberettigede"
                ].transform("sum")
                counselor_counts = temp_data.groupby("post_merger_kommune")[
                    "counselor_count"
                ].transform("sum")
            else:
                # No way to aggregate, fall back to non-aggregated values
                self.logger.warning(
                    "No post_merger_kommune found for aggregation, using raw values"
                )
                voter_counts = temp_data["stemmeberettigede"]
                counselor_counts = temp_data["counselor_count"]

            # Calculate density directly without modifying the original data structure
            result_data["density"] = voter_counts / counselor_counts

        elif year == 2009:
            result_data["density"] = (
                result_data["stemmeberettigede"] / result_data["counselor_count"]
            )

        self.logger.info(
            f"Calculated representation density for {len(result_data)} municipalities"
        )
        return result_data

    def calculate_turnout_change(
        self, data_2005: gpd.GeoDataFrame, data_2009: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Calculate turnout change between 2005 and 2009 for any pair of spatial datasets.

        Args:
            data_2005: GeoDataFrame with 2005 election data
            data_2009: GeoDataFrame with 2009 election data

        Returns:
            GeoDataFrame with turnout_change column
        """
        # Find common municipalities between the datasets
        common_names = set(data_2005["shapeName"]).intersection(
            set(data_2009["shapeName"])
        )

        # Filter to common municipalities with valid data
        data_2005_common = data_2005[data_2005["shapeName"].isin(common_names)]
        data_2009_common = data_2009[data_2009["shapeName"].isin(common_names)]

        data_2005_common = data_2005_common[data_2005_common["stemmeprocent"].notna()]
        data_2009_common = data_2009_common[data_2009_common["stemmeprocent"].notna()]

        # Merge data for comparison
        comparison = pd.merge(
            data_2005_common[["shapeName", "stemmeprocent", "geometry"]],
            data_2009_common[["shapeName", "stemmeprocent"]],
            on="shapeName",
            suffixes=("_2005", "_2009"),
        )

        # Calculate turnout change
        comparison["turnout_change"] = (
            comparison["stemmeprocent_2009"] - comparison["stemmeprocent_2005"]
        )

        # Convert to GeoDataFrame
        comparison_gdf = gpd.GeoDataFrame(
            comparison, geometry="geometry", crs=data_2005.crs
        )

        self.logger.info(
            f"Calculated turnout change for {len(comparison_gdf)} municipalities"
        )
        return comparison_gdf

    def prepare_copenhagen_turnout_change_data(self) -> gpd.GeoDataFrame:
        """Prepare turnout change data specifically for the Copenhagen metropolitan area."""
        # Get Copenhagen metropolitan data
        _, metro_2005, metro_2009 = self.get_copenhagen_metropolitan_area()

        # Ensure data is in correct projection
        metro_2005_web = (
            metro_2005.to_crs("EPSG:32632")
            if metro_2005.crs is None or metro_2005.crs.to_string() != "EPSG:32632"
            else metro_2005
        )
        metro_2009_web = (
            metro_2009.to_crs("EPSG:32632")
            if metro_2009.crs is None or metro_2009.crs.to_string() != "EPSG:32632"
            else metro_2009
        )

        # Use the common calculation method
        return self.calculate_turnout_change(metro_2005_web, metro_2009_web)
