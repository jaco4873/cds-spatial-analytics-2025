import logging
from pathlib import Path
import pandas as pd
import geopandas as gpd
from skrub import fuzzy_join


class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data")
        self.election_data_dir = self.data_dir / "kmdvalg"
        self.geodata_dir = self.data_dir / "geodata"
        self.geodata_dir.mkdir(exist_ok=True, parents=True)

    def load_election_data(self):
        """Load election data from combined CSV file and split by year.

        Returns:
            tuple: Two DataFrames containing 2005 and 2009 election data
        """
        combined_path = self.election_data_dir / "kommune_data_combined.csv"

        if not combined_path.exists():
            raise FileNotFoundError("Combined KMD Valg data file not found")

        # Explicitly specify utf-8-sig encoding when loading the CSV
        combined_df = pd.read_csv(combined_path, encoding="utf-8-sig")

        # Split by year
        election_2005_df = combined_df[combined_df["year"] == 2005]
        election_2009_df = combined_df[combined_df["year"] == 2009]

        self.logger.info(
            f"Loaded {len(election_2005_df)} records for 2005 and {len(election_2009_df)} records for 2009"
        )

        return election_2005_df, election_2009_df

    def load_geographical_data(self):
        """Load Danish municipality boundary data from local shapefile.

        Returns:
            GeoDataFrame: A geodataframe containing Danish municipality boundaries
        """
        # Define path to shapefile
        shapefile_path = str(
            self.geodata_dir
            / "geoBoundaries-DNK-ADM2-all"
            / "geoBoundaries-DNK-ADM2.shp"
        )

        try:
            self.logger.info(f"Loading municipality boundaries from: {shapefile_path}")
            # Load the shapefile using geopandas with pyogrio engine
            gdf = gpd.read_file(shapefile_path, engine="pyogrio")

            # Fix encoding issues in shapeName column
            if "shapeName" in gdf.columns:
                # Create a mapping for known encoding issues
                encoding_fixes = {
                    "Ã\x86rÃ¸": "Ærø",
                    "SÃ¸nderborg": "Sønderborg",
                    "NÃ¦stved": "Næstved",
                    "TÃ¸nder": "Tønder",
                    "FanÃ¸": "Fanø",
                    "KÃ¸ge": "Køge",
                    "SolrÃ¸d": "Solrød",
                    "SorÃ¸": "Sorø",
                    "DragÃ¸r": "Dragør",
                    "Nordfyn": "Nordfyns",
                    "IshÃ¸j": "Ishøj",
                    "VallensbÃ¦k": "Vallensbæk",
                    "TÃ¥rnby": "Tårnby",
                    "BrÃ¸ndby": "Brøndby",
                    "Copenhagen": "København",
                    "HÃ¸je-Taastrup": "Høje-Tåstrup",
                    "RÃ¸dovre": "Rødovre",
                    "HolbÃ¦k": "Holbæk",
                    "Lyngby-TaarbÃ¦k": "Lyngby-Tårbæk",
                    "FuresÃ¸": "Furesø",
                    "HÃ¸rsholm": "Hørsholm",
                    "HillerÃ¸d": "Hillerød",
                    "SamsÃ¸": "Samsø",
                    "HalsnÃ¦s": "Halsnæs",
                    "HelsingÃ¸r": "Helsingør",
                    "RingkÃ¸bing-Skjern": "Ringkøbing-Skjern",
                    "Aarhus": "Århus",
                    "MorsÃ¸": "Morsø",
                    "LÃ¦sÃ¸": "Læsø",
                    "BrÃ¸nderslev": "Brønderslev",
                    "HjÃ¸rring": "Hjørring",
                    "AllerÃ¸d": "Allerød",
                }

                # Apply the encoding fixes
                gdf["shapeName"] = gdf["shapeName"].apply(
                    lambda name: encoding_fixes.get(name, name)
                )

                # Add lowercase name column for case-insensitive matching
                gdf["name_lower"] = gdf["shapeName"].str.lower()

            self.logger.info(
                f"Successfully loaded {len(gdf)} municipalities from shapefile"
            )
            return gdf
        except Exception as e:
            self.logger.error(f"Failed to load shapefile data: {e}")
            raise FileNotFoundError(
                f"Could not load Danish municipality boundary data: {e}"
            )

    def prepare_spatial_data(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Prepare spatial data for visualization by merging election data with geographical boundaries.

        Returns:
            Tuple[GeoDataFrame, GeoDataFrame]: GeoDataFrames for 2005 and 2009 with electoral data
        """
        # Load data
        election_2005, election_2009 = self.load_election_data()
        geo_df = self.load_geographical_data()

        # Define mappings from 2005 names to 2009 names
        mapping_2005_to_2009 = {
            "Frederiksværk-Hundested": "Halsnæs",
            "Bogense": "Nordfyns",
            "Brønderslev-Dronninglund": "Brønderslev",
        }

        # Add canonical name column
        election_2005["canonical_name"] = (
            election_2005["name"]
            .map(mapping_2005_to_2009)
            .fillna(election_2005["name"])
        )
        election_2009["canonical_name"] = election_2009["name"]

        # Log the standardized names
        self.logger.info("Municipality name standardization applied:")
        for old_name, new_name in mapping_2005_to_2009.items():
            self.logger.info(f"  {old_name} (2005) → {new_name} (2009)")

        # Create normalized columns for matching
        geo_df["name_match"] = geo_df["shapeName"]
        election_2005["name_match"] = election_2005["canonical_name"]
        election_2009["name_match"] = election_2009["canonical_name"]

        # Improve fuzzy matching parameters
        self.logger.info("Performing fuzzy join for 2005 data")
        joined_2005 = fuzzy_join(
            left=geo_df,
            right=election_2005,
            left_on="name_match",
            right_on="name_match",
            suffix="_election",
            max_dist=0.3,  # Reduced threshold for stricter matching
            ref_dist="random_pairs",
            add_match_info=True,
        )

        self.logger.info("Performing fuzzy join for 2009 data")
        joined_2009 = fuzzy_join(
            left=geo_df,
            right=election_2009,
            left_on="name_match",
            right_on="name_match",
            suffix="_election",
            max_dist=0.3,  # Reduced threshold for stricter matching
            ref_dist="random_pairs",
            add_match_info=True,
        )

        # Convert back to GeoDataFrames
        merged_2005 = gpd.GeoDataFrame(joined_2005, geometry="geometry")
        merged_2009 = gpd.GeoDataFrame(joined_2009, geometry="geometry")

        # Log matching statistics
        match_rate_2005 = (
            merged_2005["name_election"].notna().sum() / len(merged_2005)
        ) * 100
        match_rate_2009 = (
            merged_2009["name_election"].notna().sum() / len(merged_2009)
        ) * 100

        self.logger.info(f"Match rate for 2005 data: {match_rate_2005:.2f}%")
        self.logger.info(f"Match rate for 2009 data: {match_rate_2009:.2f}%")

        # Log unmatched municipalities for manual inspection
        if match_rate_2005 < 100:
            unmatched = merged_2005[merged_2005["name_election"].isna()][
                "shapeName"
            ].tolist()
            self.logger.warning(f"Unmatched municipalities in 2005 data: {unmatched}")

        if match_rate_2009 < 100:
            unmatched = merged_2009[merged_2009["name_election"].isna()][
                "shapeName"
            ].tolist()
            self.logger.warning(f"Unmatched municipalities in 2009 data: {unmatched}")

        # Create a mapping to standardize display names (use 2009 names)
        display_name_mapping = {
            "Frederiksværk-Hundested": "Halsnæs",
            "Bogense": "Nordfyns",
            "Brønderslev-Dronninglund": "Brønderslev",
        }

        # Apply the display name standardization to both years for consistency
        for df in [merged_2005, merged_2009]:
            for old_name, new_name in display_name_mapping.items():
                df.loc[df["name_election"] == old_name, "name_election"] = new_name

        self.logger.info(
            "Standardized municipality names to use 2009 naming conventions"
        )

        return merged_2005, merged_2009

    def get_copenhagen_metropolitan_area(
        self,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Extract municipalities in the Copenhagen metropolitan area from prepared spatial data.

        Returns:
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: GeoDataFrames for 2005 and 2009 containing
            only municipalities in the Copenhagen metropolitan area
        """
        # Get the full spatial data
        merged_2005, merged_2009 = self.prepare_spatial_data()

        # Define municipalities in the Copenhagen metropolitan area
        copenhagen_municipalities = [
            "København",
            "Frederiksberg",
            "Dragør",
            "Tårnby",
            "Albertslund",
            "Ballerup",
            "Brøndby",
            "Gentofte",
            "Gladsaxe",
            "Glostrup",
            "Herlev",
            "Hvidovre",
            "Høje-Tåstrup",
            "Ishøj",
            "Lyngby-Tårbæk",
            "Rødovre",
            "Vallensbæk",
            "Furesø",
            "Allerød",
            "Rudersdal",
            "Egedal",
            "Hørsholm",
        ]

        self.logger.info(
            f"Filtering for {len(copenhagen_municipalities)} municipalities in Copenhagen metropolitan area"
        )

        # Filter the data to only include Copenhagen metropolitan area
        metro_2005 = merged_2005[
            merged_2005["name_election"].isin(copenhagen_municipalities)
        ]
        metro_2009 = merged_2009[
            merged_2009["name_election"].isin(copenhagen_municipalities)
        ]

        # Log the number of municipalities found
        self.logger.info(
            f"Found {len(metro_2005)} Copenhagen metro municipalities in 2005 data"
        )
        self.logger.info(
            f"Found {len(metro_2009)} Copenhagen metro municipalities in 2009 data"
        )

        # Log any missing municipalities
        if len(metro_2005) < len(copenhagen_municipalities):
            found = set(metro_2005["name_election"].dropna().unique())
            missing = set(copenhagen_municipalities) - found
            self.logger.warning(f"Missing municipalities in 2005 data: {missing}")

        if len(metro_2009) < len(copenhagen_municipalities):
            found = set(metro_2009["name_election"].dropna().unique())
            missing = set(copenhagen_municipalities) - found
            self.logger.warning(f"Missing municipalities in 2009 data: {missing}")

        return metro_2005, metro_2009
