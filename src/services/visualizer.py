from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from services.data_analyzer import DataAnalyzer
import geopandas as gpd
import random


class Visualizer:
    """Creates maps and charts to visualize municipal election data analysis."""

    def __init__(self, data_analyzer: DataAnalyzer):
        """Initialize the visualizer with a data analyzer.

        Args:
            data_analyzer: An instance of DataAnalyzer containing the analyzed data
        """
        self.logger = logging.getLogger(__name__)
        self.data_analyzer = data_analyzer
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.tile_attribution = ""
        self.cmap = plt.cm.viridis
        # Set up the visual style for publication quality
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("paper", font_scale=1.2)

    def add_compass(self, ax, size=0.25, position=(0.10, 0.875)):
        """Add a compass rose to the map using the external compass image.

        Args:
            ax: The matplotlib axis to add the compass to
            position: Tuple (x, y) position in axis coordinates (default: top right)
            size: Size of the compass (as fraction of axis size)
        """
        # Load the compass image
        compass_path = Path(__file__).parent.parent / "assets" / "compass.png"
        compass_img = mpimg.imread(compass_path)

        # Create an OffsetImage with the desired size
        imagebox = OffsetImage(compass_img, zoom=size)  # Adjust zoom factor as needed

        # Create an AnnotationBbox to place the image
        ab = AnnotationBbox(
            imagebox,
            xy=position,
            xycoords="axes fraction",
            frameon=False,
            pad=0,
            zorder=10,
        )

        # Add the image to the plot
        ax.add_artist(ab)

    def add_map_elements(self, ax, title, year=None):
        """Add common map elements like compass, scale bar, and attribution.

        Args:
            ax: The matplotlib axis
            title: Map title
            year: Optional year for title formatting
        """
        # Remove axis elements for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Add a proper scale bar with units in meters (UTM uses meters)
        scalebar = ScaleBar(
            dx=1.0,
            units="m",  # Explicitly set units to meters
            location="lower right",
            frameon=False,
            border_pad=0.5,
        )
        ax.add_artist(scalebar)

        # Add proper compass in top right corner
        self.add_compass(ax)

        # Format title with year if provided
        full_title = f"{title} ({year})" if year else title

        # Add title directly to the axes for minimal gap
        ax.set_title(full_title, fontsize=18, fontweight="bold", pad=2)
        # Add attribution text without a border
        attr_text = f"Source: KMD Valg data.    {self.tile_attribution} "
        ax.text(
            0.5,
            0.01,
            attr_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    def save_map(self, name, year=None):
        """Save the current map in PNG.

        Args:
            name: Base name for the file
            year: Optional year to append to filename
        """
        # Create filenames
        if year:
            base_name = f"{name}_{year}"
        else:
            base_name = f"{name}"

        png_path = self.output_dir / f"{base_name}.png"

        # Save in both formats
        plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        self.logger.info(f"Publication-ready {name} map saved to {png_path}")

    def prepare_spatial_data_for_mapping(self, spatial_data):
        """Prepare spatial data for mapping by ensuring correct projection.

        Args:
            spatial_data: GeoDataFrame to prepare

        Returns:
            GeoDataFrame in UTM Zone 32N projection (EPSG:25832)
        """
        # Ensure data is in UTM Zone 32N (EPSG:25832) - appropriate for Denmark
        if spatial_data.crs is None or spatial_data.crs.to_string() != "EPSG:25832":
            self.logger.info("Converting spatial data to EPSG:25832 (UTM Zone 32N)")
            spatial_data = spatial_data.to_crs("EPSG:25832")

        # Return data in UTM coordinates (better for scale accuracy)
        return spatial_data

    def add_basemap(self, ax):
        """Add a basemap to the given axis.

        Args:
            ax: Matplotlib axis with data in EPSG:25832 projection
        """
        try:
            # Specify the CRS when adding the basemap
            ctx.add_basemap(
                ax,
                source=ctx.providers.Esri.WorldGrayCanvas,
                zoom="auto",
                alpha=0.4,
                attribution=False,
                crs="EPSG:25832",  # Specify that our data is in UTM Zone 32N
            )

            self.tile_attribution = "Map tiles: © CartoDB"

        except Exception as e:
            self.logger.warning(f"Could not add basemap: {e}")
            raise e

    def create_turnout_map(self, year: int) -> None:
        """Create a choropleth map showing voter turnout by municipality for publication.

        Args:
            year: The election year (2005 or 2009)
        """
        # Prepare spatial data
        spatial_2005, spatial_2009 = self.data_analyzer.prepare_spatial_data()

        if year == 2005:
            spatial_data = spatial_2005
        elif year == 2009:
            spatial_data = spatial_2009
        else:
            raise ValueError("Year must be either 2005 or 2009")

        # Prepare data for mapping
        spatial_data_web = self.prepare_spatial_data_for_mapping(spatial_data)

        # Create the figure without constrained_layout for better control
        fig, ax = plt.subplots(1, 1, figsize=(10, 12), dpi=300)
        fig.subplots_adjust(top=0.99)  # Move subplot closer to top

        # Get data range for consistent coloring
        vmin = spatial_data_web["stemmeprocent_election"].min()
        vmax = spatial_data_web["stemmeprocent_election"].max()

        # Plot the data
        spatial_data_web.plot(
            column="stemmeprocent_election",
            ax=ax,
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            legend=True,
            legend_kwds={
                "label": "Voter Turnout (%)",
                "orientation": "horizontal",
                "shrink": 0.6,
                "pad": 0.01,
                "fraction": 0.046,
                "aspect": 30,
                "location": "bottom",
            },
            missing_kwds={"color": "lightgrey", "label": "No data"},
            edgecolor="white",
            linewidth=0.3,
            alpha=0.9,
        )

        # Add basemap
        self.add_basemap(ax)

        # Create the Copenhagen inset map with the same color scale
        copenhagen_image = self.create_copenhagen_map(
            year,
            "stemmeprocent_election",
            "stemmeprocent_election",
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # Add the Copenhagen inset directly from the image array
        self.add_copenhagen_inset_from_image(ax, copenhagen_image)

        # Add map elements
        self.add_map_elements(
            ax, title="Voter Turnout in Danish Municipalities", year=year
        )

        # Save the map
        self.save_map("turnout_map", year)

    def create_turnout_change_map(self) -> None:
        """Create a publication-quality choropleth map showing voter turnout change between 2005 and 2009."""
        # Get the prepared data from the analyzer
        comparison_gdf = self.data_analyzer.prepare_turnout_change_data()

        # Prepare data for mapping
        comparison_gdf_web = self.prepare_spatial_data_for_mapping(comparison_gdf)

        # Create the figure without constrained_layout for better control
        fig, ax = plt.subplots(1, 1, figsize=(10, 12), dpi=300)
        fig.subplots_adjust(top=0.99)  # Move subplot closer to top

        # Find the maximum absolute change for symmetric color scale
        max_change = max(
            abs(comparison_gdf["turnout_change"].min()),
            abs(comparison_gdf["turnout_change"].max()),
        )
        vmin = -max_change
        vmax = max_change

        # Plot the data with diverging colormap
        comparison_gdf_web.plot(
            column="turnout_change",
            ax=ax,
            cmap=self.cmap,
            legend=True,
            vmin=vmin,
            vmax=vmax,
            legend_kwds={
                "label": "Turnout Change (percentage points)",
                "orientation": "horizontal",
                "shrink": 0.6,
                "pad": 0.01,
                "fraction": 0.046,
                "aspect": 30,
                "location": "bottom",
            },
            missing_kwds={"color": "lightgrey", "label": "No data"},
            edgecolor="white",
            linewidth=0.3,
            alpha=0.9,
        )

        # Add basemap
        self.add_basemap(ax)

        # Create the Copenhagen inset with the same color scale
        copenhagen_image = self.create_copenhagen_map(
            2009,  # Use most recent year
            "turnout_change",
            "turnout_change",
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
        )

        self.add_copenhagen_inset_from_image(ax, copenhagen_image)

        # Add map elements
        self.add_map_elements(
            ax,
            title="Change in Voter Turnout between 2005 and 2009",
        )

        # Save the map
        self.save_map("turnout_change_map")

    def create_representation_density_map(self, year: int) -> None:
        """Create a publication-quality choropleth map showing representation density."""
        # Prepare spatial data
        spatial_2005, spatial_2009 = self.data_analyzer.prepare_spatial_data()

        if year == 2005:
            spatial_data = spatial_2005
        elif year == 2009:
            spatial_data = spatial_2009
        else:
            raise ValueError("Year must be either 2005 or 2009")

        # Calculate representation density using the refactored method
        spatial_data = self.data_analyzer.calculate_representation_density(
            spatial_data, year
        )

        # Prepare data for mapping
        spatial_data_web = self.prepare_spatial_data_for_mapping(spatial_data)

        # Create the figure without constrained_layout for better control
        fig, ax = plt.subplots(1, 1, figsize=(10, 12), dpi=300)
        fig.subplots_adjust(top=0.99)  # Move subplot closer to top

        # Create fixed bin boundaries for easier differentiation
        bounds = [0, 1000, 2000, 3000, 4000, 5000, 6000, 20000]
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, plt.cm.plasma.N)

        # Plot the data
        spatial_data_web.plot(
            column="density",
            ax=ax,
            cmap=self.cmap,
            norm=norm,
            legend=True,
            legend_kwds={
                "label": "Citizens per Councilor",
                "orientation": "horizontal",
                "shrink": 0.6,
                "pad": 0.01,
                "fraction": 0.046,
                "aspect": 30,
                "location": "bottom",
            },
            missing_kwds={"color": "lightgrey", "label": "No data"},
            edgecolor="white",
            linewidth=0.3,
            alpha=0.9,
        )
        # Add annotations for municipalities with values above 6000
        # Find the municipality with the highest density for special placement
        highest_density_row = spatial_data_web.loc[spatial_data_web["density"].idxmax()]
        highest_density_value = highest_density_row["density"]

        for _, row in spatial_data_web.iterrows():
            if row["density"] > 6000:
                # Get the centroid of the municipality
                centroid = row.geometry.centroid

                # Add offset to x-coordinate for Copenhagen
                x_offset = 0
                y_offset = 0
                if row["density"] == highest_density_value:
                    x_offset = 15000  # Offset in map units (meters for UTM)
                    y_offset = 10000

                # Add text annotation with the actual density value
                ax.text(
                    centroid.x + x_offset,
                    centroid.y + y_offset,
                    f"{int(row['density'])}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1),
                )

        # Add basemap
        self.add_basemap(ax)

        # Add Copenhagen inset with same normalization
        copenhagen_image = self.create_copenhagen_map(
            year,
            "density",
            "density",
            cmap=self.cmap,
            norm=norm,  # Pass the same norm object
        )

        self.add_copenhagen_inset_from_image(ax, copenhagen_image)

        # Add map elements
        self.add_map_elements(
            ax,
            title="Representation Density in Danish Municipalities",
            year=year,
        )

        # Save the map
        self.save_map("density_map", year)

    def create_turnout_comparison_chart(self) -> None:
        """Create a visually appealing bar chart comparing turnout before and after the reform."""
        # Instead of using turnout_analysis["overall_turnout"], calculate weighted averages for all municipalities:
        overall_2005 = (
            self.data_analyzer.election_2005_df["stemmeprocent"]
            * self.data_analyzer.election_2005_df["stemmeberettigede"]
        ).sum() / self.data_analyzer.election_2005_df["stemmeberettigede"].sum()

        overall_2009 = (
            self.data_analyzer.election_2009_df["stemmeprocent"]
            * self.data_analyzer.election_2009_df["stemmeberettigede"]
        ).sum() / self.data_analyzer.election_2009_df["stemmeberettigede"].sum()

        # Segregate municipalities by whether they were merged or not
        # First check if 'merged' column exists, otherwise use 'merged_election'
        merge_column = (
            "merged"
            if "merged" in self.data_analyzer.election_2005_df.columns
            else "merged_election"
        )

        merged_munis_2005 = self.data_analyzer.election_2005_df[
            self.data_analyzer.election_2005_df[merge_column] == True
        ]
        unmerged_munis_2005 = self.data_analyzer.election_2005_df[
            self.data_analyzer.election_2005_df[merge_column] == False
        ]

        # Calculate weighted average turnout for merged and unmerged municipalities in 2005
        merged_avg_2005 = (
            merged_munis_2005["stemmeprocent"] * merged_munis_2005["stemmeberettigede"]
        ).sum() / merged_munis_2005["stemmeberettigede"].sum()
        unmerged_avg_2005 = (
            unmerged_munis_2005["stemmeprocent"]
            * unmerged_munis_2005["stemmeberettigede"]
        ).sum() / unmerged_munis_2005["stemmeberettigede"].sum()

        # Find the unmerged municipalities in 2009 (same names as 2005)
        unmerged_names = unmerged_munis_2005["name"].tolist()
        unmerged_munis_2009 = self.data_analyzer.election_2009_df[
            self.data_analyzer.election_2009_df["name"].isin(unmerged_names)
        ]

        # Calculate weighted average turnout for unmerged municipalities in 2009
        unmerged_avg_2009 = (
            unmerged_munis_2009["stemmeprocent"]
            * unmerged_munis_2009["stemmeberettigede"]
        ).sum() / unmerged_munis_2009["stemmeberettigede"].sum()

        # All other municipalities in 2009 are merged
        merged_munis_2009 = self.data_analyzer.election_2009_df[
            ~self.data_analyzer.election_2009_df["name"].isin(unmerged_names)
        ]
        merged_avg_2009 = (
            merged_munis_2009["stemmeprocent"] * merged_munis_2009["stemmeberettigede"]
        ).sum() / merged_munis_2009["stemmeberettigede"].sum()

        # Set professional color palette
        pre_reform_color = "#2c75b3"  # Deeper blue
        post_reform_color = "#e68a2e"  # Muted orange

        # Create figure with a clean modern look
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("notebook", font_scale=1.3)

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("white")

        # Customize the grid for a cleaner look
        ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
        ax.set_axisbelow(True)  # Place gridlines behind bars

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#dddddd")
        ax.spines["bottom"].set_color("#dddddd")

        # Set up x-coordinates
        x = np.arange(3)
        width = 0.3  # Slightly narrower bars for a cleaner look

        # Create bars
        rects1 = ax.bar(
            x - width / 2,
            [overall_2005, merged_avg_2005, unmerged_avg_2005],
            width,
            label="2005 (Pre-Reform)",
            color=pre_reform_color,
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )
        rects2 = ax.bar(
            x + width / 2,
            [overall_2009, merged_avg_2009, unmerged_avg_2009],
            width,
            label="2009 (Post-Reform)",
            color=post_reform_color,
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )

        # Add labels and title
        ax.set_title(
            "Voter Turnout Comparison Before and After the 2007 Municipal Reform",
            fontsize=18,
            fontweight="bold",
            pad=20,
        )

        # Add a subtle subtitle with methodology note
        ax.text(
            0.5,
            0.97,
            "Weighted by municipality size (eligible voters)",
            transform=ax.transAxes,
            fontsize=11,
            ha="center",
            va="top",
            color="#666666",
            style="italic",
        )

        # Improve x-axis styling
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Overall", "Merged\nMunicipalities", "Unchanged\nMunicipalities"],
            fontsize=14,
        )
        ax.tick_params(axis="x", length=0, pad=10)

        # Improve y-axis
        ax.set_ylim(
            0, max([overall_2005, merged_avg_2005, unmerged_avg_2005]) * 1.15
        )  # Give room for labels
        ax.set_ylabel("Voter Turnout (%)", fontsize=14, labelpad=15)
        ax.tick_params(axis="y", labelsize=12)

        # Add value labels on the bars with improved styling
        def add_labels(rects, offset=0.5):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, offset),  # Slightly higher offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                    color="#333333",
                )

        add_labels(rects1, 3)
        add_labels(rects2, 3)

        # Enhanced legend
        leg = ax.legend(
            fontsize=13,
            frameon=True,
            framealpha=0.9,
            edgecolor="#dddddd",
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
        )

        # Add source note
        ax.text(
            0.5,
            -0.09,
            "Source: KMD Valg data. Analysis: Spatial Analytics, Aarhus University",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#666666",
        )

        # Add trend lines for each group to show the change more clearly
        for i, (y_2005, y_2009) in enumerate(
            [
                (overall_2005, overall_2009),
                (merged_avg_2005, merged_avg_2009),
                (unmerged_avg_2005, unmerged_avg_2009),
            ]
        ):
            # Add trend lines connecting the bars
            ax.plot(
                [x[i] - width / 2, x[i] + width / 2],
                [y_2005, y_2009],
                color="#444444",
                linestyle=":",
                linewidth=1.5,
                alpha=0.5,
                zorder=2,
            )

            # Add trend arrows indicating direction of change
            if y_2009 < y_2005:
                ax.annotate(
                    f"-{y_2005 - y_2009:.1f}",
                    xy=(x[i], (y_2005 + y_2009) / 2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#d32f2f",  # Red
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="white", ec="#d32f2f", alpha=0.7
                    ),
                )
            else:
                ax.annotate(
                    f"+{y_2009 - y_2005:.1f}",
                    xy=(x[i], (y_2005 + y_2009) / 2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#2e7d32",  # Green
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="white", ec="#2e7d32", alpha=0.7
                    ),
                )

        # Save the figure with high resolution
        output_path = self.output_dir / "turnout_comparison_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        self.logger.info(
            f"Publication-ready turnout comparison chart saved to {output_path}"
        )

    def create_merged_status_map(self) -> None:
        """Create a map showing municipalities by merger status with turnout change labels."""
        # Get the spatial data that contains the merged flag
        spatial_2005, spatial_2009 = self.data_analyzer.prepare_spatial_data()

        # Also get the turnout change data
        turnout_change_gdf = self.data_analyzer.prepare_turnout_change_data()

        # Prepare data for mapping - use 2005 data as base since it has merger info
        spatial_2005_web = self.prepare_spatial_data_for_mapping(spatial_2005)

        # Create the figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 12), dpi=300)
        fig.subplots_adjust(top=0.99)  # Move subplot closer to top

        # Create a binary colormap for merged status
        colors = [
            "#FDE725",
            "#3e4989",
        ]  # Viridis dark blue for unchanged, bright yellow for merged
        cmap = plt.matplotlib.colors.ListedColormap(colors)

        # Plot the data with binary coloring
        spatial_2005_web.plot(
            column="merged_election",
            ax=ax,
            cmap=cmap,
            categorical=True,
            legend=True,
            legend_kwds={
                "labels": ["Unchanged", "Merged"],
                "title": "Municipality Status",
                "frameon": True,
                "loc": "center",
                "bbox_to_anchor": (0.85, 0.4),
                "title_fontsize": 9,
                "prop": {"size": 8},
                "alignment": "left",
            },
            edgecolor="white",
            linewidth=0.5,
            alpha=0.7,
        )

        # Add basemap
        self.add_basemap(ax)

        # First merge the turnout change data with the base map
        merged_data = spatial_2005_web.merge(
            turnout_change_gdf[["shapeName", "turnout_change"]],
            left_on="shapeName",
            right_on="shapeName",
            how="left",
        )

        # Add centroid labels with turnout change values using the enhanced method
        self.add_centroid_labels(
            ax=ax,
            gdf=merged_data,
            value_column="turnout_change",
            fontsize=7,
            label_format="{:.1f}",
            positive_color="darkgreen",
            negative_color="darkred",
            show_sign=True,
            alpha=0.7,
            pad=1,
            jitter=False,
        )

        # Create and add Copenhagen inset with both merger status coloring and turnout change labels
        copenhagen_image = self.create_copenhagen_map(
            2005,  # Using 2005 data since we're showing merger status
            "merged_election",
            "merged_election",
            cmap=cmap,
            categorical=True,
            add_labels=True,
            label_column="turnout_change",
            label_kwargs={
                "fontsize": 18,
                "label_format": "{:.1f}",
                "positive_color": "darkgreen",
                "negative_color": "darkred",
                "show_sign": True,
            },
        )

        self.add_copenhagen_inset_from_image(
            ax, copenhagen_image, position=(0.8, 0.75), size=0.1
        )

        # Add map elements
        self.add_map_elements(
            ax,
            title="Municipalities by Merger Status with Voter Turnout Change (2005-2009)",
        )

        # Save the map
        self.save_map("merged_status_map")

    def create_copenhagen_map(
        self,
        year: int,
        column: str,
        title: str,
        cmap=None,
        vmin=None,
        vmax=None,
        norm=None,
        categorical=False,
        add_labels=False,
        label_column=None,
        label_kwargs=None,
    ) -> np.ndarray:
        """Create a standalone map of the Copenhagen metropolitan area with transparent background.

        Args:
            year: The election year
            column: Column name to visualize
            title: Title for legend/reference
            cmap: Colormap to use
            vmin: Minimum value for color normalization
            vmax: Maximum value for color normalization
            norm: Custom normalization
            categorical: Whether the data is categorical
            add_labels: Whether to add centroid labels
            label_column: Column to use for label values (defaults to same as 'column')
            label_kwargs: Additional kwargs to pass to add_centroid_labels

        Returns:
            Numpy array with the generated image
        """
        # Get the regular Copenhagen metro data first
        metro_2005, metro_2009 = (
            self.data_analyzer.data_loader.get_copenhagen_metropolitan_area()
        )
        metro_data = metro_2005 if year == 2005 else metro_2009
        metro_data_web = self.prepare_spatial_data_for_mapping(metro_data)

        # Special handling for different column types
        if column == "turnout_change" or (
            add_labels and label_column == "turnout_change"
        ):
            # Get the prepared turnout change data for Copenhagen
            copenhagen_turnout_change = (
                self.data_analyzer.prepare_copenhagen_turnout_change_data()
            )
            # If we're using turnout_change as the main column, use that data directly
            if column == "turnout_change":
                metro_data_web = copenhagen_turnout_change
            # Otherwise merge the turnout change data into our base map for labels
            elif add_labels and label_column == "turnout_change":
                # Merge the turnout change column into the base metro data
                metro_data_web = metro_data_web.merge(
                    copenhagen_turnout_change[["shapeName", "turnout_change"]],
                    on="shapeName",
                    how="left",
                )

        # Special handling for density column
        if column == "density":
            # Calculate representation density using the refactored method
            metro_data_web = self.data_analyzer.calculate_representation_density(
                metro_data_web, year
            )

        # Create the figure with transparent background
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300, facecolor="none")
        fig.patch.set_alpha(0)  # Transparent figure background

        # Use the provided colormap or default to the instance cmap
        map_cmap = cmap if cmap is not None else self.cmap

        # Plot the data with consistent color normalization
        plot_kwargs = {
            "column": column,
            "ax": ax,
            "cmap": map_cmap,
            "legend": False,
            "edgecolor": "white",
            "linewidth": 0.3,
            "alpha": 0.9,
        }

        # Add normalization parameters if provided
        if categorical:
            plot_kwargs["categorical"] = True
        elif norm is not None:
            plot_kwargs["norm"] = norm
        elif vmin is not None and vmax is not None:
            plot_kwargs["vmin"] = vmin
            plot_kwargs["vmax"] = vmax

        metro_data_web.plot(**plot_kwargs)

        # Add centroid labels if requested
        if add_labels:
            # Use the specified label column or fall back to the main column
            label_col = label_column if label_column is not None else column

            # Check if the label column exists in the data
            if label_col in metro_data_web.columns:
                # Default label kwargs
                default_kwargs = {
                    "fontsize": 5,  # Smaller font for inset
                    "pad": 0.3,  # Smaller padding for inset
                    "alpha": 0.85,  # Slightly more opaque for visibility
                }

                # Merge with provided kwargs if any
                if label_kwargs is not None:
                    default_kwargs.update(label_kwargs)

                # Add the labels
                self.add_centroid_labels(
                    ax=ax, gdf=metro_data_web, value_column=label_col, **default_kwargs
                )
            else:
                self.logger.warning(
                    f"Label column '{label_col}' not found in Copenhagen data. Skipping labels."
                )

        # Style the map with minimal styling to reduce borders
        ax.set_title("", fontsize=0)  # Remove title to keep inset clean
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_frame_on(False)  # Removes the frame
        ax.margins(0)  # Minimize margins

        # Apply tight layout to reduce whitespace
        fig.tight_layout(pad=0)

        # Convert to image array with transparency preserved
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        self.logger.info("Copenhagen map created in memory with consistent coloring")
        return image_array

    def add_copenhagen_inset_from_image(
        self, ax, image_array, position=(0.8, 0.7), size=0.1
    ):
        """Add a Copenhagen inset map using an in-memory image with transparency.

        Args:
            ax: The main plot axis
            image_array: Numpy array of the Copenhagen map image
            position: (x, y) position in axis coordinates (0-1 range)
            size: Size of the inset as fraction of the figure
        """
        try:
            # Create an OffsetImage with the desired size
            imagebox = OffsetImage(image_array, zoom=size)
            imagebox.image.axes = ax  # Connect the image to the axes

            # Create an AnnotationBbox with a visible square border but transparent background
            ab = AnnotationBbox(
                imagebox,
                xy=position,
                xycoords="axes fraction",
                frameon=True,  # Show frame
                pad=0.1,  # Small padding around the image
                bboxprops=dict(
                    edgecolor="black",  # Border color
                    linewidth=1,  # Border width
                    boxstyle="square",  # Square border
                    facecolor="none",  # Transparent face
                    alpha=1.0,  # Fully opaque border
                ),
                zorder=10,
            )

            # Add the image to the plot
            ax.add_artist(ab)
            self.logger.info("Added transparent Copenhagen inset with border")

        except Exception as e:
            self.logger.error(f"Failed to add Copenhagen inset: {e}")

    def add_centroid_labels(
        self,
        ax,
        gdf,
        value_column,
        fontsize=7,
        label_format="{:.1f}",
        positive_color="darkgreen",
        negative_color="darkred",
        show_sign=True,
        alpha=0.7,
        pad=1,
        categorical=False,
        category_colors=None,
        use_fixed_color=None,
        format_func=None,
        jitter=True,
        jitter_seed=42,
        jitter_max_attempts=100,
        jitter_radius=None,
    ):
        """Add centroid labels to municipalities on a map.

        Args:
            ax: Matplotlib axis to add labels to
            gdf: GeoDataFrame containing municipality geometries and values
            value_column: Column name containing the values to display
            fontsize: Font size for labels
            label_format: Format string for the label values
            positive_color: Color for positive values
            negative_color: Color for negative values
            show_sign: Whether to show + sign for positive values
            alpha: Transparency of label background
            pad: Padding around the label text
            categorical: Whether the value column contains categorical data
            category_colors: Dictionary mapping category values to colors
            use_fixed_color: If provided, use this color for all labels
            format_func: Custom function to format label text and determine color
            jitter: Whether to apply jittering to avoid label overlap
            jitter_seed: Random seed for reproducible jittering
            jitter_max_attempts: Maximum attempts to find non-overlapping position
            jitter_radius: Maximum jittering radius in map units (if None, auto-calculated)
        """
        self.logger.info(f"Adding centroid labels for {len(gdf)} municipalities")

        # Set random seed for reproducible jittering
        random.seed(jitter_seed)

        # Store text objects and their bounding boxes for collision detection
        placed_texts = []

        # Calculate a default jitter radius if not provided (as % of map width)
        if jitter and jitter_radius is None:
            bounds = gdf.total_bounds  # Get the bounds of the entire dataset
            map_width = bounds[2] - bounds[0]
            jitter_radius = map_width * 0.03  # 3% of map width as default radius

        # Add centroid labels with values
        for _, row in gdf.iterrows():
            if pd.notna(row[value_column]):
                # Get the centroid of the municipality
                centroid = row.geometry.centroid
                value = row[value_column]

                # Determine the label text and color based on the data type and provided options
                if format_func is not None:
                    # Use custom formatting function if provided
                    label, color = format_func(value)
                elif categorical:
                    # Handle categorical data
                    label = str(value)
                    if category_colors and value in category_colors:
                        color = category_colors[value]
                    else:
                        color = "gray"  # Default color for categories
                else:
                    # Handle numeric data with sign formatting
                    if value > 0 and show_sign:
                        label = f"+{label_format.format(value)}"
                        color = positive_color
                    else:
                        label = f"{label_format.format(value)}"
                        color = negative_color if value < 0 else positive_color

                # Override color if fixed color is provided
                if use_fixed_color:
                    color = use_fixed_color

                # Apply jittering to avoid overlaps if enabled
                x, y = centroid.x, centroid.y

                if jitter and placed_texts:
                    # Start with the centroid position
                    best_position = (x, y)
                    lowest_overlap = float("inf")

                    # Try different positions to minimize overlap
                    for attempt in range(jitter_max_attempts):
                        # Generate a random position within the jitter radius
                        angle = random.uniform(0, 2 * 3.14159)  # Random angle
                        distance = random.uniform(0, jitter_radius)  # Random distance
                        test_x = x + distance * np.cos(angle)
                        test_y = y + distance * np.sin(angle)

                        # Create a test text to check for overlap
                        test_text = ax.text(
                            test_x,
                            test_y,
                            label,
                            fontsize=fontsize,
                            ha="center",
                            va="center",
                        )

                        # Get the bounding box
                        test_bbox = test_text.get_window_extent(
                            renderer=ax.figure.canvas.get_renderer()
                        )

                        # Calculate overlap with existing texts
                        overlap = 0
                        for placed_text, _ in placed_texts:
                            placed_bbox = placed_text.get_window_extent(
                                renderer=ax.figure.canvas.get_renderer()
                            )
                            # Check if bounding boxes intersect
                            if test_bbox.overlaps(placed_bbox):
                                # Calculate the overlap area
                                overlap_width = min(test_bbox.x1, placed_bbox.x1) - max(
                                    test_bbox.x0, placed_bbox.x0
                                )
                                overlap_height = min(
                                    test_bbox.y1, placed_bbox.y1
                                ) - max(test_bbox.y0, placed_bbox.y0)
                                overlap += max(0, overlap_width) * max(
                                    0, overlap_height
                                )

                        # Remove the test text
                        test_text.remove()

                        # If this position has less overlap, use it
                        if overlap < lowest_overlap:
                            lowest_overlap = overlap
                            best_position = (test_x, test_y)

                        # If we found a position with no overlap, stop early
                        if lowest_overlap == 0:
                            break

                    # Use the best position found
                    x, y = best_position

                # Add text annotation with the value at the (possibly jittered) position
                text = ax.text(
                    x,
                    y,
                    label,
                    fontsize=fontsize,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        facecolor=color, alpha=alpha, pad=pad, boxstyle="round,pad=0.1"
                    ),
                )

                # Store the text and its properties for future collision detection
                if jitter:
                    placed_texts.append((text, (x, y)))

    def create_all_visualizations(self) -> None:
        """Create all visualizations at once."""
        # Make sure data is loaded
        if self.data_analyzer.election_2005_df is None:
            self.data_analyzer.load_data()

        # Generate all maps and charts
        self.create_turnout_map(2005)
        self.create_turnout_map(2009)
        self.create_turnout_change_map()
        self.create_representation_density_map(2005)
        self.create_representation_density_map(2009)
        self.create_turnout_comparison_chart()
        self.create_merged_status_map()

        self.logger.info("All visualizations created successfully")
