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
from adjustText import adjust_text


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
            GeoDataFrame in UTM Zone 32N projection (EPSG:32632)
        """
        # Ensure data is in UTM Zone 32N (EPSG:32632) - appropriate for Denmark
        if spatial_data.crs is None or spatial_data.crs.to_string() != "EPSG:32632":
            self.logger.info("Converting spatial data to EPSG:32632 (UTM Zone 32N)")
            spatial_data = spatial_data.to_crs("EPSG:32632")

        # Return data in UTM coordinates (better for scale accuracy)
        return spatial_data

    def add_basemap(self, ax):
        """Add a basemap to the given axis.

        Args:
            ax: Matplotlib axis with data in EPSG:32632 projection
        """
        try:
            # Specify the CRS when adding the basemap
            ctx.add_basemap(
                ax,
                source=ctx.providers.Esri.WorldGrayCanvas,
                zoom="auto",
                alpha=0.4,
                attribution=False,
                crs="EPSG:32632",  # Specify that our data is in UTM Zone 32N
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

        # Mask the Copenhagen area in the main map
        self.mask_copenhagen_area(ax, year)

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

        # Mask the Copenhagen area in the main map (use 2009 to match inset)
        self.mask_copenhagen_area(ax, 2009)

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

        # Get Copenhagen area for exclusion
        metro_2005, metro_2009 = (
            self.data_analyzer.data_loader.get_copenhagen_metropolitan_area()
        )
        metro_data = metro_2005 if year == 2005 else metro_2009
        metro_data_web = self.prepare_spatial_data_for_mapping(metro_data)

        # Dissolve to get a single polygon for easy containment checking
        if len(metro_data_web) > 1:
            metro_dissolved = metro_data_web.copy()
            metro_dissolved["dissolve_key"] = 1
            metro_mask = metro_dissolved.dissolve(by="dissolve_key")
        else:
            metro_mask = metro_data_web

        for _, row in spatial_data_web.iterrows():
            if row["density"] > 6000:
                # Get the centroid of the municipality
                centroid = row.geometry.centroid

                # Skip if this centroid falls within the Copenhagen area
                point_in_copenhagen = False
                for _, excl_row in metro_data_web.iterrows():
                    if excl_row.geometry.contains(centroid):
                        point_in_copenhagen = True
                        break

                if point_in_copenhagen:
                    continue

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

        # Mask the Copenhagen area in the main map
        self.mask_copenhagen_area(ax, year)

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
            self.data_analyzer.election_2005_df[merge_column] == True  # noqa: E712
        ]
        unmerged_munis_2005 = self.data_analyzer.election_2005_df[
            self.data_analyzer.election_2005_df[merge_column] == False  # noqa: E712
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
            0.99999,
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
            ["Overall", "Merged", "Not Merged"],
            fontsize=14,
        )
        ax.tick_params(axis="x", length=0, pad=10)

        # Improve y-axis
        ax.set_ylim(
            0, max([overall_2005, merged_avg_2005, unmerged_avg_2005]) * 1.15
        )  # Give room for labels
        ax.set_ylabel("Voter Turnout (%)", fontsize=14, labelpad=15)
        ax.tick_params(axis="y", labelsize=12)

        # Add value labels on the bars
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
            "#4174ae",
            "#d98e43",
        ]
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
                "bbox_to_anchor": (0.85, 0.35),
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

        # Mask the Copenhagen area in the main map
        self.mask_copenhagen_area(ax, 2005)

        # First merge the turnout change data with the base map
        merged_data = spatial_2005_web.merge(
            turnout_change_gdf[["shapeName", "turnout_change"]],
            left_on="shapeName",
            right_on="shapeName",
            how="left",
        )

        # Get Copenhagen metropolitan area for label exclusion
        metro_2005, _ = (
            self.data_analyzer.data_loader.get_copenhagen_metropolitan_area()
        )
        metro_data_web = self.prepare_spatial_data_for_mapping(metro_2005)

        # Add centroid labels with turnout change values using adjustText
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
            pad=0.5,
            adjust_text_args={
                "force_points": 0.2,
                "force_text": 0.8,
                "expand_points": (1.5, 1.5),
                "arrowprops": dict(
                    arrowstyle="->", color="gray", alpha=0.9, lw=0.8, zorder=15
                ),
            },
            exclude_geometries=metro_data_web,  # Exclude labels in the Copenhagen area
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
                "adjust_text_args": {
                    "expand_points": (2.0, 2.0),
                    "force_text": 0.8,
                    "arrowprops": dict(
                        arrowstyle="->", color="gray", alpha=0.9, lw=0.8, zorder=15
                    ),
                },
            },
        )

        self.add_copenhagen_inset_from_image(ax, copenhagen_image)

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

        # Special handling for high density values in the inset
        if column == "density":
            # Find municipalities with density > 6000 in the Copenhagen area
            high_density_munis = metro_data_web[metro_data_web["density"] > 6000]

            # Add special labels for high density municipalities
            for _, row in high_density_munis.iterrows():
                centroid = row.geometry.centroid

                # Add label with density value
                ax.text(
                    centroid.x + 4000,
                    centroid.y,
                    f"{int(row['density'])}",
                    fontsize=20,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1),
                    zorder=5,
                )

        # Add centroid labels if requested
        if add_labels:
            # Use the specified label column or fall back to the main column
            label_col = label_column if label_column is not None else column

            # Check if the label column exists in the data
            if label_col in metro_data_web.columns:
                # Default label kwargs with a larger fontsize for better visibility in the inset
                default_kwargs = {
                    "fontsize": 7,  # Larger font for inset visibility
                    "pad": 0.3,  # Smaller padding for inset
                    "alpha": 0.85,  # Slightly more opaque for visibility
                }

                # Merge with provided kwargs if any
                if label_kwargs is not None:
                    default_kwargs.update(label_kwargs)

                # Add the labels (no exclusion here - we want all labels in the inset)
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

        # Add a smaller scale bar suitable for the inset
        inset_scalebar = ScaleBar(
            dx=1.0,
            units="m",  # UTM uses meters
            location="lower right",
            frameon=False,
            scale_loc="top",
            font_properties={"size": 20},
            length_fraction=0.22,
            height_fraction=0.015,
        )
        ax.add_artist(inset_scalebar)

        # Apply tight layout to reduce whitespace
        fig.tight_layout(pad=0)

        # Convert to image array with transparency preserved
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        self.logger.info("Copenhagen map created in memory with consistent coloring")
        return image_array

    def add_copenhagen_inset_from_image(
        self, ax, image_array, position=(0.8, 0.75), size=0.1, add_connectors=True
    ):
        """Add a Copenhagen inset map using an in-memory image with transparency.

        Args:
            ax: The main plot axis
            image_array: Numpy array of the Copenhagen map image
            position: (x, y) position in axis coordinates (0-1 range)
            size: Size of the inset as fraction of the figure
            add_connectors: Whether to add connector lines to the masked area
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

            # Add connector lines if requested
            if add_connectors:
                # Get the Copenhagen metropolitan area for the current year
                metro_2005, metro_2009 = (
                    self.data_analyzer.data_loader.get_copenhagen_metropolitan_area()
                )
                # Get the most recently used metro data
                metro_data = (
                    metro_2005
                    if hasattr(ax, "year") and ax.year == 2005
                    else metro_2009
                )
                metro_data_web = self.prepare_spatial_data_for_mapping(metro_data)

                # Dissolve to get a single polygon if needed
                if len(metro_data_web) > 1:
                    metro_dissolved = metro_data_web.copy()
                    metro_dissolved["dissolve_key"] = 1
                    metro_mask = metro_dissolved.dissolve(by="dissolve_key")
                else:
                    metro_mask = metro_data_web

                # Add the connector lines with specific target points
                self.add_inset_connectors(
                    ax,
                    metro_mask.geometry,
                    source_points=[
                        (0.611, 0.43),
                        (0.625, 0.34),
                    ],
                    target_points=[
                        (0.633, 0.535),
                        (0.9675, 0.535),
                    ],
                )

            self.logger.info(
                "Added transparent Copenhagen inset with border and connectors"
            )

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
        adjust_text_args=None,
        exclude_geometries=None,
    ):
        """Add centroid labels to municipalities on a map with optimized placement.

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
            adjust_text_args: Dictionary of arguments to pass to adjust_text
            exclude_geometries: Optional GeoDataFrame with geometries to exclude from labeling
        """
        self.logger.info(f"Adding centroid labels for {len(gdf)} municipalities")

        # Store all text objects for adjustText
        texts = []

        # Add centroid labels with values
        for _, row in gdf.iterrows():
            if pd.notna(row[value_column]):
                # Get the centroid of the municipality
                centroid = row.geometry.centroid

                # Skip if this centroid falls within any of the exclusion geometries
                if exclude_geometries is not None:
                    # Check if the centroid is within any of the exclusion geometries
                    point_in_exclusion = False
                    for _, excl_row in exclude_geometries.iterrows():
                        if excl_row.geometry.contains(centroid):
                            point_in_exclusion = True
                            break

                    # Skip this label if the point is in an exclusion zone
                    if point_in_exclusion:
                        continue

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

                # Add text annotation at the centroid position
                text = ax.text(
                    centroid.x,
                    centroid.y,
                    label,
                    fontsize=fontsize,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        facecolor=color, alpha=alpha, pad=pad, boxstyle="round,pad=0.1"
                    ),
                    zorder=5,  # Set labels above map
                )

                # Collect text objects for adjustText
                texts.append(text)

        # Apply adjustText to optimize label placement
        if texts:
            self.logger.info("Applying adjustText to optimize label placement")

            # Default parameters for adjustText
            default_adjust_params = {
                "expand_points": (1.5, 1.5),
                "force_points": 0.1,  # Lower force from points
                "force_text": 0.5,  # Higher force between texts
                "lim": 500,  # More iterations for better placement
                "only_move": {
                    "points": "xy",
                    "text": "xy",
                },  # Allow movement of both points and text
                "arrowprops": dict(
                    arrowstyle="->", color="gray", alpha=1, lw=0.8
                ),  # Use arrow style
                "avoid_self": True,  # Avoid overlaps between texts
            }

            # Update with user-provided parameters if any
            if adjust_text_args:
                default_adjust_params.update(adjust_text_args)

            # Run the adjustText algorithm
            adjust_text(texts, **default_adjust_params)

    def mask_copenhagen_area(self, ax, year):
        """Mask out the Copenhagen area in the main map by overlaying a semi-transparent grey polygon.

        Args:
            ax: The matplotlib axis to add the mask to
            year: The election year (2005 or 2009)
        """
        self.logger.info("Masking Copenhagen metropolitan area in main map")

        # Store the year on the axis for reference when adding connectors
        ax.year = year

        # Get the Copenhagen metropolitan area data
        metro_2005, metro_2009 = (
            self.data_analyzer.data_loader.get_copenhagen_metropolitan_area()
        )
        metro_data = metro_2005 if year == 2005 else metro_2009

        # Ensure the data is in the correct projection
        metro_data_web = self.prepare_spatial_data_for_mapping(metro_data)

        # Dissolve all the municipalities into a single polygon if more than one
        if len(metro_data_web) > 1:
            # Create a copy to avoid modifying the original
            metro_dissolved = metro_data_web.copy()
            # Add a constant column to dissolve by
            metro_dissolved["dissolve_key"] = 1
            # Dissolve to get a single polygon
            metro_mask = metro_dissolved.dissolve(by="dissolve_key")
        else:
            metro_mask = metro_data_web

        # Plot the mask with a semi-transparent grey fill and no border
        metro_mask.plot(
            ax=ax,
            color="grey",
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,  # Place above the main map but below labels and other annotations
        )

        self.logger.info("Copenhagen area masked successfully")

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

    def add_inset_connectors(
        self,
        ax,
        mask_geometry,
        source_points=None,
        target_points=None,
        line_style="--",
        line_color="black",
        alpha=0.5,
        linewidth=0.8,
    ):
        """Add connector lines between the masked area and the inset map.

        Args:
            ax: The matplotlib axis
            mask_geometry: The geometry of the masked area
            source_points: List of (x,y) tuples for source points, or None to use mask bounds
            target_points: List of (x,y) tuples for target points on the inset
            line_style: Style of the connector lines
            line_color: Color of the connector lines
            alpha: Transparency of the lines
            linewidth: Width of the lines
        """
        # Get the bounding box of the mask geometry
        bounds = mask_geometry.bounds
        if len(bounds) == 0:
            return

        # For multi-geometry masks, use the first one's bounds
        if isinstance(bounds, pd.DataFrame):
            minx, miny, maxx, maxy = bounds.iloc[0]
        else:
            minx, miny, maxx, maxy = bounds

        # If source points not provided, use default points from the mask bounds
        if source_points is None:
            source_points = [(maxx, maxy), (maxx, miny)]  # top-right, bottom-right

        # If target points not provided, raise an error
        if target_points is None:
            raise ValueError("Target points must be provided for inset connectors")

        # Ensure we have the same number of source and target points
        if len(source_points) != len(target_points):
            raise ValueError("Number of source and target points must match")

        # Draw connector lines between each pair of source and target points
        for i, ((src_x, src_y), (tgt_x, tgt_y)) in enumerate(
            zip(source_points, target_points)
        ):
            # Convert source point from axes fraction to data coordinates if needed
            if (
                isinstance(src_x, float)
                and 0 <= src_x <= 1
                and isinstance(src_y, float)
                and 0 <= src_y <= 1
            ):
                x_data, y_data = ax.transAxes.transform((src_x, src_y))
                src_x, src_y = ax.transData.inverted().transform((x_data, y_data))

            # Convert target point from axes fraction to data coordinates if needed
            if (
                isinstance(tgt_x, float)
                and 0 <= tgt_x <= 1
                and isinstance(tgt_y, float)
                and 0 <= tgt_y <= 1
            ):
                x_data, y_data = ax.transAxes.transform((tgt_x, tgt_y))
                tgt_x, tgt_y = ax.transData.inverted().transform((x_data, y_data))

            ax.plot(
                [src_x, tgt_x],
                [src_y, tgt_y],
                linestyle=line_style,
                color=line_color,
                alpha=alpha,
                linewidth=linewidth,
                zorder=9,
            )
