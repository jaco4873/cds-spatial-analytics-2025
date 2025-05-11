import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd


from services.data_analyzer import DataAnalyzer


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

    def add_compass(self, ax, size=0.25, position=(0.85, 0.85)):
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
        self.add_compass(ax, position=(0.90, 0.90))

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

        # Plot the data
        spatial_data_web.plot(
            column="stemmeprocent_election",
            ax=ax,
            cmap=self.cmap,
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

        # Plot the data with diverging colormap
        comparison_gdf_web.plot(
            column="turnout_change",
            ax=ax,
            cmap=self.cmap,
            legend=True,
            vmin=-max_change,
            vmax=max_change,
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

        # Add map elements
        self.add_map_elements(
            ax,
            title="Change in Voter Turnout between 2005 and 2009",
        )

        # Save the map
        self.save_map("turnout_change_map")

    def create_representation_density_map(self, year: int) -> None:
        """Create a publication-quality choropleth map showing representation density (citizens per councilor).

        Args:
            year: The election year (2005 or 2009)
            vmin: Optional minimum value for color scale
            vmax: Optional maximum value for color scale
        """
        # Prepare spatial data
        spatial_2005, spatial_2009 = self.data_analyzer.prepare_spatial_data()

        if year == 2005:
            spatial_data = spatial_2005
        elif year == 2009:
            spatial_data = spatial_2009
        else:
            raise ValueError("Year must be either 2005 or 2009")

        # Calculate representation density
        spatial_data["density"] = spatial_data["stemmeberettigede_election"] / (
            self.data_analyzer.TOTAL_COUNCILORS_2005
            / len(self.data_analyzer.election_2005_df)
            if year == 2005
            else self.data_analyzer.TOTAL_COUNCILORS_2009
            / len(self.data_analyzer.election_2009_df)
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
        merged_munis_2005 = self.data_analyzer.election_2005_df[
            self.data_analyzer.election_2005_df["merged"] == True
        ]
        unmerged_munis_2005 = self.data_analyzer.election_2005_df[
            self.data_analyzer.election_2005_df["merged"] == False
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
            legend=True,
            categorical=True,
            legend_kwds={
                "labels": ["Unchanged", "Merged"],
                "title": "Municipality Status",
                "frameon": True,
                "loc": "center",
                "bbox_to_anchor": (0.90, 0.75),
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

        # Add centroid labels for turnout change
        # First merge the turnout change data with the base map
        merged_data = spatial_2005_web.merge(
            turnout_change_gdf[["shapeName", "turnout_change"]],
            left_on="shapeName",
            right_on="shapeName",
            how="left",
        )

        # Add centroid labels with turnout change values
        for _, row in merged_data.iterrows():
            if pd.notna(row["turnout_change"]):
                # Get the centroid of the municipality
                centroid = row.geometry.centroid

                # Format the label with sign
                if row["turnout_change"] > 0:
                    label = f"+{row['turnout_change']:.1f}"
                    color = "darkgreen"
                else:
                    label = f"{row['turnout_change']:.1f}"
                    color = "darkred"

                # Add text annotation with the turnout change value
                ax.text(
                    centroid.x,
                    centroid.y,
                    label,
                    fontsize=7,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        facecolor=color, alpha=0.7, pad=1, boxstyle="round,pad=0.1"
                    ),
                )

        # Add map elements
        self.add_map_elements(
            ax,
            title="Municipalities by Merger Status with Voter Turnout Change (2005-2009)",
        )

        # Save the map
        self.save_map("merged_status_map")

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
