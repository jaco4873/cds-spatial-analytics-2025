import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import contextily as ctx
from matplotlib.patches import Circle
from matplotlib_scalebar.scalebar import ScaleBar


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

        # Set up the visual style for publication quality
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("paper", font_scale=1.2)

        # Color scheme for publications
        self.sequential_cmap = plt.cm.YlOrRd
        self.diverging_cmap = plt.cm.RdBu_r

    def add_compass(self, ax, position=(0.90, 0.90), size=0.07):
        """Add a proper compass rose to the map.

        Args:
            ax: The matplotlib axis to add the compass to
            position: Tuple (x, y) position in axis coordinates (default: top right)
            size: Size of the compass (radius as fraction of axis size)
        """
        x, y = position
        radius = size

        # Create a circle for the compass background
        circle = Circle(
            (x, y),
            radius,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="black",
            alpha=0.7,
            zorder=10,
        )
        ax.add_patch(circle)

        # Add compass directions
        directions = {
            "N": (0, radius * 0.7),
            "S": (0, -radius * 0.7),
            "E": (radius * 0.7, 0),
            "W": (-radius * 0.7, 0),
        }

        for label, (dx, dy) in directions.items():
            ax.annotate(
                label,
                xy=(x + dx, y + dy),
                xycoords=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                zorder=11,
            )

        # Add compass lines
        ax.plot(
            [x, x],
            [y - radius * 0.6, y + radius * 0.6],
            color="black",
            transform=ax.transAxes,
            linewidth=1.5,
            zorder=11,
        )
        ax.plot(
            [x - radius * 0.6, x + radius * 0.6],
            [y, y],
            color="black",
            transform=ax.transAxes,
            linewidth=1.5,
            zorder=11,
        )

        # North pointer
        ax.annotate(
            "",
            xy=(x, y + radius * 0.6),
            xytext=(x, y),
            arrowprops=dict(facecolor="black", width=2, headwidth=5, headlength=5),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            zorder=12,
        )

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
        attr_text = (
            f"Source: KMD Valg data. Cartography: Spatial Analytics, Aarhus University. "
            f"{self.tile_attribution} "
        )
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
            cmap=self.sequential_cmap,
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
            cmap=self.diverging_cmap,
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

    def create_representation_density_map(
        self, year: int, vmin=None, vmax=None
    ) -> None:
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
        bounds = [0, 2000, 4000, 6000, 8000, 10000, 20000]
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, plt.cm.viridis.N)

        # Plot the data
        spatial_data_web.plot(
            column="density",
            ax=ax,
            cmap=plt.cm.viridis,
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
        """Create a bar chart comparing turnout before and after the reform."""
        # Get the analysis data
        turnout_analysis = self.data_analyzer.analyze_voter_turnout()

        # Extract data for the chart
        overall_2005 = turnout_analysis["overall_turnout"]["2005"]
        overall_2009 = turnout_analysis["overall_turnout"]["2009"]

        avg_2005 = turnout_analysis["average_turnout"]["2005"]
        avg_2009 = turnout_analysis["average_turnout"]["2009"]

        unchanged_2005 = turnout_analysis["unchanged_municipalities"][
            "avg_turnout_2005"
        ]
        unchanged_2009 = turnout_analysis["unchanged_municipalities"][
            "avg_turnout_2009"
        ]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set up x-coordinates
        x = np.arange(3)
        width = 0.35

        # Create bars
        rects1 = ax.bar(
            x - width / 2,
            [overall_2005, avg_2005, unchanged_2005],
            width,
            label="2005 (Pre-Reform)",
        )
        rects2 = ax.bar(
            x + width / 2,
            [overall_2009, avg_2009, unchanged_2009],
            width,
            label="2009 (Post-Reform)",
        )

        # Add labels and title
        ax.set_title(
            "Voter Turnout Comparison Before and After the 2007 Municipal Reform"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["Overall", "Average Municipal", "Unchanged Municipalities"])
        ax.legend()

        # Add value labels on the bars
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        add_labels(rects1)
        add_labels(rects2)

        # Save the figure
        output_path = self.output_dir / "turnout_comparison_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Turnout comparison chart saved to {output_path}")

    def create_municipality_reduction_chart(self) -> None:
        """Create a chart showing the reduction in municipalities and councilors."""
        # Get the analysis data
        representation_analysis = self.data_analyzer.analyze_representation_density()

        # Extract data
        muni_2005 = representation_analysis["municipalities_count"]["2005"]
        muni_2009 = representation_analysis["municipalities_count"]["2009"]

        councilors_2005 = representation_analysis["total_councilors"]["2005"]
        councilors_2009 = representation_analysis["total_councilors"]["2009"]

        # Calculate percentage reductions
        muni_reduction = (muni_2005 - muni_2009) / muni_2005 * 100
        councilor_reduction = (
            (councilors_2005 - councilors_2009) / councilors_2005 * 100
        )

        # Create figure
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Set up axes
        color1 = "tab:blue"
        color2 = "tab:red"

        # First axis - absolute numbers
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Count", color=color1)
        ax1.bar(
            [0, 1],
            [muni_2005, muni_2009],
            0.4,
            label="Municipalities",
            color=color1,
            alpha=0.7,
        )
        ax1.bar(
            [2, 3],
            [councilors_2005, councilors_2009],
            0.4,
            label="Councilors",
            color=color2,
            alpha=0.7,
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xticks([0, 1, 2, 3])
        ax1.set_xticklabels(["2005", "2009", "2005", "2009"])

        # Add value labels
        for i, v in enumerate([muni_2005, muni_2009, councilors_2005, councilors_2009]):
            ax1.text(i, v + 50, str(v), ha="center")

        # Add reduction percentages
        ax1.text(
            0.5,
            muni_2009 - 300
            if muni_2009 > 300
            else muni_2009 / 2,  # Adjust position for small values
            f"-{muni_reduction:.1f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
        ax1.text(
            2.5,
            councilors_2009 - 300
            if councilors_2009 > 300
            else councilors_2009 / 2,  # Adjust position for small values
            f"-{councilor_reduction:.1f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

        # Set title
        plt.title("Reduction in Municipalities and Councilors after the 2007 Reform")
        plt.legend()

        # Save the figure
        output_path = self.output_dir / "municipality_reduction_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Municipality reduction chart saved to {output_path}")

    def create_all_visualizations(self) -> None:
        """Create all visualizations at once."""
        # Make sure data is loaded
        if self.data_analyzer.election_2005_df is None:
            self.data_analyzer.load_data()

        # Generate all maps and charts
        self.create_turnout_map(2005)
        self.create_turnout_map(2009)
        self.create_turnout_change_map()
        self.create_representation_density_map(2005, vmin=0, vmax=20000)
        self.create_representation_density_map(2009, vmin=0, vmax=20000)
        self.create_turnout_comparison_chart()
        self.create_municipality_reduction_chart()

        self.logger.info("All visualizations created successfully")
