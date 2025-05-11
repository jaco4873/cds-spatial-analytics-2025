import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import logging


class MunicipalVisualizer:
    """Handles visualization of municipal reform analysis."""

    def __init__(self, data=None):
        self.logger = logging.getLogger(__name__)
        self.data = data
        self.setup_style()

    def setup_style(self):
        """Set up matplotlib plotting style."""
        plt.style.use("seaborn-v0_8-whitegrid")

    def set_data(self, data):
        """Set the data to be visualized."""
        self.data = data

    def plot_municipal_mergers(self):
        """Create map of municipal mergers."""
        if self.data is None:
            self.logger.error("No data available for visualization.")
            return None

        plt.figure(figsize=(12, 10))

        # Create a custom categorical colormap with 2 colors
        merge_colors = ListedColormap(["green", "red"])

        # Count merged vs unchanged
        merged_count = self.data["merged"].sum()
        unchanged_count = len(self.data) - merged_count

        # Create the map
        merged_map = self.data.plot(
            column="merged",
            cmap=merge_colors,
            legend=False,  # We'll create our own legend
            figsize=(12, 10),
        )

        # Add a custom legend
        legend_elements = [
            Patch(
                facecolor="green", label=f"Unchanged ({unchanged_count} municipalities)"
            ),
            Patch(facecolor="red", label=f"Merged ({merged_count} municipalities)"),
        ]
        merged_map.legend(handles=legend_elements, loc="lower right")

        merged_map.set_title(
            "Map of Municipal Mergers in Denmark (2007 Reform)", fontsize=16
        )
        plt.savefig("output/municipal_mergers_map.png", dpi=300, bbox_inches="tight")
        return merged_map

    def plot_turnout_change(self):
        """Create map of voter turnout change."""
        if self.data is None:
            self.logger.error("No data available for visualization.")
            return None

        plt.figure(figsize=(12, 10))

        # Center the colormap around 0 for fair comparison
        vmin = min(
            self.data["turnout_change"].min(), -self.data["turnout_change"].max()
        )
        vmax = max(
            self.data["turnout_change"].max(), -self.data["turnout_change"].min()
        )

        # Use a diverging colormap without white in the middle
        turnout_map = self.data.plot(
            column="turnout_change",
            cmap="coolwarm",  # Continuous spectrum from blue to red
            legend=True,
            vmin=vmin,
            vmax=vmax,
            legend_kwds={
                "label": "Percentage Point Change in Voter Turnout (2005-2009)"
            },
            figsize=(12, 10),
        )

        # Add text for legend interpretation
        plt.text(
            0.02,
            0.05,
            "Red = Decreased Turnout\nBlue = Increased Turnout",
            transform=turnout_map.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        turnout_map.set_title(
            "Changes in Municipal Voter Turnout After 2007 Reform", fontsize=16
        )
        plt.savefig("output/turnout_change_map.png", dpi=300, bbox_inches="tight")
        return turnout_map

    def plot_turnout_comparison(self):
        """Create boxplot comparing turnout between merged and unchanged municipalities."""
        if self.data is None:
            self.logger.error("No data available for visualization.")
            return None

        plt.figure(figsize=(10, 6))
        turnout_comp = self.data.boxplot(
            column="turnout_change",
            by="merged",
            grid=False,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
        )
        plt.title("Voter Turnout Change by Municipal Status", fontsize=16)
        plt.suptitle("")  # Remove default title
        plt.ylabel("Percentage Point Change in Turnout (2005-2009)", fontsize=12)
        plt.xticks([1, 2], ["Unchanged", "Merged"], fontsize=12)
        plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)  # Reference line at 0
        plt.savefig(
            "output/turnout_comparison_boxplot.png", dpi=300, bbox_inches="tight"
        )

        # Calculate and log average turnout change by group
        turnout_by_group = self.data.groupby("merged")["turnout_change"].agg(
            ["mean", "median", "std", "count"]
        )
        self.logger.info("\nVoter Turnout Change by Municipal Status:")
        self.logger.info(turnout_by_group)

        return turnout_comp

    def plot_turnout_2005(self):
        """Create map of 2005 voter turnout."""
        if self.data is None:
            self.logger.error("No data available for visualization.")
            return None

        plt.figure(figsize=(12, 10))
        turnout2005_map = self.data.plot(
            column="VoterTurnout2005_pct",
            cmap="Blues",  # Blue sequential colormap - darker blue for higher turnout
            legend=True,
            legend_kwds={"label": "Voter Turnout (2005, %)"},
            figsize=(12, 10),
        )

        # Add text explanation
        plt.text(
            0.02,
            0.05,
            "Darker Blue = Higher Turnout",
            transform=turnout2005_map.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        turnout2005_map.set_title(
            "Municipal Voter Turnout in 2005 (Pre-Reform)", fontsize=16
        )
        plt.savefig("output/turnout_2005_map.png", dpi=300, bbox_inches="tight")
        return turnout2005_map

    def plot_boundary_turnout_correlation(self):
        """Create scatterplot showing relationship between boundary changes and turnout."""
        if self.data is None:
            self.logger.error("No data available for visualization.")
            return None

        plt.figure(figsize=(10, 6))
        plt.scatter(self.data["dist"], self.data["turnout_change"], alpha=0.7)
        plt.title(
            "Relationship Between Boundary Changes and Voter Turnout Change",
            fontsize=16,
        )
        plt.xlabel("Municipal Boundary Change Measure (dist)", fontsize=12)
        plt.ylabel("Percentage Point Change in Turnout (2005-2009)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Add trend line if we have enough data
        valid_data = self.data.dropna(subset=["dist", "turnout_change"])
        if len(valid_data) > 1:
            z = np.polyfit(valid_data["dist"], valid_data["turnout_change"], 1)
            p = np.poly1d(z)
            plt.plot(self.data["dist"], p(self.data["dist"]), "r--", alpha=0.8)

            # Calculate correlation
            correlation = valid_data["dist"].corr(valid_data["turnout_change"])
            plt.text(
                0.05,
                0.95,
                f"Correlation: {correlation:.4f}",
                transform=plt.gca().transAxes,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=12,
            )

        plt.savefig("output/dist_turnout_correlation.png", dpi=300, bbox_inches="tight")
        return plt.gca()

    def plot_turnout_changes_ranked(self):
        """Create plot of ranked turnout changes to identify outliers."""
        if self.data is None:
            self.logger.error("No data available for visualization.")
            return None

        plt.figure(figsize=(12, 6))

        # Sort municipalities by turnout change for easier visualization
        sorted_municipalities = self.data.sort_values("turnout_change")

        # Create rank column for x-axis
        sorted_municipalities["rank"] = range(len(sorted_municipalities))

        # Plot sorted turnout changes
        plt.scatter(
            sorted_municipalities["rank"],
            sorted_municipalities["turnout_change"],
            alpha=0.7,
        )
        plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)  # Reference line at 0
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.title("Ranked Voter Turnout Changes (2005-2009)", fontsize=16)
        plt.xlabel("Municipality Rank", fontsize=12)
        plt.ylabel("Percentage Point Change in Turnout", fontsize=12)

        # Label extreme outliers
        for idx, row in sorted_municipalities.head(3).iterrows():  # Lowest 3
            plt.annotate(
                row["MunicipalityName"],
                (row["rank"], row["turnout_change"]),
                xytext=(5, -15),
                textcoords="offset points",
            )

        for idx, row in sorted_municipalities.tail(3).iterrows():  # Highest 3
            plt.annotate(
                row["MunicipalityName"],
                (row["rank"], row["turnout_change"]),
                xytext=(5, 10),
                textcoords="offset points",
            )

        plt.savefig("output/turnout_change_ranked.png", dpi=300, bbox_inches="tight")
        return plt.gca()
