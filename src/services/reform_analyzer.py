import logging


class MunicipalReformAnalyzer:
    """Analyzes the effects of Denmark's 2007 Municipal Reform."""

    def __init__(self, data=None):
        self.logger = logging.getLogger(__name__)
        self.data = data
        self.results = {}

    def set_data(self, data):
        """Set the data to be analyzed."""
        self.data = data

    def analyze_mergers(self):
        """Analyze municipal merger patterns."""
        if self.data is None:
            self.logger.error("No data available for analysis.")
            return None

        # Count merged vs unchanged municipalities
        merged_count = self.data["merged"].sum()
        unchanged_count = len(self.data) - merged_count

        self.results["merger_counts"] = {
            "merged": merged_count,
            "unchanged": unchanged_count,
        }

        return self.results["merger_counts"]

    def analyze_turnout_change(self):
        """Analyze voter turnout changes after the reform."""
        if self.data is None:
            self.logger.error("No data available for analysis.")
            return None

        # Calculate turnout change statistics
        turnout_stats = self.data["turnout_change"].describe()

        # Calculate turnout change by merger status
        turnout_by_group = self.data.groupby("merged")["turnout_change"].agg(
            ["mean", "median", "std", "count"]
        )

        self.results["turnout_change"] = {
            "overall_stats": turnout_stats,
            "by_merger_status": turnout_by_group,
        }

        return self.results["turnout_change"]

    def analyze_boundary_turnout_correlation(self):
        """Analyze correlation between boundary changes and turnout changes."""
        if self.data is None:
            self.logger.error("No data available for analysis.")
            return None

        # Calculate correlation between boundary changes (dist) and turnout change
        valid_data = self.data.dropna(subset=["dist", "turnout_change"])
        if len(valid_data) > 1:
            correlation = valid_data["dist"].corr(valid_data["turnout_change"])

            self.results["boundary_turnout_correlation"] = {
                "correlation": correlation,
                "n_observations": len(valid_data),
            }

            return self.results["boundary_turnout_correlation"]
        else:
            self.logger.warning("Not enough valid data to calculate correlation.")
            return None

    def identify_outliers(self):
        """Identify municipalities with extreme turnout changes."""
        if self.data is None:
            self.logger.error("No data available for analysis.")
            return None

        # Sort municipalities by turnout change
        sorted_data = self.data.sort_values("turnout_change")

        # Get top and bottom 3 municipalities
        bottom_3 = sorted_data.head(3)[["MunicipalityName", "turnout_change"]]
        top_3 = sorted_data.tail(3)[["MunicipalityName", "turnout_change"]]

        self.results["outliers"] = {
            "largest_decreases": bottom_3,
            "largest_increases": top_3,
        }

        return self.results["outliers"]

    def run_full_analysis(self):
        """Run all analysis methods and compile results."""
        if self.data is None:
            self.logger.error("No data available for analysis.")
            return None

        self.analyze_mergers()
        self.analyze_turnout_change()
        self.analyze_boundary_turnout_correlation()
        self.identify_outliers()

        self.logger.info("Analysis complete.")
        return self.results
