import logging
import os
from services.data_loader import DataLoader
from services.reform_analyzer import MunicipalReformAnalyzer
from services.reform_visualizer import MunicipalVisualizer


def setup_logging():
    """Set up logging configuration."""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("logs/municipal_analysis.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def main():
    """Run the municipal reform analysis."""
    logger = setup_logging()
    logger.info("Starting Denmark's 2007 Municipal Reform Analysis")

    # Initialize components
    data_loader = DataLoader()
    analyzer = MunicipalReformAnalyzer()
    visualizer = MunicipalVisualizer()

    # Load and prepare data
    data_loader.load_municipal_data()
    data_loader.clean_data()
    data_loader.load_voter_data()
    municipalities = data_loader.prepare_data()

    # Set data for analysis and visualization
    analyzer.set_data(municipalities)
    visualizer.set_data(municipalities)

    # Run analysis
    logger.info("Running analysis...")
    results = analyzer.run_full_analysis()

    # Log key results
    logger.info(f"Merger counts: {results['merger_counts']}")
    logger.info(
        f"Correlation between boundary changes and turnout: "
        f"{results.get('boundary_turnout_correlation', {}).get('correlation', 'N/A')}"
    )

    # Create visualizations
    logger.info("Creating visualizations...")
    visualizer.plot_municipal_mergers()
    visualizer.plot_turnout_change()
    visualizer.plot_turnout_comparison()
    visualizer.plot_turnout_2005()
    visualizer.plot_boundary_turnout_correlation()
    visualizer.plot_turnout_changes_ranked()

    logger.info("Analysis complete. Visualizations saved to the 'output' directory.")


if __name__ == "__main__":
    main()
