#!/usr/bin/env python3
import logging
import json
from pathlib import Path
import click

from services.data_loader import DataLoader
from services.data_analyzer import DataAnalyzer
from services.visualizer import Visualizer


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(),
    default="output",
    help="Directory to save output files",
)
@click.option(
    "--report-file",
    type=str,
    default="municipal_reform_report.json",
    help="Filename for the JSON report",
)
@click.option(
    "--visualize",
    is_flag=True,
    help="Generate visualization charts and maps",
)
def main(output_dir, report_file, visualize):
    """Run the municipal reform analysis."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting municipal reform analysis")

    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)

    # Initialize components
    data_loader = DataLoader()
    data_analyzer = DataAnalyzer(data_loader)

    try:
        # Load data
        logger.info("Loading data...")
        data_analyzer.load_data()

        # Generate analysis report
        logger.info("Generating analysis report...")
        report = data_analyzer.generate_analysis_report()

        # Save report to JSON file
        report_path = output_dir_path / report_file
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Analysis report saved to {report_path}")

        # Generate visualizations if requested
        if visualize:
            logger.info("Generating visualizations...")
            visualizer = Visualizer(data_analyzer)
            visualizer.create_all_visualizations()
            logger.info("Visualizations completed")

        # Log summary findings
        logger.info("===== SUMMARY FINDINGS =====")
        logger.info(
            f"Municipalities before reform: {report['summary']['municipalities_before_reform']}"
        )
        logger.info(
            f"Municipalities after reform: {report['summary']['municipalities_after_reform']}"
        )
        logger.info(
            f"Reduction in municipalities: {report['summary']['reduction_percentage']:.1f}%"
        )
        logger.info(
            f"Total councilors before reform: {report['summary']['total_councilors_before']}"
        )
        logger.info(
            f"Total councilors after reform: {report['summary']['total_councilors_after']}"
        )
        logger.info(
            f"Reduction in councilors: {report['summary']['councilors_reduction_percentage']:.1f}%"
        )
        logger.info(
            f"Overall voter turnout before reform: {report['summary']['overall_turnout_before']:.2f}%"
        )
        logger.info(
            f"Overall voter turnout after reform: {report['summary']['overall_turnout_after']:.2f}%"
        )
        logger.info(
            f"Change in voter turnout: {report['summary']['overall_turnout_change']:.2f}%"
        )
        logger.info(
            f"Representation density before reform: {report['summary']['representation_density_before']:.1f} citizens per councilor"
        )
        logger.info(
            f"Representation density after reform: {report['summary']['representation_density_after']:.1f} citizens per councilor"
        )
        logger.info(
            f"Increase in representation density: {report['summary']['representation_density_increase_percentage']:.1f}%"
        )

        logger.info("Municipal reform analysis completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    main()  # Click handles the exit code
