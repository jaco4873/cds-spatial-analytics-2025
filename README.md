# Danish Municipal Reform Analysis

This project analyzes the democratic consequences of Denmark's 2007 municipal reform, which reduced the number of municipalities from 271 to 98 and cut the number of local councilors from 4,597 to 2,520.

## Overview

The analysis evaluates the democratic impact of the reform by examining:

1. Changes in voter turnout between 2005 (pre-reform) and 2009 (post-reform)
2. Changes in representation density (citizens per councilor)
3. Spatial patterns in democratic participation

## Data Sources

- **Election Data**: Municipal election results from 2005 and 2009 (KMD Valg)
- **Geographical Data**: Municipal boundaries from geoBoundaries

## Project Structure

```
.
├── data/                  # Data directory
│   └── kmdvalg/           # Election data
├── src/                   # Source code
│   ├── services/          # Service modules
│   │   ├── data_loader.py # Data loading service
│   │   ├── data_analyzer.py # Analysis service
│   │   └── visualizer.py  # Visualization service
│   └── run_municipal_analysis.py # Main script
├── output/                # Generated output directory
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository
2. Create a virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the analysis script:

```bash
python src/run_municipal_analysis.py
```

### Command Line Options

- `--output-dir OUTPUT_DIR`: Directory to save output files (default: "output")
- `--skip-visualizations`: Skip creating visualizations
- `--report-file REPORT_FILE`: Filename for the JSON report (default: "municipal_reform_report.json")

## Outputs

The analysis generates:

1. A JSON report with detailed findings
2. Choropleth maps showing voter turnout and representation density
3. Comparative charts of democratic indicators before and after the reform

## Contributing

This project was created for a bachelor-level course in spatial analytics at Aarhus University, Denmark.
