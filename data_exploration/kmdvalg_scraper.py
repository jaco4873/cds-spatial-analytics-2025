import requests
from bs4 import BeautifulSoup
import pandas as pd
import chardet
from urllib.parse import urljoin
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("scraper.log")],
)
logger = logging.getLogger(__name__)


def fetch_html(url) -> str:
    """Fetch HTML content with proper encoding detection."""
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Detect encoding automatically
        raw_content = response.content
        detected = chardet.detect(raw_content)
        encoding = detected["encoding"]
        logger.info(f"Detected encoding: {encoding} for {url}")

        # Decode content with detected encoding
        decoded = raw_content.decode(encoding)
        return decoded
    except Exception as e:
        logger.error(f"Error fetching the URL: {e}")
        return None


def extract_kommune_links(html_content, base_url):
    """Extract kommune information from HTML content with recursive handling of nested pages."""
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    kommune_data = []

    for area in soup.find_all("area"):
        name = area.get("alt", "")
        href = area.get("href", "")

        if not href or not name:
            continue

        # Check if this Hovedstadens kommuner
        if name == "Hovedstadens kommuner" or (href and href.endswith("h.htm")):
            logger.info(
                f"Found container page: {name}, fetching nested kommuner from {href}"
            )

            # Construct full URL for the nested page
            nested_url = urljoin(base_url, href)

            # Fetch and parse the nested page
            nested_html = fetch_html(nested_url)
            if nested_html:
                # Extract kommuner from nested page and add them
                nested_kommuner = extract_kommune_links(nested_html, nested_url)
                kommune_data.extend(nested_kommuner)
                logger.info(f"Added {len(nested_kommuner)} nested kommuner from {name}")
            else:
                logger.warning(f"Failed to fetch nested page: {nested_url}")
        else:
            # Regular kommune, add it to the list
            kommune_data.append(
                {
                    "name": name.strip(),
                    "href": href.strip(),
                }
            )

    return kommune_data


def fetch_kommune_page(base_url, kommune_href, save_path=None):
    """
    Fetch a specific kommune page and optionally save it to a file.

    Args:
        base_url: Base URL of the election site
        kommune_href: Href of the kommune to fetch
        save_path: Path to save the HTML content (optional)

    Returns:
        The HTML content of the kommune page
    """
    # Construct full URL
    kommune_url = urljoin(base_url, kommune_href)
    logger.info(f"Fetching kommune page: {kommune_url}")

    # Fetch the page
    html_content = fetch_html(kommune_url)

    # Save the content if requested
    if save_path and html_content:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Saved kommune page to {save_path}")

    return html_content


def extract_kommune_stats(html_content):
    """
    Extract voting statistics from a kommune page

    Args:
        html_content: HTML content of kommune page

    Returns:
        dict: Dictionary with kommune voting statistics
    """
    if not html_content:
        return {}

    soup = BeautifulSoup(html_content, "html.parser")
    stats = {}

    # Find the table with voting statistics
    stat_tables = soup.find_all("table", attrs={"bgcolor": "#EFEFEF"})

    if stat_tables and len(stat_tables) > 0:
        # Process the first relevant table
        rows = stat_tables[0].find_all("tr")

        for row in rows:
            cells = row.find_all("td")
            if len(cells) == 2:
                label = cells[0].get_text().strip()
                value = cells[1].get_text().strip()

                if "Stemmeberettigede" in label:
                    stats["stemmeberettigede"] = value
                elif "Stemmeprocent" in label:
                    stats["stemmeprocent"] = value
                elif "Optalte stemmer" in label:
                    stats["optalte_stemmer"] = value

    # Check if this is a merged kommune by looking for "Kommuner" heading
    merged = False
    merged_municipalities = []

    font_headers = soup.find_all("font", attrs={"color": "#FFFFFF"})
    for header in font_headers:
        bold_tags = header.find_all("b")
        for bold in bold_tags:
            if bold.get_text().strip() == "Kommuner":
                merged = True

                # Find the table containing municipality links
                table = bold.find_parent("table")
                if table:
                    # Find the row with kommune links
                    link_row = table.find("td", attrs={"bgcolor": "#CCCCCC"})
                    if link_row:
                        # Extract all kommune links
                        for link in link_row.find_all("a"):
                            kommune_name = link.get_text().strip()
                            if kommune_name:
                                merged_municipalities.append(kommune_name)

    # Check for "Afstemningsområder" (voting areas) which confirms it's not merged
    if not merged:
        for header in font_headers:
            bold_tags = header.find_all("b")
            for bold in bold_tags:
                if "Afstemningsområder" in bold.get_text().strip():
                    merged = False

    # Add the merger information to the stats
    stats["merged"] = merged
    stats["merged_municipalities"] = merged_municipalities if merged else []

    return stats


def extract_kommune_name_from_page(html_content):
    """
    Extract the kommune name directly from the kommune page HTML
    where encoding is properly handled.

    Args:
        html_content: HTML content of kommune page

    Returns:
        str: Properly encoded kommune name
    """
    if not html_content:
        return None

    soup = BeautifulSoup(html_content, "html.parser")

    # Strategy 1: Check the title first
    title = soup.find("title")

    title_text = title.get_text().strip()
    # Case-insensitive search for "kommune"
    if "kommune" in title_text.lower():
        # Extract kommune name from title (e.g., "Kommunalvalg Ærø kommune" → "Ærø Kommune")
        parts = title_text.split()
        for i, part in enumerate(parts):
            if "kommune" in part.lower() and i > 0:
                return f"{parts[i - 1]}"

    else:
        logger.warning("No title found in the HTML content")

    return None


def scrape_all_kommuner(kommune_df, base_url, year):
    """
    Scrape voting data for all kommuner in the DataFrame

    Args:
        kommune_df: DataFrame with kommune names and hrefs
        base_url: Base URL for constructing full kommune URLs
        year: Election year for the data being scraped

    Returns:
        DataFrame: Enhanced DataFrame with voting statistics
    """
    # Calculate base directory from the URL
    base_directory = "/".join(base_url.split("/")[:-1]) + "/"

    # Create columns for the statistics
    kommune_df["stemmeberettigede"] = None
    kommune_df["stemmeprocent"] = None
    kommune_df["optalte_stemmer"] = None
    kommune_df["merged"] = None
    kommune_df["merged_municipalities"] = None
    kommune_df["year"] = year  # Add year column

    # Process each kommune
    for index, row in kommune_df.iterrows():
        kommune_name = row["name"]
        kommune_href = row["href"]

        logger.info(
            f"Processing {kommune_name} ({index + 1}/{len(kommune_df)}) for year {year}"
        )

        # Fetch the kommune page
        html_content = fetch_html(urljoin(base_directory, kommune_href))

        if html_content:
            # Extract statistics
            stats = extract_kommune_stats(html_content)

            # Check if name contains the placeholder character "ï¿½"
            if "ï¿½" in kommune_name:
                corrected_name = extract_kommune_name_from_page(html_content)
                if corrected_name:
                    logger.info(
                        f"Corrected name from '{kommune_name}' to '{corrected_name}'"
                    )
                    kommune_df.at[index, "name"] = corrected_name

            # Update the DataFrame with stats data
            for key, value in stats.items():
                if key in ["merged", "merged_municipalities"] and year != 2005:
                    # Skip merger data for non-2005 years
                    continue
                kommune_df.at[index, key] = value

    return kommune_df


def scrape_election_year(year):
    """
    Scrape data for a specific election year

    Args:
        year: Election year to scrape

    Returns:
        DataFrame: DataFrame with kommune voting data
    """
    base_url = f"https://www.kmdvalg.dk/kv/{year}/adk.htm"
    logger.info(f"Starting scraper for election year {year}")

    # Define output dir
    output_dir = Path("data/kmdvalg")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file with year
    output_file = f"{output_dir}/kommune_data_{year}.csv"

    # Always fetch fresh kommune links from the website
    logger.info(f"Fetching kommune links for {year} from website...")
    html_content = fetch_html(base_url)

    if not html_content:
        logger.error(f"Failed to fetch HTML content for {year}")
        return None

    # Extract kommune data
    kommune_data = extract_kommune_links(html_content, base_url)
    df = pd.DataFrame(kommune_data)
    df = df.drop_duplicates(subset=["name", "href"])
    logger.info(f"Found {len(df)} kommune links for {year}")

    # Scrape voting data for all kommuner
    enhanced_df = scrape_all_kommuner(df, base_url, year)

    # Before saving, normalize Danish characters manually
    for idx, row in enhanced_df.iterrows():
        for col in ["name"]:
            if isinstance(row[col], str):
                # Ensure proper encoding of Danish characters
                enhanced_df.at[idx, col] = (
                    row[col].encode("latin1", errors="replace").decode("latin1")
                )

    # Save with UTF-8
    enhanced_df.to_csv(output_file, index=False, encoding="utf-8")
    logger.info(f"Saved enhanced DataFrame to {output_file}")

    # Display results
    logger.info(f"\nCompleted scraping voting data for all kommuner in {year}")
    logger.info(f"\n{enhanced_df.head()}")

    # Show some statistics
    stats_columns = ["stemmeberettigede", "stemmeprocent", "optalte_stemmer"]
    missing_data = enhanced_df[stats_columns].isna().sum()
    logger.info(f"\nMissing data count: {missing_data}")

    valid_counts = len(enhanced_df) - missing_data
    logger.info(f"Successfully scraped data: {valid_counts}")

    return enhanced_df


def main():
    """Main function to scrape multiple election years and combine results"""
    years = [2005, 2009]
    all_data = []

    for year in years:
        year_df = scrape_election_year(year)
        if year_df is not None:
            all_data.append(year_df)

    # Combine all data frames
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Fix encoding for Danish characters
        for idx, row in combined_df.iterrows():
            if isinstance(row["name"], str):
                name = row["name"]
                name = name.replace("ï¿½", "ø").replace("ï¿½", "æ").replace("ï¿½", "å")
                combined_df.at[idx, "name"] = name

        # Save the combined data
        output_dir = Path("data/kmdvalg")
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_output_file = f"{output_dir}/kommune_data_combined.csv"
        combined_df.to_csv(combined_output_file, index=False, encoding="utf-8-sig")
        logger.info(
            f"Saved combined data with {len(combined_df)} records to {combined_output_file}"
        )


if __name__ == "__main__":
    main()
