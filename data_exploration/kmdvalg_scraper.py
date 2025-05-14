import requests
from bs4 import BeautifulSoup
import chardet
from urllib.parse import urljoin
import logging
import re
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("kmdvalg_scraper.log")],
)
logger = logging.getLogger(__name__)


# ===== UTILITY FUNCTIONS =====


def fetch_html(url) -> str:
    """Fetch HTML content with proper encoding detection."""
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Detect encoding automatically
        raw_content = response.content
        detected = chardet.detect(raw_content)
        encoding = detected["encoding"]

        # Decode content with detected encoding
        decoded = raw_content.decode(encoding)
        return decoded
    except Exception as e:
        logger.error(f"Error fetching the URL: {e}")
        return None


def save_html(html_content, file_path):
    """Save HTML content to a file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Successfully saved HTML to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving HTML to file: {e}")
        return False


def clean_text(text):
    """Clean text by removing unwanted whitespace and control characters."""
    if not text:
        return ""

    # Replace newlines, carriage returns and multiple spaces with a single space
    cleaned = re.sub(r"[\n\r\t]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


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

    return stats


def extract_counselor_count(html_content, base_directory):
    """
    Extract the number of counselors from a kommune page by finding and processing
    the 'Kommunalbestyrelse' link.

    Args:
        html_content: HTML content of the kommune page
        base_directory: Base URL for constructing full URLs

    Returns:
        int: Number of counselors or None if not found
    """
    if not html_content:
        return None

    soup = BeautifulSoup(html_content, "html.parser")

    # Look for the link to the council page ('Kommunalbestyrelse')
    council_link = None

    for link in soup.find_all("a"):
        href = link.get("href", "")
        # Check if the link URL contains "km" which is the prefix for council pages
        if href and (href.startswith("km") or href.startswith("am")):
            council_link = href
            break

    if not council_link:
        logger.warning("Could not find council page link")
        return None

    # Fetch the council page
    council_url = urljoin(base_directory, council_link)
    logger.info(f"Fetching council page: {council_url}")
    council_html = fetch_html(council_url)

    if not council_html:
        logger.warning(f"Failed to fetch council page: {council_url}")
        return None

    # Parse the council page to count the number of counselors
    council_soup = BeautifulSoup(council_html, "html.parser")

    # Method 1: Look for the highest mandate number in the table
    highest_mandate = 0

    # Find all cells that might contain mandate numbers
    mandate_cells = council_soup.find_all("td", attrs={"width": "3%"})
    for cell in mandate_cells:
        # Get text and try to convert to integer
        mandate_text = cell.get_text().strip()
        try:
            mandate_num = int(mandate_text)
            highest_mandate = max(highest_mandate, mandate_num)
        except (ValueError, TypeError):
            # If not a valid number, skip
            continue

    if highest_mandate > 0:
        return highest_mandate

    # If method 1 fails, return None
    return None


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
    if title:
        title_text = title.get_text().strip()
        # Case-insensitive search for "kommune"
        if "kommune" in title_text.lower():
            # Extract kommune name from title (e.g., "Kommunalvalg Ærø kommune" → "Ærø")
            parts = title_text.split()
            for i, part in enumerate(parts):
                if "kommune" in part.lower() and i > 0:
                    return f"{parts[i - 1]}"
    else:
        logger.warning("No title found in the HTML content")

    return None


# ===== 2001-SPECIFIC FUNCTIONS =====


def extract_area_tags(html_content, base_url):
    """
    Extract all AREA tags from an amt page (2001-specific)

    Args:
        html_content: HTML content of the amt page
        base_url: Base URL for resolving relative links

    Returns:
        List of dictionaries with kommune name and href
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    kommune_data = []

    # Find all AREA tags
    for area in soup.find_all("area"):
        name = area.get("alt", "")
        href = area.get("href", "")

        if not href or not name:
            continue

        # Clean the name
        cleaned_name = clean_text(name)

        # Add to the list
        kommune_data.append(
            {
                "name": cleaned_name,
                "href": href.strip(),
            }
        )

    return kommune_data


def fetch_main_page_2001(year):
    """
    Fetch the main election page for 2001

    Args:
        year: Election year

    Returns:
        HTML content of the main page
    """
    base_url = f"https://www.kmdvalg.dk/kv/{year}/"
    logger.info(f"Fetching main election page for {year}: {base_url}")

    # Fetch the page
    html_content = fetch_html(base_url)

    if not html_content:
        logger.error(f"Failed to fetch content from {base_url}")
        return None

    return html_content


def extract_amt_links(html_content, base_url):
    """
    Extract all amt links from the main page (2001-specific)

    Args:
        html_content: HTML content of the main page
        base_url: Base URL for resolving relative links

    Returns:
        List of dictionaries with amt name and href
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    amt_data = []

    # Find all image map areas which contain the amt links
    for area in soup.find_all("area"):
        name = area.get("alt", "")
        href = area.get("href", "")

        if not href or not name:
            continue

        # Clean the name
        cleaned_name = clean_text(name)

        # Add to the list
        amt_data.append(
            {
                "name": cleaned_name,
                "href": href.strip(),
            }
        )

    return amt_data


def fetch_all_amt_pages(amt_links, year):
    """
    Fetch all amt pages and extract kommune links (2001-specific)

    Args:
        amt_links: List of amt links
        year: Election year

    Returns:
        List of dictionaries with kommune data including amt information
    """
    base_url = f"https://www.kmdvalg.dk/kv/{year}/"
    all_kommuner = []

    for amt in amt_links:
        amt_name = amt["name"]
        amt_href = amt["href"]
        amt_url = urljoin(base_url, amt_href)

        logger.info(f"Fetching amt page: {amt_name} from {amt_url}")

        # Fetch the amt page
        amt_html = fetch_html(amt_url)

        if not amt_html:
            logger.error(f"Failed to fetch amt page: {amt_url}")
            continue

        # Extract kommune links from the amt page
        kommune_data = extract_area_tags(amt_html, base_url)

        logger.info(f"Found {len(kommune_data)} kommune links in {amt_name}")

        # Add to the flat list of all kommuner with amt info
        for kommune in kommune_data:
            kommune_with_amt = kommune.copy()
            kommune_with_amt["amt"] = amt_name
            all_kommuner.append(kommune_with_amt)

    return all_kommuner


def process_kommune_2001(kommune_data, year, stats_counter):
    """
    Process a kommune by fetching its page and extracting data (2001-specific)

    Args:
        kommune_data: Dictionary with kommune information
        year: Election year
        stats_counter: Dictionary to track success/failure statistics

    Returns:
        Dictionary with enhanced kommune data
    """
    base_url = f"https://www.kmdvalg.dk/kv/{year}/"
    kommune_name = kommune_data["name"]
    kommune_href = kommune_data["href"]
    amt_name = kommune_data["amt"]

    # Full URL for the kommune page
    kommune_url = urljoin(base_url, kommune_href)

    logger.info(f"Processing kommune: {kommune_name} in {amt_name} from {kommune_url}")

    # Fetch the kommune page
    html_content = fetch_html(kommune_url)

    if not html_content:
        logger.error(f"Failed to fetch HTML for {kommune_name}")
        stats_counter["html_fetch_failures"] += 1
        return kommune_data

    # Get basic statistics
    stats = extract_kommune_stats(html_content)

    # Get counselor count
    counselor_count = extract_counselor_count(html_content, base_url)

    # Add collected data to kommune_data
    enhanced_data = kommune_data.copy()
    enhanced_data.update(stats)

    # Track statistics
    if stats.get("stemmeberettigede") is None:
        stats_counter["missing_stemmeberettigede"] += 1
    if stats.get("stemmeprocent") is None:
        stats_counter["missing_stemmeprocent"] += 1
    if stats.get("optalte_stemmer") is None:
        stats_counter["missing_optalte_stemmer"] += 1

    if counselor_count:
        enhanced_data["counselor_count"] = counselor_count
    else:
        stats_counter["missing_counselor_count"] += 1

    # Add year
    enhanced_data["year"] = year

    # Track fully successful extraction
    if (
        stats.get("stemmeberettigede") is not None
        and stats.get("stemmeprocent") is not None
        and stats.get("optalte_stemmer") is not None
        and counselor_count is not None
    ):
        stats_counter["fully_successful"] += 1

    return enhanced_data


# ===== 2005/2009-SPECIFIC FUNCTIONS =====


def extract_kommune_links(html_content, base_url):
    """
    Extract kommune information from HTML content with recursive handling of nested pages.
    Used for 2005/2009 structure.
    """
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


def check_merged_kommune(html_content):
    """
    Check if a kommune is a merged kommune (2005/2009-specific)

    Args:
        html_content: HTML content of kommune page

    Returns:
        tuple: (is_merged, list of merged municipalities)
    """
    if not html_content:
        return False, []

    soup = BeautifulSoup(html_content, "html.parser")
    merged = False
    merged_municipalities = []

    # Check if this is a merged kommune by looking for "Kommuner" heading
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

    return merged, merged_municipalities


def scrape_all_kommuner(kommune_df, base_url, year):
    """
    Scrape voting data for all kommuner in the DataFrame (2005/2009-specific)

    Args:
        kommune_df: DataFrame with kommune names and hrefs
        base_url: Base URL for constructing full kommune URLs
        year: Election year

    Returns:
        DataFrame: Enhanced DataFrame with voting statistics
    """
    # Calculate base directory from the URL
    base_directory = "/".join(base_url.split("/")[:-1]) + "/"

    # Create columns for the statistics
    kommune_df["stemmeberettigede"] = None
    kommune_df["stemmeprocent"] = None
    kommune_df["optalte_stemmer"] = None
    kommune_df["counselor_count"] = None  # New column for counselor count
    kommune_df["year"] = year  # Add year column

    # Add merge-related columns only for 2005+
    if year >= 2005:
        kommune_df["merged"] = None
        kommune_df["merged_municipalities"] = None

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
                kommune_df.at[index, key] = value

            # Extract merger information for 2005+
            if year >= 2005:
                merged, merged_municipalities = check_merged_kommune(html_content)
                kommune_df.at[index, "merged"] = merged
                kommune_df.at[index, "merged_municipalities"] = merged_municipalities

            # Extract counselor count
            counselor_count = extract_counselor_count(html_content, base_directory)
            if counselor_count:
                kommune_df.at[index, "counselor_count"] = counselor_count

    return kommune_df


# ===== UNIFIED SCRAPING FUNCTIONS =====


def scrape_election_year(year):
    """
    Scrape data for a specific election year

    Args:
        year: Election year to scrape

    Returns:
        DataFrame: DataFrame with kommune voting data
    """
    # Define output dir
    output_dir = Path("data/kmdvalg")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file with year
    output_file = f"{output_dir}/kommune_data_{year}.csv"

    if year == 2001:
        return scrape_election_year_2001(output_file)
    else:
        # 2005/2009 process
        base_url = f"https://www.kmdvalg.dk/kv/{year}/adk.htm"
        logger.info(f"Starting scraper for election year {year}")

        # Fetch kommune links from the website
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

        # Apply hardcoded fixes for specific years
        apply_hardcoded_fixes(enhanced_df, year)

        # Fix encoding for Danish characters
        fix_danish_encoding(enhanced_df)

        # Calculate and log counselors
        total_counselors = enhanced_df["counselor_count"].dropna().sum()
        logger.info(
            f"Total number of counselors for year {year}: {int(total_counselors)}"
        )

        # Save with UTF-8
        enhanced_df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"Saved enhanced DataFrame to {output_file}")

        return enhanced_df


def scrape_election_year_2001(output_file):
    """
    Scrape data for the 2001 election year

    Args:
        output_file: Path to save the output CSV

    Returns:
        DataFrame with kommune voting data
    """
    year = 2001
    logger.info(f"Starting scraper for election year {year}")

    # Initialize statistics counter
    stats_counter = {
        "total_kommuner": 0,
        "fully_successful": 0,
        "html_fetch_failures": 0,
        "missing_stemmeberettigede": 0,
        "missing_stemmeprocent": 0,
        "missing_optalte_stemmer": 0,
        "missing_counselor_count": 0,
    }

    # Fetch main election page
    main_html = fetch_main_page_2001(year)

    if not main_html:
        logger.error(f"Failed to fetch main election page for {year}")
        return None

    # Extract amt links
    base_url = f"https://www.kmdvalg.dk/kv/{year}/"
    amt_links = extract_amt_links(main_html, base_url)
    logger.info(f"Found {len(amt_links)} amt links for {year}")

    # Fetch all amt pages and extract kommune links
    all_kommuner = fetch_all_amt_pages(amt_links, year)
    total_kommuner = len(all_kommuner)
    stats_counter["total_kommuner"] = total_kommuner
    logger.info(f"Found {total_kommuner} kommune links for {year}")

    # Process each kommune
    processed_kommuner = []

    for i, kommune in enumerate(all_kommuner):
        logger.info(f"Processing kommune {i + 1}/{total_kommuner}: {kommune['name']}")
        enhanced_kommune = process_kommune_2001(kommune, year, stats_counter)
        processed_kommuner.append(enhanced_kommune)

    # Convert to DataFrame
    df = pd.DataFrame(processed_kommuner)

    # Apply hardcoded fixes for 2001
    apply_hardcoded_fixes(df, 2001)

    # Fix encoding for Danish characters
    fix_danish_encoding(df)

    # Calculate and log the sum of counselors for this year
    total_counselors = df["counselor_count"].dropna().sum()
    counselors_found = df["counselor_count"].count()
    logger.info(
        f"Total number of counselors for year {year}: {int(total_counselors)} (found for {counselors_found}/{total_kommuner} kommuner)"
    )

    # Save with UTF-8
    df.to_csv(output_file, index=False, encoding="utf-8")
    logger.info(f"Saved DataFrame to {output_file}")

    # Log statistics summary
    log_statistics_summary(stats_counter, df)

    return df


def fix_danish_encoding(df):
    """Fix encoding for Danish characters in a DataFrame"""
    for idx, row in df.iterrows():
        if isinstance(row["name"], str):
            # Ensure proper encoding of Danish characters
            df.at[idx, "name"] = (
                row["name"]
                .replace("ï¿½", "ø")
                .replace("ï¿½", "æ")
                .replace("ï¿½", "å")
                .encode("latin1", errors="replace")
                .decode("latin1")
            )


def apply_hardcoded_fixes(df, year):
    """Apply hardcoded fixes for specific situations"""
    if year == 2005:
        # Existing fixes for Læsø and Fanø
        laeso_idx = df[df["name"].str.contains("Læsø", na=False, case=False)].index
        if not laeso_idx.empty:
            df.at[laeso_idx[0], "counselor_count"] = 9
            logger.info(
                "Hardcoded counselor count for Læsø kommune: 9 (source: statistikbanken.dk/VALGK3)"
            )

        fano_idx = df[df["name"].str.contains("Fanø", na=False, case=False)].index
        if not fano_idx.empty:
            df.at[fano_idx[0], "counselor_count"] = 11
            logger.info(
                "Hardcoded counselor count for Fanø kommune: 11 (source: statistikbanken.dk/VALGK3)"
            )

    # For our 2001-to-2005 mapping, create a lookup for missing municipalities
    if year == 2001:
        # Create a dictionary of municipalities we couldn't find in 2001 that should exist
        missing_municipalities = {
            "Rønne": {
                "amt": "Bornholms Amt",
                "counselor_count": 21,
            },
            "Bredebro": {
                "amt": "Sønderjyllands Amt",
                "counselor_count": 13,
            },
            "Rougsø": {
                "amt": "Århus Amt",
                "counselor_count": 15,
            },
            "Stenlille": {
                "amt": "Vestsjællands Amt",
                "counselor_count": 13,
            },
            "Ullerslev": {
                "amt": "Fyns Amt",
                "counselor_count": 13,
            },
            "Løgstør": {
                "amt": "Nordjyllands Amt",
                "counselor_count": 19,
            },
            "Thyborøn-Harboøre": {
                "amt": "Ringkøbing Amt",
                "counselor_count": 13,
            },
            "Sydfalster": {
                "amt": "Storstrøms Amt",
                "counselor_count": 15,
            },
            "Jelling": {
                "amt": "Vejle Amt",
                "counselor_count": 13,
            },
            "Langeskov": {
                "amt": "Fyns Amt",
                "counselor_count": 13,
            },
            "Nykøbing-Rørvig": {
                "amt": "Vestsjællands Amt",
                "counselor_count": 15,
            },
            "Ørbæk": {
                "amt": "Fyns Amt",
                "counselor_count": 15,
            },
            "Holmsland": {
                "amt": "Ringkøbing Amt",
                "counselor_count": 11,
            },
            "Broby": {
                "amt": "Fyns Amt",
                "counselor_count": 15,
            },
            "Aulum-Haderup": {
                "amt": "Ringkøbing Amt",
                "counselor_count": 15,
            },
            "Lunderskov": {
                "amt": "Vejle Amt",
                "counselor_count": 13,
            },
            "Læsø": {
                "amt": "Nordjyllands Amt",
                "counselor_count": 9,
            },
            "Fanø": {
                "amt": "Ribe Amt",
                "counselor_count": 11,
            },
        }

        # Check if any of these missing municipalities should be added
        for missing_name, data in missing_municipalities.items():
            if missing_name not in df["name"].values:
                logger.info(f"Adding missing municipality: {missing_name}")

                # Create a new row for this municipality
                new_row = {
                    "name": missing_name,
                    "href": "",  # Placeholder href
                    "amt": data["amt"],
                    "counselor_count": data["counselor_count"],
                    "year": year,
                    # Other columns will be set to NaN by default
                }

                # Append to the dataframe
                df.loc[len(df)] = new_row


def log_statistics_summary(stats_counter, df):
    """Log a summary of the scraping statistics"""
    # Show some statistics
    stats_columns = ["stemmeberettigede", "stemmeprocent", "optalte_stemmer"]
    missing_data = df[stats_columns].isna().sum()
    logger.info(f"\nMissing data count: {missing_data}")

    valid_counts = len(df) - missing_data
    logger.info(f"Successfully scraped data: {valid_counts}")

    # Log statistics summary
    logger.info("\n===== SCRAPING STATISTICS =====")
    logger.info(f"Total kommuner processed: {stats_counter['total_kommuner']}")

    if stats_counter["total_kommuner"] > 0:
        success_percentage = (
            stats_counter["fully_successful"] / stats_counter["total_kommuner"] * 100
        )
        logger.info(
            f"Fully successful extractions: {stats_counter['fully_successful']}/{stats_counter['total_kommuner']} "
            + f"({success_percentage:.1f}%)"
        )

    logger.info(f"HTML fetch failures: {stats_counter['html_fetch_failures']}")
    logger.info(
        f"Missing stemmeberettigede: {stats_counter['missing_stemmeberettigede']}"
    )
    logger.info(f"Missing stemmeprocent: {stats_counter['missing_stemmeprocent']}")
    logger.info(f"Missing optalte_stemmer: {stats_counter['missing_optalte_stemmer']}")
    logger.info(f"Missing counselor_count: {stats_counter['missing_counselor_count']}")
    logger.info("==============================")


def main():
    """Main function to scrape multiple election years and combine results"""
    years = [2001, 2005, 2009]
    all_data = []

    for year in years:
        year_df = scrape_election_year(year)
        if year_df is not None:
            all_data.append(year_df)

    # Combine all data frames
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Fix encoding for Danish characters (final pass)
        fix_danish_encoding(combined_df)

        # Save the combined data
        output_dir = Path("data/kmdvalg")
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_output_file = f"{output_dir}/kommune_data_combined.csv"
        combined_df.to_csv(combined_output_file, index=False, encoding="utf-8-sig")
        logger.info(
            f"Saved combined data with {len(combined_df)} records to {combined_output_file}"
        )

        # Print summary of combined data
        logger.info("\n===== COMBINED DATA SUMMARY =====")
        by_year = combined_df.groupby("year").size()
        logger.info(f"Records by year:\n{by_year}")
        logger.info("===============================")


if __name__ == "__main__":
    main()
