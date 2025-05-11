from bs4 import BeautifulSoup
import pandas as pd
from data_exploration.kmdvalg_scraper import fetch_html, extract_kommune_name_from_page


def test_kommune_name_extraction(urls):
    """
    Test kommune name extraction with specific URLs.

    Args:
        urls: List of URLs to test

    Returns:
        DataFrame with test results
    """
    results = []

    for url in urls:
        print(f"Testing URL: {url}")

        # Fetch HTML content
        html_content = fetch_html(url)

        if html_content:
            # Extract kommune name
            kommune_name = extract_kommune_name_from_page(html_content)

            # Get raw soup object for diagnostic information
            soup = BeautifulSoup(html_content, "html.parser")
            title = soup.find("title")
            title_text = title.get_text().strip() if title else "No title found"

            # Store results
            results.append(
                {
                    "url": url,
                    "extracted_name": kommune_name,
                    "page_title": title_text,
                    "successful": kommune_name is not None,
                }
            )
        else:
            results.append(
                {
                    "url": url,
                    "extracted_name": None,
                    "page_title": "Failed to fetch",
                    "successful": False,
                }
            )

    # Convert to DataFrame for better visualization
    results_df = pd.DataFrame(results)
    print(f"\nSuccess rate: {results_df['successful'].mean() * 100:.1f}%")
    return results_df


# Usage example
if __name__ == "__main__":
    test_urls = [
        "https://www.kmdvalg.dk/kv/2009/k84733492.htm",
        "https://www.kmdvalg.dk/kv/2009/k84979269.htm",
        "https://www.kmdvalg.dk/kv/2009/k84982260.htm",
        "https://www.kmdvalg.dk/kv/2009/k84982217.htm",
        "https://www.kmdvalg.dk/kv/2009/k84979316.htm",
        # Add more test URLs here
    ]

    results = test_kommune_name_extraction(test_urls)
    print(results)
