import pandas as pd
import ast
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load the datasets
df_2001 = pd.read_csv("data/kmdvalg/kommune_data_2001.csv")
df_2005 = pd.read_csv("data/kmdvalg/kommune_data_2005.csv")


# Function for normalizing municipality names
def normalize_name(name):
    # Remove common additions and standardize format
    name = re.sub(r"\s+Kommune(\s+\(Udenfor amtsinddeling\))?$", "", name)
    # Replace "aa" with "å" for better matching
    name = name.replace("aa", "å")
    # Remove punctuation
    name = name.replace(".", "")
    return name


# Add manual corrections for known edge cases
manual_corrections = {
    "Københavns Kommune (Udenfor amtsinddeling)": "København",
    "Frederiksberg Kommune (Udenfor amtsinddeling)": "Frederiksberg",
    "Høje-Taastrup": "Høje-Tåstrup",
    "Lyngby-Taarbæk": "Lyngby-Tårbæk",
    "Nykøbing F.": "Nykøbing F",
    "Norborg": "Nordborg",
    # Bornholm municipalities
    "Nexø": "Bornholm",
    "Aakirkeby": "Bornholm",
    "Allinge-Gudhjem": "Bornholm",
    "Hasle": "Bornholm",
    "Rønne": "Bornholm",
}

# Apply manual corrections to 2001 dataset
df_2001["corrected_name"] = df_2001["name"].apply(
    lambda x: manual_corrections.get(x, x)
)

# Normalize names in the 2001 dataset
df_2001["normalized_name"] = df_2001["corrected_name"].apply(normalize_name)

# Create a mapping dataframe from the 2005 dataset
mapping_rows = []

# Process the 2005 dataset to extract merger information
for _, row in df_2005.iterrows():
    if row["merged"]:  # If this is a merged municipality
        # Convert string representation to list
        merged_list = ast.literal_eval(row["merged_municipalities"])

        # Add each pre-merger municipality to the mapping
        for pre_merger in merged_list:
            mapping_rows.append({"pre_merger": pre_merger, "post_merger": row["name"]})
    else:
        # For non-merged municipalities, map to themselves
        mapping_rows.append({"pre_merger": row["name"], "post_merger": row["name"]})

# Create a mapping dataframe
df_mapping = pd.DataFrame(mapping_rows)
df_mapping["normalized_pre_merger"] = df_mapping["pre_merger"].apply(normalize_name)

logger.info(f"Created mapping with {len(df_mapping)} pre-merger municipalities")

# Perform exact join instead of fuzzy join
joined_df = pd.merge(
    df_2001,
    df_mapping,
    left_on="normalized_name",
    right_on="normalized_pre_merger",
    how="left",
    suffixes=("", "_mapping"),
)

# Add post_merger_kommune column
joined_df["post_merger_kommune"] = joined_df["post_merger"]

# Add a match quality column
joined_df["match_quality"] = "no_match"
joined_df.loc[joined_df["post_merger"].notna(), "match_quality"] = "exact"

# Count statistics
exact_matches = (joined_df["match_quality"] == "exact").sum()
no_matches = (joined_df["match_quality"] == "no_match").sum()

# Print results
logger.info(f"Total municipalities in 2001 dataset: {len(df_2001)}")
logger.info(f"Total matched with exact match: {exact_matches}")
logger.info(f"Total unmatched municipalities: {no_matches}")

# Print unmatched municipalities
if no_matches > 0:
    logger.info("\nUnmatched municipalities from 2001 dataset:")
    unmatched = joined_df[joined_df["match_quality"] == "no_match"]["name"].tolist()
    for name in unmatched:
        logger.info(f"  {name}")

# Check for any missing pre-merger municipalities from the 2005 dataset
all_pre_merger = set(df_mapping["pre_merger"])
matched_pre_merger = set(joined_df[joined_df["match_quality"] == "exact"]["pre_merger"])
missing_pre_merger = all_pre_merger - matched_pre_merger

if missing_pre_merger:
    logger.info(
        "\nPre-merger municipalities from 2005 dataset without a match in 2001:"
    )
    for name in missing_pre_merger:
        post = df_mapping[df_mapping["pre_merger"] == name]["post_merger"].iloc[0]
        logger.info(f"  {name} -> {post}")

# Save the enhanced dataset (select only the necessary columns)
output_df = df_2001.copy()
output_df["post_merger_kommune"] = joined_df["post_merger_kommune"]
output_df["match_quality"] = joined_df[
    "match_quality"
]  # Keep for statistics calculation

# Calculate aggregated counselor counts (ignoring NAs and no matches)
valid_matches = output_df[output_df["match_quality"] != "no_match"]
counselor_counts = (
    valid_matches.groupby("post_merger_kommune")["counselor_count"].sum().reset_index()
)

# Remove match_quality column before saving
output_df = output_df.drop(columns=["match_quality"])

output_df.to_csv("data/kmdvalg/kommune_data_2001_with_post_merger.csv", index=False)
counselor_counts.to_csv("data/kmdvalg/aggregated_counselor_counts.csv", index=False)

logger.info("Saved enhanced dataset to kommune_data_2001_with_post_merger.csv")
logger.info("Saved aggregated counselor counts to aggregated_counselor_counts.csv")
