import os
import re
import pandas as pd

DATASET_DIR = "dataset"


def normalize_filename(filename):
    return re.sub(r"[^a-z0-9]", "", filename.lower())


def find_file(possible_names):
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError("Dataset folder not found!")

    actual_files = os.listdir(DATASET_DIR)

    for wanted in possible_names:
        wanted_normalized = normalize_filename(wanted)

        for actual in actual_files:
            if normalize_filename(actual) == wanted_normalized:
                path = os.path.join(DATASET_DIR, actual)
                print("Found:", actual)
                return path

    raise FileNotFoundError(
        "Could not find dataset. Tried: " + str(possible_names)
    )


def clean_name(series):
    return (
        series.astype(str)
        .str.strip()
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )


def prepare_training_data():

    print("\nLoading your datasets...\n")

    # Locate the three existing datasets
    dfsi_path = find_file([
        "DFSI_clean.csv",
        "DFSI-cleaned.csv",
        "DFSI_cleaned.csv",
        "DFSI - cleaned.csv"
    ])

    impact_path = find_file([
        "District_FloodImpact_clean.csv",
        "District-FloodImpact-cleaned.csv",
        "district - flood impact - cleaned.csv",
        "district-flood-impact-cleaned.csv"
    ])

    area_path = find_file([
        "District_FloodedArea.csv",
        "district-flooded-area.csv",
        "district - flooded area.csv"
    ])

    # Read datasets
    dfsi = pd.read_csv(dfsi_path)
    impact = pd.read_csv(impact_path)
    area = pd.read_csv(area_path)

    # Clean column headings
    for frame in [dfsi, impact, area]:
        frame.columns = frame.columns.str.strip()

    print("\nDFSI columns:", list(dfsi.columns))
    print("Impact columns:", list(impact.columns))
    print("Area columns:", list(area.columns))

    # Required columns
    required_dfsi = [
        "Dist_Name",
        "State_Name",
        "DFSI"
    ]

    required_impact = [
        "Dist_Name",
        "Human_fatality",
        "Human_injured",
        "Population",
        "Mean_Flood_Durationt"
    ]

    required_area = [
        "Dist_Name",
        "Corrected_Percent_Flooded_Area"
    ]

    # Validate columns
    for column in required_dfsi:
        if column not in dfsi.columns:
            raise ValueError("Missing DFSI column: " + column)

    for column in required_impact:
        if column not in impact.columns:
            raise ValueError("Missing impact column: " + column)

    for column in required_area:
        if column not in area.columns:
            raise ValueError("Missing area column: " + column)

    # Normalize district names
    dfsi["district_key"] = clean_name(dfsi["Dist_Name"])
    impact["district_key"] = clean_name(impact["Dist_Name"])
    area["district_key"] = clean_name(area["Dist_Name"])

    # Convert numeric columns
    dfsi["DFSI"] = pd.to_numeric(
        dfsi["DFSI"], errors="coerce"
    )

    impact_columns = [
        "Human_fatality",
        "Human_injured",
        "Population",
        "Mean_Flood_Durationt"
    ]

    for column in impact_columns:
        impact[column] = pd.to_numeric(
            impact[column], errors="coerce"
        )

    area["Corrected_Percent_Flooded_Area"] = pd.to_numeric(
        area["Corrected_Percent_Flooded_Area"],
        errors="coerce"
    )

    # Remove missing DFSI values
    dfsi = dfsi.dropna(subset=["DFSI"])

    # Exclude duplicate district names to avoid ambiguous matches
    dfsi = dfsi[
        ~dfsi["district_key"].duplicated(keep=False)
    ]

    impact = impact[
        ~impact["district_key"].duplicated(keep=False)
    ]

    area = area[
        ~area["district_key"].duplicated(keep=False)
    ]

    print("\nMatching district records...")

    # Merge the three datasets
    merged = dfsi[
        ["Dist_Name", "State_Name", "district_key", "DFSI"]
    ].merge(
        impact[
            ["district_key"] + impact_columns
        ],
        on="district_key",
        how="inner",
        validate="one_to_one"
    ).merge(
        area[
            ["district_key", "Corrected_Percent_Flooded_Area"]
        ],
        on="district_key",
        how="inner",
        validate="one_to_one"
    )

    print("Matched records:", len(merged))

    if len(merged) < 10:
        raise ValueError(
            "Not enough reliable district matches. "
            "At least 10 records are required."
        )

    # Features
    features = [
        "Human_fatality",
        "Human_injured",
        "Population",
        "Mean_Flood_Durationt",
        "Corrected_Percent_Flooded_Area"
    ]

    # Fill missing feature values using medians
    for column in features:
        median_value = merged[column].median()

        if pd.isna(median_value):
            raise ValueError(
                "Cannot calculate median for " + column
            )

        merged[column] = merged[column].fillna(median_value)

    # Create historical risk categories from DFSI
    low_limit = merged["DFSI"].quantile(0.33)
    high_limit = merged["DFSI"].quantile(0.67)

    def assign_severity(value):

        if value <= low_limit:
            return "Low"

        elif value <= high_limit:
            return "Medium"

        else:
            return "High"

    merged["severity"] = merged["DFSI"].apply(
        assign_severity
    )

    # Prepare final training dataset
    output = merged[
        features + ["severity"]
    ].copy()

    # Save CSV
    output_path = os.path.join(
        DATASET_DIR,
        "disaster_data.csv"
    )

    output.to_csv(output_path, index=False)

    print("\n================================")
    print("TRAINING DATA CREATED SUCCESSFULLY")
    print("================================")

    print("Total records:", len(output))

    print("\nRisk category distribution:")
    print(output["severity"].value_counts())

    print("\nSaved file:", output_path)


if __name__ == "__main__":
    prepare_training_data()

   