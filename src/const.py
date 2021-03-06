from pathlib import Path

INPUT_DATA_DIR = Path("../data/input")
OUTPUT_DATA_DIR = Path("../data/output")
PROCESSED_DATA_DIR = Path("../data/processed/")
FEATURE_DIR = Path("../data/features")
FEATURE_CUSTOM_DATASET_DIR = Path("../configs/feature/")
LOG_DIR = Path("../logs")

SAMPLE_SUB_PATH = INPUT_DATA_DIR / "atmacup10__sample_submission.csv"

ID_COLS = ["object_id", "art_series_id"]
TARGET_COLS = ["likes"]

NUMERICAL_FEATURES = [
    "dating_sorting_date",
    "dating_period",
    "dating_year_early",
    "dating_year_late",
]

CATEGORICAL_FEATURES = [
    "principal_maker",
    "principal_or_first_maker",
    "copyright_holder",
    "acquisition_method",
    "acquisition_credit_line",
    "dating_presenting_date",
]


TEXT_COLS = ["title", "description", "long_title", "sub_title", "more_title"]

DATE_COLS = ["acquisition_date"]
