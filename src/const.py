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
    # train / test
    "dating_sorting_date",
    "dating_period",
    "dating_year_early",
    "dating_year_late",
    # color
    "percentage",
    # palette
    "ratio",
    "color_r",
    "color_g",
    "color_b",
]

CATEGORICAL_FEATURES = [
    # train / test
    "principal_maker",
    "principal_or_first_maker",
    "copyright_holder",
    "acquisition_method",
    "acquisition_credit_line",
    "dating_presenting_date",
    # maker
    "place_of_birth",
    "date_of_birth",
    "date_of_death",
    "place_of_death",
    "nationality",
    # material
    "material_name",
    # principal_maker
    "qualification",
    "roles",
    "productionPlaces",
    "maker_name",
    # principal_maker_occupation
    "principal_maker_occupation_auctioneer",
    "principal_maker_occupation_bookseller",
    "principal_maker_occupation_designer",
    "principal_maker_occupation_draughtsman",
    "principal_maker_occupation_goldsmith",
    "principal_maker_occupation_instrument_maker",
    "principal_maker_occupation_jeweler",
    "principal_maker_occupation_painter",
    "principal_maker_occupation_print_maker",
    "principal_maker_occupation_printer",
    "principal_maker_occupation_publisher",
    "principal_maker_occupation_sculptor",
    # technique
    "technique_name",
    # historical_person
    "historical_person_name",
    # object_collection
    "object_collection_name",
    # production_place
    "production_place_name",
    # color
    "hex",
]


TEXT_COLS = ["title", "description", "long_title", "sub_title", "more_title"]

DATE_COLS = ["acquisition_date"]
