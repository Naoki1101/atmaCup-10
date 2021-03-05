import sys

import pandas as pd

sys.path.append("../src")
import const


def main():
    train_df = pd.read_feather(const.INPUT_DATA_DIR / "train.feather")

    for col in const.ID_COLS + const.TARGET_COLS:
        train_df[[col]].to_feather(const.FEATURE_DIR / f"{col}.feather")


if __name__ == "__main__":
    main()
