import sys

import pandas as pd

sys.path.append("../src")
import const
from feature_utils import save_features


def get_features(df: pd.DataFrame):
    features_df = pd.DataFrame()

    df["acquisition_year"] = pd.to_datetime(df["acquisition_date"])
    features_df["acquisition_year"] = df["acquisition_year"].dt.year

    features_df["acquisition_year_diff_dating_year_early"] = (
        df["acquisition_year"].dt.year - df["dating_year_early"]
    )
    features_df["acquisition_year_diff_dating_year_late"] = (
        df["acquisition_year"].dt.year - df["dating_year_late"]
    )

    return features_df


def main():
    train_df = pd.read_feather(const.INPUT_DATA_DIR / "train.feather")
    test_df = pd.read_feather(const.INPUT_DATA_DIR / "test.feather")

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    whole_features_df = get_features(whole_df)

    train_features_df = whole_features_df.iloc[: len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df) :]

    save_features(train_features_df, data_type="train")
    save_features(test_features_df, data_type="test")


if __name__ == "__main__":
    main()
