import sys

import pandas as pd

sys.path.append("../src")
import const
from feature_utils import save_features


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    features_df = pd.DataFrame()

    for col in const.TEXT_COLS:
        features_df[f"count_word_{col}"] = df[col].apply(
            lambda x: len(str(x).split(" "))
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
