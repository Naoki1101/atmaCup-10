import sys

import pandas as pd

sys.path.append("../src")
import const
from function import CountEncoder
from feature_utils import save_features


def get_features(df: pd.DataFrame, maker_df: pd.DataFrame):
    features_df = pd.DataFrame()

    for col in ["place_of_birth", "place_of_death"]:
        le = CountEncoder()
        maker_df[f"ce_{col}"] = le.fit_transform(maker_df[col])

        m2n = dict(maker_df[["name", f"ce_{col}"]].values)
        features_df[f"ce_{col}"] = df["principal_maker"].map(m2n)

    return features_df


def main():
    train_df = pd.read_feather(const.INPUT_DATA_DIR / "train.feather")
    test_df = pd.read_feather(const.INPUT_DATA_DIR / "test.feather")
    maker_df = pd.read_feather(const.INPUT_DATA_DIR / "maker.feather")

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    whole_features_df = get_features(whole_df, maker_df)

    train_features_df = whole_features_df.iloc[: len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df) :]

    save_features(train_features_df, data_type="train")
    save_features(test_features_df, data_type="test")


if __name__ == "__main__":
    main()
