import sys

import pandas as pd

sys.path.append("../src")
import const
from feature_utils import save_features


def get_features(df: pd.DataFrame, principal_maker_df: pd.DataFrame):
    features_df = pd.DataFrame()

    for col in ["qualification", "roles"]:
        crosstab_df = pd.crosstab(
            principal_maker_df["object_id"], principal_maker_df["qualification"]
        )
        crosstab_df.columns = [
            col.replace(" ", "_") for col in list(crosstab_df.columns)
        ]

        for crosstab_col in crosstab_df.columns:
            le = dict(crosstab_df[crosstab_col])
            features_df[f"principal_maker_{crosstab_col}"] = df["object_id"].map(le)

    return features_df


def main():
    train_df = pd.read_feather(const.INPUT_DATA_DIR / "train.feather")
    test_df = pd.read_feather(const.INPUT_DATA_DIR / "test.feather")
    principal_maker_df = pd.read_csv(const.INPUT_DATA_DIR / "principal_maker.csv")

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    whole_features_df = get_features(whole_df, principal_maker_df)

    train_features_df = whole_features_df.iloc[: len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df) :]

    save_features(train_features_df, data_type="train")
    save_features(test_features_df, data_type="test")


if __name__ == "__main__":
    main()
