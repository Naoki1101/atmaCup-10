import sys

import pandas as pd

sys.path.append("../src")
import const
from feature_utils import save_features


def preprocess_principal_maker_occupation(
    principal_maker_occupation_df: pd.DataFrame,
) -> pd.DataFrame:
    principal_maker_occupation_df = pd.crosstab(
        index=principal_maker_occupation_df["id"],
        columns=principal_maker_occupation_df["name"],
    ).reset_index()

    principal_maker_occupation_df.columns = ["id"] + [
        f"principal_maker_occupation_{col}"
        for col in principal_maker_occupation_df.columns[1:]
    ]

    principal_maker_df = pd.read_csv("../data/input/principal_maker.csv")
    le = dict(principal_maker_df[["id", "object_id"]].values)
    principal_maker_occupation_df["object_id"] = principal_maker_occupation_df[
        "id"
    ].map(le)

    return principal_maker_occupation_df


def get_features(df: pd.DataFrame, principal_maker_occupation_df: pd.DataFrame):
    features_df = pd.DataFrame()

    for col in principal_maker_occupation_df.columns:
        if col != "object_id":
            le = dict(principal_maker_occupation_df[["object_id", col]].values)
            features_df[f"principal_maker_occupation_{col}"] = df["object_id"].map(le)

    return features_df


def main():
    train_df = pd.read_feather(const.INPUT_DATA_DIR / "train.feather")
    test_df = pd.read_feather(const.INPUT_DATA_DIR / "test.feather")
    principal_maker_occupation_df = pd.read_feather(
        const.INPUT_DATA_DIR / "principal_maker_occupation.feather"
    )
    principal_maker_occupation_df = preprocess_principal_maker_occupation(
        principal_maker_occupation_df
    )

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    whole_features_df = get_features(whole_df, principal_maker_occupation_df)

    train_features_df = whole_features_df.iloc[: len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df) :]

    save_features(train_features_df, data_type="train")
    save_features(test_features_df, data_type="test")


if __name__ == "__main__":
    main()
