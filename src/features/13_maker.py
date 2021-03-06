import sys

import pandas as pd

sys.path.append("../src")
import const
from feature_utils import save_features


def preprocess_maker(maker_df: pd.DataFrame):
    maker_df["equal_birth_and_death_place"] = (
        maker_df["place_of_birth"] == maker_df["date_of_birth"]
    ).astype(int)

    maker_df["year_of_birth"] = maker_df["date_of_birth"].apply(
        lambda x: int(x[:4]) if type(x) == str else x
    )
    maker_df["year_of_death"] = maker_df["date_of_death"].apply(
        lambda x: int(x[:4]) if type(x) == str else x
    )
    maker_df["age_of_death"] = maker_df["year_of_death"] - maker_df["year_of_birth"]

    return maker_df


def get_features(df: pd.DataFrame, maker_df: pd.DataFrame):
    features_df = pd.DataFrame()

    for col in [
        "equal_birth_and_death_place",
        "year_of_birth",
        "year_of_death",
        "age_of_death",
    ]:
        le = dict(maker_df[["name", col]].values)
        features_df[col] = df["principal_maker"].map(le)

    return features_df


def main():
    train_df = pd.read_feather(const.INPUT_DATA_DIR / "train.feather")
    test_df = pd.read_feather(const.INPUT_DATA_DIR / "test.feather")
    maker_df = pd.read_feather(const.INPUT_DATA_DIR / "maker.feather")
    maker_df = preprocess_maker(maker_df)

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    whole_features_df = get_features(whole_df, maker_df)

    train_features_df = whole_features_df.iloc[: len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df) :]

    save_features(train_features_df, data_type="train")
    save_features(test_features_df, data_type="test")


if __name__ == "__main__":
    main()
