import sys

import pandas as pd

sys.path.append("../src")
import const
from feature_utils import save_features


def preprocess_object_collection(object_collection_df: pd.DataFrame) -> pd.DataFrame:
    object_collection_df = pd.crosstab(
        object_collection_df["object_id"], object_collection_df["name"]
    )
    object_collection_df.columns = [
        col.replace(" ", "_") for col in list(object_collection_df.columns)
    ]

    return object_collection_df


def get_features(df: pd.DataFrame, object_collection_df: pd.DataFrame):
    features_df = pd.DataFrame()

    for col in object_collection_df.columns:
        le = dict(object_collection_df[col])
        features_df[col] = df["object_id"].map(le)

    return features_df


def main():
    train_df = pd.read_feather(const.INPUT_DATA_DIR / "train.feather")
    test_df = pd.read_feather(const.INPUT_DATA_DIR / "test.feather")
    object_collection_df = pd.read_feather(
        const.INPUT_DATA_DIR / "object_collection.feather"
    )
    object_collection_df = preprocess_object_collection(object_collection_df)

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    whole_features_df = get_features(whole_df, object_collection_df)

    train_features_df = whole_features_df.iloc[: len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df) :]

    save_features(train_features_df, data_type="train")
    save_features(test_features_df, data_type="test")


if __name__ == "__main__":
    main()
