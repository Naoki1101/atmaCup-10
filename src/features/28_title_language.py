import sys

import pandas as pd
import pycld2 as cld2
import texthero as hero
from texthero import preprocessing

sys.path.append("../src")
import const
from function import LabelEncoder
from feature_utils import save_features


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    custom_pipeline = [
        preprocessing.fillna,
        preprocessing.lowercase,
        preprocessing.remove_digits,
        preprocessing.remove_punctuation,
        preprocessing.remove_diacritics,
        preprocessing.remove_whitespace,
        preprocessing.remove_stopwords,
    ]

    df["title_normalized"] = hero.clean(df["title"], custom_pipeline)

    return df


def get_features(df):
    features_df = pd.DataFrame()

    df["title_language"] = df["title_normalized"].map(lambda x: cld2.detect(x)[2][0][1])

    le = LabelEncoder()
    features_df["le_title_language"] = le.fit_transform(df["title_language"])

    return features_df


def main():
    train_df = pd.read_feather(const.INPUT_DATA_DIR / "train.feather")
    test_df = pd.read_feather(const.INPUT_DATA_DIR / "test.feather")

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    whole_df = preprocess(whole_df)
    whole_features_df = get_features(whole_df)

    train_features_df = whole_features_df.iloc[: len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df) :]

    save_features(train_features_df, data_type="train")
    save_features(test_features_df, data_type="test")


if __name__ == "__main__":
    main()
