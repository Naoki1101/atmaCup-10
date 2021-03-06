import re
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.append("../src")
import const
from feature_utils import save_features


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    parsed_sub_title = parse_sub_title(df)
    df["sub_title_d"] = parsed_sub_title[:, 0]
    df["sub_title_h"] = parsed_sub_title[:, 1]
    df["sub_title_l"] = parsed_sub_title[:, 2]
    df["sub_title_t"] = parsed_sub_title[:, 3]
    df["sub_title_w"] = parsed_sub_title[:, 4]
    return df


def parse_sub_title(df: pd.DataFrame):
    sub_title_array = np.repeat(np.nan, 5 * len(df)).reshape(-1, 5)
    unit_array = np.ones((len(df), 5))

    p = r"\d+\.*\d*"

    for idx, row in enumerate(df["sub_title"].values):
        if type(row) == str:
            values = row.replace(" ", "").split("×")

            for v in values:
                if "d" in v:
                    col_idx = 0
                elif "h" in v:
                    col_idx = 1
                elif "l" in v:
                    col_idx = 2
                elif "t" in v:
                    col_idx = 3
                elif "w" in v:
                    col_idx = 4

                if re.search(p, v):
                    sub_title_array[idx, col_idx] = float(re.search(p, v).group(0))
                else:
                    sub_title_array[idx, col_idx] = None

                if "cm" in v:
                    unit_array[idx, col_idx] = 100

    sub_title_array = sub_title_array * unit_array

    return sub_title_array


def get_features(df: pd.DataFrame):
    features_df = pd.DataFrame()

    sub_title_cols = [
        "sub_title_d",
        "sub_title_h",
        "sub_title_l",
        "sub_title_t",
        "sub_title_w",
    ]

    for col in sub_title_cols:
        features_df[col] = df[col]

    comb_list = list(combinations(sub_title_cols, 2))
    for col1, col2 in comb_list:
        features_df[f"{col1}_multi_{col2}"] = df[col1] * df[col2]

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