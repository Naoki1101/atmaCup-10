import sys

import pandas as pd

sys.path.append("../src")
import const
from function import Aggregation
from feature_utils import save_features

s = 1e-5


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    features_df = pd.DataFrame()

    for cat_col in const.CATEGORICAL_FEATURES:
        for num_col in const.NUMERICAL_FEATURES:
            agg = Aggregation(
                by=cat_col,
                columns=num_col,
                aggs={"min", "max", "mean", "median", "std"},
            )
            agg_df = agg.fit_transform(df)

            for agg_col in agg_df.columns:
                features_df[f"agg_{agg_col}_by_{cat_col}"] = agg_df[agg_col]

            features_df[f"agg_{num_col}_div_mean_by_{cat_col}"] = df[num_col] / (
                agg_df[f"{num_col}_mean"] + s
            )
            features_df[f"agg_{num_col}_div_median_by_{cat_col}"] = df[num_col] / (
                agg_df[f"{num_col}_median"] + s
            )
            features_df[f"agg_{num_col}_div_min_by_{cat_col}"] = df[num_col] / (
                agg_df[f"{num_col}_min"] + s
            )
            features_df[f"agg_{num_col}_div_max_by_{cat_col}"] = df[num_col] / (
                agg_df[f"{num_col}_max"] + s
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
