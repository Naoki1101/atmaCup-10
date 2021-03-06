import sys
import colorsys

import numpy as np
import pandas as pd

sys.path.append("../src")
import const
from function import Aggregation
from feature_utils import save_features


def preprocess_palette(color_df: pd.DataFrame) -> pd.DataFrame:
    color_df["color_r_ratio"] = color_df["ratio"] * color_df["color_r"]
    color_df["color_g_ratio"] = color_df["ratio"] * color_df["color_g"]
    color_df["color_b_ratio"] = color_df["ratio"] * color_df["color_b"]

    hsv_array = get_hsv(color_df)
    color_df["color_h"] = hsv_array[:, 0]
    color_df["color_s"] = hsv_array[:, 1]
    color_df["color_v"] = hsv_array[:, 2]

    color_df["color_h_ratio"] = color_df["ratio"] * color_df["color_h"]
    color_df["color_s_ratio"] = color_df["ratio"] * color_df["color_s"]
    color_df["color_v_ratio"] = color_df["ratio"] * color_df["color_v"]

    return color_df


def get_hsv(color_df):
    hsv_array = np.zeros((len(color_df), 3))

    for idx in color_df.index:
        row = color_df.iloc[idx]
        hsv_array[idx, :] = colorsys.rgb_to_hsv(
            row["color_r"], row["color_g"], row["color_b"]
        )

    return hsv_array


def get_features(df: pd.DataFrame, palette_df: pd.DataFrame):
    features_df = pd.DataFrame()

    for col in [
        "ratio",
        "color_r",
        "color_g",
        "color_b",
        "color_h",
        "color_s",
        "color_v",
    ]:
        agg = Aggregation(
            by="object_id",
            columns=col,
            aggs={"min", "max", "mean", "median", "std"},
        )
        agg_df = agg.fit_transform(palette_df)
        agg_df["object_id"] = palette_df["object_id"]

        for agg_col in agg_df.columns:
            if agg_col != "object_id":
                le = dict(agg_df[["object_id", agg_col]].values)
                features_df[f"agg_palette_{agg_col}_by_object_id"] = df[
                    "object_id"
                ].map(le)

    for col in [
        "color_r_ratio",
        "color_g_ratio",
        "color_b_ratio",
        "color_h_ratio",
        "color_s_ratio",
        "color_v_ratio",
    ]:
        agg = Aggregation(
            by="object_id",
            columns=col,
            aggs={"sum"},
        )
        agg_df = agg.fit_transform(palette_df)
        agg_df["object_id"] = palette_df["object_id"]

        for agg_col in agg_df.columns:
            if agg_col != "object_id":
                le = dict(agg_df[["object_id", agg_col]].values)
                features_df[f"agg_palette_{agg_col}_by_object_id"] = df[
                    "object_id"
                ].map(le)

    return features_df


def main():
    train_df = pd.read_feather(const.INPUT_DATA_DIR / "train.feather")
    test_df = pd.read_feather(const.INPUT_DATA_DIR / "test.feather")
    palette_df = pd.read_feather(const.INPUT_DATA_DIR / "palette.feather")
    palette_df = preprocess_palette(palette_df)

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    whole_features_df = get_features(whole_df, palette_df)

    train_features_df = whole_features_df.iloc[: len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df) :]

    save_features(train_features_df, data_type="train")
    save_features(test_features_df, data_type="test")


if __name__ == "__main__":
    main()
