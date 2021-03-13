import sys

import pandas as pd

sys.path.append("../src")
import const
from function import Word2Vec, Aggregation
from feature_utils import save_features

OUT_DIM = 20


def get_features(df: pd.DataFrame, material_df: pd.DataFrame) -> pd.DataFrame:
    features_df = pd.DataFrame()

    w2v = Word2Vec()
    w2v_array, vocab_keys = w2v.get_w2v_array(
        material_df, index_name="object_id", token_name="name", out_dim=OUT_DIM
    )

    for col_idx in range(OUT_DIM):
        le = dict(zip(vocab_keys, w2v_array[:, col_idx]))
        material_df[f"w2v_{col_idx}"] = material_df["name"].map(le)

    for col_idx in range(OUT_DIM):
        agg = Aggregation(
            by="object_id",
            columns=f"w2v_{col_idx}",
            aggs={"min", "max", "mean", "median", "std"},
        )
        agg_df = agg.fit_transform(material_df)
        agg_df["object_id"] = material_df["object_id"]

        for agg_col in agg_df.columns:
            if agg_col != "object_id":
                le = dict(agg_df[["object_id", agg_col]].values)
                features_df[f"w2v_agg_material_{agg_col}_by_object_id"] = df[
                    "object_id"
                ].map(le)

    return features_df


def main():
    train_df = pd.read_feather(const.INPUT_DATA_DIR / "train.feather")
    test_df = pd.read_feather(const.INPUT_DATA_DIR / "test.feather")
    material_df = pd.read_feather(const.INPUT_DATA_DIR / "material.feather")

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    whole_features_df = get_features(whole_df, material_df)

    train_features_df = whole_features_df.iloc[: len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df) :]

    save_features(train_features_df, data_type="train")
    save_features(test_features_df, data_type="test")


if __name__ == "__main__":
    main()
