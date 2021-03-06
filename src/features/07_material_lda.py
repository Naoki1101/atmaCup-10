import sys

import pandas as pd

sys.path.append("../src")
import const
from function import LDA
from feature_utils import save_features

TOPIC_NUM = 10


def get_features(df: pd.DataFrame, material_df: pd.DataFrame):
    features_df = pd.DataFrame()

    lda = LDA(n_topics=TOPIC_NUM, n_iter=1_000, random_state=0)
    topic_array, index_list = lda.get_topic_array(
        material_df, index_name="object_id", token_name="name"
    )
    topic_df = pd.DataFrame(
        topic_array, columns=[f"topic_{i}" for i in range(TOPIC_NUM)]
    )
    topic_df.index = index_list

    for i, col in enumerate(topic_df.columns):
        le = dict(topic_df[col])
        features_df[f"lda_material_by_object_id_{i}"] = df["object_id"].map(le)

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
