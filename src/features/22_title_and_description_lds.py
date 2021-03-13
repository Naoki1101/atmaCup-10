import sys

import pandas as pd

sys.path.append("../src")
import const
from function import LDA
from feature_utils import save_features

TOPIC_NUM = 10


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    features_df = pd.DataFrame()

    df["description_and_long_title"] = df["description"] + df["long_title"]

    for col in [
        "title",
        "description",
        "long_title",
        "more_title",
        "description_and_long_title",
    ]:
        lda = LDA(n_topics=TOPIC_NUM, n_iter=1_000, random_state=0)
        topic_array, index_list = lda.get_topic_array(
            df, index_name="object_id", token_name=col
        )

        for i in range(TOPIC_NUM):
            features_df[f"lda_{col}_by_object_id_{i}"] = topic_array[:, i]

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
