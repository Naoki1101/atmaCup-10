import sys

import pandas as pd

sys.path.append("../src")
import const
from function import LDA
from feature_utils import save_features

TOPIC_NUM = 10


def preprocess_historical_person(historical_person_df: pd.DataFrame) -> pd.DataFrame:
    historical_person_df["name"] = historical_person_df["name"].apply(
        lambda x: x.replace("(", "( ")
    )
    historical_person_df["name"] = historical_person_df["name"].apply(
        lambda x: x.replace(")", " )")
    )

    return historical_person_df


def get_features(df: pd.DataFrame, historical_person_df: pd.DataFrame):
    features_df = pd.DataFrame()

    lda = LDA(n_topics=TOPIC_NUM, n_iter=1_000, random_state=0)
    topic_array, index_list = lda.get_topic_array(
        historical_person_df, index_name="object_id", token_name="name"
    )
    topic_df = pd.DataFrame(
        topic_array, columns=[f"topic_{i}" for i in range(TOPIC_NUM)]
    )
    topic_df.index = index_list

    for i, col in enumerate(topic_df.columns):
        le = dict(topic_df[col])
        features_df[f"lda_historical_person_name_by_object_id_{i}"] = (
            df["object_id"].map(le).fillna(-1)
        )

    return features_df


def main():
    train_df = pd.read_feather(const.INPUT_DATA_DIR / "train.feather")
    test_df = pd.read_feather(const.INPUT_DATA_DIR / "test.feather")
    historical_person_df = pd.read_feather(
        const.INPUT_DATA_DIR / "historical_person.feather"
    )
    historical_person_df = preprocess_historical_person(historical_person_df)

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)
    whole_features_df = get_features(whole_df, historical_person_df)

    train_features_df = whole_features_df.iloc[: len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df) :]

    save_features(train_features_df, data_type="train")
    save_features(test_features_df, data_type="test")


if __name__ == "__main__":
    main()
