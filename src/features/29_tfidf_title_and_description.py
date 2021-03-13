import sys

import pandas as pd
import texthero as hero
from texthero import preprocessing

sys.path.append("../src")
import const
from function import TfidfVectorizer, Decomposer
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

    for col in ["title", "description", "long_title"]:
        df[col].fillna("NaN", inplace=True)
        df[f"{col}_normalized"] = hero.clean(df[col], custom_pipeline)

    return df


def get_features(df):
    features_df = pd.DataFrame()

    for col in ["title", "description", "long_title"]:
        tfidf = TfidfVectorizer()
        tfidf_sparse_matrix = tfidf.get_tfidf_array(
            df=df, index_name="object_id", token_name=f"{col}_normalized"
        )

        for method in ["TSNE", "UMAP"]:
            decomp = Decomposer(method=method, n_components=2)
            tfidf_decomp_array = decomp.fit_transform(tfidf_sparse_matrix)

            for i in range(2):
                features_df[f"tfidf_{col}_{method.lower()}_{i}"] = tfidf_decomp_array[
                    :, i
                ]

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
