import sys

import numpy as np
import pandas as pd
import texthero as hero
from texthero import preprocessing
from tqdm import tqdm

sys.path.append("../src")
import const
from function import BertSequenceVectorizer, Decomposer
from feature_utils import save_features

n_comp = 2


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

    for col in ["title", "description", "long_title", "acquisition_credit_line"]:
        df[col].fillna("NaN", inplace=True)
        df[f"{col}_normalized"] = hero.clean(df[col], custom_pipeline)

    return df


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    features_df = pd.DataFrame()

    for col in tqdm(["title", "description", "long_title", "acquisition_credit_line"]):
        BSV = BertSequenceVectorizer()
        df[f"{col}_feature"] = df[f"{col}_normalized"].apply(lambda x: BSV.vectorize(x))

        for method in ["TSNE", "UMAP"]:
            decomp = Decomposer(method=method, n_components=n_comp)
            bert_decomp_array = decomp.fit_transform(np.stack(df[f"{col}_feature"]))

            for i in range(n_comp):
                features_df[f"bert_{col}_{method.lower()}_{i}"] = bert_decomp_array[
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
