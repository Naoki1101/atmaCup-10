import dataclasses
from typing import List, Tuple

import pandas as pd
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer


@dataclasses.dataclass
class TfidfVectorizer:
    def _get_token(
        self, df: pd.DataFrame, index_name: str, token_name: str, unique: bool = True
    ) -> Tuple[List, List]:
        token_list = []

        for index, sample_df in df.groupby(index_name):
            if unique:
                token = " ".join(list(sample_df[token_name].astype(str).unique()))
            else:
                token = " ".join(list(sample_df[token_name].astype(str).values))
            token_list.append(token)

        return token_list

    def get_tfidf_array(
        self,
        df: pd.DataFrame,
        index_name: str,
        token_name: str,
        unique: bool = True,
    ) -> csr_matrix:
        token_list = self._get_token(df, index_name, token_name, unique)

        vectorizer = SklearnTfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        sparse_matrix = vectorizer.fit_transform(token_list)

        return sparse_matrix
