import dataclasses
from typing import List, Tuple

import pandas as pd
import lda as lda_model
from sklearn.feature_extraction.text import CountVectorizer


@dataclasses.dataclass
class LDA:
    n_topics: int = 10
    n_iter: int = 1_000
    random_state: int = 0

    def _get_token(
        self, df: pd.DataFrame, index_name: str, token_name: str, unique: bool = True
    ) -> Tuple[List, List]:
        index_list = []
        token_list = []

        for index, sample_df in df.groupby(index_name):
            index_list.append(index)
            if unique:
                token = " ".join(list(sample_df[token_name].astype(str).unique()))
            else:
                token = " ".join(list(sample_df[token_name].astype(str).values))
            token_list.append(token)

        return index_list, token_list

    def get_topic_array(
        self, df: pd.DataFrame, index_name: str, token_name: str, unique: bool = True
    ) -> Tuple[List, List]:
        index_list, token_list = self._get_token(df, index_name, token_name, unique)

        vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        X = vectorizer.fit_transform(token_list)

        model = lda_model.LDA(
            n_topics=self.n_topics,
            n_iter=self.n_iter,
            random_state=self.random_state,
            alpha=0.5,
            eta=0.5,
        )
        model.fit(X)
        topic_array = model.transform(X)

        return topic_array, index_list
