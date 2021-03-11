import dataclasses
from typing import List, Union

import numpy as np
import pandas as pd
from gensim.models import word2vec


@dataclasses.dataclass
class Word2Vec:
    n_iter: int = 1_000
    window: int = 128
    seed: int = 0
    min_count: int = 1
    workers: int = -1

    def _get_token(
        self, df: pd.DataFrame, index_name: str, token_name: str, unique: bool = True
    ) -> Union[List, List]:
        index_list = []
        token_list = []

        for index, sample_df in df.groupby(index_name):
            index_list.append(index)
            if unique:
                token = list(sample_df[token_name].astype(str).unique())
            else:
                token = list(sample_df[token_name].astype(str).values)
            token_list.append(token)

        return index_list, token_list

    def get_w2v_array(
        self,
        df: pd.DataFrame,
        index_name: str,
        token_name: str,
        unique: bool = True,
        out_dim: int = 8,
    ) -> Union[List, List]:
        index_list, token_list = self._get_token(df, index_name, token_name, unique)

        token_list += self._get_reverse_token(token_list)
        index_list *= 2

        model = word2vec.Word2Vec(
            token_list,
            size=out_dim,
            iter=self.n_iter,
            window=self.window,
            seed=self.seed,
            min_count=self.min_count,
            workers=self.workers,
        )

        vocab_keys = list(model.wv.vocab.keys())
        w2v_array = np.zeros((len(vocab_keys), out_dim))

        for i, v in enumerate(vocab_keys):
            w2v_array[i, :] = model.wv[v]

        return w2v_array, vocab_keys

    def _get_reverse_token(self, token_list: List[List[str]]) -> List[List[str]]:
        reverse_token_list = []

        for token in token_list:
            reverse_token_list.append(token[::-1])

        return reverse_token_list
