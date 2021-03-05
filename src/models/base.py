import dataclasses
from typing import List
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd


@dataclasses.dataclass
class Model(metaclass=ABCMeta):
    @abstractmethod
    def fit(
        self,
        tr_x: pd.DataFrame,
        tr_y: np.array,
        va_x: pd.DataFrame = None,
        va_y: np.array = None,
        cat_features: List = [],
    ) -> None:
        """
        学習を行う
        Parameters
        ----------
        tr_x : pd.DataFrame
            学習データの説明変数
        tr_y : np.array
            学習データの目的変数
        va_x : pd.DataFrame, optional
            評価データの説明変数, by default None
        va_y : np.array, optional
            評価データの目的変数, by default None
        cat_features : List, optional
            カテゴリ変数のカラムをまとめたリスト, by default None
        """
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame, cat_features: List = None) -> None:
        """
        推論を行う
        Parameters
        ----------
        te_x : pd.DataFrame
            テストデータの説明変数
        cat_features : List, optional
            カテゴリ変数のカラムをまとめたリスト, by default None
        """
        pass