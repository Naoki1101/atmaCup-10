import dataclasses

import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

from utils.data import DataHandler

dh = DataHandler()


@dataclasses.dataclass
class _BaseEncoder(metaclass=ABCMeta):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def fit_transform(self):
        pass


@dataclasses.dataclass
class LabelEncoder(_BaseEncoder):
    init_value: int = 0

    def fit(self, feat_array: pd.Series) -> None:
        self.levels = np.unique(feat_array.astype(str))
        self.encoder = {v: self.init_value + i for i, v in enumerate(self.levels)}

    def transform(self, feat_array: pd.Series) -> pd.Series:
        encoded_feat_array = feat_array.map(self.encoder)
        return encoded_feat_array

    def fit_transform(self, feat_array: pd.Series) -> pd.Series:
        self.fit(feat_array)
        encoded_feat_array = self.transform(feat_array)

        return encoded_feat_array


@dataclasses.dataclass
class CountEncoder(_BaseEncoder):
    def fit(self, feat_array: pd.Series) -> None:
        self.levels = feat_array.value_counts().index
        self.encoder = dict(feat_array.value_counts())

    def transform(self, feat_array: pd.Series) -> pd.Series:
        encoded_feat_array = feat_array.map(self.encoder)
        return encoded_feat_array

    def fit_transform(self, feat_array: pd.Series) -> pd.Series:
        self.fit(feat_array)
        encoded_feat_array = self.transform(feat_array)

        return encoded_feat_array


@dataclasses.dataclass
class TargetEncoder(_BaseEncoder):
    folds: pd.DataFrame

    def __post_init__(self):
        super(TargetEncoder, self).__init__()
        self.all_fold = self.folds.columns
        self.feature_name = None
        self.target_name = None
        self.encoder = {}

    def fit(self, values: pd.Series, target: pd.Series) -> None:
        self.feature_name = values.name
        self.target_name = target.name
        df = pd.concat([values, target], axis=1)
        for col in self.all_fold:
            df_fold = self._get_df(df, col)
            self.encoder[col] = self._get_encoder(df_fold)

    def transform(self, values: pd.Series) -> pd.Series:
        values_encoded = np.zeros(len(values))
        for fold_, encoder in self.encoder.items():
            values_encoded += values.map(encoder) / len(self.all_fold)
        return values_encoded

    def fit_transform(self, values: pd.Series, target: pd.Series) -> pd.Series:
        self.feature_name = values.name
        self.target_name = target.name
        values_encoded = np.zeros(len(values))
        df = pd.concat([values, target], axis=1)
        for col in self.all_fold:
            val_idx = self.folds[self.folds[col] > 0].index
            train_df = self._get_df(df, col)
            self.encoder[col] = self._get_encoder(train_df)
            values_encoded[val_idx] += values[val_idx].map(self.encoder[col])
        return values_encoded

    def _get_df(self, df, col):
        return df[self.folds[col] == 0]

    def _get_encoder(self, df):
        return df.groupby(self.feature_name)[self.target_name].mean().to_dict()


class OneHotEncoder:
    def __init__(self):
        self.cat_features = None

    def fit(self, cat_features):
        self.cat_features = cat_features

    def transform(self, df):
        return pd.get_dummies(data=df, columns=self.cat_features)

    def fit_transform(self, df, cat_features):
        self.cat_features = cat_features
        return pd.get_dummies(data=df, columns=self.cat_features)