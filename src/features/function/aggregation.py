import dataclasses
from typing import List, Dict

import pandas as pd


@dataclasses.dataclass
class Aggregation:
    by: str
    columns: str
    aggs: Dict

    def __post_init__(self):
        self.agg_df = None
        self.output_df = pd.DataFrame()

    def fit(self, df: pd.DataFrame) -> None:
        gp = df.groupby(by=self.by)
        self.agg_df = gp[self.columns].agg(self.aggs).add_prefix(f"{self.columns}_")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.agg_df.columns:
            encoder = dict(self.agg_df[col])
            self.output_df[col] = df[self.by].map(encoder)
        return self.output_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        self.output_df = self.transform(df)
        return self.output_df

    def get_columns(self) -> List:
        return [f"{self.columns}_{agg}" for agg in self.aggs]