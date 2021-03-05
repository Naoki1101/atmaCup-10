import os
import json
import yaml
from datetime import datetime
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import dataclasses
import numpy as np
import pandas as pd
import joblib
from collections import OrderedDict
from easydict import EasyDict as edict


class DataProcessor(metaclass=ABCMeta):
    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def save(self, path: str, data) -> None:
        pass


@dataclasses.dataclass
class YmlPrrocessor(DataProcessor):
    def load(self, path: str) -> edict:
        """
        yamlをロードする
        Parameters
        ----------
        path: str
            取得データのpath
        Returns
        -------
        yaml_file: edict
            yaml
        """
        with open(path, "r") as yf:
            yaml_file = yaml.load(yf, Loader=yaml.SafeLoader)
        yaml_file = edict(yaml_file)
        return yaml_file

    def save(self, path: str, data: edict) -> None:
        """
        yamlを保存する
        Parameters
        ----------
        path: str
            yamlデータの保存先
        data: edict
            保存するデータ
        """

        def represent_odict(dumper, instance):
            return dumper.represent_mapping("tag:yaml.org,2002:map", instance.items())

        yaml.add_representer(OrderedDict, represent_odict)
        yaml.add_representer(edict, represent_odict)

        with open(path, "w") as yf:
            yf.write(yaml.dump(OrderedDict(data), default_flow_style=False))


@dataclasses.dataclass
class CsvProcessor(DataProcessor):
    sep: str = ","

    def load(self, path: str) -> pd.DataFrame:
        """
        csvをロードする
        Parameters
        ----------
        path: str
            取得データのpath
        sep: str
            区切り文字
            tsvデータの場合は、sep='\t'を指定する
        Returns
        -------
        data: pd.DataFrame
            抽出したdataframe
        """
        data = pd.read_csv(path, sep=self.sep)
        return data

    def save(self, path: str, data: pd.DataFrame) -> None:
        """
        csv形式で保存する
        Parameters
        ----------
        path: str
            データの保存先path
        data: pd.DataFrame
            保存するデータ
        """
        data.to_csv(path, index=False)


@dataclasses.dataclass
class FeatherProcessor(DataProcessor):
    def load(self, path: str) -> pd.DataFrame:
        """
        featherをロードする
        Parameters
        ----------
        path: str
            取得データのpath
        Returns
        -------
        data: pd.DataFrame
            抽出したdataframe
        """
        data = pd.read_feather(path)
        return data

    def save(self, path: str, data: pd.DataFrame):
        """
        feather形式で保存する
        Parameters
        ----------
        path: str
            データの保存先path
        data: pd.DataFrame
            保存するデータ
        """
        data.to_feather(path)


@dataclasses.dataclass
class PickleProcessor(DataProcessor):
    def load(self, path: str):
        """
        pickleをロードする
        Parameters
        ----------
        path: str
            取得データのpath
        Returns
        -------
        data: いろいろ
            抽出したデータ
        """
        data = joblib.load(path)
        return data

    def save(self, path: str, data) -> None:
        """
        pickle形式で保存する
        Parameters
        ----------
        path: str
            データの保存先path
        data: いろいろ
            保存するデータ
        """
        joblib.dump(data, path, compress=True)


@dataclasses.dataclass
class NpyProcessor(DataProcessor):
    def load(self, path: str) -> np.array:
        """
        npyをロードする
        Parameters
        ----------
        path: str
            取得データのpath
        Returns
        -------
        data: np.array
            抽出したデータ
        """
        data = np.load(path)
        return data

    def save(self, path: str, data: np.array) -> None:
        """
        npy形式で保存する
        Parameters
        ----------
        path: str
            データの保存先path
        data: np.array
            保存するデータ
        """
        np.save(path, data)


@dataclasses.dataclass
class JsonProcessor(DataProcessor):
    def load(self, path: str) -> OrderedDict:
        """
        jsonをロードする
        Parameters
        ----------
        path: str
            取得データのpath
        Returns
        -------
        data: OrderedDict
            抽出したデータ
        """
        with open(path, "r") as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
        return data

    def save(self, path: str, data: Dict) -> None:
        """
        json形式で保存する
        Parameters
        ----------
        path: str
            データの保存先path
        data: OrderedDict
            保存するデータ
        """
        with open(path, "w") as f:
            json.dump(data, f, indent=4)


@dataclasses.dataclass
class SqlProcessor(DataProcessor):
    def load(self, path: str) -> str:
        """
        queryをロードする
        Parameters
        ----------
        path: str
            SQLクエリのpath
        Returns
        -------
        query: str
            SQLクエリ
        """
        with open(path, "r") as f:
            query = f.read()

        return query

    def save(self, path: str, data: str) -> None:
        pass


@dataclasses.dataclass
class DataHandler:
    def __post_init__(self):
        self.data_encoder = {
            ".yml": YmlPrrocessor(),
            ".csv": CsvProcessor(sep=","),
            ".tsv": CsvProcessor(sep="\t"),
            ".feather": FeatherProcessor(),
            ".pkl": PickleProcessor(),
            ".npy": NpyProcessor(),
            ".json": JsonProcessor(),
            ".sql": SqlProcessor(),
        }

    def load(self, path: str):
        """
        データをロードする
        Parameters
        ----------
        path: str
            取得データのpath
        Returns
        -------
        data: いろいろ
        """
        extension = self._extract_extension(path)
        data = self.data_encoder[extension].load(path)
        return data

    def save(self, path: str, data) -> None:
        """
        データを保存する
        Parameters
        ----------
        path: str
            データの保存先path
        data: いろいろ
            保存するデータ
        """
        extension = self._extract_extension(path)
        self.data_encoder[extension].save(path, data)

    def _extract_extension(self, path: str) -> str:
        """
        拡張子を取得する
        Parameters
        ----------
        path: str
            取得データのpath
        Returns
        -------
        extention: str
            拡張子
        """
        extention = os.path.splitext(path)[1]
        return extention


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    データの省メモリ化
    Parameters
    ----------
    df: pd.DataFrame
        データ
    Returns
    -------
    df: pd.DataFrame
        データ
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    # print("column = ", len(df.columns))
    for i, col in enumerate(df.columns):
        try:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)

                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)

                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)

                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df[col] = df[col].astype(np.int32)

                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float32)

                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)

                    else:
                        df[col] = df[col].astype(np.float32)

        except:
            continue

    end_mem = df.memory_usage().sum() / 1024 ** 2
    decreased_mem = 100 * (start_mem - end_mem) / start_mem
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(decreased_mem))

    return df


def transform_dtype(df: pd.DataFrame, dtype_dict: Dict) -> pd.DataFrame:
    """
    dataframeの型を一括で変更する
    Parameters
    ----------
    df : pd.DataFrame
        dataframe
    dtype_dict : Dict
        カラム名 -> 型の辞書
    Returns
    -------
    df: pd.DataFrame
        dataframe
    """
    df = df[list(dtype_dict.keys())]
    df = df.astype(dtype_dict)
    df.reset_index(drop=True, inplace=True)
    return df