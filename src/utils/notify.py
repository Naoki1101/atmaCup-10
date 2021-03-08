import os
import datetime
from typing import List, Dict

import requests
import dataclasses
import pandas as pd
from notion.client import NotionClient, CollectionRowBlock
from notion.collection import NotionDate


@dataclasses.dataclass
class Notion:
    params: str

    def __post_init__(self) -> None:
        self.client = NotionClient(token_v2=self.params.token)
        self.url = self.params.url

    def get_table(self, dropna: bool = False) -> pd.DataFrame:
        """
        テーブル情報を取得する
        Parameters
        ----------
        dropna: bool
            欠損しているデータも取得するかどうか
        Returns
        -------
        table_df: pd.DataFrame
            抽出したdataframe
        """
        table = self.client.get_collection_view(self.url)

        rows = []
        for row in table.collection.get_rows():
            rows.append(self._get_row_item(row))

        table_df = pd.DataFrame(rows, columns=list(row.get_all_properties().keys()))
        if dropna:
            table_df = table_df.dropna().reset_index(drop=True)
        return table_df

    def _get_row_item(self, row) -> List:
        """
        レコードごとのデータを取得する
        Parameters
        ----------
        dropna: bool
            欠損しているデータも取得するかどうか
        Returns
        -------
        items: List
            抽出したデータのリスト
        """
        items = []
        for col, item in row.get_all_properties().items():
            type_ = type(item)
            item = row.get_property(identifier=col)
            if type_ not in [list, NotionDate]:
                items.append(item)
            elif type_ == list:
                items.append(" ".join(item))
            elif type_ == NotionDate:
                items.append(item.__dict__["start"])
        return items

    def insert_rows(self, item_dict: Dict) -> None:
        """
        レコードごとのデータを取得する
        Parameters
        ----------
        item_dict: Dict
        """
        table = self.client.get_collection_view(self.url)
        row = self._create_new_record(table)

        for col_name, value in item_dict.items():
            row.set_property(identifier=col_name, val=value)

    def _create_new_record(self, table):
        """
        ???
        Parameters
        ----------
        table: Dict
        Returns
        -------
        row: ???
            ???
        """
        row_id = self.client.create_record(
            "block", parent=table.collection, type="page"
        )
        row = CollectionRowBlock(self.client, row_id)

        with self.client.as_atomic_transaction():
            for view in self.client.get_block(table.get("parent_id")).views:
                view.set("page_sort", view.get("page_sort", []) + [row_id])

        return row


@dataclasses.dataclass
class Notificator:
    run_name: str
    model_name: str
    cv: float
    process_time: float
    comment: str
    params: Dict

    def send_line(self) -> None:
        """
        LINEに通知を送る
        """
        if self.params.line.token is not None:
            endpoint = "https://notify-api.line.me/api/notify"
            message = (
                f"""\n{self.run_name}\ncv: {self.cv}\ntime: {self.process_time}[min]"""
            )
            payload = {"message": message}
            headers = {"Authorization": "Bearer {}".format(self.params.line.token)}
            requests.post(endpoint, data=payload, headers=headers)

    def send_notion(self) -> None:
        """
        Notionに通知を送る
        """
        if os.environ.get("NOTION_TOKEN") is not None:
            notion = Notion(params=self.params.notion)
            notion.insert_rows(
                {
                    # "name": self.run_name,
                    # "created": datetime.datetime.now(),
                    "model": self.model_name,
                    "local_cv": self.cv,
                    "time": self.process_time,
                    "comment": self.comment,
                }
            )
