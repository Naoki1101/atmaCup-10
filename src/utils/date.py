import dataclasses
from datetime import datetime
from dateutil.relativedelta import relativedelta


@dataclasses.dataclass
class DateMethod:
    """
    文字列のdateをあれこれする
    """

    def s2d(self, date: str, format: str = "%Y%m%d") -> datetime:
        """
        文字列の日付をdatetime型に変換する
        Parameters
        ----------
        format : str, optional
            日付の型, by default '%Y%m%d'
        Returns
        -------
        parsed_date: datetime
            datetime型の日付
        """
        parsed_date = datetime.strptime(date, format)
        return parsed_date

    def d2s(self, date: datetime, format: str = "%Y%m%d") -> str:
        """
        datetime型の日付を文字列に変換
        Parameters
        ----------
        date : datetime
            日付
        format : str, optional
            日付の型, by default '%Y%m%d'
        Returns
        -------
        date_str: str
            文字列の日付
        """
        date_str = date.strftime(format)
        return date_str

    def add(
        self,
        date: str,
        format: str = "%Y%m%d",
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
    ) -> str:
        """
        日付を加算する
        Parameters
        ----------
        date : str
            基準とする日付
        format : str, optional
            日付の型, by default '%Y%m%d'
        years : int, optional
            送らせる年数, by default 0
        months : int, optional
            送らせる月数, by default 0
        weeks : int, optional
            送らせる週数, by default 0
        days : int, optional
            送らせる日数, by default 0
        Returns
        -------
        added_date_str: str
            加算された日付
        """
        parsed_date = self.s2d(date, format=format)
        added_date = parsed_date + relativedelta(
            years=years, months=months, weeks=weeks, days=days
        )
        added_date_str = self.d2s(added_date, format=format)
        return added_date_str

    def truncate(self, date: str, level: str, format: str = "%Y%m%d") -> str:
        """
        日付を切り捨てる
        Parameters
        ----------
        date : str
            日付
        level : str
            どの単位で切り捨てるか `month` or `year`
        format : str, optional
            [description], by default '%Y%m%d'
        Returns
        -------
        trancated_date_str: str
            切り捨てた日付
        """
        parsed_date = self.s2d(date=date, format="%Y%m%d")

        if level == "month":
            trancated_date = parsed_date.replace(day=1)

        elif level == "year":
            trancated_date = parsed_date.replace(month=1, day=1)

        trancated_date_str = self.d2s(trancated_date, format=format)
        return trancated_date_str

    def today(self, format: str = "%Y%m%d") -> str:
        """
        今日の日付を取得
        Parameters
        ----------
        format : str, optional
            日付の型, by default '%Y%m%d'
        Returns
        -------
        today_str: str
            今日の日付
        """
        today = datetime.now()
        today_str = self.d2s(date=today, format=format)
        return today_str

    def diff(self, date1: str, date2: str) -> int:
        """
        日付間の日数を計算
        Parameters
        ----------
        date1 : str
            日付1
        date2 : str
            日付2
        Returns
        -------
        diff_days: int
            日付間の日数
        """
        datetime1 = self.s2d(date1)
        datetime2 = self.s2d(date2)

        diff_datetime = datetime1 - datetime2
        diff_days = diff_datetime.days
        return diff_days

    def change_format(
        self, date: str, in_format: str = "%Y%m%d", out_format: str = "%Y%m%d"
    ) -> str:
        parsed_date = self.s2d(date, format=in_format)
        parsed_date_str = self.d2s(parsed_date, format=out_format)
        return parsed_date_str