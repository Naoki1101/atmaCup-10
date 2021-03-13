import sys
import logging
import dataclasses
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm.callback import _format_eval_result
import optuna.integration.lightgbm as lgb_tuner

sys.path.append("../src")
import factory
from .base import Model


@dataclasses.dataclass
class LightGBM(Model):
    cfg: Dict

    def fit(
        self,
        tr_x: pd.DataFrame,
        tr_y: pd.Series,
        va_x: Optional[pd.DataFrame] = None,
        va_y: Optional[pd.Series] = None,
        cat_features: List = [],
    ) -> None:
        """
        LightGBMの学習を行う
        Parameters
        ----------
        tr_x : pd.DataFrame
            学習データの説明変数
        tr_y : pd.Series
            学習データの目的変数
        va_x : pd.DataFrame, optional
            評価用データの説明変数, by default None
        va_y : pd.Series, optional
            評価用データの目的変数, by default None
        cat_features : List, optional
            カテゴリ変数をまとめたリスト, by default []
        """
        validation = va_x is not None
        lgb_train = lgb.Dataset(tr_x, tr_y, categorical_feature=cat_features)
        if validation:
            lgb_eval = lgb.Dataset(
                va_x, va_y, reference=lgb_train, categorical_feature=cat_features
            )

        fobj = factory.get_lgb_objective(self.cfg.fobj)
        feval = factory.get_lgb_feval(self.cfg.feval)

        callbacks = [self._log_evaluation(period=100)]

        if self.cfg.task_type in ["binary", "classification", "regression"]:
            self.model = lgb.train(
                self.cfg.params,
                lgb_train,
                num_boost_round=self.cfg.num_boost_round,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=self.cfg.verbose_eval,
                early_stopping_rounds=self.cfg.early_stopping_rounds,
                callbacks=callbacks,
                fobj=fobj,
                feval=feval,
            )

        elif self.cfg.task_type == "optuna":
            self.model = lgb_tuner.train(
                self.cfg.params,
                lgb_train,
                num_boost_round=self.cfg.num_boost_round,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=self.cfg.verbose_eval,
                early_stopping_rounds=self.cfg.early_stopping_rounds,
                callbacks=callbacks,
            )

            print("Best params:", self.model.params)
            print("  Params: ")
            for key, value in self.model.params.items():
                print("    {}: {}".format(key, value))

    def predict(self, te_x: pd.DataFrame, cat_features: List = []) -> np.array:
        """
        LightGBMの推論を行う
        Parameters
        ----------
        te_x : pd.DataFrame
            テストデータの説明変数
        cat_features : List, optional
            カテゴリ変数をまとめたリスト, by default []
        Returns
        -------
        predict_array: np.array
            推論結果
        """
        predict_array = self.model.predict(
            te_x, num_iteration=self.model.best_iteration
        )
        predict_array = predict_array.reshape(len(predict_array), -1)
        return predict_array

    def extract_importances(self) -> np.array:
        """
        特徴量重要度を取得
        Returns
        -------
        feature_importance: np.array
            特徴量重要度
        """
        feature_importance = self.model.feature_importance(
            importance_type=self.cfg.imp_type
        )
        return feature_importance

    def _log_evaluation(
        self, period: int = 1, show_stdv: bool = True, level=logging.DEBUG
    ):
        logger = logging.getLogger("main")

        def _callback(env):
            if (
                period > 0
                and env.evaluation_result_list
                and (env.iteration + 1) % period == 0
            ):
                result = "\t".join(
                    [
                        _format_eval_result(x, show_stdv)
                        for x in env.evaluation_result_list
                    ]
                )
                logger.log(level, "[{}]\t{}".format(env.iteration + 1, result))

        _callback.order = 10
        return _callback


def lightgbm(cfg):
    model = LightGBM(cfg)
    return model
