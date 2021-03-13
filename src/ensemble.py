import argparse
import datetime
import warnings
import numpy as np
from pathlib import Path

import const
import factory
from trainer import opt_ensemble_weight
from utils import DataHandler, Notificator, Timer, make_submission, seed_everything, Git

warnings.filterwarnings("ignore")

# ===============
# Settings
# ===============
parser = argparse.ArgumentParser()
parser.add_argument("--common", default="../configs/common/compe.yml")
parser.add_argument("--notify", default="../configs/common/notify.yml")
parser.add_argument("-m", "--model")
parser.add_argument("-c", "--comment")
options = parser.parse_args()

dh = DataHandler()
cfg = dh.load(options.common)
cfg.update(dh.load(f"../configs/exp/{options.model}.yml"))

notify_params = dh.load(options.notify)

comment = options.comment
model_name = options.model
now = datetime.datetime.now()
run_name = f"{model_name}_{now:%Y%m%d%H%M%S}"

logger_path = Path(f"../logs/{run_name}")


# ===============
# Main
# ===============
def main():
    t = Timer()
    seed_everything(cfg.common.seed)

    logger_path.mkdir(exist_ok=True)

    dh.save(logger_path / "config.yml", cfg)

    with t.timer("load data"):
        train_df = dh.load("../data/input/train.feather")
        test_df = dh.load("../data/input/test.feather")

    with t.timer("drop index"):
        drop_idx = None
        if cfg.common.drop is not None:
            drop_idx = factory.get_drop_idx(cfg.common.drop)
            train_df = train_df.drop(drop_idx, axis=0).reset_index(drop=True)

    with t.timer("load oof and preds"):
        oof_list = []
        preds_list = []

        for i, log_name in enumerate(sorted(cfg.models)):
            model_oof, model_preds = factory.get_result(log_name, cfg)

            if drop_idx is not None:
                model_oof = np.delete(model_oof, drop_idx, axis=0)

            oof_list.append(model_oof)
            preds_list.append(model_preds)

    with t.timer("optimize model weight"):
        metric = factory.get_metrics(cfg.common.metrics.name)
        y_true = train_df[const.TARGET_COLS].values

        best_weight = opt_ensemble_weight(cfg, y_true, oof_list, metric)
        best_weight_array = best_weight
        print(best_weight_array)

    with t.timer("ensemble"):
        ensemble_oof = np.zeros(len(train_df))
        ensemble_preds = np.zeros(len(test_df))

        for model_idx, weight in enumerate(best_weight_array):
            ensemble_oof += oof_list[model_idx] * weight
            ensemble_preds += preds_list[model_idx] * weight

        dh.save(f"../logs/{run_name}/oof.npy", ensemble_oof)
        dh.save(f"../logs/{run_name}/raw_preds.npy", ensemble_preds)
        dh.save(f"../logs/{run_name}/best_weight.npy", best_weight_array)

        cv = metric(y_true, ensemble_oof)
        run_name_cv = f"{run_name}_{cv:.4f}"
        logger_path.rename(f"../logs/{run_name_cv}")

        print("\n\n===================================\n")
        print(f"CV: {cv:.4f}")
        print("\n===================================\n\n")

    with t.timer("make submission"):
        output_path = f"../data/output/{run_name_cv}.csv"
        make_submission(
            y_pred=ensemble_preds,
            target_name=const.TARGET_COLS[0],
            sample_path=const.SAMPLE_SUB_PATH,
            output_path=output_path,
            comp=False,
        )

    with t.timer("notify"):
        process_minutes = t.get_processing_time()
        notificator = Notificator(
            run_name=run_name_cv,
            model_name=model_name,
            cv=round(cv, 4),
            process_time=round(process_minutes, 2),
            comment=comment,
            params=notify_params,
        )
        notificator.send_line()
        notificator.send_notion()

    with t.timer("git"):
        git = Git()
        git.push(comment=run_name_cv)


if __name__ == "__main__":
    main()
